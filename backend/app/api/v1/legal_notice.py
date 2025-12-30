import uuid
import json
from datetime import datetime
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, desc

from app.core.database import get_db
from app.api.deps import get_current_user
from app.models.user import User

from app.models.contract import Contract
from app.models.legal_notice import LegalNoticeSession

from app.ai.legal_notice_agent import LegalNoticeAgent
from app.schemas.legal_notice import (
    LegalNoticeStartRequest, LegalNoticeSessionResponse,
    LegalNoticeChatRequest, LegalNoticeChatResponse,
    LegalNoticeGenerateRequest, LegalNoticeResultResponse,
    EvidenceGuideResponse, ViolationEvidenceGuide,
    LegalNoticeSessionListItem, LegalNoticeSessionListResponse,
    LegalNoticeSessionDetail
)

router = APIRouter()
agent = LegalNoticeAgent()

@router.get(
    "/evidence-guide/{contract_id}",
    response_model=EvidenceGuideResponse,
    summary="증거 수집 가이드 조회",
    description="""
    특정 계약서(`contract_id`)의 AI 분석 결과(위반 사항)를 바탕으로, 
    입증에 필요한 **필수/보조 증거 자료 목록**과 **구체적인 수집 방법(꿀팁)**을 상세 가이드로 생성하여 반환함.
    
    - 계약서 분석이 완료(`COMPLETED`)된 상태여야 함.
    - 분석 결과의 모든 위반 사항을 종합하여 가이드를 작성함.
    """,
    responses={
        404: {"description": "계약서를 찾을 수 없거나 접근 권한이 없음"},
        400: {"description": "계약서 분석이 아직 완료되지 않았음"}
    }
)
async def get_evidence_guide(
    contract_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    
    query = select(Contract).where(Contract.id == contract_id, Contract.user_id == current_user.id)
    result = await db.execute(query)
    contract = result.scalars().first()

    if not contract:
        raise HTTPException(status_code=404, detail="Contract not found")
    
    if not contract.analysis_result:
        raise HTTPException(status_code=400, detail="Analysis not ready")

    # Agent에게 분석 결과 및 계약서 전문 전달하여 가이드 생성
    guide_text = agent.generate_evidence_guide(
        analysis_result=contract.analysis_result,
        contract_text=contract.extracted_text or ""
    )
    
    return EvidenceGuideResponse(
        guides=[
            ViolationEvidenceGuide(
                violation_type="종합 증거 수집 가이드",
                severity="HIGH",
                evidence_list=[], # 텍스트 내에 상세 내용 포함됨
                guide_text=guide_text
            )
        ]
    )

@router.post(
    "/start",
    response_model=LegalNoticeSessionResponse,
    summary="내용증명 작성 세션 시작",
    description="""
    새로운 내용증명 작성 세션을 생성함.
    
    - **기능:** 발신인/수신인 기본 정보를 선택적으로 입력받아 초기화하고, AI의 첫 번째 인사말을 생성함.
    - **반환값:** 생성된 `session_id`와 초기 AI 메시지. 이후 `/chat` API에서 이 `session_id`를 사용해야 함.
    """,
    status_code=status.HTTP_201_CREATED
)
async def start_session(
    request: LegalNoticeStartRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    session_id = str(uuid.uuid4())
    
    # 초기 정보 설정 (요청에 포함된 경우)
    initial_info = {}
    if request.sender_info:
        initial_info.update(request.sender_info.model_dump(exclude_none=True))
    if request.recipient_info:
        initial_info.update(request.recipient_info.model_dump(exclude_none=True))

    # AI의 첫 인사말 설정
    first_message = "안녕하세요. 내용증명 작성을 도와드리겠습니다. 현재 겪고 계신 피해 상황(예: 임금체불, 부당해고 등)을 구체적으로 말씀해 주시겠어요?"

    new_session = LegalNoticeSession(
        id=session_id,
        contract_id=request.contract_id,
        user_id=current_user.id,
        collected_info=initial_info,
        messages=[{
            "role": "ai", 
            "content": first_message
        }],
        status="collecting"
    )
    
    db.add(new_session)
    await db.commit()
    await db.refresh(new_session)
    
    return LegalNoticeSessionResponse(
        session_id=session_id,
        status=new_session.status,
        missing_info=[],
        ai_message=first_message
    )

@router.post(
    "/chat",
    response_model=LegalNoticeChatResponse,
    summary="내용증명 정보 수집 대화",
    description="""
    AI Agent와 대화하며 내용증명 작성에 필요한 필수 정보를 수집함.
    
    - **동작 방식:**
        1. 사용자의 메시지(`user_message`)를 분석하여 이름, 날짜, 피해 사실 등의 정보를 추출(`extracted_info`)하여 저장함.
        2. 아직 수집되지 않은 필수 정보가 있다면, 이를 묻는 다음 질문(`ai_message`)을 생성함.
        3. 모든 필수 정보가 수집되면 `is_complete: true`를 반환함.
    """,
    responses={
        404: {"description": "유효하지 않은 세션 ID"}
    }
)
async def chat_session(
    request: LegalNoticeChatRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):

    query = select(LegalNoticeSession).where(LegalNoticeSession.id == request.session_id)
    result = await db.execute(query)
    session = result.scalars().first()

    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # 1. 사용자 메시지 저장 (리스트 복사 후 추가)
    messages = list(session.messages) 
    messages.append({"role": "user", "content": request.user_message})
    
    # 2. AI Agent 호출하여 정보 추출 및 답변 생성
    result_agent = agent.chat_for_collection(
        current_info=session.collected_info,
        user_message=request.user_message,
        history=messages
    )
    
    # 3. 추출된 정보 업데이트
    if result_agent.get("extracted_info"):
        current_info = dict(session.collected_info)
        current_info.update(result_agent["extracted_info"])
        session.collected_info = current_info
    
    # 4. AI 응답 저장
    ai_message = result_agent.get("ai_message", "계속 진행하겠습니다.")
    messages.append({"role": "ai", "content": ai_message})
    session.messages = messages
    
    # 5. 상태 업데이트 (완료 시)
    if result_agent.get("is_complete"):
        session.status = "generating"
        
    await db.commit()
    
    return LegalNoticeChatResponse(
        session_id=session.id,
        ai_message=ai_message,
        is_complete=result_agent.get("is_complete", False),
        collected_info=session.collected_info
    )

@router.post(
    "/generate",
    response_model=LegalNoticeResultResponse,
    summary="내용증명서 최종 생성",
    description="""
    수집 완료된 정보와 계약서 분석 결과를 종합하여 **최종 내용증명서(Markdown)**를 생성함.
    
    - **전제 조건:** `/chat` API를 통해 `is_complete: true` 상태가 되어야 함 (권장).
    - **생성 내용:** 육하원칙에 따른 피해 사실, 법적 근거, 구체적 요구사항, 법적 조치 예고 등이 포함된 엄중한 어조의 문서.
    """,
    responses={
        404: {"description": "세션을 찾을 수 없음"}
    }
)
async def generate_legal_notice(
    request: LegalNoticeGenerateRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):

    query_session = select(LegalNoticeSession).where(LegalNoticeSession.id == request.session_id)
    result_session = await db.execute(query_session)
    session = result_session.scalars().first()

    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    query_contract = select(Contract).where(Contract.id == session.contract_id)
    result_contract = await db.execute(query_contract)
    contract = result_contract.scalars().first()

    analysis_summary = ""
    analysis_result = {}
    contract_text = ""
    if contract:
        if contract.analysis_result:
            analysis_result = contract.analysis_result
            analysis_summary = json.dumps(contract.analysis_result, ensure_ascii=False)[:3000]
        if contract.extracted_text:
            contract_text = contract.extracted_text

    # Agent에게 내용증명 작성 요청
    final_content = agent.write_legal_notice(
        collected_info=session.collected_info,
        analysis_summary=analysis_summary
    )

    # 증거수집 전략 생성 (분석 결과 + 계약서 전문 + 피해현황 종합)
    evidence_guide = ""
    if analysis_result:
        evidence_guide = agent.generate_evidence_guide(
            analysis_result=analysis_result,
            contract_text=contract_text,
            collected_info=session.collected_info
        )

    session.final_content = final_content
    session.evidence_guide = evidence_guide
    session.status = "completed"

    await db.commit()
    
    return LegalNoticeResultResponse(
        title=f"내용증명서_{session.collected_info.get('sender_name', '본인')}",
        content=final_content,
        generated_at=datetime.now()
    )


# --- 세션 관리 API ---

@router.get(
    "/sessions",
    response_model=LegalNoticeSessionListResponse,
    summary="내용증명 세션 목록 조회",
    description="현재 사용자의 내용증명 세션 목록을 조회합니다."
)
async def list_sessions(
    skip: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
    status_filter: Optional[str] = Query(None, description="상태 필터 (collecting, generating, completed)"),
    search: Optional[str] = Query(None, description="검색어 (제목)"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    # Base query
    base_query = select(LegalNoticeSession).where(
        LegalNoticeSession.user_id == current_user.id
    )

    # Status filter
    if status_filter:
        base_query = base_query.where(LegalNoticeSession.status == status_filter)

    # Search filter
    if search:
        base_query = base_query.where(LegalNoticeSession.title.ilike(f"%{search}%"))

    # Count total
    count_query = select(func.count()).select_from(base_query.subquery())
    total_result = await db.execute(count_query)
    total = total_result.scalar() or 0

    # Get items with pagination
    items_query = base_query.order_by(desc(LegalNoticeSession.created_at)).offset(skip).limit(limit)
    result = await db.execute(items_query)
    sessions = result.scalars().all()

    # Get contract titles
    contract_ids = [s.contract_id for s in sessions if s.contract_id]
    contracts_map = {}
    if contract_ids:
        contracts_query = select(Contract).where(Contract.id.in_(contract_ids))
        contracts_result = await db.execute(contracts_query)
        for c in contracts_result.scalars().all():
            contracts_map[c.id] = c.title

    # Build response items
    items = []
    for session in sessions:
        items.append(LegalNoticeSessionListItem(
            id=session.id,
            title=session.title,
            status=session.status,
            contract_id=session.contract_id,
            contract_title=contracts_map.get(session.contract_id) if session.contract_id else None,
            created_at=session.created_at,
            updated_at=session.updated_at
        ))

    # Calculate stats
    stats_query = select(
        LegalNoticeSession.status,
        func.count(LegalNoticeSession.id)
    ).where(
        LegalNoticeSession.user_id == current_user.id
    ).group_by(LegalNoticeSession.status)

    stats_result = await db.execute(stats_query)
    stats = {"collecting": 0, "generating": 0, "completed": 0, "total": 0}
    for row in stats_result:
        stats[row[0]] = row[1]
        stats["total"] += row[1]

    return LegalNoticeSessionListResponse(
        items=items,
        total=total,
        stats=stats
    )


@router.get(
    "/sessions/{session_id}",
    response_model=LegalNoticeSessionDetail,
    summary="내용증명 세션 상세 조회",
    description="특정 세션의 상세 정보를 조회합니다."
)
async def get_session(
    session_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    query = select(LegalNoticeSession).where(
        LegalNoticeSession.id == session_id,
        LegalNoticeSession.user_id == current_user.id
    )
    result = await db.execute(query)
    session = result.scalars().first()

    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    # Get contract title if exists
    contract_title = None
    if session.contract_id:
        contract_query = select(Contract).where(Contract.id == session.contract_id)
        contract_result = await db.execute(contract_query)
        contract = contract_result.scalars().first()
        if contract:
            contract_title = contract.title

    return LegalNoticeSessionDetail(
        id=session.id,
        title=session.title,
        status=session.status,
        contract_id=session.contract_id,
        contract_title=contract_title,
        messages=session.messages or [],
        collected_info=session.collected_info or {},
        final_content=session.final_content,
        evidence_guide=session.evidence_guide,
        damage_summary=session.damage_summary,
        created_at=session.created_at,
        updated_at=session.updated_at
    )


@router.delete(
    "/sessions/{session_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="내용증명 세션 삭제",
    description="특정 세션을 삭제합니다."
)
async def delete_session(
    session_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    query = select(LegalNoticeSession).where(
        LegalNoticeSession.id == session_id,
        LegalNoticeSession.user_id == current_user.id
    )
    result = await db.execute(query)
    session = result.scalars().first()

    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    await db.delete(session)
    await db.commit()

    return None


@router.patch(
    "/sessions/{session_id}/contract",
    response_model=LegalNoticeSessionDetail,
    summary="세션에 계약서 연결",
    description="진행 중인 세션에 분석된 계약서를 연결합니다."
)
async def link_contract_to_session(
    session_id: str,
    contract_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    # Get session
    session_query = select(LegalNoticeSession).where(
        LegalNoticeSession.id == session_id,
        LegalNoticeSession.user_id == current_user.id
    )
    session_result = await db.execute(session_query)
    session = session_result.scalars().first()

    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    # Verify contract exists and belongs to user
    contract_query = select(Contract).where(
        Contract.id == contract_id,
        Contract.user_id == current_user.id
    )
    contract_result = await db.execute(contract_query)
    contract = contract_result.scalars().first()

    if not contract:
        raise HTTPException(status_code=404, detail="Contract not found")

    # Update session
    session.contract_id = contract_id

    # Generate evidence guide if contract has analysis (preliminary - will be regenerated at /generate)
    if contract.analysis_result:
        session.evidence_guide = agent.generate_evidence_guide(
            analysis_result=contract.analysis_result,
            contract_text=contract.extracted_text or "",
            collected_info=session.collected_info
        )

    await db.commit()
    await db.refresh(session)

    return LegalNoticeSessionDetail(
        id=session.id,
        title=session.title,
        status=session.status,
        contract_id=session.contract_id,
        contract_title=contract.title,
        messages=session.messages or [],
        collected_info=session.collected_info or {},
        final_content=session.final_content,
        evidence_guide=session.evidence_guide,
        damage_summary=session.damage_summary,
        created_at=session.created_at,
        updated_at=session.updated_at
    )