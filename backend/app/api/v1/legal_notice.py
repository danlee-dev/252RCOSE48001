import uuid
import json
from datetime import datetime
from typing import List, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

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
    EvidenceGuideResponse, ViolationEvidenceGuide
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

    # Agent에게 분석 결과 전달하여 가이드 생성
    guide_text = agent.generate_evidence_guide(contract.analysis_result)
    
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
    if contract and contract.analysis_result:
        # 분석 결과를 문자열로 요약 (너무 길면 잘릴 수 있으니 3000자로 제한)
        analysis_summary = json.dumps(contract.analysis_result, ensure_ascii=False)[:3000]
    
    # Agent에게 작성 요청
    final_content = agent.write_legal_notice(
        collected_info=session.collected_info,
        analysis_summary=analysis_summary
    )
    
    session.final_content = final_content
    session.status = "completed"
    
    await db.commit()
    
    return LegalNoticeResultResponse(
        title=f"내용증명서_{session.collected_info.get('sender_name', '본인')}",
        content=final_content,
        generated_at=datetime.now()
    )