"""
채팅 API - Dify Agent 연동
계약서 분석 결과를 기반으로 사용자와 대화하는 에이전트
"""

import os
import json
import requests
from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, AsyncGenerator
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from app.api.deps import get_db
from app.models.contract import Contract

router = APIRouter()

# Dify API 설정
DIFY_API_URL = os.getenv("DIFY_API_URL", "https://api.dify.ai/v1")
# DIFY_API_KEY 또는 DIFY_CHAT_API_KEY 사용 (호환성)
DIFY_CHAT_API_KEY = os.getenv("DIFY_CHAT_API_KEY") or os.getenv("DIFY_API_KEY", "")


class ChatRequest(BaseModel):
    """채팅 요청"""
    message: str = Field(..., description="사용자 메시지", example="포괄임금제가 왜 위험한가요?")
    conversation_id: Optional[str] = Field(None, description="대화 ID (이어서 대화할 때)")


class ChatResponse(BaseModel):
    """채팅 응답"""
    answer: str = Field(..., description="AI 답변")
    conversation_id: str = Field(..., description="대화 ID")
    message_id: str = Field(..., description="메시지 ID")
    sources: Optional[List[Dict[str, Any]]] = Field(None, description="참고 자료")


class ConversationInfo(BaseModel):
    """대화 정보"""
    id: str
    name: str
    created_at: int


@router.post(
    "/{contract_id}",
    response_model=ChatResponse,
    summary="계약서 기반 채팅",
    description="""
    특정 계약서의 분석 결과를 컨텍스트로 하여 Dify Agent와 대화합니다.

    첫 메시지: conversation_id를 비워두면 새 대화가 시작됩니다.
    이어서 대화: 이전 응답의 conversation_id를 전달하면 대화가 이어집니다.
    """
)
async def chat_with_agent(
    contract_id: int,
    request: ChatRequest,
    db: AsyncSession = Depends(get_db)
):
    """계약서 기반 채팅"""

    # 1. 계약서 조회
    stmt = select(Contract).where(Contract.id == contract_id)
    result = await db.execute(stmt)
    contract = result.scalar_one_or_none()

    if not contract:
        raise HTTPException(status_code=404, detail="계약서를 찾을 수 없습니다")

    if contract.status != "COMPLETED":
        raise HTTPException(status_code=400, detail="분석이 완료되지 않은 계약서입니다")

    # 2. Dify API 호출
    if not DIFY_CHAT_API_KEY:
        raise HTTPException(status_code=500, detail="DIFY_CHAT_API_KEY가 설정되지 않았습니다")

    # 분석 결과를 JSON 문자열로 변환 (토큰 절약을 위해 일부만)
    analysis_summary = ""
    if contract.analysis_result:
        analysis_data = contract.analysis_result
        # 주요 정보만 추출
        summary_parts = []
        if "summary" in analysis_data:
            summary_parts.append(f"요약: {analysis_data['summary']}")
        if "risk_level" in analysis_data:
            summary_parts.append(f"위험도: {analysis_data['risk_level']}")
        if "risk_clauses" in analysis_data:
            clauses = analysis_data["risk_clauses"][:5]  # 상위 5개만
            for c in clauses:
                summary_parts.append(f"- [{c.get('level', 'N/A')}] {c.get('text', '')[:100]}")
        analysis_summary = "\n".join(summary_parts)

    # 계약서 원문 (처음 2000자만)
    extracted_text = contract.extracted_text[:2000] if contract.extracted_text else ""

    payload = {
        "inputs": {
            "analysis_result": analysis_summary,
            "extracted_text": extracted_text
        },
        "query": request.message,
        "user": f"user_{contract.user_id}",
        "response_mode": "blocking",
        "auto_generate_name": True
    }

    if request.conversation_id:
        payload["conversation_id"] = request.conversation_id

    headers = {
        "Authorization": f"Bearer {DIFY_CHAT_API_KEY}",
        "Content-Type": "application/json"
    }

    try:
        response = requests.post(
            f"{DIFY_API_URL}/chat-messages",
            headers=headers,
            json=payload,
            timeout=60
        )
        response.raise_for_status()
        dify_response = response.json()

        # 참고 자료 추출
        sources = None
        if "metadata" in dify_response and "retriever_resources" in dify_response["metadata"]:
            sources = dify_response["metadata"]["retriever_resources"]

        return ChatResponse(
            answer=dify_response.get("answer", ""),
            conversation_id=dify_response.get("conversation_id", ""),
            message_id=dify_response.get("message_id", dify_response.get("id", "")),
            sources=sources
        )

    except requests.exceptions.Timeout:
        raise HTTPException(status_code=504, detail="Dify API 응답 시간 초과")
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=502, detail=f"Dify API 호출 실패: {str(e)}")


@router.get(
    "/{contract_id}/conversations",
    summary="대화 목록 조회",
    description="해당 계약서와 관련된 대화 목록을 조회합니다."
)
async def list_conversations(
    contract_id: int,
    db: AsyncSession = Depends(get_db)
):
    """대화 목록 조회 (Dify API 활용)"""

    # 계약서 확인
    stmt = select(Contract).where(Contract.id == contract_id)
    result = await db.execute(stmt)
    contract = result.scalar_one_or_none()

    if not contract:
        raise HTTPException(status_code=404, detail="계약서를 찾을 수 없습니다")

    if not DIFY_CHAT_API_KEY:
        raise HTTPException(status_code=500, detail="DIFY_CHAT_API_KEY가 설정되지 않았습니다")

    headers = {
        "Authorization": f"Bearer {DIFY_CHAT_API_KEY}",
        "Content-Type": "application/json"
    }

    try:
        response = requests.get(
            f"{DIFY_API_URL}/conversations",
            headers=headers,
            params={
                "user": f"user_{contract.user_id}",
                "limit": 20
            },
            timeout=30
        )
        response.raise_for_status()
        return response.json()

    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=502, detail=f"Dify API 호출 실패: {str(e)}")


@router.get(
    "/{contract_id}/conversations/{conversation_id}/messages",
    summary="대화 내역 조회",
    description="특정 대화의 메시지 내역을 조회합니다."
)
async def get_conversation_messages(
    contract_id: int,
    conversation_id: str,
    db: AsyncSession = Depends(get_db)
):
    """대화 내역 조회"""

    # 계약서 확인
    stmt = select(Contract).where(Contract.id == contract_id)
    result = await db.execute(stmt)
    contract = result.scalar_one_or_none()

    if not contract:
        raise HTTPException(status_code=404, detail="계약서를 찾을 수 없습니다")

    if not DIFY_CHAT_API_KEY:
        raise HTTPException(status_code=500, detail="DIFY_CHAT_API_KEY가 설정되지 않았습니다")

    headers = {
        "Authorization": f"Bearer {DIFY_CHAT_API_KEY}",
        "Content-Type": "application/json"
    }

    try:
        response = requests.get(
            f"{DIFY_API_URL}/messages",
            headers=headers,
            params={
                "user": f"user_{contract.user_id}",
                "conversation_id": conversation_id,
                "limit": 100
            },
            timeout=30
        )
        response.raise_for_status()
        return response.json()

    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=502, detail=f"Dify API 호출 실패: {str(e)}")
