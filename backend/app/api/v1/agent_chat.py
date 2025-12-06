"""
LangGraph Agent Chat API with SSE Streaming

Endpoints:
- POST /api/v1/agent/chat/{contract_id}/stream - Stream chat response
- POST /api/v1/agent/chat/{contract_id} - Non-streaming chat (fallback)
"""

import json
import asyncio
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from app.core.database import get_db
from app.models.contract import Contract
from app.models.user import User
from app.api.deps import get_current_user
from app.ai.langgraph_agent import get_chat_agent, StreamEvent


router = APIRouter(prefix="/agent", tags=["Agent Chat"])


def build_analysis_summary(analysis_result: dict) -> str:
    """
    Build a structured summary of the contract analysis for agent context.

    This summary allows the AI agent to understand all detected risks,
    legal violations, and suggestions without the user having to repeat them.
    """
    if not analysis_result:
        return ""

    parts = []

    # 1. Overall risk level
    risk_level = analysis_result.get("risk_level", "")
    if risk_level:
        parts.append(f"[전체 위험도] {risk_level}")

    # 2. Underpayment amounts
    stress_test = analysis_result.get("stress_test", {})
    monthly = stress_test.get("total_underpayment", 0)
    annual = stress_test.get("annual_underpayment", 0)
    if monthly or annual:
        parts.append(f"[체불 예상액] 월 {monthly:,}원 / 연 {annual:,}원")

    # 3. Risk violations summary
    violations = stress_test.get("violations", [])
    if violations:
        parts.append(f"\n[발견된 위험 조항 {len(violations)}건]")

        # Group by severity
        high_risks = []
        medium_risks = []
        low_risks = []

        for v in violations:
            clause_num = v.get("clause_number", "")
            v_type = v.get("type", "")
            severity = v.get("severity", "").upper()
            description = v.get("description", "")[:150]
            legal_basis = v.get("legal_basis", "")
            suggestion = v.get("suggestion", "")[:100]

            entry = f"  - [{clause_num}] {v_type}"
            if description:
                entry += f"\n    사유: {description}"
            if legal_basis:
                entry += f"\n    법적근거: {legal_basis}"
            if suggestion:
                entry += f"\n    수정제안: {suggestion}..."

            if severity in ["HIGH", "CRITICAL"]:
                high_risks.append(entry)
            elif severity == "MEDIUM":
                medium_risks.append(entry)
            else:
                low_risks.append(entry)

        if high_risks:
            parts.append(f"\n* HIGH 위험 ({len(high_risks)}건):")
            parts.extend(high_risks[:5])  # Top 5 high risks
            if len(high_risks) > 5:
                parts.append(f"  ... 외 {len(high_risks) - 5}건")

        if medium_risks:
            parts.append(f"\n* MEDIUM 위험 ({len(medium_risks)}건):")
            parts.extend(medium_risks[:3])
            if len(medium_risks) > 3:
                parts.append(f"  ... 외 {len(medium_risks) - 3}건")

        if low_risks:
            parts.append(f"\n* LOW 위험 ({len(low_risks)}건)")

    # 4. Analysis summary text
    summary = analysis_result.get("analysis_summary", "")
    if summary:
        parts.append(f"\n[분석 요약]\n{summary[:300]}")

    return "\n".join(parts)


class ChatRequest(BaseModel):
    message: str
    conversation_id: Optional[str] = None
    include_contract: bool = True


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatHistoryRequest(BaseModel):
    message: str
    history: list[ChatMessage] = []
    include_contract: bool = True


@router.post("/{contract_id}/stream")
async def stream_chat(
    contract_id: int,
    request: ChatRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Stream chat response using Server-Sent Events (SSE)

    Event types:
    - step: Current processing step (analyzing, generating)
    - tool: Tool execution status (searching, complete)
    - token: Response token for streaming text
    - done: Response complete with full text
    - error: Error occurred
    """

    # Get contract
    result = await db.execute(
        select(Contract).where(
            Contract.id == contract_id,
            Contract.user_id == current_user.id
        )
    )
    contract = result.scalar_one_or_none()

    if not contract:
        raise HTTPException(status_code=404, detail="Contract not found")

    # Get contract text
    contract_text = ""
    if request.include_contract and contract.extracted_text:
        contract_text = contract.extracted_text

    # Get agent
    agent = get_chat_agent()

    async def event_generator():
        """Generate SSE events"""
        try:
            async for event in agent.chat_stream(
                message=request.message,
                contract_text=contract_text,
                contract_id=contract_id
            ):
                yield event.to_sse()
                await asyncio.sleep(0.01)  # Small delay for smooth streaming

        except Exception as e:
            error_event = StreamEvent("error", {"message": str(e)})
            yield error_event.to_sse()

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
        }
    )


@router.post("/{contract_id}/stream/history")
async def stream_chat_with_history(
    contract_id: int,
    request: ChatHistoryRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Stream chat response with conversation history
    """

    # Get contract
    result = await db.execute(
        select(Contract).where(
            Contract.id == contract_id,
            Contract.user_id == current_user.id
        )
    )
    contract = result.scalar_one_or_none()

    if not contract:
        raise HTTPException(status_code=404, detail="Contract not found")

    # Get contract text
    contract_text = ""
    if request.include_contract and contract.extracted_text:
        contract_text = contract.extracted_text

    # Build comprehensive analysis summary for agent context
    analysis_summary = ""
    if contract.analysis_result:
        analysis_summary = build_analysis_summary(contract.analysis_result)
        print(f">>> [agent_chat] Analysis summary length: {len(analysis_summary)} chars")

    # Convert history
    chat_history = [{"role": msg.role, "content": msg.content} for msg in request.history]

    # Get agent
    agent = get_chat_agent()

    async def event_generator():
        """Generate SSE events"""
        try:
            async for event in agent.chat_stream(
                message=request.message,
                contract_text=contract_text,
                contract_id=contract_id,
                chat_history=chat_history,
                analysis_summary=analysis_summary
            ):
                yield event.to_sse()
                await asyncio.sleep(0.01)

        except Exception as e:
            error_event = StreamEvent("error", {"message": str(e)})
            yield error_event.to_sse()

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )


@router.post("/{contract_id}")
async def chat_sync(
    contract_id: int,
    request: ChatRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Non-streaming chat endpoint (fallback for clients that don't support SSE)
    """

    # Get contract
    result = await db.execute(
        select(Contract).where(
            Contract.id == contract_id,
            Contract.user_id == current_user.id
        )
    )
    contract = result.scalar_one_or_none()

    if not contract:
        raise HTTPException(status_code=404, detail="Contract not found")

    # Get contract text
    contract_text = ""
    if request.include_contract and contract.extracted_text:
        contract_text = contract.extracted_text

    # Get agent and collect response
    agent = get_chat_agent()

    full_response = ""
    tool_results = []

    async for event in agent.chat_stream(
        message=request.message,
        contract_text=contract_text,
        contract_id=contract_id
    ):
        if event.event_type == "token":
            full_response += event.data.get("content", "")
        elif event.event_type == "tool":
            if event.data.get("status") == "complete":
                tool_results.append(event.data.get("tool", ""))
        elif event.event_type == "done":
            full_response = event.data.get("full_response", full_response)

    return {
        "answer": full_response,
        "tools_used": tool_results,
        "contract_id": contract_id
    }


# Health check for agent
@router.get("/health")
async def agent_health():
    """Check if agent is ready"""
    try:
        agent = get_chat_agent()
        return {"status": "healthy", "agent": "ready"}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}
