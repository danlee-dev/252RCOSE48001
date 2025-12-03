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

    # Get analysis result summary for search query context
    analysis_summary = ""
    if contract.analysis_result:
        try:
            result = contract.analysis_result
            # Extract key violations for context
            if "stress_test" in result and "violations" in result["stress_test"]:
                violations = result["stress_test"]["violations"][:5]
                violation_texts = []
                for v in violations:
                    v_type = v.get("type", "")
                    v_desc = v.get("description", "")[:80]
                    violation_texts.append(f"- {v_type}: {v_desc}")
                if violation_texts:
                    analysis_summary = "발견된 위험 조항:\n" + "\n".join(violation_texts)
        except Exception:
            pass

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
