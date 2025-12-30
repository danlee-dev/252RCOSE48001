"""
Advanced AI Analysis API Endpoints
- 고급 AI 분석 기능을 위한 API 엔드포인트
- Stress Test, Redlining, Reasoning Trace 등 개별 기능 접근 제공
"""

from typing import Dict, Any, Optional, List
from fastapi import APIRouter, Depends, HTTPException, Query, Body
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from app.core.database import get_db
from app.api import deps
from app.models.user import User
from app.models.contract import Contract

# AI 모듈 imports
from app.ai.legal_stress_test import LegalStressTest
from app.ai.redlining import GenerativeRedlining
from app.ai.judge import LLMJudge
from app.ai.reasoning_trace import ReasoningTracer
from app.ai.pii_masking import PIIMasker
from app.ai.hyde import HyDEGenerator
from app.ai.constitutional_ai import ConstitutionalAI


router = APIRouter()


# -------------------------------------------------------------------------
# Request/Response Models
# -------------------------------------------------------------------------

class StressTestRequest(BaseModel):
    """Legal Stress Test 요청"""
    contract_text: str = Field(..., description="분석할 계약서 텍스트")
    monthly_hours: Optional[int] = Field(209, description="월 근로시간 (기본값: 209시간)")
    include_details: bool = Field(True, description="상세 분석 포함 여부")


class StressTestResponse(BaseModel):
    """Legal Stress Test 응답"""
    violations: List[Dict[str, Any]]
    total_underpayment: float
    annual_underpayment: float
    risk_score: float
    summary: str


class RedliningRequest(BaseModel):
    """Redlining 요청"""
    contract_text: str = Field(..., description="분석할 계약서 텍스트")
    output_format: str = Field("json", description="출력 형식: json, diff, html")


class RedliningResponse(BaseModel):
    """Redlining 응답"""
    changes: List[Dict[str, Any]]
    change_count: int
    high_risk_count: int
    revised_text: Optional[str] = None
    diff_view: Optional[str] = None


class JudgeRequest(BaseModel):
    """LLM Judge 요청"""
    analysis: str = Field(..., description="평가할 AI 분석 결과")
    context: str = Field("", description="분석에 사용된 컨텍스트")


class JudgeResponse(BaseModel):
    """LLM Judge 응답"""
    overall_score: float
    confidence_level: str
    is_reliable: bool
    verdict: str
    scores: Dict[str, float]
    recommendations: List[str]


class PIIMaskRequest(BaseModel):
    """PII 마스킹 요청"""
    text: str = Field(..., description="마스킹할 텍스트")
    format_preserving: bool = Field(True, description="형식 유지 마스킹")


class PIIMaskResponse(BaseModel):
    """PII 마스킹 응답"""
    masked_text: str
    pii_count: int
    pii_types: Dict[str, int]


class HyDERequest(BaseModel):
    """HyDE 검색 강화 요청"""
    query: str = Field(..., description="검색 쿼리")
    prompt_type: str = Field("labor_law", description="프롬프트 유형: labor_law, contract, risk")


class HyDEResponse(BaseModel):
    """HyDE 응답"""
    original_query: str
    primary_document: str
    enhanced_query: str


class ConstitutionalReviewRequest(BaseModel):
    """Constitutional AI 리뷰 요청"""
    response: str = Field(..., description="검토할 AI 응답")
    context: str = Field("", description="컨텍스트")


class ConstitutionalReviewResponse(BaseModel):
    """Constitutional AI 응답"""
    original_response: str
    revised_response: str
    was_revised: bool
    critique: str
    principles_violated: List[str]


# -------------------------------------------------------------------------
# API Endpoints
# -------------------------------------------------------------------------

@router.post("/stress-test",
             response_model=StressTestResponse,
             summary="Legal Stress Test - 수치 시뮬레이션",
             description="계약서의 임금, 근로시간 조건을 수치 시뮬레이션하여 법적 위반 여부를 분석합니다.")
async def run_stress_test(
    request: StressTestRequest,
    current_user: User = Depends(deps.get_current_user)
):
    """
    Neuro-Symbolic AI 기반 수치 시뮬레이션

    - 최저임금 미달 검사
    - 연장근로수당 계산
    - 주휴수당 검증
    - 연차휴가 일수 검증
    """
    try:
        stress_test = LegalStressTest()
        result = stress_test.run(request.contract_text)

        return StressTestResponse(
            violations=result.violations,
            total_underpayment=result.total_underpayment,
            annual_underpayment=result.annual_underpayment,
            risk_score=result.risk_score,
            summary=result.summary
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Stress Test 실행 오류: {str(e)}")


@router.post("/redlining",
             response_model=RedliningResponse,
             summary="Generative Redlining - 수정 제안",
             description="계약서의 위험 조항을 탐지하고 수정 제안을 생성합니다.")
async def run_redlining(
    request: RedliningRequest,
    current_user: User = Depends(deps.get_current_user)
):
    """
    Generative AI 기반 계약서 수정 제안

    - 위험 조항 탐지
    - 법적 근거 기반 수정안 생성
    - Git-style Diff 출력 지원
    """
    try:
        redliner = GenerativeRedlining()
        result = redliner.redline(request.contract_text)

        response = RedliningResponse(
            changes=[
                {
                    "type": c.change_type.value,
                    "original": c.original_text,
                    "revised": c.revised_text,
                    "reason": c.reason,
                    "legal_basis": c.legal_basis,
                    "severity": c.severity
                }
                for c in result.changes
            ],
            change_count=result.change_count,
            high_risk_count=len(result.high_risk_changes)
        )

        if request.output_format == "diff":
            response.diff_view = result.get_unified_diff()
        elif request.output_format == "html":
            response.diff_view = result.get_html_diff()

        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Redlining 실행 오류: {str(e)}")


@router.post("/judge",
             response_model=JudgeResponse,
             summary="LLM-as-a-Judge - 신뢰도 평가",
             description="AI 분석 결과의 신뢰도를 평가합니다.")
async def run_judge(
    request: JudgeRequest,
    current_user: User = Depends(deps.get_current_user)
):
    """
    LLM-as-a-Judge 신뢰도 평가

    - 정확성, 일관성, 완전성, 관련성, 법적 근거 평가
    - 팩트 체크 (선택적)
    - 신뢰도 배지 생성
    """
    try:
        judge = LLMJudge(strict_mode=True)
        result = judge.evaluate(request.analysis, request.context)

        return JudgeResponse(
            overall_score=result.overall_score,
            confidence_level=result.confidence_level,
            is_reliable=result.is_reliable,
            verdict=result.verdict,
            scores=result.get_score_summary(),
            recommendations=result.recommendations
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Judge 평가 오류: {str(e)}")


@router.post("/pii-mask",
             response_model=PIIMaskResponse,
             summary="PII Masking - 개인정보 비식별화",
             description="텍스트에서 개인정보를 탐지하고 마스킹합니다.")
async def mask_pii(
    request: PIIMaskRequest,
    current_user: User = Depends(deps.get_current_user)
):
    """
    Privacy-Preserving AI

    - 주민등록번호, 전화번호, 이메일, 계좌번호 등 탐지
    - 형식 유지 마스킹 (Format-Preserving)
    - 역마스킹 지원
    """
    try:
        masker = PIIMasker()
        result = masker.mask(request.text, format_preserving=request.format_preserving)

        pii_types = {}
        for pii in result.pii_entities:
            pii_type = pii["type"]
            pii_types[pii_type] = pii_types.get(pii_type, 0) + 1

        return PIIMaskResponse(
            masked_text=result.masked_text,
            pii_count=len(result.pii_entities),
            pii_types=pii_types
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PII 마스킹 오류: {str(e)}")


@router.post("/hyde",
             response_model=HyDEResponse,
             summary="HyDE - 검색 쿼리 강화",
             description="구어체 질문을 법률 전문 용어로 변환하여 검색을 강화합니다.")
async def enhance_query(
    request: HyDERequest,
    current_user: User = Depends(deps.get_current_user)
):
    """
    Hypothetical Document Embeddings

    - 구어체 -> 법률 전문 용어 변환
    - 가상의 완벽한 법률 답변 생성
    - Semantic Gap 해소
    """
    try:
        hyde = HyDEGenerator()
        result = hyde.generate(request.query, prompt_type=request.prompt_type)

        return HyDEResponse(
            original_query=result.original_query,
            primary_document=result.primary_document,
            enhanced_query=hyde.enhance_query(request.query)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"HyDE 처리 오류: {str(e)}")


@router.post("/constitutional-review",
             response_model=ConstitutionalReviewResponse,
             summary="Constitutional AI - 헌법적 검토",
             description="AI 응답이 노동법 원칙에 부합하는지 검토하고 수정합니다.")
async def constitutional_review(
    request: ConstitutionalReviewRequest,
    current_user: User = Depends(deps.get_current_user)
):
    """
    Constitutional AI (Anthropic RLAIF)

    - 노동법 원칙 준수 검토
    - Critique and Revise 워크플로우
    - 원칙 위반 시 자동 수정
    """
    try:
        constitutional_ai = ConstitutionalAI()
        result = constitutional_ai.review(request.response, request.context)

        return ConstitutionalReviewResponse(
            original_response=result.original_response,
            revised_response=result.revised_response,
            was_revised=result.was_revised,
            critique=result.critique,
            principles_violated=result.principles_violated
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Constitutional AI 검토 오류: {str(e)}")


@router.get("/contract/{contract_id}/reasoning-trace",
            summary="Reasoning Trace - 추론 과정 시각화",
            description="계약서 분석의 추론 과정을 시각화합니다.")
async def get_reasoning_trace(
    contract_id: int,
    format: str = Query("mermaid", description="출력 형식: mermaid, d3, cytoscape"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(deps.get_current_user)
):
    """
    Explainable AI (XAI)

    - 추론 과정 그래프 생성
    - Mermaid, D3.js, Cytoscape.js 형식 지원
    - 근거 문서 연결 시각화
    """
    # 계약서 조회
    stmt = select(Contract).where(
        Contract.id == contract_id,
        Contract.user_id == current_user.id
    )
    result = await db.execute(stmt)
    contract = result.scalar_one_or_none()

    if not contract:
        raise HTTPException(status_code=404, detail="계약서를 찾을 수 없습니다.")

    if not contract.analysis_result:
        raise HTTPException(status_code=400, detail="분석 결과가 없습니다.")

    # Reasoning Trace 추출
    analysis_result = contract.analysis_result
    reasoning_trace = analysis_result.get("reasoning_trace")

    if not reasoning_trace:
        raise HTTPException(status_code=400, detail="Reasoning Trace가 없습니다.")

    if format == "mermaid":
        return {"format": "mermaid", "diagram": reasoning_trace.get("mermaid_diagram", "")}
    elif format == "d3":
        return {"format": "d3", "data": reasoning_trace.get("d3_data", {})}
    elif format == "cytoscape":
        return {"format": "cytoscape", "elements": reasoning_trace.get("cytoscape_data", {})}
    else:
        return {"format": "json", "trace": reasoning_trace}


@router.get("/contract/{contract_id}/analysis-detail",
            summary="상세 분석 결과 조회",
            description="계약서의 상세 분석 결과를 조회합니다.")
async def get_analysis_detail(
    contract_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(deps.get_current_user)
):
    """
    상세 분석 결과 조회

    - Stress Test 결과
    - Redlining 제안
    - Judge 평가
    - Reasoning Trace
    """
    stmt = select(Contract).where(
        Contract.id == contract_id,
        Contract.user_id == current_user.id
    )
    result = await db.execute(stmt)
    contract = result.scalar_one_or_none()

    if not contract:
        raise HTTPException(status_code=404, detail="계약서를 찾을 수 없습니다.")

    if not contract.analysis_result:
        raise HTTPException(status_code=400, detail="분석 결과가 없습니다.")

    return {
        "contract_id": contract.id,
        "title": contract.title,
        "status": contract.status,
        "risk_level": contract.risk_level,
        "analysis": contract.analysis_result,
        "created_at": contract.created_at.isoformat() if contract.created_at else None
    }


@router.get("/pipeline-info",
            summary="파이프라인 정보 조회",
            description="Advanced AI Pipeline의 구성 요소 정보를 조회합니다.")
async def get_pipeline_info():
    """
    파이프라인 구성 요소 정보
    """
    return {
        "version": "1.0.0",
        "components": [
            {
                "name": "PII Masking",
                "description": "개인정보 비식별화 (주민등록번호, 전화번호, 이메일 등)",
                "reference": "Privacy-Preserving NLP"
            },
            {
                "name": "HyDE",
                "description": "Hypothetical Document Embeddings - 검색 쿼리 강화",
                "reference": "Gao et al., 2022"
            },
            {
                "name": "RAPTOR",
                "description": "Recursive Abstractive Processing for Tree-Organized Retrieval",
                "reference": "ICLR 2024"
            },
            {
                "name": "Graph-Guided CRAG",
                "description": "Corrective RAG with Knowledge Graph Expansion",
                "reference": "CRAG Paper + Neo4j"
            },
            {
                "name": "Constitutional AI",
                "description": "노동법 원칙 기반 자기 비판 및 수정",
                "reference": "Anthropic RLAIF"
            },
            {
                "name": "Legal Stress Test",
                "description": "Neuro-Symbolic AI 기반 수치 시뮬레이션",
                "reference": "Domain-Specific Neuro-Symbolic"
            },
            {
                "name": "Generative Redlining",
                "description": "위험 조항 탐지 및 수정 제안 생성",
                "reference": "Legal Contract Review"
            },
            {
                "name": "LLM-as-a-Judge",
                "description": "AI 분석 결과 신뢰도 평가",
                "reference": "MT-Bench & Chatbot Arena"
            },
            {
                "name": "Vision RAG",
                "description": "표/이미지 기반 정보 추출 (GPT-4o Vision)",
                "reference": "Multimodal RAG"
            },
            {
                "name": "Reasoning Trace",
                "description": "추론 과정 시각화 (XAI)",
                "reference": "Explainable AI"
            },
            {
                "name": "DSPy Self-Evolution",
                "description": "동적 프롬프트 최적화 및 자가 진화",
                "reference": "Stanford DSPy"
            }
        ],
        "supported_formats": ["PDF", "Image (PNG, JPG)", "Text"],
        "llm_models": {
            "hybrid_mode": True,
            "retrieval": {
                "model": "gemini-2.5-flash-lite",
                "provider": "google",
                "use_case": "document_summarization, search_enhancement"
            },
            "reasoning": {
                "model": "gpt-5-mini",
                "provider": "openai",
                "use_case": "legal_analysis, response_generation"
            },
            "embedding": "nlpai-lab/KURE-v1"
        }
    }
