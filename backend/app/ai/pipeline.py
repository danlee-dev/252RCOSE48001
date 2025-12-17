"""
Advanced AI Pipeline (통합 파이프라인) - V2
- LLM 기반 조항 분할 + CRAG 통합 분석
- Graph+Vector DB 검색 결과를 실제 분석에 활용

Pipeline Flow V2:
1. PII Masking (개인정보 비식별화)
2. LLM Clause Extraction (조항 분할 + 값 추출)
3. Clause-by-Clause CRAG Analysis (조항별 법률 검색 + LLM 위반 분석)
4. RAPTOR Indexing (계층적 요약)
5. Constitutional AI Review (메타 평가)
6. LLM-as-a-Judge (신뢰도 평가)
7. Reasoning Trace (추론 시각화)

Legacy (선택적):
- Legal Stress Test (정규식 기반 - 폴백용)
- Generative Redlining (정규식 기반 - 폴백용)
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime
import time

from app.core.config import settings
from app.core.llm_client import get_llm_client, HybridLLMClient, LLMTask
from app.core.pipeline_logger import PipelineLogger, get_pipeline_logger
from app.core.token_usage_tracker import TokenUsageTracker, track_token_usage

# AI 모듈 imports
from .preprocessor import ContractPreprocessor
from .pii_masking import PIIMasker, MaskingResult
from .hyde import HyDEGenerator
from .raptor import RAPTORIndexer, ContractRAPTOR
from .constitutional_ai import ConstitutionalAI, ConstitutionalReview
from .crag import GraphGuidedCRAG, CRAGResult
from .legal_stress_test import LegalStressTest, StressTestResult
from .redlining import GenerativeRedlining, RedlineResult
from .judge import LLMJudge, JudgmentResult
from .reasoning_trace import ReasoningTracer, ReasoningTrace
from .dspy_optimizer import DSPyOptimizer
from .clause_analyzer import LLMClauseAnalyzer, ClauseAnalysisResult, ViolationLocationMapper


@dataclass
class PipelineConfig:
    """파이프라인 설정"""
    # V2: LLM 기반 조항 분석 (권장)
    enable_llm_clause_analysis: bool = True  # LLM 조항 분할 + CRAG 통합 분석

    # 공통
    enable_pii_masking: bool = True
    enable_raptor: bool = True
    enable_constitutional_ai: bool = True
    enable_judge: bool = True
    enable_reasoning_trace: bool = True
    enable_dspy: bool = False  # 기본 비활성화 (성능)

    # Legacy (LLM 분석 비활성화 시 폴백)
    enable_hyde: bool = True
    enable_crag: bool = True
    enable_stress_test: bool = False  # LLM 분석이 대체
    enable_redlining: bool = False    # LLM 분석이 대체

    # 하이브리드 LLM 모델 설정
    retrieval_model: str = settings.LLM_RETRIEVAL_MODEL  # Gemini
    reasoning_model: str = settings.LLM_REASONING_MODEL  # OpenAI
    hyde_model: str = settings.LLM_HYDE_MODEL  # gpt-4o (HyDE 생성용)
    embedding_model: str = "nlpai-lab/KURE-v1"

    # 검색 설정
    search_top_k: int = 5
    crag_max_hops: int = 2


@dataclass
class PipelineResult:
    """파이프라인 실행 결과"""
    contract_id: str
    original_text: str
    masked_text: str = ""

    # 분석 결과
    analysis_summary: str = ""
    risk_level: str = "Low"  # High, Medium, Low
    risk_score: float = 0.0

    # V2: LLM 기반 조항 분석 결과
    clause_analysis: Optional[ClauseAnalysisResult] = None

    # 세부 결과
    pii_masking: Optional[MaskingResult] = None
    stress_test: Optional[StressTestResult] = None  # Legacy 또는 clause_analysis에서 변환
    redlining: Optional[RedlineResult] = None       # Legacy 또는 clause_analysis에서 변환
    constitutional_review: Optional[ConstitutionalReview] = None
    judgment: Optional[JudgmentResult] = None
    reasoning_trace: Optional[ReasoningTrace] = None

    # 컨텍스트
    retrieved_docs: List[Dict[str, Any]] = field(default_factory=list)

    # 메타데이터
    processing_time: float = 0.0
    pipeline_version: str = "2.0.0"  # V2
    timestamp: datetime = field(default_factory=datetime.now)

    # 토큰 사용량 (비용 추적)
    token_usage: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        # V2: clause_analysis 결과를 stress_test 형식으로 변환 (프론트엔드 호환성)
        stress_test_data = None
        redlining_data = None

        if self.clause_analysis and self.clause_analysis.violations:
            # LLM 조항 분석 결과를 기존 형식으로 변환
            stress_test_data = {
                "violations": [
                    {
                        "type": v.violation_type,
                        "severity": v.severity.value,
                        "description": v.description,
                        "legal_basis": v.legal_basis,
                        "current_value": v.current_value,
                        "legal_standard": v.legal_standard,
                        "suggestion": v.suggestion,
                        "suggested_text": v.suggested_text,  # 수정된 조항 텍스트 (에디터용)
                        "clause_number": v.clause.clause_number,
                        "sources": v.crag_sources,
                        "original_text": v.clause.original_text,  # 원본 조항 텍스트 (하이라이팅용)
                        "matched_text": getattr(v, 'matched_text', None),  # 텍스트 기반 하이라이팅용
                        "start_index": v.clause.position.get("start", -1),  # 하이라이팅 시작 위치
                        "end_index": v.clause.position.get("end", -1),  # 하이라이팅 끝 위치
                    }
                    for v in self.clause_analysis.violations
                ],
                "total_underpayment": self.clause_analysis.total_underpayment,
                "annual_underpayment": self.clause_analysis.annual_underpayment
            }
            # Redlining 형식으로도 변환 (수정 제안용)
            redlining_data = {
                "change_count": len(self.clause_analysis.violations),
                "changes": [
                    {
                        "type": "modify",
                        "original": v.clause.original_text[:200],
                        "revised": v.suggestion,
                        "reason": v.description,
                        "severity": v.severity.value
                    }
                    for v in self.clause_analysis.violations
                    if v.suggestion
                ]
            }
        elif self.stress_test:
            # Legacy: 기존 stress_test 결과 사용
            stress_test_data = {
                "violations": [
                    {
                        "type": v["type"],
                        "severity": v["severity"],
                        "description": v["description"],
                        "legal_basis": v.get("legal_basis", ""),
                        "current_value": v.get("current_value"),
                        "legal_standard": v.get("legal_standard")
                    }
                    for v in self.stress_test.violations
                ],
                "total_underpayment": self.stress_test.total_underpayment,
                "annual_underpayment": self.stress_test.annual_underpayment
            }

        if not redlining_data and self.redlining:
            # Legacy: 기존 redlining 결과 사용
            redlining_data = {
                "change_count": self.redlining.change_count,
                "changes": [
                    {
                        "type": c.change_type.value,
                        "original": c.original_text[:100],
                        "revised": c.revised_text[:100],
                        "reason": c.reason,
                        "severity": c.severity
                    }
                    for c in self.redlining.changes
                ]
            }

        return {
            "contract_id": self.contract_id,
            "risk_level": self.risk_level,
            "risk_score": self.risk_score,
            "analysis_summary": self.analysis_summary,
            "pipeline_version": self.pipeline_version,
            "stress_test": stress_test_data,
            "redlining": redlining_data,
            "judgment": {
                "overall_score": self.judgment.overall_score if self.judgment else 0,
                "confidence_level": self.judgment.confidence_level if self.judgment else "Medium",
                "is_reliable": self.judgment.is_reliable if self.judgment else True,
                "verdict": self.judgment.verdict if self.judgment else "",
                "recommendations": self.judgment.recommendations if self.judgment else []
            } if self.judgment else None,
            "constitutional_review": {
                "is_constitutional": self.constitutional_review.is_constitutional if self.constitutional_review else True,
                "has_violations": self.constitutional_review.has_violations if self.constitutional_review else False,
                "high_severity_count": self.constitutional_review.high_severity_count if self.constitutional_review else 0,
                "critiques": [
                    {
                        "principle": c.principle.value,
                        "violation_detected": c.violation_detected,
                        "critique": c.critique,
                        "severity": c.severity,
                        "suggestion": c.suggestion
                    }
                    for c in (self.constitutional_review.critiques if self.constitutional_review else [])
                    if c.violation_detected
                ],
                "revised_response": self.constitutional_review.revised_response if self.constitutional_review else None
            } if self.constitutional_review else None,
            "reasoning_trace": self.reasoning_trace.to_dict() if self.reasoning_trace else None,
            "retrieved_docs": [
                {"source": d.get("source", ""), "text": d.get("text", "")[:200]}
                for d in self.retrieved_docs[:5]
            ],
            "processing_time": self.processing_time,
            "timestamp": self.timestamp.isoformat(),
            "token_usage": self.token_usage
        }


class AdvancedAIPipeline:
    """
    고급 AI 분석 파이프라인

    사용법:
        pipeline = AdvancedAIPipeline()
        result = pipeline.analyze(contract_text)
    """

    def __init__(
        self,
        config: PipelineConfig = None,
        es_client: Optional[Any] = None,
        neo4j_driver: Optional[Any] = None,
        llm_client: Optional[HybridLLMClient] = None
    ):
        """
        Args:
            config: 파이프라인 설정
            es_client: Elasticsearch 클라이언트
            neo4j_driver: Neo4j 드라이버
            llm_client: 하이브리드 LLM 클라이언트
        """
        self.config = config or PipelineConfig()

        # 외부 서비스 클라이언트
        self.es_client = es_client
        self.neo4j_driver = neo4j_driver

        # 하이브리드 LLM 클라이언트 초기화 (Gemini + OpenAI)
        self.hybrid_llm = llm_client or get_llm_client()

        # OpenAI 클라이언트 (기존 모듈 호환성)
        self.llm_client = self.hybrid_llm.openai_client

        # AI 모듈 초기화
        self._init_modules()

    def _init_modules(self):
        """AI 모듈 초기화 (하이브리드 LLM 적용)"""
        self.preprocessor = ContractPreprocessor()
        self.pii_masker = PIIMasker()

        # CRAG 초기화 (clause_analyzer에서도 사용)
        self.crag = GraphGuidedCRAG(
            es_client=self.es_client,
            neo4j_driver=self.neo4j_driver,
            llm_client=self.llm_client,
            model=self.config.reasoning_model
        )

        # V2: LLM 기반 조항 분석기 (CRAG 통합)
        self.clause_analyzer = LLMClauseAnalyzer(
            crag=self.crag,
            llm_client=self.llm_client,
            model=self.config.reasoning_model,
            enable_crag=self.config.enable_crag
        )

        # Retrieval/Summarization 모듈
        self.hyde = HyDEGenerator(
            llm_client=self.llm_client,
            model=self.config.hyde_model  # gpt-4o (HyDE 생성용, temperature 지원)
        )
        self.raptor = RAPTORIndexer(
            llm_client=self.llm_client,
            model=self.config.reasoning_model
        )
        self.contract_raptor = ContractRAPTOR(self.raptor)

        # Reasoning/Analysis 모듈
        self.constitutional_ai = ConstitutionalAI(
            llm_client=self.llm_client,
            model=settings.LLM_CONSTITUTIONAL_MODEL  # Gemini 사용 (system role 지원)
        )
        self.stress_test = LegalStressTest(
            llm_client=self.llm_client
        )
        self.redliner = GenerativeRedlining(
            llm_client=self.llm_client,
            model=self.config.reasoning_model
        )
        self.judge = LLMJudge(
            llm_client=self.llm_client,
            model=settings.LLM_JUDGE_MODEL  # Gemini 사용 (temperature 지원)
        )
        self.tracer = ReasoningTracer(
            neo4j_driver=self.neo4j_driver
        )
        self.dspy = DSPyOptimizer()

    def analyze(
        self,
        contract_text: str,
        contract_id: str = None,
        file_path: str = None
    ) -> PipelineResult:
        """
        계약서 종합 분석 실행

        Args:
            contract_text: 계약서 텍스트
            contract_id: 계약서 ID
            file_path: 원본 파일 경로 (Vision 분석용)

        Returns:
            PipelineResult: 분석 결과
        """
        pipeline_start = time.time()
        contract_id = contract_id or f"contract_{int(time.time())}"

        result = PipelineResult(
            contract_id=contract_id,
            original_text=contract_text
        )

        # 파이프라인 로거 초기화
        logger = get_pipeline_logger(contract_id)
        logger.log_step("Pipeline", "started", input_summary=f"Text length: {len(contract_text)} chars")

        # 입력 계약서 전문 로깅
        logger.log_detail(
            step_name="Pipeline",
            category="InputContract",
            data=contract_text,
            description=f"입력 계약서 전문 ({len(contract_text)} chars)"
        )

        # 토큰 사용량 추적기 초기화
        token_tracker = TokenUsageTracker(contract_id, save_to_file=True)
        TokenUsageTracker.set_active(token_tracker)

        # 모든 AI 모듈에 contract_id 설정 (토큰 추적용)
        self.hyde.contract_id = contract_id
        self.raptor.contract_id = contract_id
        self.crag.contract_id = contract_id
        self.constitutional_ai.contract_id = contract_id
        self.judge.contract_id = contract_id
        self.redliner.contract_id = contract_id
        self.clause_analyzer.contract_id = contract_id

        # 상세 로깅을 위해 logger 전달
        self.clause_analyzer.pipeline_logger = logger

        try:
            # 1. PII Masking (개인정보 비식별화)
            if self.config.enable_pii_masking:
                step_start = time.time()
                logger.log_step("PII_Masking", "started", input_summary=f"Input: {len(contract_text)} chars")
                try:
                    result.pii_masking = self.pii_masker.mask(contract_text)
                    result.masked_text = result.pii_masking.masked_text
                    working_text = result.masked_text
                    logger.log_step(
                        "PII_Masking", "success",
                        output_summary=f"Masked {result.pii_masking.pii_count} items",
                        duration_ms=(time.time() - step_start) * 1000,
                        details={"masked_count": result.pii_masking.pii_count}
                    )
                except Exception as e:
                    logger.log_error("PII_Masking", e, (time.time() - step_start) * 1000)
                    working_text = contract_text
            else:
                working_text = contract_text

            # 2. 청킹
            step_start = time.time()
            logger.log_step("Chunking", "started", input_summary=f"Text: {len(working_text)} chars")
            chunks = self.preprocessor.chunk_text(working_text)
            logger.log_step(
                "Chunking", "success",
                output_summary=f"Created {len(chunks)} chunks",
                duration_ms=(time.time() - step_start) * 1000
            )

            # 3. HyDE Enhanced Search (검색 강화)
            retrieved_docs = []
            if self.config.enable_hyde and self.config.enable_crag:
                # HyDE
                step_start = time.time()
                logger.log_step("HyDE", "started", input_summary=working_text[:200])
                try:
                    hyde_result = self.hyde.generate(
                        working_text[:500],
                        prompt_type="contract"
                    )
                    logger.log_step(
                        "HyDE", "success",
                        output_summary=f"Generated {len(hyde_result.hypothetical_documents)} docs, strategy: {hyde_result.strategy_used.value}",
                        duration_ms=(time.time() - step_start) * 1000,
                        details={
                            "primary_document": hyde_result.primary_document[:500] if hyde_result.primary_document else "",
                            "complexity": hyde_result.query_complexity.value
                        }
                    )
                except Exception as e:
                    logger.log_error("HyDE", e, (time.time() - step_start) * 1000)
                    hyde_result = None

                # CRAG
                if hyde_result:
                    step_start = time.time()
                    logger.log_step("CRAG", "started", input_summary="Retrieve and correct")
                    try:
                        from .crag import RetrievedDocument
                        crag_result = self.crag.retrieve_and_correct_sync(
                            hyde_result.primary_document,
                            [],
                            max_graph_hops=self.config.crag_max_hops
                        )
                        retrieved_docs = [
                            {"source": d.source, "text": d.text, "score": d.score}
                            for d in crag_result.all_docs
                        ]
                        result.retrieved_docs = retrieved_docs
                        logger.log_step(
                            "CRAG", "success",
                            output_summary=f"Retrieved {len(retrieved_docs)} docs, iterations: {crag_result.correction_iterations}",
                            duration_ms=(time.time() - step_start) * 1000,
                            details={"retrieval_quality": str(crag_result.quality), "confidence": crag_result.confidence_score}
                        )
                    except Exception as e:
                        logger.log_error("CRAG", e, (time.time() - step_start) * 1000)

            # 4. V2: LLM 기반 조항 분석 (CRAG 통합) - 핵심 분석 단계
            if self.config.enable_llm_clause_analysis:
                step_start = time.time()
                logger.log_step("ClauseAnalysis", "started", input_summary="LLM-based clause extraction and analysis")
                try:
                    result.clause_analysis = self.clause_analyzer.analyze(contract_text)
                    logger.log_step(
                        "ClauseAnalysis", "success",
                        output_summary=f"Found {result.clause_analysis.violation_count} violations, "
                                       f"{result.clause_analysis.high_severity_count} high-severity, "
                                       f"underpayment: {result.clause_analysis.annual_underpayment:,}",
                        duration_ms=(time.time() - step_start) * 1000,
                        details={
                            "clause_count": len(result.clause_analysis.clauses),
                            "violation_count": result.clause_analysis.violation_count,
                            "high_severity_count": result.clause_analysis.high_severity_count,
                            "annual_underpayment": result.clause_analysis.annual_underpayment
                        }
                    )

                    # 상세 로깅: 추출된 조항 전체
                    logger.log_detail(
                        step_name="ClauseAnalysis",
                        category="ExtractedClauses",
                        data=[{
                            "number": c.clause_number,
                            "type": c.clause_type.value,
                            "title": c.title,
                            "text": c.original_text,
                            "values": c.extracted_values,
                            "position": c.position
                        } for c in result.clause_analysis.clauses],
                        description=f"추출된 조항 {len(result.clause_analysis.clauses)}개"
                    )

                    # 상세 로깅: 발견된 위반 전체
                    logger.log_detail(
                        step_name="ClauseAnalysis",
                        category="DetectedViolations",
                        data=[{
                            "clause_number": v.clause.clause_number,
                            "type": v.violation_type,
                            "severity": v.severity.value,
                            "description": v.description,
                            "legal_basis": v.legal_basis,
                            "current_value": v.current_value,
                            "legal_standard": v.legal_standard,
                            "suggestion": v.suggestion,
                            "suggested_text": v.suggested_text,
                            "confidence": v.confidence,
                            "crag_sources": v.crag_sources
                        } for v in result.clause_analysis.violations],
                        description=f"발견된 위반 {result.clause_analysis.violation_count}개"
                    )

                    # 4-1. ViolationLocationMapper: 위치 매핑 및 수정안 생성 (Gemini)
                    if result.clause_analysis.violations:
                        step_start = time.time()
                        logger.log_step("LocationMapping", "started", input_summary="Mapping violation locations with Gemini")
                        try:
                            location_mapper = ViolationLocationMapper()

                            # violations를 dict 형태로 변환
                            violations_dict = []
                            for i, v in enumerate(result.clause_analysis.violations):
                                violations_dict.append({
                                    "id": f"v_{i}_{v.clause.clause_number}",
                                    "type": v.violation_type,
                                    "clause_number": v.clause.clause_number,
                                    "description": v.description,
                                    "suggestion": v.suggestion,
                                    "original_text": v.clause.original_text,
                                })

                            # Gemini로 위치 매핑 및 suggested_text 생성
                            mapped_violations = location_mapper.map_violation_locations(
                                contract_text,
                                violations_dict,
                                contract_id
                            )

                            # 결과를 clause_analysis.violations에 반영
                            for i, v in enumerate(result.clause_analysis.violations):
                                vid = f"v_{i}_{v.clause.clause_number}"
                                mapped = next((m for m in mapped_violations if m.get("id") == vid), None)
                                if mapped:
                                    # 위치 정보 업데이트
                                    if mapped.get("start_index") is not None and mapped.get("end_index") is not None:
                                        v.clause.position = {
                                            "start": mapped["start_index"],
                                            "end": mapped["end_index"]
                                        }
                                    # suggested_text 업데이트
                                    if mapped.get("suggested_text"):
                                        v.suggested_text = mapped["suggested_text"]
                                    # matched_text 업데이트 (텍스트 기반 하이라이팅용)
                                    if mapped.get("matched_text"):
                                        v.matched_text = mapped["matched_text"]

                            mapped_count = sum(1 for v in result.clause_analysis.violations
                                             if v.clause.position.get("start", -1) >= 0)
                            logger.log_step(
                                "LocationMapping", "success",
                                output_summary=f"Mapped {mapped_count}/{len(result.clause_analysis.violations)} violations",
                                duration_ms=(time.time() - step_start) * 1000
                            )
                        except Exception as e:
                            logger.log_error("LocationMapping", e, (time.time() - step_start) * 1000)

                except Exception as e:
                    logger.log_error("ClauseAnalysis", e, (time.time() - step_start) * 1000)
                    # 폴백: Legacy 분석 활성화
                    self.config.enable_stress_test = True
                    self.config.enable_redlining = True

            # 5. RAPTOR (계층적 요약)
            if self.config.enable_raptor:
                step_start = time.time()
                logger.log_step("RAPTOR", "started", input_summary=f"Building tree from {len(chunks)} chunks")
                try:
                    raptor_tree = self.contract_raptor.build_contract_tree(
                        working_text, chunks
                    )
                    # RAPTOR 트리는 검색용으로 유지, analysis_summary는 _generate_summary()로 생성
                    # (사용자 요청: 요약을 더 간결하게, 조항별 상세는 아래 별도 표시)
                    logger.log_step(
                        "RAPTOR", "success",
                        output_summary=f"Tree built with {len(raptor_tree.nodes)} nodes, {len(raptor_tree.root_ids)} roots",
                        duration_ms=(time.time() - step_start) * 1000,
                        details={"summary": result.analysis_summary[:300] if result.analysis_summary else ""}
                    )
                except Exception as e:
                    logger.log_error("RAPTOR", e, (time.time() - step_start) * 1000)

            # 6. Legacy: Legal Stress Test (LLM 분석 실패 시 폴백)
            if self.config.enable_stress_test and not result.clause_analysis:
                step_start = time.time()
                logger.log_step("StressTest", "started", input_summary="Running legacy stress test (fallback)")
                try:
                    result.stress_test = self.stress_test.run(contract_text)
                    logger.log_step(
                        "StressTest", "success",
                        output_summary=f"Found {len(result.stress_test.violations)} violations, underpayment: {result.stress_test.annual_underpayment:,}",
                        duration_ms=(time.time() - step_start) * 1000,
                        details={
                            "violations": result.stress_test.violations,
                            "total_underpayment": result.stress_test.total_underpayment,
                            "annual_underpayment": result.stress_test.annual_underpayment
                        }
                    )
                except Exception as e:
                    logger.log_error("StressTest", e, (time.time() - step_start) * 1000)

            # 7. Legacy: Generative Redlining (LLM 분석 실패 시 폴백)
            if self.config.enable_redlining and not result.clause_analysis:
                step_start = time.time()
                logger.log_step("Redlining", "started", input_summary="Generating revision suggestions (fallback)")
                try:
                    result.redlining = self.redliner.redline(working_text)
                    logger.log_step(
                        "Redlining", "success",
                        output_summary=f"Generated {result.redlining.change_count} changes, {len(result.redlining.high_risk_changes)} high-risk",
                        duration_ms=(time.time() - step_start) * 1000,
                        details={
                            "changes": [
                                {"type": c.change_type.value, "original": c.original_text[:100], "revised": c.revised_text[:100]}
                                for c in result.redlining.changes[:5]
                            ]
                        }
                    )
                except Exception as e:
                    logger.log_error("Redlining", e, (time.time() - step_start) * 1000)

            # 7. 위험도 계산
            step_start = time.time()
            result.risk_level, result.risk_score = self._calculate_risk(result)
            logger.log_step(
                "RiskCalculation", "success",
                output_summary=f"Risk: {result.risk_level} (score: {result.risk_score:.2f})",
                duration_ms=(time.time() - step_start) * 1000
            )

            # 8. 분석 요약 생성
            if not result.analysis_summary:
                result.analysis_summary = self._generate_summary(result)

            # 9. Constitutional AI Review (헌법적 검토)
            if self.config.enable_constitutional_ai:
                step_start = time.time()
                logger.log_step("ConstitutionalAI", "started", input_summary="Reviewing for labor law principles")
                try:
                    result.constitutional_review = self.constitutional_ai.review(
                        result.analysis_summary,
                        context=working_text[:2000]
                    )
                    logger.log_step(
                        "ConstitutionalAI", "success",
                        output_summary=f"Constitutional: {result.constitutional_review.is_constitutional}, violations: {result.constitutional_review.has_violations}",
                        duration_ms=(time.time() - step_start) * 1000,
                        details={
                            "is_constitutional": result.constitutional_review.is_constitutional,
                            "has_violations": result.constitutional_review.has_violations,
                            "critiques_count": len(result.constitutional_review.critiques)
                        }
                    )
                except Exception as e:
                    logger.log_error("ConstitutionalAI", e, (time.time() - step_start) * 1000)

            # 10. LLM-as-a-Judge (신뢰도 평가)
            if self.config.enable_judge:
                step_start = time.time()
                logger.log_step("Judge", "started", input_summary="Evaluating analysis reliability")
                try:
                    context = "\n".join([d.get("text", "") for d in retrieved_docs[:3]])
                    result.judgment = self.judge.evaluate(
                        result.analysis_summary,
                        context=context
                    )
                    logger.log_step(
                        "Judge", "success",
                        output_summary=f"Score: {result.judgment.overall_score}, reliable: {result.judgment.is_reliable}",
                        duration_ms=(time.time() - step_start) * 1000,
                        details={
                            "overall_score": result.judgment.overall_score,
                            "confidence_level": result.judgment.confidence_level,
                            "is_reliable": result.judgment.is_reliable
                        }
                    )
                except Exception as e:
                    logger.log_error("Judge", e, (time.time() - step_start) * 1000)

            # 11. Reasoning Trace (추론 시각화)
            if self.config.enable_reasoning_trace:
                step_start = time.time()
                logger.log_step("ReasoningTrace", "started", input_summary="Building reasoning trace")
                try:
                    analysis_result = {
                        "risk_patterns": [
                            {"name": v["type"], "explanation": v["description"], "severity": v["severity"]}
                            for v in (result.stress_test.violations if result.stress_test else [])
                        ],
                        "conclusion": result.analysis_summary
                    }
                    result.reasoning_trace = self.tracer.trace_analysis(
                        working_text[:500],
                        analysis_result,
                        retrieved_docs
                    )
                    logger.log_step(
                        "ReasoningTrace", "success",
                        output_summary=f"Trace built with {len(result.reasoning_trace.nodes)} nodes",
                        duration_ms=(time.time() - step_start) * 1000
                    )
                except Exception as e:
                    logger.log_error("ReasoningTrace", e, (time.time() - step_start) * 1000)

            # 12. DSPy Feedback 기록 (자가 진화)
            if self.config.enable_dspy:
                step_start = time.time()
                logger.log_step("DSPy", "started", input_summary="Recording feedback")
                try:
                    self.dspy.record_feedback(
                        query=working_text[:500],
                        response=result.analysis_summary,
                        feedback_type="positive" if result.risk_score < 0.5 else "neutral",
                        metadata={"risk_level": result.risk_level}
                    )
                    logger.log_step(
                        "DSPy", "success",
                        output_summary="Feedback recorded",
                        duration_ms=(time.time() - step_start) * 1000
                    )
                except Exception as e:
                    logger.log_error("DSPy", e, (time.time() - step_start) * 1000)

            # 파이프라인 완료
            result.processing_time = time.time() - pipeline_start
            logger.log_step(
                "Pipeline", "success",
                output_summary=f"Completed in {result.processing_time:.2f}s, risk: {result.risk_level}",
                duration_ms=result.processing_time * 1000,
                details={"risk_level": result.risk_level, "risk_score": result.risk_score}
            )

        except Exception as e:
            result.analysis_summary = f"분석 중 오류 발생: {str(e)}"
            result.risk_level = "Unknown"
            result.processing_time = time.time() - pipeline_start
            logger.log_error("Pipeline", e, result.processing_time * 1000)

        finally:
            # 토큰 사용량 요약 저장
            try:
                from dataclasses import asdict
                token_summary = token_tracker.get_summary()
                result.token_usage = {
                    "total_input_tokens": token_summary.total_input_tokens,
                    "total_output_tokens": token_summary.total_output_tokens,
                    "total_cached_tokens": token_summary.total_cached_tokens,
                    "total_tokens": token_summary.total_tokens,
                    "total_cost_usd": token_summary.total_cost_usd,
                    "total_cost_krw": token_summary.total_cost_krw,
                    "total_llm_calls": token_summary.total_llm_calls,
                    "by_model": token_summary.by_model,
                    "by_module": token_summary.by_module
                }
                token_tracker.save_log()
                token_tracker.print_summary()
            except Exception as token_error:
                logger.log_step("TokenUsage", "error", error_message=str(token_error))
            finally:
                TokenUsageTracker.remove_active(contract_id)

        # 로그 저장
        logger.save()

        return result

    def _calculate_risk(self, result: PipelineResult) -> tuple:
        """위험도 계산"""
        risk_score = 0.0

        # V2: LLM 조항 분석 결과 반영 (우선)
        if result.clause_analysis and result.clause_analysis.violations:
            from .clause_analyzer import ViolationSeverity

            for v in result.clause_analysis.violations:
                if v.severity == ViolationSeverity.CRITICAL:
                    risk_score += 0.35
                elif v.severity == ViolationSeverity.HIGH:
                    risk_score += 0.25
                elif v.severity == ViolationSeverity.MEDIUM:
                    risk_score += 0.12
                elif v.severity == ViolationSeverity.LOW:
                    risk_score += 0.05

            # 체불액 반영
            if result.clause_analysis.annual_underpayment >= 1_000_000:
                risk_score += 0.2
            elif result.clause_analysis.annual_underpayment >= 500_000:
                risk_score += 0.1

        else:
            # Legacy: Stress Test 결과 반영
            if result.stress_test:
                high_count = len([v for v in result.stress_test.violations
                                  if v.get("severity") in ["High", "HIGH", "CRITICAL"]])
                medium_count = len([v for v in result.stress_test.violations
                                    if v.get("severity") in ["Medium", "MEDIUM"]])
                risk_score += high_count * 0.3 + medium_count * 0.15

                # 체불액 반영
                if result.stress_test.annual_underpayment >= 1_000_000:
                    risk_score += 0.2
                elif result.stress_test.annual_underpayment >= 500_000:
                    risk_score += 0.1

            # Redlining 결과 반영
            if result.redlining:
                high_changes = len(result.redlining.high_risk_changes)
                risk_score += high_changes * 0.15

        # 점수 정규화
        risk_score = min(1.0, risk_score)

        # 레벨 결정
        if risk_score >= 0.6:
            risk_level = "High"
        elif risk_score >= 0.3:
            risk_level = "Medium"
        else:
            risk_level = "Low"

        return risk_level, risk_score

    def _generate_summary(self, result: PipelineResult) -> str:
        """분석 요약 생성 - 계약서 분석 결과를 사용자 친화적으로 요약"""
        parts = []

        # 위험도 레벨에 따른 도입 문구
        if result.risk_level == "High":
            parts.append("이 계약서에서 주의가 필요한 조항들이 발견되었습니다.")
        elif result.risk_level == "Medium":
            parts.append("이 계약서에서 일부 검토가 필요한 조항들이 있습니다.")
        else:
            parts.append("이 계약서는 전반적으로 양호합니다.")

        # V2: LLM 조항 분석 결과 (우선)
        if result.clause_analysis and result.clause_analysis.violations:
            from .clause_analyzer import ViolationSeverity

            violation_count = result.clause_analysis.violation_count
            high_count = result.clause_analysis.high_severity_count

            if high_count > 0:
                parts.append(f"법적 기준 위반 {violation_count}건이 발견되었으며, 그 중 {high_count}건은 심각한 수준입니다.")
            else:
                parts.append(f"법적 기준 위반 {violation_count}건이 발견되었습니다.")

            # 주요 위반 유형 나열 (최대 3개)
            violation_types = list(set(v.violation_type for v in result.clause_analysis.violations[:3]))
            if violation_types:
                parts.append(f"주요 문제: {', '.join(violation_types)}.")

            # 체불액 정보
            if result.clause_analysis.annual_underpayment > 0:
                parts.append(f"연간 예상 체불액은 약 {result.clause_analysis.annual_underpayment:,}원입니다.")

        else:
            # Legacy: StressTest 결과 (법적 기준 위반)
            if result.stress_test and result.stress_test.violations:
                violation_count = len(result.stress_test.violations)
                high_count = sum(1 for v in result.stress_test.violations
                               if v.get("severity") in ["CRITICAL", "HIGH", "High"])

                if high_count > 0:
                    parts.append(f"법적 기준 위반 {violation_count}건이 발견되었으며, 그 중 {high_count}건은 심각한 수준입니다.")
                else:
                    parts.append(f"법적 기준 위반 {violation_count}건이 발견되었습니다.")

                # 체불액 정보
                if result.stress_test.annual_underpayment > 0:
                    parts.append(f"연간 예상 체불액은 약 {result.stress_test.annual_underpayment:,}원입니다.")

            # Redlining 결과 (불공정 조항)
            if result.redlining and result.redlining.change_count > 0:
                high_risk = len(result.redlining.high_risk_changes) if result.redlining.high_risk_changes else 0

                if high_risk > 0:
                    parts.append(f"불공정 조항 {result.redlining.change_count}건 중 {high_risk}건이 고위험으로 분류되어 수정을 권장합니다.")
                else:
                    parts.append(f"검토가 필요한 조항 {result.redlining.change_count}건에 대해 수정을 제안합니다.")

        # 결과가 없는 경우
        if len(parts) == 1:
            parts.append("특별한 법적 위험이 발견되지 않았습니다.")

        return " ".join(parts)

    def analyze_file(self, file_path: str) -> PipelineResult:
        """
        파일 분석 (PDF/이미지)

        Args:
            file_path: 파일 경로

        Returns:
            PipelineResult
        """
        # PDF인 경우 텍스트 추출
        if file_path.lower().endswith(".pdf"):
            text = self.preprocessor.extract_text(file_path)
        else:
            # 이미지인 경우 Vision 분석
            from .vision_parser import VisionParser
            parser = VisionParser(llm_client=self.llm_client)
            vision_result = parser.parse_image(file_path)
            text = vision_result.raw_text

        return self.analyze(text, file_path=file_path)


# API 통합용 함수
def create_pipeline(
    es_client: Optional[Any] = None,
    neo4j_driver: Optional[Any] = None,
    config: PipelineConfig = None
) -> AdvancedAIPipeline:
    """파이프라인 팩토리 함수"""
    return AdvancedAIPipeline(
        config=config,
        es_client=es_client,
        neo4j_driver=neo4j_driver
    )


def quick_analyze(contract_text: str) -> Dict[str, Any]:
    """빠른 분석 (간소화 버전)"""
    config = PipelineConfig(
        enable_raptor=False,
        enable_reasoning_trace=False,
        enable_dspy=False
    )
    pipeline = AdvancedAIPipeline(config=config)
    result = pipeline.analyze(contract_text)
    return result.to_dict()
