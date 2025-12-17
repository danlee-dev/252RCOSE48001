"""
Graph-Guided CRAG (Corrective Retrieval-Augmented Generation) - Production Grade
- 다단계 검색 품질 평가 (Scoring Rubric 기반)
- 적응형 검색 전략 (Query Rewriting, Decomposition)
- 그래프 기반 지식 확장 및 자가 보정
- 신뢰도 기반 문서 필터링 및 랭킹

Reference: "Corrective Retrieval Augmented Generation" (CRAG), Yan et al. 2024
"""

import os
import json
import hashlib
import time
from typing import List, Dict, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
from functools import lru_cache
import asyncio
from concurrent.futures import ThreadPoolExecutor

from app.core.config import settings
from app.core.token_usage_tracker import record_llm_usage


class RetrievalQuality(Enum):
    """검색 품질 등급 (세분화)"""
    EXCELLENT = "excellent"     # 완벽한 답변 가능
    GOOD = "good"               # 충분한 정보
    CORRECT = "correct"         # 관련있음
    PARTIAL = "partial"         # 부분적 관련
    AMBIGUOUS = "ambiguous"     # 모호함
    WEAK = "weak"               # 약한 관련성
    INCORRECT = "incorrect"     # 관련없음
    HARMFUL = "harmful"         # 잘못된 정보 포함


class CorrectionStrategy(Enum):
    """보정 전략"""
    NONE = "none"                       # 보정 불필요
    REFINE = "refine"                   # 정제만
    AUGMENT = "augment"                 # 그래프 확장
    REWRITE = "rewrite"                 # 쿼리 재작성
    DECOMPOSE = "decompose"             # 쿼리 분해
    FALLBACK = "fallback"               # 폴백 검색
    MULTI_HOP = "multi_hop"             # 다단계 추론


class DocumentRelevance(Enum):
    """문서 관련성"""
    HIGHLY_RELEVANT = "highly_relevant"
    RELEVANT = "relevant"
    MARGINALLY_RELEVANT = "marginally_relevant"
    NOT_RELEVANT = "not_relevant"
    CONTRADICTORY = "contradictory"


@dataclass
class RetrievedDocument:
    """검색된 문서 (확장)"""
    id: str
    text: str
    source: str
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    relevance: DocumentRelevance = DocumentRelevance.RELEVANT
    confidence: float = 0.0
    extracted_info: str = ""                # 추출된 핵심 정보
    supporting_evidence: List[str] = field(default_factory=list)
    contradicting_evidence: List[str] = field(default_factory=list)
    legal_references: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text[:500],
            "source": self.source,
            "score": self.score,
            "relevance": self.relevance.value,
            "confidence": self.confidence,
            "extracted_info": self.extracted_info,
            "legal_references": self.legal_references,
        }


@dataclass
class QualityEvaluation:
    """품질 평가 결과"""
    quality: RetrievalQuality
    confidence: float
    reasoning: str
    missing_info: List[str] = field(default_factory=list)
    suggested_queries: List[str] = field(default_factory=list)
    correction_strategy: CorrectionStrategy = CorrectionStrategy.NONE
    document_scores: Dict[str, float] = field(default_factory=dict)
    rubric_scores: Dict[str, float] = field(default_factory=dict)


@dataclass
class CRAGResult:
    """CRAG 처리 결과 (확장)"""
    query: str
    original_query: str                     # 원본 쿼리
    rewritten_queries: List[str] = field(default_factory=list)  # 재작성된 쿼리들
    initial_docs: List[RetrievedDocument] = field(default_factory=list)
    quality_evaluation: Optional[QualityEvaluation] = None
    corrected_docs: List[RetrievedDocument] = field(default_factory=list)
    graph_expanded_docs: List[RetrievedDocument] = field(default_factory=list)
    reasoning_trace: List[str] = field(default_factory=list)
    final_context: str = ""
    correction_iterations: int = 0          # 보정 반복 횟수
    total_docs_processed: int = 0
    processing_time_ms: float = 0.0
    confidence_score: float = 0.0           # 최종 신뢰도

    @property
    def quality(self) -> RetrievalQuality:
        if self.quality_evaluation:
            return self.quality_evaluation.quality
        return RetrievalQuality.AMBIGUOUS

    @property
    def all_docs(self) -> List[RetrievedDocument]:
        """모든 문서 (중복 제거, 점수순 정렬)"""
        seen = set()
        result = []
        all_documents = self.initial_docs + self.corrected_docs + self.graph_expanded_docs

        for doc in all_documents:
            if doc.id not in seen:
                seen.add(doc.id)
                result.append(doc)

        # 신뢰도 * 점수로 정렬
        result.sort(key=lambda d: d.confidence * d.score, reverse=True)
        return result

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "original_query": self.original_query,
            "quality": self.quality.value,
            "confidence_score": self.confidence_score,
            "correction_iterations": self.correction_iterations,
            "initial_count": len(self.initial_docs),
            "corrected_count": len(self.corrected_docs),
            "graph_expanded_count": len(self.graph_expanded_docs),
            "total_count": len(self.all_docs),
            "processing_time_ms": self.processing_time_ms,
            "reasoning_trace": self.reasoning_trace,
            "documents": [d.to_dict() for d in self.all_docs[:10]],
        }


class ScoringRubric:
    """품질 평가 루브릭"""

    LEGAL_RELEVANCE_RUBRIC = """
[법적 관련성 평가 루브릭]

점수 5 (Excellent):
- 질문에 직접적으로 답변 가능한 법령 조항 또는 판례 포함
- 구체적인 법적 근거와 해석 제시
- 유사 사례에 대한 판단 기준 명시

점수 4 (Good):
- 관련 법령 또는 판례 포함
- 질문의 대부분에 답변 가능
- 일부 추가 정보 필요

점수 3 (Adequate):
- 관련 법적 개념 언급
- 간접적으로 답변에 도움
- 상당한 추가 검색 필요

점수 2 (Marginal):
- 법적 맥락은 맞으나 직접 관련 없음
- 배경 정보 수준
- 핵심 정보 부재

점수 1 (Poor):
- 관련성 거의 없음
- 다른 법적 영역 또는 일반 정보

점수 0 (Irrelevant):
- 완전히 관련 없음
- 잘못된 정보 포함 가능
"""

    FACTUAL_ACCURACY_RUBRIC = """
[사실 정확성 평가 루브릭]

점수 5: 법령 조문 또는 판례 원문 직접 인용
점수 4: 법령/판례 내용 정확히 설명
점수 3: 대체로 정확하나 세부 사항 부정확
점수 2: 부분적으로만 정확
점수 1: 대부분 부정확
점수 0: 잘못된 정보 또는 오해 유발
"""

    COMPLETENESS_RUBRIC = """
[완전성 평가 루브릭]

점수 5: 질문의 모든 측면 완벽히 다룸
점수 4: 주요 측면 대부분 다룸
점수 3: 핵심 측면은 다루나 일부 누락
점수 2: 부분적으로만 다룸
점수 1: 단편적 정보만 제공
점수 0: 질문과 무관
"""


class GraphGuidedCRAG:
    """
    Graph-Guided CRAG 구현 (Production Grade)

    사용법:
        crag = GraphGuidedCRAG(es_client, neo4j_driver)
        result = await crag.retrieve_and_correct(query, initial_docs)
    """

    # 품질 평가 프롬프트 (확장)
    QUALITY_EVALUATION_PROMPT = """당신은 법률 검색 결과 품질 평가 전문가입니다.

[평가 대상 질문]
{query}

[검색된 문서들]
{documents}

[평가 루브릭]
{rubric}

위 루브릭에 따라 검색 결과를 평가하고 다음 형식으로 응답하세요:

{{
    "quality": "excellent/good/correct/partial/ambiguous/weak/incorrect/harmful",
    "confidence": 0.0-1.0,
    "reasoning": "평가 이유 (한국어로 상세히)",
    "rubric_scores": {{
        "legal_relevance": 0-5,
        "factual_accuracy": 0-5,
        "completeness": 0-5
    }},
    "document_evaluations": [
        {{
            "doc_id": "문서ID",
            "relevance": "highly_relevant/relevant/marginally_relevant/not_relevant/contradictory",
            "key_info": "이 문서에서 추출된 핵심 정보"
        }}
    ],
    "missing_info": ["부족한 정보 목록"],
    "suggested_queries": ["추가 검색이 필요한 쿼리들"],
    "correction_strategy": "none/refine/augment/rewrite/decompose/fallback/multi_hop"
}}"""

    # 쿼리 재작성 프롬프트
    QUERY_REWRITE_PROMPT = """당신은 법률 검색 쿼리 최적화 전문가입니다.

원본 쿼리: {query}

검색 결과가 불충분합니다. 다음을 수행하세요:

1. 법률 용어로 변환: 일상어를 법률 용어로
2. 구체화: 모호한 표현을 구체화
3. 분해: 복합 질문을 단순 질문들로

응답 형식:
{{
    "rewritten_query": "최적화된 쿼리",
    "decomposed_queries": ["분해된 하위 쿼리들"],
    "legal_terms": ["관련 법률 용어들"],
    "search_keywords": ["핵심 검색 키워드"]
}}"""

    # 지식 정제 프롬프트 (확장)
    KNOWLEDGE_REFINEMENT_PROMPT = """다음 문서에서 질문과 관련된 핵심 정보를 추출하세요.

[질문]
{query}

[문서]
{document}

[지시사항]
1. 직접적으로 관련된 법령 조항 추출
2. 관련 판례 요지 추출
3. 핵심 법적 개념 정리
4. 적용 가능한 해석 기준 제시

[응답 형식]
{{
    "key_info": "핵심 정보 요약",
    "legal_articles": ["관련 법령 조항"],
    "precedents": ["관련 판례"],
    "legal_concepts": ["법적 개념"],
    "applicability": "적용 가능성 평가"
}}"""

    def __init__(
        self,
        es_client: Optional[Any] = None,
        neo4j_driver: Optional[Any] = None,
        llm_client: Optional[Any] = None,
        model: str = None,
        quality_threshold: float = 0.7,
        max_correction_iterations: int = 3,
        enable_caching: bool = True,
        cache_ttl_seconds: int = 3600,
        contract_id: Optional[str] = None
    ):
        """
        Args:
            es_client: Elasticsearch 클라이언트
            neo4j_driver: Neo4j 드라이버
            llm_client: OpenAI 클라이언트
            model: 평가에 사용할 LLM 모델 (기본값: settings.LLM_CRAG_MODEL)
            quality_threshold: 품질 임계값
            max_correction_iterations: 최대 보정 반복 횟수
            enable_caching: 캐싱 활성화
            cache_ttl_seconds: 캐시 TTL
            contract_id: 계약서 ID (토큰 추적용)
        """
        self.es_client = es_client
        self.neo4j_driver = neo4j_driver
        self.model = model if model else settings.LLM_CRAG_MODEL
        self.quality_threshold = quality_threshold
        self.max_correction_iterations = max_correction_iterations
        self.enable_caching = enable_caching
        self.cache_ttl_seconds = cache_ttl_seconds
        self.contract_id = contract_id

        # 평가 캐시
        self._evaluation_cache: Dict[str, Tuple[QualityEvaluation, datetime]] = {}

        # 스레드 풀
        self._executor = ThreadPoolExecutor(max_workers=4)

        if llm_client is None:
            try:
                from openai import OpenAI
                self.llm_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            except ImportError:
                self.llm_client = None
        else:
            self.llm_client = llm_client

        # 루브릭 초기화
        self.rubric = ScoringRubric()

    def _is_reasoning_model(self) -> bool:
        """reasoning 모델 여부 확인 (temperature 미지원)"""
        reasoning_keywords = ["o1", "o3", "gpt-5"]
        return any(kw in self.model.lower() for kw in reasoning_keywords)

    async def retrieve_and_correct(
        self,
        query: str,
        initial_docs: List[RetrievedDocument],
        max_graph_hops: int = 2
    ) -> CRAGResult:
        """
        검색 결과 평가 및 보정 (비동기)

        Args:
            query: 검색 쿼리
            initial_docs: 초기 검색 결과
            max_graph_hops: 그래프 확장 최대 홉

        Returns:
            CRAGResult: 보정된 검색 결과
        """
        import time
        start_time = time.time()

        result = CRAGResult(
            query=query,
            original_query=query,
            initial_docs=initial_docs
        )

        result.reasoning_trace.append(f"초기 검색 문서 수: {len(initial_docs)}")
        result.total_docs_processed = len(initial_docs)

        # 1. 캐시 확인
        cache_key = self._generate_cache_key(query, initial_docs)
        cached_eval = self._get_cached_evaluation(cache_key)

        # 2. 품질 평가 (캐시 미스 시)
        if cached_eval:
            evaluation = cached_eval
            result.reasoning_trace.append("캐시된 평가 결과 사용")
        else:
            evaluation = await self._evaluate_quality_async(query, initial_docs)
            if self.enable_caching:
                self._cache_evaluation(cache_key, evaluation)

        result.quality_evaluation = evaluation
        result.reasoning_trace.append(
            f"검색 품질: {evaluation.quality.value} "
            f"(신뢰도: {evaluation.confidence:.2f})"
        )

        # 3. 보정 전략에 따른 처리
        iteration = 0
        current_docs = initial_docs
        current_query = query

        while iteration < self.max_correction_iterations:
            strategy = evaluation.correction_strategy

            if strategy == CorrectionStrategy.NONE:
                # 충분함 -> 정제만
                result.corrected_docs = await self._refine_documents_async(
                    current_query, current_docs
                )
                result.reasoning_trace.append("품질 충분: 지식 정제 완료")
                break

            elif strategy == CorrectionStrategy.REFINE:
                result.corrected_docs = await self._refine_documents_async(
                    current_query, current_docs
                )
                break

            elif strategy == CorrectionStrategy.REWRITE:
                # 쿼리 재작성
                rewritten = await self._rewrite_query_async(current_query)
                result.rewritten_queries.append(rewritten["rewritten_query"])
                current_query = rewritten["rewritten_query"]
                result.reasoning_trace.append(f"쿼리 재작성: {current_query}")

                # 재검색
                new_docs = await self._search_with_query_async(current_query)
                current_docs = self._merge_documents(current_docs, new_docs)

            elif strategy == CorrectionStrategy.DECOMPOSE:
                # 쿼리 분해
                rewritten = await self._rewrite_query_async(current_query)
                sub_queries = rewritten.get("decomposed_queries", [])
                result.rewritten_queries.extend(sub_queries)

                for sub_q in sub_queries[:3]:
                    sub_docs = await self._search_with_query_async(sub_q)
                    current_docs = self._merge_documents(current_docs, sub_docs)
                    result.reasoning_trace.append(f"하위 쿼리 검색: {sub_q}")

            elif strategy == CorrectionStrategy.AUGMENT:
                # 그래프 확장
                graph_docs = await self._expand_with_graph_async(
                    current_query,
                    current_docs,
                    evaluation.missing_info,
                    max_graph_hops
                )
                result.graph_expanded_docs.extend(graph_docs)
                result.reasoning_trace.append(
                    f"그래프 확장: {len(graph_docs)}건 추가"
                )

            elif strategy == CorrectionStrategy.FALLBACK:
                # 폴백 검색
                fallback_docs = await self._fallback_search_async(current_query)
                current_docs = self._merge_documents(current_docs, fallback_docs)
                result.reasoning_trace.append(
                    f"폴백 검색: {len(fallback_docs)}건"
                )

            elif strategy == CorrectionStrategy.MULTI_HOP:
                # 다단계 추론
                for suggested_q in evaluation.suggested_queries[:2]:
                    hop_docs = await self._search_with_query_async(suggested_q)
                    current_docs = self._merge_documents(current_docs, hop_docs)
                    result.reasoning_trace.append(
                        f"다단계 검색: {suggested_q}"
                    )

            # 재평가
            iteration += 1
            result.correction_iterations = iteration

            if iteration < self.max_correction_iterations:
                evaluation = await self._evaluate_quality_async(
                    current_query, current_docs
                )
                result.quality_evaluation = evaluation
                result.reasoning_trace.append(
                    f"재평가 (반복 {iteration}): {evaluation.quality.value}"
                )

        # 4. 최종 정제 및 컨텍스트 생성
        if not result.corrected_docs:
            result.corrected_docs = await self._refine_documents_async(
                current_query, current_docs
            )

        result.total_docs_processed = len(result.all_docs)
        result.final_context = self._build_context(result.all_docs)
        result.confidence_score = self._calculate_final_confidence(result)

        # 5. 처리 시간 기록
        result.processing_time_ms = (time.time() - start_time) * 1000

        return result

    def retrieve_and_correct_sync(
        self,
        query: str,
        initial_docs: List[RetrievedDocument],
        max_graph_hops: int = 2
    ) -> CRAGResult:
        """동기 버전 - 별도 스레드에서 실행하여 event loop 충돌 방지"""
        import concurrent.futures

        def run_async():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(
                    self.retrieve_and_correct(query, initial_docs, max_graph_hops)
                )
            finally:
                loop.close()

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(run_async)
            return future.result()

    # ========== 비동기 헬퍼 메서드 ==========

    async def _evaluate_quality_async(
        self,
        query: str,
        docs: List[RetrievedDocument]
    ) -> QualityEvaluation:
        """비동기 품질 평가"""
        return await asyncio.get_event_loop().run_in_executor(
            self._executor,
            self._evaluate_quality,
            query,
            docs
        )

    async def _refine_documents_async(
        self,
        query: str,
        docs: List[RetrievedDocument]
    ) -> List[RetrievedDocument]:
        """비동기 문서 정제"""
        return await asyncio.get_event_loop().run_in_executor(
            self._executor,
            self._refine_knowledge,
            query,
            docs
        )

    async def _rewrite_query_async(self, query: str) -> Dict[str, Any]:
        """비동기 쿼리 재작성"""
        return await asyncio.get_event_loop().run_in_executor(
            self._executor,
            self._rewrite_query,
            query
        )

    async def _expand_with_graph_async(
        self,
        query: str,
        anchor_docs: List[RetrievedDocument],
        missing_info: List[str],
        max_hops: int
    ) -> List[RetrievedDocument]:
        """비동기 그래프 확장"""
        return await asyncio.get_event_loop().run_in_executor(
            self._executor,
            self._expand_with_graph,
            query,
            anchor_docs,
            missing_info,
            max_hops
        )

    async def _fallback_search_async(self, query: str) -> List[RetrievedDocument]:
        """비동기 폴백 검색"""
        return await asyncio.get_event_loop().run_in_executor(
            self._executor,
            self._fallback_search,
            query
        )

    async def _search_with_query_async(self, query: str) -> List[RetrievedDocument]:
        """비동기 쿼리 검색"""
        return await asyncio.get_event_loop().run_in_executor(
            self._executor,
            self._search_with_query,
            query
        )

    # ========== 핵심 메서드 ==========

    def _evaluate_quality(
        self,
        query: str,
        docs: List[RetrievedDocument]
    ) -> QualityEvaluation:
        """검색 결과 품질 평가 (루브릭 기반)"""
        if not docs:
            return QualityEvaluation(
                quality=RetrievalQuality.INCORRECT,
                confidence=0.0,
                reasoning="검색 결과 없음",
                missing_info=["전체 정보 필요"],
                correction_strategy=CorrectionStrategy.FALLBACK
            )

        # 점수 기반 1차 평가
        avg_score = sum(d.score for d in docs) / len(docs)

        if avg_score < 0.2:
            return QualityEvaluation(
                quality=RetrievalQuality.INCORRECT,
                confidence=avg_score,
                reasoning="검색 점수가 매우 낮음",
                correction_strategy=CorrectionStrategy.REWRITE
            )

        # LLM 기반 상세 평가
        if self.llm_client is not None:
            return self._llm_evaluate_quality(query, docs)

        # 폴백: 점수 기반 간단 평가
        return self._score_based_evaluation(avg_score)

    def _llm_evaluate_quality(
        self,
        query: str,
        docs: List[RetrievedDocument]
    ) -> QualityEvaluation:
        """LLM 기반 품질 평가"""
        llm_start = time.time()
        try:
            doc_texts = "\n\n---\n\n".join([
                f"[문서 {i+1}] (ID: {d.id}, 점수: {d.score:.2f})\n"
                f"출처: {d.source}\n{d.text[:800]}"
                for i, d in enumerate(docs[:7])
            ])

            rubric = (
                self.rubric.LEGAL_RELEVANCE_RUBRIC +
                self.rubric.FACTUAL_ACCURACY_RUBRIC +
                self.rubric.COMPLETENESS_RUBRIC
            )

            prompt = self.QUALITY_EVALUATION_PROMPT.format(
                query=query,
                documents=doc_texts,
                rubric=rubric
            )

            # reasoning 모델은 temperature 미지원
            if self._is_reasoning_model():
                response = self.llm_client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "user", "content": f"당신은 법률 검색 결과 품질 평가 전문가입니다. 루브릭에 따라 엄격하게 평가하세요.\n\n{prompt}"}
                    ],
                    response_format={"type": "json_object"}
                )
            else:
                response = self.llm_client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": "당신은 법률 검색 결과 품질 평가 전문가입니다. "
                                       "루브릭에 따라 엄격하게 평가하세요."
                        },
                        {"role": "user", "content": prompt}
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.1
                )

            llm_duration = (time.time() - llm_start) * 1000

            # 토큰 사용량 기록
            if response.usage and self.contract_id:
                cached = 0
                if hasattr(response.usage, 'prompt_tokens_details') and response.usage.prompt_tokens_details:
                    cached = getattr(response.usage.prompt_tokens_details, 'cached_tokens', 0) or 0
                record_llm_usage(
                    contract_id=self.contract_id,
                    module="crag.evaluate_quality",
                    model=self.model,
                    input_tokens=response.usage.prompt_tokens,
                    output_tokens=response.usage.completion_tokens,
                    cached_tokens=cached,
                    duration_ms=llm_duration
                )

            result = json.loads(response.choices[0].message.content)

            # 품질 등급 매핑
            quality_map = {
                "excellent": RetrievalQuality.EXCELLENT,
                "good": RetrievalQuality.GOOD,
                "correct": RetrievalQuality.CORRECT,
                "partial": RetrievalQuality.PARTIAL,
                "ambiguous": RetrievalQuality.AMBIGUOUS,
                "weak": RetrievalQuality.WEAK,
                "incorrect": RetrievalQuality.INCORRECT,
                "harmful": RetrievalQuality.HARMFUL,
            }

            quality = quality_map.get(
                result.get("quality", "ambiguous").lower(),
                RetrievalQuality.AMBIGUOUS
            )

            # 보정 전략 매핑
            strategy_map = {
                "none": CorrectionStrategy.NONE,
                "refine": CorrectionStrategy.REFINE,
                "augment": CorrectionStrategy.AUGMENT,
                "rewrite": CorrectionStrategy.REWRITE,
                "decompose": CorrectionStrategy.DECOMPOSE,
                "fallback": CorrectionStrategy.FALLBACK,
                "multi_hop": CorrectionStrategy.MULTI_HOP,
            }

            strategy = strategy_map.get(
                result.get("correction_strategy", "augment").lower(),
                CorrectionStrategy.AUGMENT
            )

            # 문서별 관련성 업데이트
            doc_evals = result.get("document_evaluations", [])
            for eval_item in doc_evals:
                doc_id = eval_item.get("doc_id", "")
                for doc in docs:
                    if doc.id == doc_id:
                        relevance_str = eval_item.get("relevance", "relevant")
                        doc.relevance = DocumentRelevance[relevance_str.upper()]
                        doc.extracted_info = eval_item.get("key_info", "")
                        break

            return QualityEvaluation(
                quality=quality,
                confidence=result.get("confidence", 0.5),
                reasoning=result.get("reasoning", ""),
                missing_info=result.get("missing_info", []),
                suggested_queries=result.get("suggested_queries", []),
                correction_strategy=strategy,
                rubric_scores=result.get("rubric_scores", {})
            )

        except Exception as e:
            print(f"Quality evaluation error: {e}")
            avg_score = sum(d.score for d in docs) / len(docs) if docs else 0
            return self._score_based_evaluation(avg_score)

    def _score_based_evaluation(self, avg_score: float) -> QualityEvaluation:
        """점수 기반 간단 평가"""
        if avg_score >= 0.85:
            return QualityEvaluation(
                quality=RetrievalQuality.EXCELLENT,
                confidence=avg_score,
                reasoning="높은 검색 점수",
                correction_strategy=CorrectionStrategy.NONE
            )
        elif avg_score >= 0.7:
            return QualityEvaluation(
                quality=RetrievalQuality.GOOD,
                confidence=avg_score,
                reasoning="충분한 검색 점수",
                correction_strategy=CorrectionStrategy.REFINE
            )
        elif avg_score >= 0.5:
            return QualityEvaluation(
                quality=RetrievalQuality.PARTIAL,
                confidence=avg_score,
                reasoning="부분적 관련성",
                correction_strategy=CorrectionStrategy.AUGMENT
            )
        else:
            return QualityEvaluation(
                quality=RetrievalQuality.WEAK,
                confidence=avg_score,
                reasoning="낮은 관련성",
                correction_strategy=CorrectionStrategy.REWRITE
            )

    def _rewrite_query(self, query: str) -> Dict[str, Any]:
        """쿼리 재작성"""
        if self.llm_client is None:
            return {"rewritten_query": query, "decomposed_queries": []}

        llm_start = time.time()
        try:
            prompt = self.QUERY_REWRITE_PROMPT.format(query=query)

            # reasoning 모델은 temperature 미지원
            if self._is_reasoning_model():
                response = self.llm_client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_object"}
                )
            else:
                response = self.llm_client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_object"},
                    temperature=0.3
                )

            llm_duration = (time.time() - llm_start) * 1000

            # 토큰 사용량 기록
            if response.usage and self.contract_id:
                cached = 0
                if hasattr(response.usage, 'prompt_tokens_details') and response.usage.prompt_tokens_details:
                    cached = getattr(response.usage.prompt_tokens_details, 'cached_tokens', 0) or 0
                record_llm_usage(
                    contract_id=self.contract_id,
                    module="crag.rewrite_query",
                    model=self.model,
                    input_tokens=response.usage.prompt_tokens,
                    output_tokens=response.usage.completion_tokens,
                    cached_tokens=cached,
                    duration_ms=llm_duration
                )

            return json.loads(response.choices[0].message.content)

        except Exception as e:
            print(f"Query rewrite error: {e}")
            return {"rewritten_query": query, "decomposed_queries": []}

    def _refine_knowledge(
        self,
        query: str,
        docs: List[RetrievedDocument]
    ) -> List[RetrievedDocument]:
        """지식 정제: 관련 부분 추출 및 신뢰도 계산"""
        refined = []

        for doc in docs:
            # 이미 추출된 정보가 있으면 재사용
            if doc.extracted_info:
                refined_text = doc.extracted_info
            elif self.llm_client is not None:
                refined_text = self._extract_relevant_part(query, doc.text)
            else:
                refined_text = doc.text

            # 신뢰도 계산
            confidence = self._calculate_document_confidence(doc, query)

            refined.append(RetrievedDocument(
                id=doc.id,
                text=refined_text,
                source=doc.source,
                score=doc.score,
                relevance=doc.relevance,
                confidence=confidence,
                extracted_info=refined_text,
                legal_references=doc.legal_references,
                metadata={**doc.metadata, "refined": True}
            ))

        return refined

    def _extract_relevant_part(self, query: str, text: str) -> str:
        """LLM으로 관련 부분 추출"""
        llm_start = time.time()
        try:
            prompt = self.KNOWLEDGE_REFINEMENT_PROMPT.format(
                query=query,
                document=text[:3000]
            )

            # reasoning 모델은 temperature 미지원
            if self._is_reasoning_model():
                response = self.llm_client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_object"},
                    max_tokens=800
                )
            else:
                response = self.llm_client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_object"},
                    max_tokens=800,
                    temperature=0.2
                )

            llm_duration = (time.time() - llm_start) * 1000

            # 토큰 사용량 기록
            if response.usage and self.contract_id:
                cached = 0
                if hasattr(response.usage, 'prompt_tokens_details') and response.usage.prompt_tokens_details:
                    cached = getattr(response.usage.prompt_tokens_details, 'cached_tokens', 0) or 0
                record_llm_usage(
                    contract_id=self.contract_id,
                    module="crag.extract_relevant",
                    model=self.model,
                    input_tokens=response.usage.prompt_tokens,
                    output_tokens=response.usage.completion_tokens,
                    cached_tokens=cached,
                    duration_ms=llm_duration
                )

            result = json.loads(response.choices[0].message.content)
            return result.get("key_info", text[:500])

        except Exception:
            return text[:500]

    def _calculate_document_confidence(
        self,
        doc: RetrievedDocument,
        query: str
    ) -> float:
        """문서 신뢰도 계산"""
        confidence = doc.score

        # 관련성에 따른 가중치
        relevance_weights = {
            DocumentRelevance.HIGHLY_RELEVANT: 1.0,
            DocumentRelevance.RELEVANT: 0.8,
            DocumentRelevance.MARGINALLY_RELEVANT: 0.5,
            DocumentRelevance.NOT_RELEVANT: 0.2,
            DocumentRelevance.CONTRADICTORY: 0.1,
        }
        confidence *= relevance_weights.get(doc.relevance, 0.5)

        # 법적 참조가 있으면 신뢰도 상승
        if doc.legal_references:
            confidence = min(1.0, confidence + 0.1)

        # 출처에 따른 가중치
        source_weights = {
            "law": 1.0,
            "precedent": 0.95,
            "interpretation": 0.85,
            "article": 0.7,
        }
        source_type = doc.metadata.get("type", "")
        confidence *= source_weights.get(source_type, 0.75)

        return min(1.0, max(0.0, confidence))

    def _expand_with_graph(
        self,
        query: str,
        anchor_docs: List[RetrievedDocument],
        missing_info: List[str],
        max_hops: int
    ) -> List[RetrievedDocument]:
        """그래프 DB로 지식 확장"""
        expanded = []

        if self.neo4j_driver is None:
            return expanded

        try:
            with self.neo4j_driver.session() as session:
                # 1. 앵커 문서에서 관련 법령/판례 탐색
                anchor_ids = [d.id for d in anchor_docs[:5]]

                if anchor_ids:
                    expanded.extend(
                        self._traverse_from_anchors(session, anchor_ids, max_hops)
                    )

                # 2. 위험 패턴 검색
                expanded.extend(
                    self._search_risk_patterns(session, query)
                )

                # 3. 부족한 정보 검색
                for info in missing_info[:3]:
                    expanded.extend(
                        self._search_by_concept(session, info)
                    )

                # 4. 법령 조항 직접 검색
                expanded.extend(
                    self._search_legal_articles(session, query)
                )

        except Exception as e:
            print(f"Graph expansion error: {e}")

        return expanded

    def _traverse_from_anchors(
        self,
        session,
        anchor_ids: List[str],
        max_hops: int
    ) -> List[RetrievedDocument]:
        """앵커 문서에서 그래프 탐색"""
        docs = []

        # 2-hop 탐색: 문서 -> 법령 -> 관련 판례
        query = """
        MATCH path = (d:Document)-[:CITES*1..{max_hops}]->(related)
        WHERE d.id IN $anchor_ids
        AND related.id <> d.id
        AND (related:Law OR related:Precedent OR related:Document)
        RETURN DISTINCT related.id AS id,
               related.content AS text,
               related.source AS source,
               labels(related)[0] AS type,
               length(path) AS hops
        ORDER BY hops ASC
        LIMIT 5
        """.replace("{max_hops}", str(max_hops))

        try:
            result = session.run(query, anchor_ids=anchor_ids)
            for record in result:
                doc_type = record["type"].lower()
                docs.append(RetrievedDocument(
                    id=record["id"],
                    text=record["text"] or "",
                    source=record["source"] or "graph",
                    score=0.85 - (record["hops"] * 0.1),
                    relevance=DocumentRelevance.RELEVANT,
                    metadata={
                        "type": doc_type,
                        "source": "graph_traverse",
                        "hops": record["hops"]
                    }
                ))
        except Exception as e:
            print(f"Graph traverse error: {e}")

        return docs

    def _search_risk_patterns(
        self,
        session,
        query: str
    ) -> List[RetrievedDocument]:
        """위험 패턴 검색"""
        docs = []

        # 노동법 관련 위험 키워드
        risk_keywords = [
            "포괄임금", "위약금", "해고", "최저임금", "연장근로",
            "야간근로", "휴일근로", "퇴직금", "4대보험", "연차"
        ]
        matched_keywords = [kw for kw in risk_keywords if kw in query]

        if not matched_keywords:
            return docs

        pattern_query = """
        MATCH (r:RiskPattern)-[:HAS_CASE]->(p:Precedent)
        WHERE any(trigger IN r.triggers WHERE $query CONTAINS trigger)
        RETURN r.name AS pattern_name,
               r.explanation AS explanation,
               r.riskLevel AS risk_level,
               r.legalBasis AS legal_basis,
               p.content AS precedent_text,
               p.id AS precedent_id
        LIMIT 3
        """

        try:
            result = session.run(pattern_query, query=query)
            for record in result:
                docs.append(RetrievedDocument(
                    id=f"risk_{record['pattern_name']}",
                    text=f"[위험 패턴: {record['pattern_name']}]\n"
                         f"위험도: {record['risk_level']}\n"
                         f"법적 근거: {record['legal_basis']}\n"
                         f"설명: {record['explanation']}\n"
                         f"관련 판례: {record['precedent_text'][:500] if record['precedent_text'] else '없음'}",
                    source="risk_pattern",
                    score=0.9,
                    relevance=DocumentRelevance.HIGHLY_RELEVANT,
                    legal_references=[record.get('legal_basis', '')],
                    metadata={
                        "pattern": record["pattern_name"],
                        "risk_level": record["risk_level"],
                        "source": "graph_risk_pattern"
                    }
                ))
        except Exception as e:
            print(f"Risk pattern search error: {e}")

        return docs

    def _search_by_concept(
        self,
        session,
        concept: str
    ) -> List[RetrievedDocument]:
        """법적 개념으로 검색"""
        docs = []

        query = """
        MATCH (c:Concept)-[:DEFINED_BY]->(law:Law)
        WHERE c.name CONTAINS $concept OR c.keywords CONTAINS $concept
        RETURN c.name AS concept_name,
               c.definition AS definition,
               law.article AS article,
               law.content AS law_content
        LIMIT 2
        """

        try:
            result = session.run(query, concept=concept)
            for record in result:
                docs.append(RetrievedDocument(
                    id=f"concept_{record['concept_name']}",
                    text=f"[법적 개념: {record['concept_name']}]\n"
                         f"정의: {record['definition']}\n"
                         f"근거 조항: {record['article']}\n"
                         f"조문 내용: {record['law_content'][:300] if record['law_content'] else ''}",
                    source="legal_concept",
                    score=0.8,
                    relevance=DocumentRelevance.RELEVANT,
                    legal_references=[record.get('article', '')],
                    metadata={"concept": record["concept_name"]}
                ))
        except Exception as e:
            print(f"Concept search error: {e}")

        return docs

    def _search_legal_articles(
        self,
        session,
        query: str
    ) -> List[RetrievedDocument]:
        """법령 조항 직접 검색"""
        docs = []

        # 법령 조항 패턴 추출
        import re
        law_patterns = [
            r'근로기준법\s*제?\s*(\d+)\s*조',
            r'최저임금법\s*제?\s*(\d+)\s*조',
            r'근로자퇴직급여보장법\s*제?\s*(\d+)\s*조',
        ]

        search_articles = []
        for pattern in law_patterns:
            matches = re.findall(pattern, query)
            for match in matches:
                search_articles.append(match)

        if not search_articles:
            return docs

        article_query = """
        MATCH (law:Law)
        WHERE law.articleNumber IN $articles
        RETURN law.name AS law_name,
               law.articleNumber AS article_num,
               law.content AS content,
               law.interpretation AS interpretation
        LIMIT 3
        """

        try:
            result = session.run(article_query, articles=search_articles)
            for record in result:
                docs.append(RetrievedDocument(
                    id=f"law_{record['law_name']}_{record['article_num']}",
                    text=f"[{record['law_name']} 제{record['article_num']}조]\n"
                         f"{record['content']}\n\n"
                         f"해석: {record['interpretation'] or ''}",
                    source="law",
                    score=0.95,
                    relevance=DocumentRelevance.HIGHLY_RELEVANT,
                    legal_references=[f"{record['law_name']} 제{record['article_num']}조"],
                    metadata={"type": "law", "article": record['article_num']}
                ))
        except Exception as e:
            print(f"Legal article search error: {e}")

        return docs

    def _fallback_search(self, query: str) -> List[RetrievedDocument]:
        """폴백 검색 (Elasticsearch BM25)"""
        if self.es_client is None:
            return []

        try:
            result = self.es_client.search(
                index="docscanner_chunks",
                body={
                    "query": {
                        "multi_match": {
                            "query": query,
                            "fields": ["text^2", "title", "keywords"],
                            "type": "best_fields",
                            "fuzziness": "AUTO"
                        }
                    },
                    "size": 5
                }
            )

            docs = []
            for hit in result["hits"]["hits"]:
                docs.append(RetrievedDocument(
                    id=hit["_id"],
                    text=hit["_source"].get("text", ""),
                    source=hit["_source"].get("source", "elasticsearch"),
                    score=min(1.0, hit["_score"] / 10),
                    metadata={"source": "fallback_bm25"}
                ))

            return docs

        except Exception as e:
            print(f"Fallback search error: {e}")
            return []

    def _search_with_query(self, query: str) -> List[RetrievedDocument]:
        """쿼리로 검색"""
        # Elasticsearch 사용
        return self._fallback_search(query)

    def _merge_documents(
        self,
        existing: List[RetrievedDocument],
        new_docs: List[RetrievedDocument]
    ) -> List[RetrievedDocument]:
        """문서 병합 (중복 제거)"""
        seen = {d.id for d in existing}
        merged = list(existing)

        for doc in new_docs:
            if doc.id not in seen:
                seen.add(doc.id)
                merged.append(doc)

        return merged

    def _build_context(self, docs: List[RetrievedDocument]) -> str:
        """최종 컨텍스트 문자열 생성"""
        if not docs:
            return ""

        context_parts = []
        for i, doc in enumerate(docs[:10]):
            source_info = f"[출처: {doc.source}]" if doc.source else ""
            confidence_info = f"(신뢰도: {doc.confidence:.2f})" if doc.confidence else ""

            # 정제된 텍스트 사용
            text = doc.extracted_info if doc.extracted_info else doc.text

            # 법적 참조 추가
            legal_refs = ""
            if doc.legal_references:
                legal_refs = f"\n법적 근거: {', '.join(doc.legal_references)}"

            context_parts.append(
                f"[문서 {i+1}] {source_info} {confidence_info}\n{text}{legal_refs}"
            )

        return "\n\n---\n\n".join(context_parts)

    def _calculate_final_confidence(self, result: CRAGResult) -> float:
        """최종 신뢰도 계산"""
        if not result.all_docs:
            return 0.0

        # 문서들의 평균 신뢰도
        avg_confidence = sum(d.confidence for d in result.all_docs) / len(result.all_docs)

        # 품질 등급에 따른 가중치
        quality_weights = {
            RetrievalQuality.EXCELLENT: 1.0,
            RetrievalQuality.GOOD: 0.9,
            RetrievalQuality.CORRECT: 0.8,
            RetrievalQuality.PARTIAL: 0.6,
            RetrievalQuality.AMBIGUOUS: 0.5,
            RetrievalQuality.WEAK: 0.3,
            RetrievalQuality.INCORRECT: 0.1,
            RetrievalQuality.HARMFUL: 0.0,
        }
        quality_weight = quality_weights.get(result.quality, 0.5)

        # 보정 반복 횟수에 따른 패널티 (많이 반복할수록 원래 결과가 불충분)
        iteration_penalty = max(0, 1 - (result.correction_iterations * 0.1))

        return avg_confidence * quality_weight * iteration_penalty

    # ========== 캐싱 메서드 ==========

    def _generate_cache_key(
        self,
        query: str,
        docs: List[RetrievedDocument]
    ) -> str:
        """캐시 키 생성"""
        doc_ids = sorted([d.id for d in docs[:5]])
        content = f"{query}|{'|'.join(doc_ids)}"
        return hashlib.md5(content.encode()).hexdigest()

    def _get_cached_evaluation(
        self,
        cache_key: str
    ) -> Optional[QualityEvaluation]:
        """캐시된 평가 조회"""
        if not self.enable_caching:
            return None

        if cache_key in self._evaluation_cache:
            evaluation, timestamp = self._evaluation_cache[cache_key]
            elapsed = (datetime.now() - timestamp).total_seconds()
            if elapsed < self.cache_ttl_seconds:
                return evaluation

        return None

    def _cache_evaluation(
        self,
        cache_key: str,
        evaluation: QualityEvaluation
    ):
        """평가 결과 캐싱"""
        if self.enable_caching:
            self._evaluation_cache[cache_key] = (evaluation, datetime.now())

            # 오래된 캐시 정리
            if len(self._evaluation_cache) > 100:
                oldest_keys = sorted(
                    self._evaluation_cache.keys(),
                    key=lambda k: self._evaluation_cache[k][1]
                )[:50]
                for key in oldest_keys:
                    del self._evaluation_cache[key]


class CRAGWorkflow:
    """
    CRAG 워크플로우 (Dify 통합용)
    """

    def __init__(self, crag: GraphGuidedCRAG):
        self.crag = crag

    def execute(
        self,
        query: str,
        initial_results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        CRAG 워크플로우 실행 (동기)

        Args:
            query: 검색 쿼리
            initial_results: 초기 검색 결과 (dict 형태)

        Returns:
            워크플로우 결과
        """
        # Dict를 RetrievedDocument로 변환
        initial_docs = [
            RetrievedDocument(
                id=r.get("id", f"doc_{i}"),
                text=r.get("text", ""),
                source=r.get("source", ""),
                score=r.get("score", 0.5),
                metadata=r.get("metadata", {})
            )
            for i, r in enumerate(initial_results)
        ]

        # CRAG 실행 (동기)
        result = self.crag.retrieve_and_correct_sync(query, initial_docs)

        return result.to_dict()

    async def execute_async(
        self,
        query: str,
        initial_results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """비동기 실행"""
        initial_docs = [
            RetrievedDocument(
                id=r.get("id", f"doc_{i}"),
                text=r.get("text", ""),
                source=r.get("source", ""),
                score=r.get("score", 0.5),
                metadata=r.get("metadata", {})
            )
            for i, r in enumerate(initial_results)
        ]

        result = await self.crag.retrieve_and_correct(query, initial_docs)
        return result.to_dict()


# ========== 편의 함수 ==========

def create_crag(
    es_client: Optional[Any] = None,
    neo4j_driver: Optional[Any] = None
) -> GraphGuidedCRAG:
    """CRAG 인스턴스 생성"""
    return GraphGuidedCRAG(
        es_client=es_client,
        neo4j_driver=neo4j_driver
    )


def evaluate_retrieval_quality(
    query: str,
    documents: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """검색 품질 간단 평가"""
    crag = GraphGuidedCRAG()

    docs = [
        RetrievedDocument(
            id=d.get("id", f"doc_{i}"),
            text=d.get("text", ""),
            source=d.get("source", ""),
            score=d.get("score", 0.5)
        )
        for i, d in enumerate(documents)
    ]

    evaluation = crag._evaluate_quality(query, docs)

    return {
        "quality": evaluation.quality.value,
        "confidence": evaluation.confidence,
        "reasoning": evaluation.reasoning,
        "missing_info": evaluation.missing_info,
        "suggested_strategy": evaluation.correction_strategy.value,
    }
