"""
HyDE (Hypothetical Document Embeddings) - Production-Grade Implementation
- 다중 가상 문서 생성 및 앙상블
- 법률 도메인 특화 프롬프트 템플릿
- 의미적 유사도 기반 가중치 앙상블
- 쿼리 복잡도 분석 및 적응형 생성
- 캐싱 및 배치 처리 최적화

Reference: Gao et al., "Precise Zero-Shot Dense Retrieval without Relevance Labels"
"""

import os
import hashlib
import asyncio
import time
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import numpy as np
from collections import OrderedDict
import json

from app.core.config import settings
from app.core.token_usage_tracker import record_llm_usage


class QueryComplexity(Enum):
    """쿼리 복잡도 등급"""
    SIMPLE = "simple"           # 단순 키워드 질문
    MODERATE = "moderate"       # 일반적 법률 질문
    COMPLEX = "complex"         # 복합적 법률 해석 필요
    EXPERT = "expert"           # 전문가 수준 분석 필요


class HyDEStrategy(Enum):
    """HyDE 생성 전략"""
    SINGLE = "single"               # 단일 문서 생성
    ENSEMBLE = "ensemble"           # 다중 문서 앙상블
    ADAPTIVE = "adaptive"           # 쿼리 복잡도 기반 적응형
    MULTI_PERSPECTIVE = "multi_perspective"  # 다관점 생성


@dataclass
class HyDEResult:
    """HyDE 처리 결과 (고도화)"""
    original_query: str
    hypothetical_documents: List[str]  # 다중 문서 지원
    primary_document: str              # 대표 문서
    ensemble_embedding: Optional[np.ndarray] = None
    individual_embeddings: List[np.ndarray] = field(default_factory=list)
    query_complexity: QueryComplexity = QueryComplexity.MODERATE
    strategy_used: HyDEStrategy = HyDEStrategy.SINGLE
    confidence_scores: List[float] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    processing_time_ms: float = 0.0
    cache_hit: bool = False

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        # 대표 문서 설정
        if self.hypothetical_documents and not self.primary_document:
            self.primary_document = self.hypothetical_documents[0]


@dataclass
class LRUCache:
    """LRU 캐시 구현"""
    max_size: int = 1000
    ttl_seconds: int = 3600  # 1시간
    _cache: OrderedDict = field(default_factory=OrderedDict)
    _timestamps: Dict[str, datetime] = field(default_factory=dict)

    def get(self, key: str) -> Optional[Any]:
        if key not in self._cache:
            return None
        # TTL 체크
        if datetime.now() - self._timestamps[key] > timedelta(seconds=self.ttl_seconds):
            del self._cache[key]
            del self._timestamps[key]
            return None
        # LRU 업데이트
        self._cache.move_to_end(key)
        return self._cache[key]

    def set(self, key: str, value: Any):
        if key in self._cache:
            self._cache.move_to_end(key)
        else:
            if len(self._cache) >= self.max_size:
                oldest_key = next(iter(self._cache))
                del self._cache[oldest_key]
                del self._timestamps[oldest_key]
        self._cache[key] = value
        self._timestamps[key] = datetime.now()


class HyDEGenerator:
    """
    HyDE (Hypothetical Document Embeddings) 생성기 - 상용화 버전

    주요 특징:
    1. 다중 가상 문서 앙상블: 3-5개 문서 생성 후 가중 평균
    2. 적응형 전략: 쿼리 복잡도에 따른 자동 조절
    3. 다관점 생성: 법률/판례/행정해석 관점
    4. 신뢰도 점수: 각 문서의 품질 평가
    5. 캐싱: LRU 캐시로 반복 쿼리 최적화

    사용법:
        hyde = HyDEGenerator(strategy=HyDEStrategy.ENSEMBLE)
        result = hyde.generate("포괄임금제가 유효한가요?")
        print(f"앙상블 문서 수: {len(result.hypothetical_documents)}")
    """

    # 근로기준법 전문가 관점 프롬프트
    LABOR_LAW_EXPERT_PROMPT = """당신은 30년 경력의 대한민국 노동법 전문 변호사입니다.
다음 질문에 대해 근로기준법, 최저임금법, 산업안전보건법 등 관련 법령과
대법원 판례, 고용노동부 행정해석을 근거로 전문적인 법률 의견서를 작성하세요.

[질문]
{query}

[법률 의견서 형식]
1. 쟁점 분석
   - 법적 쟁점 정리
   - 적용 법률 조항

2. 관련 법령
   - 근로기준법 제X조 (조문 내용)
   - 기타 관련 법률

3. 판례 및 행정해석
   - 대법원 XXXX.XX.XX. 선고 XXXX다XXXXX 판결
   - 고용노동부 행정해석 (관련 있는 경우)

4. 법적 결론
   - 유효성 판단
   - 근로자 권리 보호 관점

5. 권고사항
   - 실무적 대응 방안

[법률 의견서]"""

    # 판례 분석가 관점 프롬프트
    PRECEDENT_ANALYST_PROMPT = """당신은 법원 판례 연구관입니다.
다음 질문과 관련된 핵심 판례들을 분석하여 법률적 기준을 제시하세요.

[질문]
{query}

[판례 분석 형식]
1. 대법원 판례 요지
   - 판결 번호 및 일자
   - 핵심 판시 사항
   - 적용 법리

2. 하급심 판례 동향
   - 일관된 해석 여부
   - 최근 경향

3. 실무상 시사점
   - 유의사항
   - 증명책임 소재

[판례 분석]"""

    # 행정해석 관점 프롬프트
    ADMINISTRATIVE_PROMPT = """당신은 고용노동부 근로개선정책과 담당관입니다.
다음 질문에 대해 고용노동부의 공식 행정해석 및 지침을 기반으로 답변하세요.

[질문]
{query}

[행정해석 형식]
1. 관련 법령 조항
   - 법령명 및 조항
   - 시행령/시행규칙 관련 조항

2. 고용노동부 해석
   - 기존 행정해석 사례
   - 유권해석 기준

3. 근로감독 실무 기준
   - 위반 판단 기준
   - 시정 지시 사항

4. 결론
   - 행정해석 요약
   - 실무 권고

[행정해석]"""

    # 계약서 조항 분석 프롬프트 (고도화)
    CONTRACT_ANALYSIS_PROMPT = """당신은 공인노무사이자 계약서 검토 전문가입니다.
다음 계약서 조항을 근로기준법 관점에서 심층 분석하세요.

[계약서 조항]
{query}

[분석 항목]
1. 조항의 법적 성격
   - 조항 유형 (임금/근로시간/휴가/해고 등)
   - 강행규정 vs 임의규정

2. 근로기준법 위반 여부
   - 관련 조항: 근로기준법 제X조
   - 위반 여부 판단
   - 위반 시 효력 (무효/취소 등)

3. 위험도 평가
   - 위험 수준: High/Medium/Low
   - 근로자 불이익 정도
   - 분쟁 가능성

4. 판례 및 행정해석
   - 유사 사례 판결
   - 고용노동부 해석

5. 수정 권고
   - 문제점 요약
   - 구체적 수정안

[분석 결과]"""

    # 위험 패턴 탐지 프롬프트 (고도화)
    RISK_DETECTION_PROMPT = """당신은 근로계약서 위험조항 탐지 AI 시스템입니다.
다음 텍스트에서 근로자에게 불리한 조항, 법적 위험 요소를 체계적으로 탐지하세요.

[분석 대상]
{query}

[탐지 대상 위험 패턴]
1. 임금 관련 위험
   - 포괄임금제 (실질 연장근로 미반영)
   - 최저임금 미달 (2025년 시급 10,030원 기준)
   - 연장/야간/휴일근로수당 미지급
   - 주휴수당 미포함

2. 근로시간 관련 위험
   - 주 52시간 초과
   - 휴게시간 미부여
   - 야간/휴일근로 상시화

3. 계약 종료 관련 위험
   - 부당해고 조항
   - 과도한 위약금/손해배상 예정
   - 경업금지 과도

4. 기타 위험
   - 불명확한 근로조건
   - 근로자 권리 포기 조항
   - 일방적 변경 조항

[분석 형식]
각 위험 패턴에 대해:
- 패턴명:
- 위험도: High/Medium/Low
- 해당 문구:
- 법적 근거:
- 설명:

[위험 분석]"""

    # 구어체 → 법률 용어 확장 매핑 (고도화)
    LEGAL_TERMINOLOGY_MAP = {
        # 해고/퇴직 관련
        "잘라": ["해고", "근로관계 종료", "계약 해지", "해촉", "근로기준법 제23조", "부당해고"],
        "짤려": ["해고", "근로관계 종료", "계약 해지"],
        "나가라": ["퇴직 권고", "해고 통보", "권고사직"],
        "퇴사": ["퇴직", "사직", "근로관계 종료", "자발적 이직"],

        # 임금 관련
        "월급": ["임금", "급여", "보수", "근로의 대가", "근로기준법 제43조"],
        "돈": ["임금", "급여", "수당", "보수"],
        "안줘": ["임금 체불", "급여 미지급", "근로기준법 제43조 위반"],
        "덜줘": ["임금 삭감", "급여 감액", "불이익 변경"],
        "시급": ["시간급", "시간당 임금", "최저임금법"],

        # 근로시간 관련
        "야근": ["연장근로", "시간외근로", "근로기준법 제53조", "연장근로수당"],
        "밤일": ["야간근로", "야간근무", "22시-06시 근로"],
        "주말": ["휴일근로", "휴일근무", "주휴일"],
        "쉬는날": ["휴일", "휴가", "연차유급휴가", "근로기준법 제60조"],

        # 계약 관련
        "계약": ["근로계약", "고용계약", "근로기준법 제17조"],
        "각서": ["서약서", "약정서", "동의서"],
        "사인": ["서명", "날인", "기명날인"],

        # 당사자 관련
        "사장": ["사용자", "사업주", "사업자", "대표이사"],
        "부장": ["관리자", "중간관리자", "사용자 대리인"],
        "알바": ["단시간근로", "시간제근로", "파트타임", "근로기준법 제18조"],
        "정직원": ["정규직", "기간의 정함이 없는 근로자"],

        # 수당 관련
        "수당": ["연장근로수당", "야간근로수당", "휴일근로수당", "가산임금"],
        "보너스": ["상여금", "성과급", "정기상여"],

        # 위험 패턴 키워드
        "포괄": ["포괄임금제", "포괄임금약정", "고정OT"],
        "최저": ["최저임금", "최저임금법", "시급 10,030원"],
        "위약금": ["위약 예정", "손해배상 예정", "근로기준법 제20조"],
    }

    def __init__(
        self,
        llm_client: Optional[Any] = None,
        model: str = None,
        temperature: float = 0.3,
        embedding_model: Optional[Any] = None,
        strategy: HyDEStrategy = HyDEStrategy.ADAPTIVE,
        num_documents: int = 3,
        enable_cache: bool = True,
        cache_size: int = 1000,
        cache_ttl: int = 3600,
        contract_id: Optional[str] = None
    ):
        """
        Args:
            llm_client: OpenAI 클라이언트 (legacy, Gemini 사용 시 무시됨)
            model: LLM 모델명 (기본값: settings.LLM_HYDE_MODEL)
            temperature: 생성 온도
            embedding_model: 임베딩 모델 (sentence-transformers)
            strategy: HyDE 생성 전략
            num_documents: 앙상블 시 생성할 문서 수
            enable_cache: 캐싱 활성화
            cache_size: 캐시 최대 크기
            cache_ttl: 캐시 TTL (초)
            contract_id: 계약서 ID (토큰 추적용)
        """
        self.model = model if model else settings.LLM_HYDE_MODEL
        self.temperature = temperature
        self.strategy = strategy
        self.num_documents = num_documents
        self.enable_cache = enable_cache
        self.contract_id = contract_id
        self.llm_client = llm_client  # OpenAI fallback

        # 캐시 초기화
        if enable_cache:
            self._cache = LRUCache(max_size=cache_size, ttl_seconds=cache_ttl)
        else:
            self._cache = None

        # Gemini safety settings (완전 완화 - 계약서 분석은 합법적 용도)
        self.safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
        ]

        # Gemini 클라이언트 초기화 (우선 사용)
        self._gemini_model = None
        if "gemini" in self.model.lower():
            try:
                import google.generativeai as genai
                genai.configure(api_key=settings.GEMINI_API_KEY)
                self._gemini_model = genai.GenerativeModel(self.model)
            except ImportError:
                print("google-generativeai package not installed, falling back to OpenAI")
            except Exception as e:
                print(f"Gemini initialization error: {e}")

        # OpenAI fallback
        if self._gemini_model is None and self.llm_client is None:
            try:
                from openai import OpenAI
                self.llm_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            except ImportError:
                self.llm_client = None

        # 임베딩 모델 초기화
        if embedding_model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self.embedding_model = SentenceTransformer("nlpai-lab/KURE-v1")
            except ImportError:
                self.embedding_model = None
        else:
            self.embedding_model = embedding_model

    def generate(
        self,
        query: str,
        prompt_type: str = "labor_law",
        generate_embedding: bool = True,
        force_refresh: bool = False
    ) -> HyDEResult:
        """
        HyDE 문서 생성 (고도화)

        Args:
            query: 원본 질문/조항
            prompt_type: 프롬프트 유형
            generate_embedding: 임베딩 생성 여부
            force_refresh: 캐시 무시

        Returns:
            HyDEResult: 고도화된 HyDE 결과
        """
        import time
        start_time = time.time()

        # 캐시 확인
        cache_key = self._generate_cache_key(query, prompt_type)
        if self._cache and not force_refresh:
            cached = self._cache.get(cache_key)
            if cached:
                cached.cache_hit = True
                return cached

        # 쿼리 복잡도 분석
        complexity = self._analyze_query_complexity(query)

        # 전략 결정
        effective_strategy = self._determine_strategy(complexity)

        # 가상 문서 생성
        if effective_strategy == HyDEStrategy.SINGLE:
            documents = [self._generate_single_document(query, prompt_type)]
            confidence_scores = [1.0]

        elif effective_strategy == HyDEStrategy.ENSEMBLE:
            documents, confidence_scores = self._generate_ensemble_documents(
                query, prompt_type, self.num_documents
            )

        elif effective_strategy == HyDEStrategy.MULTI_PERSPECTIVE:
            documents, confidence_scores = self._generate_multi_perspective_documents(query)

        else:  # ADAPTIVE
            if complexity == QueryComplexity.SIMPLE:
                documents = [self._generate_single_document(query, prompt_type)]
                confidence_scores = [1.0]
            elif complexity == QueryComplexity.EXPERT:
                documents, confidence_scores = self._generate_multi_perspective_documents(query)
            else:
                documents, confidence_scores = self._generate_ensemble_documents(
                    query, prompt_type, self.num_documents
                )

        # 임베딩 생성
        individual_embeddings = []
        ensemble_embedding = None

        if generate_embedding and self.embedding_model is not None:
            individual_embeddings = [
                self._generate_embedding(doc) for doc in documents
            ]
            # 가중 앙상블 임베딩
            if len(individual_embeddings) > 1:
                ensemble_embedding = self._weighted_ensemble_embedding(
                    individual_embeddings, confidence_scores
                )
            elif individual_embeddings:
                ensemble_embedding = individual_embeddings[0]

        processing_time = (time.time() - start_time) * 1000

        result = HyDEResult(
            original_query=query,
            hypothetical_documents=documents,
            primary_document=documents[0] if documents else "",
            ensemble_embedding=ensemble_embedding,
            individual_embeddings=individual_embeddings,
            query_complexity=complexity,
            strategy_used=effective_strategy,
            confidence_scores=confidence_scores,
            metadata={
                "prompt_type": prompt_type,
                "model": self.model,
                "temperature": self.temperature,
                "num_documents": len(documents)
            },
            processing_time_ms=processing_time,
            cache_hit=False
        )

        # 캐시 저장
        if self._cache:
            self._cache.set(cache_key, result)

        return result

    async def generate_async(
        self,
        query: str,
        prompt_type: str = "labor_law",
        generate_embedding: bool = True
    ) -> HyDEResult:
        """비동기 HyDE 생성"""
        return await asyncio.to_thread(
            self.generate, query, prompt_type, generate_embedding
        )

    def generate_batch(
        self,
        queries: List[str],
        prompt_type: str = "labor_law",
        generate_embedding: bool = True,
        max_concurrent: int = 5
    ) -> List[HyDEResult]:
        """배치 HyDE 생성"""
        results = []
        for query in queries:
            results.append(self.generate(query, prompt_type, generate_embedding))
        return results

    def _analyze_query_complexity(self, query: str) -> QueryComplexity:
        """쿼리 복잡도 분석"""
        # 길이 기반 점수
        length_score = min(len(query) / 200, 1.0)

        # 법률 용어 기반 점수
        legal_terms = [
            "제", "조", "항", "호", "근로기준법", "대법원", "판결",
            "해석", "위반", "무효", "취소", "손해배상", "청구"
        ]
        term_count = sum(1 for term in legal_terms if term in query)
        term_score = min(term_count / 5, 1.0)

        # 복합 질문 여부
        question_words = ["어떻게", "왜", "무엇", "언제", "누가", "어디"]
        question_count = sum(1 for word in question_words if word in query)

        # 종합 점수
        total_score = (length_score * 0.3) + (term_score * 0.5) + (question_count * 0.1)

        if total_score < 0.2:
            return QueryComplexity.SIMPLE
        elif total_score < 0.5:
            return QueryComplexity.MODERATE
        elif total_score < 0.8:
            return QueryComplexity.COMPLEX
        else:
            return QueryComplexity.EXPERT

    def _determine_strategy(self, complexity: QueryComplexity) -> HyDEStrategy:
        """복잡도에 따른 전략 결정"""
        if self.strategy != HyDEStrategy.ADAPTIVE:
            return self.strategy

        strategy_map = {
            QueryComplexity.SIMPLE: HyDEStrategy.SINGLE,
            QueryComplexity.MODERATE: HyDEStrategy.ENSEMBLE,
            QueryComplexity.COMPLEX: HyDEStrategy.ENSEMBLE,
            QueryComplexity.EXPERT: HyDEStrategy.MULTI_PERSPECTIVE
        }
        return strategy_map[complexity]

    def _generate_single_document(self, query: str, prompt_type: str) -> str:
        """단일 가상 문서 생성"""
        prompt_template = self._get_prompt_template(prompt_type)
        prompt = prompt_template.format(query=query)
        return self._call_llm(prompt)

    def _generate_ensemble_documents(
        self,
        query: str,
        prompt_type: str,
        num_docs: int
    ) -> Tuple[List[str], List[float]]:
        """앙상블 가상 문서 생성"""
        documents = []
        confidence_scores = []

        # 다양한 temperature로 생성
        temperatures = [0.2, 0.4, 0.6, 0.3, 0.5][:num_docs]
        prompt_template = self._get_prompt_template(prompt_type)
        prompt = prompt_template.format(query=query)

        for i, temp in enumerate(temperatures):
            doc = self._call_llm(prompt, temperature=temp)
            documents.append(doc)
            # 낮은 temperature일수록 높은 신뢰도
            confidence = 1.0 - (temp * 0.5)
            confidence_scores.append(confidence)

        return documents, confidence_scores

    def _generate_multi_perspective_documents(
        self,
        query: str
    ) -> Tuple[List[str], List[float]]:
        """다관점 가상 문서 생성"""
        documents = []
        confidence_scores = []

        # 세 가지 관점에서 생성
        perspectives = [
            (self.LABOR_LAW_EXPERT_PROMPT, 1.0),      # 법률 전문가
            (self.PRECEDENT_ANALYST_PROMPT, 0.9),     # 판례 분석가
            (self.ADMINISTRATIVE_PROMPT, 0.85),       # 행정해석 관점
        ]

        for prompt_template, confidence in perspectives:
            prompt = prompt_template.format(query=query)
            doc = self._call_llm(prompt)
            documents.append(doc)
            confidence_scores.append(confidence)

        return documents, confidence_scores

    def _is_reasoning_model(self) -> bool:
        """reasoning 모델 여부 확인 (temperature 미지원)"""
        reasoning_keywords = ["o1", "o3", "gpt-5"]
        return any(kw in self.model.lower() for kw in reasoning_keywords)

    def _call_llm(self, prompt: str, temperature: float = None) -> str:
        """LLM 호출"""
        if self._gemini_model is None and self.llm_client is None:
            return self._fallback_expansion(prompt)

        llm_start = time.time()
        try:
            # Gemini 사용 (우선)
            if self._gemini_model is not None:
                full_prompt = "당신은 대한민국 노동법 전문가입니다. 정확하고 상세한 법률 정보를 제공합니다.\n\n" + prompt
                result = self._gemini_model.generate_content(
                    full_prompt,
                    generation_config={
                        "temperature": temperature or self.temperature,
                        "max_output_tokens": 1500
                    },
                    safety_settings=self.safety_settings
                )
                llm_duration = (time.time() - llm_start) * 1000

                # 토큰 사용량 기록
                if self.contract_id and hasattr(result, 'usage_metadata'):
                    usage = result.usage_metadata
                    record_llm_usage(
                        contract_id=self.contract_id,
                        module="hyde.generate",
                        model=self.model,
                        input_tokens=getattr(usage, 'prompt_token_count', 0),
                        output_tokens=getattr(usage, 'candidates_token_count', 0),
                        cached_tokens=getattr(usage, 'cached_content_token_count', 0),
                        duration_ms=llm_duration
                    )

                # 응답 안전하게 추출 (SAFETY 차단 대응)
                if result.candidates and len(result.candidates) > 0:
                    candidate = result.candidates[0]
                    if candidate.content and candidate.content.parts:
                        return candidate.content.parts[0].text

                # 차단된 경우 OpenAI fallback 시도
                if self.llm_client is not None:
                    print(f">>> [HyDE] Gemini blocked, falling back to OpenAI")
                    return self._call_llm_openai(prompt, temperature)

                return self._fallback_expansion(prompt)

            # OpenAI fallback
            else:
                # reasoning 모델은 temperature 미지원, system message도 user로 통합
                if self._is_reasoning_model():
                    response = self.llm_client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {"role": "user", "content": f"당신은 대한민국 노동법 전문가입니다. 정확하고 상세한 법률 정보를 제공합니다.\n\n{prompt}"}
                        ],
                        max_completion_tokens=1500
                    )
                else:
                    response = self.llm_client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {
                                "role": "system",
                                "content": "당신은 대한민국 노동법 전문가입니다. 정확하고 상세한 법률 정보를 제공합니다."
                            },
                            {"role": "user", "content": prompt}
                        ],
                        temperature=temperature or self.temperature,
                        max_completion_tokens=1500
                    )

                llm_duration = (time.time() - llm_start) * 1000

                # 토큰 사용량 기록
                if response.usage and self.contract_id:
                    cached = 0
                    if hasattr(response.usage, 'prompt_tokens_details') and response.usage.prompt_tokens_details:
                        cached = getattr(response.usage.prompt_tokens_details, 'cached_tokens', 0) or 0
                    record_llm_usage(
                        contract_id=self.contract_id,
                        module="hyde.generate",
                        model=self.model,
                        input_tokens=response.usage.prompt_tokens,
                        output_tokens=response.usage.completion_tokens,
                        cached_tokens=cached,
                        duration_ms=llm_duration
                    )

                return response.choices[0].message.content
        except Exception as e:
            print(f"HyDE LLM error: {e}")
            return self._fallback_expansion(prompt)

    def _get_prompt_template(self, prompt_type: str) -> str:
        """프롬프트 템플릿 반환"""
        templates = {
            "labor_law": self.LABOR_LAW_EXPERT_PROMPT,
            "contract": self.CONTRACT_ANALYSIS_PROMPT,
            "risk": self.RISK_DETECTION_PROMPT,
            "precedent": self.PRECEDENT_ANALYST_PROMPT,
            "administrative": self.ADMINISTRATIVE_PROMPT,
        }
        return templates.get(prompt_type, self.LABOR_LAW_EXPERT_PROMPT)

    def _fallback_expansion(self, prompt: str) -> str:
        """LLM 없이 키워드 기반 확장"""
        expanded_parts = [prompt]

        for colloquial, legal_terms in self.LEGAL_TERMINOLOGY_MAP.items():
            if colloquial in prompt.lower():
                expanded_parts.append(f"\n[관련 법률 용어] {', '.join(legal_terms)}")

        return "\n".join(expanded_parts)

    def _generate_embedding(self, text: str) -> Optional[np.ndarray]:
        """텍스트 임베딩 생성"""
        if self.embedding_model is None:
            return None
        try:
            return self.embedding_model.encode(text, normalize_embeddings=True)
        except Exception as e:
            print(f"Embedding error: {e}")
            return None

    def _weighted_ensemble_embedding(
        self,
        embeddings: List[np.ndarray],
        weights: List[float]
    ) -> np.ndarray:
        """가중 앙상블 임베딩"""
        valid_embeddings = [e for e in embeddings if e is not None]
        if not valid_embeddings:
            return None

        # 가중치 정규화
        total_weight = sum(weights[:len(valid_embeddings)])
        normalized_weights = [w / total_weight for w in weights[:len(valid_embeddings)]]

        # 가중 평균
        ensemble = np.zeros_like(valid_embeddings[0])
        for emb, weight in zip(valid_embeddings, normalized_weights):
            ensemble += emb * weight

        # L2 정규화
        norm = np.linalg.norm(ensemble)
        if norm > 0:
            ensemble = ensemble / norm

        return ensemble

    def _generate_cache_key(self, query: str, prompt_type: str) -> str:
        """캐시 키 생성"""
        content = f"{query}:{prompt_type}:{self.strategy.value}"
        return hashlib.md5(content.encode()).hexdigest()

    def enhance_query(self, query: str, include_all_documents: bool = False) -> str:
        """
        검색 쿼리 강화

        Args:
            query: 원본 검색 쿼리
            include_all_documents: 모든 가상 문서 포함 여부

        Returns:
            강화된 검색 쿼리
        """
        result = self.generate(query, prompt_type="labor_law")

        if include_all_documents and len(result.hypothetical_documents) > 1:
            docs_text = "\n\n---\n\n".join(result.hypothetical_documents)
            return f"{query}\n\n[다관점 법률 분석]\n{docs_text}"
        else:
            return f"{query}\n\n[법률 정보]\n{result.primary_document}"


class HyDESearchEnhancer:
    """
    HyDE 기반 검색 강화 클래스 (상용화 버전)

    Elasticsearch와 통합하여 검색 품질 향상
    """

    def __init__(
        self,
        hyde_generator: HyDEGenerator,
        embedding_model: Any,
        search_client: Any
    ):
        self.hyde = hyde_generator
        self.embedding_model = embedding_model
        self.search_client = search_client

    def search(
        self,
        query: str,
        index: str = "docscanner_chunks",
        top_k: int = 5,
        use_hyde: bool = True,
        use_ensemble: bool = True,
        rerank: bool = True
    ) -> List[Dict[str, Any]]:
        """
        HyDE 강화 검색

        Args:
            query: 검색 쿼리
            index: Elasticsearch 인덱스
            top_k: 반환할 결과 수
            use_hyde: HyDE 적용 여부
            use_ensemble: 앙상블 임베딩 사용
            rerank: 결과 재순위화

        Returns:
            검색 결과 리스트
        """
        if use_hyde:
            hyde_result = self.hyde.generate(query, prompt_type="labor_law")

            if use_ensemble and hyde_result.ensemble_embedding is not None:
                search_embedding = hyde_result.ensemble_embedding
            else:
                search_embedding = self.embedding_model.encode(
                    hyde_result.primary_document,
                    normalize_embeddings=True
                )
        else:
            search_embedding = self.embedding_model.encode(
                query, normalize_embeddings=True
            )

        # Elasticsearch KNN 검색
        search_body = {
            "knn": {
                "field": "embedding",
                "query_vector": search_embedding.tolist(),
                "k": top_k * 2 if rerank else top_k,
                "num_candidates": top_k * 10
            }
        }

        results = self.search_client.search(index=index, body=search_body)

        hits = [
            {
                "score": hit["_score"],
                "source": hit["_source"].get("source", ""),
                "text": hit["_source"].get("text", ""),
                "metadata": hit["_source"].get("metadata", {})
            }
            for hit in results["hits"]["hits"]
        ]

        # 재순위화: 원본 쿼리와의 추가 유사도 계산
        if rerank and len(hits) > top_k:
            query_embedding = self.embedding_model.encode(
                query, normalize_embeddings=True
            )
            for hit in hits:
                text_embedding = self.embedding_model.encode(
                    hit["text"][:500], normalize_embeddings=True
                )
                query_sim = np.dot(query_embedding, text_embedding)
                hit["rerank_score"] = hit["score"] * 0.6 + query_sim * 0.4

            hits.sort(key=lambda x: x.get("rerank_score", x["score"]), reverse=True)

        return hits[:top_k]


# 편의 함수
def create_hyde_generator(
    model: str = None,
    strategy: HyDEStrategy = HyDEStrategy.ADAPTIVE,
    num_documents: int = 3
) -> HyDEGenerator:
    """HyDE 생성기 팩토리 함수"""
    return HyDEGenerator(
        model=model,  # None이면 settings.LLM_HYDE_MODEL 사용
        strategy=strategy,
        num_documents=num_documents
    )


def generate_hypothetical_document(
    query: str,
    prompt_type: str = "labor_law"
) -> str:
    """간편 가상 문서 생성"""
    generator = HyDEGenerator()
    result = generator.generate(query, prompt_type)
    return result.primary_document
