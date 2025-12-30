"""
DSPy Dynamic Few-Shot (Self-Evolving) - Production Grade
- 품질 기반 예시 선택 (A/B 테스트 기반 성능 평가)
- 적응형 임계값 시스템 (동적 피드백 임계값)
- 멀티 암 밴딧 기반 프롬프트 최적화
- 분산 피드백 수집 및 온라인 학습

Reference: Stanford NLP - DSPy: Compiling Declarative Language Model Calls into Self-Improving Pipelines
"""

import os
import json
import hashlib
import random
from typing import Dict, Any, Optional, List, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from enum import Enum
import math
from collections import defaultdict
import threading
from concurrent.futures import ThreadPoolExecutor


class ExampleQuality(Enum):
    """예시 품질 등급"""
    EXCELLENT = "excellent"     # 모범 사례
    GOOD = "good"               # 좋은 예시
    ADEQUATE = "adequate"       # 적절한 예시
    MARGINAL = "marginal"       # 경계선 예시
    POOR = "poor"               # 나쁜 예시


class FeedbackSource(Enum):
    """피드백 소스"""
    USER_EXPLICIT = "user_explicit"     # 사용자 명시적 피드백
    USER_IMPLICIT = "user_implicit"     # 사용자 암묵적 피드백 (클릭, 시간 등)
    SYSTEM_AUTO = "system_auto"         # 시스템 자동 평가
    EXPERT_REVIEW = "expert_review"     # 전문가 검토
    AB_TEST = "ab_test"                 # A/B 테스트 결과


@dataclass
class FeedbackRecord:
    """피드백 기록 (확장)"""
    id: str
    query: str
    response: str
    context: str = ""
    feedback_type: str = "positive"
    feedback_score: float = 1.0
    feedback_source: FeedbackSource = FeedbackSource.USER_EXPLICIT
    quality_grade: ExampleQuality = ExampleQuality.ADEQUATE
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    response_time_ms: float = 0.0
    token_count: int = 0
    prompt_version: int = 1
    was_edited: bool = False            # 사용자가 응답 수정했는지

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "query": self.query,
            "response": self.response,
            "context": self.context[:200] if self.context else "",
            "feedback_type": self.feedback_type,
            "feedback_score": self.feedback_score,
            "feedback_source": self.feedback_source.value,
            "quality_grade": self.quality_grade.value,
            "timestamp": self.timestamp.isoformat(),
            "prompt_version": self.prompt_version,
        }


@dataclass
class FewShotExample:
    """Few-Shot 예시 (확장)"""
    id: str
    query: str
    response: str
    context: str = ""
    score: float = 1.0
    category: str = "general"
    quality_grade: ExampleQuality = ExampleQuality.ADEQUATE
    usage_count: int = 0
    success_rate: float = 1.0           # 이 예시 사용 시 성공률
    avg_feedback_score: float = 1.0     # 평균 피드백 점수
    last_used: datetime = field(default_factory=datetime.now)
    created_at: datetime = field(default_factory=datetime.now)
    legal_category: str = ""            # 노동법 세부 카테고리
    complexity: str = "medium"          # simple/medium/complex

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "query": self.query,
            "response": self.response[:500],
            "context": self.context[:200] if self.context else "",
            "score": self.score,
            "category": self.category,
            "quality_grade": self.quality_grade.value,
            "usage_count": self.usage_count,
            "success_rate": self.success_rate,
            "legal_category": self.legal_category,
            "complexity": self.complexity,
        }


@dataclass
class OptimizedPrompt:
    """최적화된 프롬프트 (확장)"""
    base_prompt: str
    few_shot_examples: List[FewShotExample] = field(default_factory=list)
    optimization_score: float = 0.0
    version: int = 1
    last_updated: datetime = field(default_factory=datetime.now)
    ab_test_id: str = ""                # A/B 테스트 ID
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    is_active: bool = True

    def to_full_prompt(self, query: str = "", context: str = "") -> str:
        """전체 프롬프트 생성"""
        parts = [self.base_prompt]

        if self.few_shot_examples:
            parts.append("\n\n[성공 사례 예시]")
            for i, ex in enumerate(self.few_shot_examples[:3]):
                parts.append(f"\n--- 예시 {i+1} (품질: {ex.quality_grade.value}) ---")
                parts.append(f"질문: {ex.query}")
                if ex.context:
                    parts.append(f"컨텍스트: {ex.context[:300]}...")
                parts.append(f"답변: {ex.response}")

        if context:
            parts.append(f"\n\n[검색된 컨텍스트]\n{context}")

        if query:
            parts.append(f"\n\n[현재 질문]\n{query}")

        parts.append("\n\n위 예시들을 참고하여 현재 질문에 답변하세요.")

        return "\n".join(parts)


class AdaptiveThreshold:
    """적응형 임계값 시스템"""

    def __init__(
        self,
        initial_threshold: float = 0.7,
        min_threshold: float = 0.5,
        max_threshold: float = 0.95,
        adaptation_rate: float = 0.1
    ):
        self.threshold = initial_threshold
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
        self.adaptation_rate = adaptation_rate
        self.history: List[Tuple[float, bool]] = []  # (score, was_successful)

    def update(self, score: float, was_successful: bool):
        """피드백 기반 임계값 업데이트"""
        self.history.append((score, was_successful))

        # 최근 100개만 유지
        if len(self.history) > 100:
            self.history = self.history[-100:]

        # 성공률 기반 조정
        recent = self.history[-20:] if len(self.history) >= 20 else self.history

        above_threshold = [(s, succ) for s, succ in recent if s >= self.threshold]
        below_threshold = [(s, succ) for s, succ in recent if s < self.threshold]

        if above_threshold:
            above_success_rate = sum(1 for _, s in above_threshold if s) / len(above_threshold)

            # 성공률이 낮으면 임계값 상향
            if above_success_rate < 0.7:
                self.threshold = min(
                    self.max_threshold,
                    self.threshold + self.adaptation_rate
                )
            # 성공률이 높고 임계값 이하 성공도 많으면 하향
            elif above_success_rate > 0.9 and below_threshold:
                below_success_rate = sum(1 for _, s in below_threshold if s) / len(below_threshold)
                if below_success_rate > 0.5:
                    self.threshold = max(
                        self.min_threshold,
                        self.threshold - self.adaptation_rate / 2
                    )

    def get_threshold(self) -> float:
        return self.threshold


class MultiArmedBandit:
    """멀티 암 밴딧 기반 프롬프트 선택"""

    def __init__(self, exploration_rate: float = 0.1):
        self.exploration_rate = exploration_rate
        self.arms: Dict[str, Dict[str, Any]] = {}  # prompt_id -> {trials, successes, ucb}
        self._lock = threading.Lock()

    def add_arm(self, arm_id: str):
        """새 프롬프트 추가"""
        with self._lock:
            if arm_id not in self.arms:
                self.arms[arm_id] = {
                    "trials": 0,
                    "successes": 0,
                    "total_score": 0.0,
                    "avg_score": 0.5,  # 초기 낙관적 추정
                }

    def select_arm(self) -> str:
        """UCB1 알고리즘으로 프롬프트 선택"""
        if not self.arms:
            return ""

        with self._lock:
            total_trials = sum(a["trials"] for a in self.arms.values())

            # 탐색: 시도 안 한 arm 우선
            for arm_id, data in self.arms.items():
                if data["trials"] == 0:
                    return arm_id

            # UCB1 계산
            def ucb_score(arm_id: str) -> float:
                data = self.arms[arm_id]
                if data["trials"] == 0:
                    return float('inf')

                exploitation = data["avg_score"]
                exploration = math.sqrt(2 * math.log(total_trials) / data["trials"])
                return exploitation + self.exploration_rate * exploration

            return max(self.arms.keys(), key=ucb_score)

    def update_arm(self, arm_id: str, score: float, success: bool):
        """결과 업데이트"""
        with self._lock:
            if arm_id in self.arms:
                self.arms[arm_id]["trials"] += 1
                self.arms[arm_id]["total_score"] += score
                if success:
                    self.arms[arm_id]["successes"] += 1
                self.arms[arm_id]["avg_score"] = (
                    self.arms[arm_id]["total_score"] / self.arms[arm_id]["trials"]
                )

    def get_best_arm(self) -> str:
        """현재 최고 성과 프롬프트"""
        with self._lock:
            if not self.arms:
                return ""

            valid_arms = {k: v for k, v in self.arms.items() if v["trials"] >= 5}
            if not valid_arms:
                return max(self.arms.keys(), key=lambda k: self.arms[k]["trials"])

            return max(valid_arms.keys(), key=lambda k: valid_arms[k]["avg_score"])


class DSPyOptimizer:
    """
    DSPy 기반 프롬프트 최적화기 (Production Grade)

    사용법:
        optimizer = DSPyOptimizer()
        optimizer.record_feedback(query, response, "positive")
        optimized = optimizer.get_optimized_prompt("contract_analysis")
    """

    # 기본 프롬프트 템플릿 (확장)
    BASE_PROMPTS = {
        "contract_analysis": """당신은 대한민국 노동법 전문 AI 변호사입니다.
근로계약서를 분석하여 위험 조항을 식별하고 법적 조언을 제공합니다.

[분석 프레임워크]
1. 근로기준법 위반 여부 확인
   - 최저임금법 준수
   - 근로시간 규정 준수
   - 휴게시간 및 휴일 규정

2. 불공정 조항 식별
   - 위약금 예정 조항
   - 과도한 경쟁금지 조항
   - 부당한 해고 조건

3. 필수 기재사항 확인
   - 임금, 근로시간, 휴일
   - 근무 장소, 업무 내용

4. 개선 제안
   - 구체적인 수정 방안
   - 법적 근거 제시

[응답 형식]
- 위험도: High/Medium/Low
- 위반 조항: 구체적 명시
- 법적 근거: 관련 법령 조항
- 개선 방안: 실행 가능한 제안

정확하고 근거있는 분석을 제공하세요.""",

        "risk_detection": """당신은 계약서 위험 탐지 전문가입니다.

[위험 패턴 체크리스트]
1. 포괄임금제
   - 연장/야간/휴일근로 포함 여부
   - 법정 가산수당 지급 여부

2. 부당 해고 조항
   - 정당한 사유 없는 해고
   - 절차적 정당성

3. 과도한 위약금
   - 손해배상 예정 금지
   - 교육비 반환 조건

4. 최저임금 미달
   - 시급 계산 방식
   - 수당 포함 여부

5. 경쟁 금지 조항
   - 기간 및 범위의 합리성
   - 보상 조건

위험 요소를 구체적으로 지적하고 법적 근거를 제시하세요.
반드시 대법원 판례를 참조하여 분석하세요.""",

        "legal_qa": """당신은 노동법 Q&A 전문가입니다.

[답변 원칙]
1. 정확성: 법령 조문과 판례 기반
2. 이해용이성: 일반인도 이해 가능
3. 실용성: 실제 적용 가능한 조언
4. 균형성: 근로자와 사용자 양측 관점

[답변 구조]
1. 핵심 답변 (1-2문장)
2. 법적 근거 (관련 법령)
3. 상세 설명 (필요시)
4. 주의사항 (예외 상황)

질문에 맞는 정확하고 실용적인 답변을 제공하세요.""",

        "clause_rewrite": """당신은 근로계약서 조항 수정 전문가입니다.

주어진 문제 조항을 법적으로 유효하고 공정한 조항으로 수정하세요.

[수정 원칙]
1. 근로기준법 준수
2. 양측 권리 균형
3. 명확한 표현
4. 실무적 적용 가능성

[수정 형식]
기존 조항: [원문]
수정 조항: [수정안]
수정 사유: [법적 근거 및 이유]""",
    }

    # 법적 카테고리
    LEGAL_CATEGORIES = [
        "임금", "근로시간", "휴일휴가", "해고퇴직",
        "4대보험", "계약조건", "징계", "기타"
    ]

    def __init__(
        self,
        storage_path: str = None,
        min_examples: int = 3,
        max_examples: int = 10,
        feedback_threshold: float = 0.7,
        enable_ab_testing: bool = True,
        enable_online_learning: bool = True
    ):
        """
        Args:
            storage_path: 피드백/예시 저장 경로
            min_examples: 최소 Few-Shot 예시 수
            max_examples: 최대 Few-Shot 예시 수
            feedback_threshold: 예시로 사용할 최소 점수
            enable_ab_testing: A/B 테스트 활성화
            enable_online_learning: 온라인 학습 활성화
        """
        self.storage_path = Path(storage_path or "data/dspy_feedback")
        self.storage_path.mkdir(parents=True, exist_ok=True)

        self.min_examples = min_examples
        self.max_examples = max_examples
        self.enable_ab_testing = enable_ab_testing
        self.enable_online_learning = enable_online_learning

        # 적응형 임계값
        self.threshold = AdaptiveThreshold(initial_threshold=feedback_threshold)

        # 멀티 암 밴딧
        self.bandit = MultiArmedBandit()

        # 메모리 캐시
        self._feedback_cache: List[FeedbackRecord] = []
        self._example_cache: Dict[str, List[FewShotExample]] = {}
        self._prompt_cache: Dict[str, OptimizedPrompt] = {}
        self._prompt_versions: Dict[str, List[OptimizedPrompt]] = {}

        # 동시성 제어
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=2)

        # 통계
        self._stats = defaultdict(int)

        # 저장된 데이터 로드
        self._load_data()

        # 밴딧에 기본 프롬프트 등록
        for prompt_type in self.BASE_PROMPTS:
            self.bandit.add_arm(f"{prompt_type}_v1")

    def record_feedback(
        self,
        query: str,
        response: str,
        feedback_type: str = "positive",
        context: str = "",
        feedback_score: float = None,
        feedback_source: FeedbackSource = FeedbackSource.USER_EXPLICIT,
        metadata: Dict[str, Any] = None,
        response_time_ms: float = 0.0,
        was_edited: bool = False
    ) -> FeedbackRecord:
        """
        사용자 피드백 기록

        Args:
            query: 원본 질문
            response: AI 응답
            feedback_type: 피드백 유형 (positive/negative)
            context: 컨텍스트
            feedback_score: 점수 (-1.0 ~ 1.0)
            feedback_source: 피드백 소스
            metadata: 추가 메타데이터
            response_time_ms: 응답 시간
            was_edited: 사용자가 응답 수정했는지

        Returns:
            FeedbackRecord
        """
        if feedback_score is None:
            if feedback_type == "positive":
                feedback_score = 1.0 if not was_edited else 0.7
            else:
                feedback_score = -0.5

        # 품질 등급 결정
        quality_grade = self._determine_quality_grade(
            feedback_score, feedback_source, was_edited
        )

        record = FeedbackRecord(
            id=self._generate_id(query, response),
            query=query,
            response=response,
            context=context,
            feedback_type=feedback_type,
            feedback_score=feedback_score,
            feedback_source=feedback_source,
            quality_grade=quality_grade,
            metadata=metadata or {},
            response_time_ms=response_time_ms,
            was_edited=was_edited,
            prompt_version=self._get_current_version(
                metadata.get("prompt_type", "contract_analysis") if metadata else "contract_analysis"
            )
        )

        with self._lock:
            self._feedback_cache.append(record)

        # 적응형 임계값 업데이트
        is_success = feedback_score >= 0.5
        self.threshold.update(feedback_score, is_success)

        # 밴딧 업데이트
        if metadata and "prompt_id" in metadata:
            self.bandit.update_arm(metadata["prompt_id"], feedback_score, is_success)

        # 예시 후보 추가 (비동기)
        if feedback_score >= self.threshold.get_threshold():
            self._executor.submit(self._add_example_candidate, record)

        # 통계 업데이트
        self._stats["total_feedback"] += 1
        self._stats[f"feedback_{feedback_type}"] += 1

        # 주기적 저장
        if len(self._feedback_cache) % 10 == 0:
            self._executor.submit(self._save_data)

        return record

    def get_optimized_prompt(
        self,
        prompt_type: str = "contract_analysis",
        query: str = "",
        context: str = "",
        category: str = None,
        complexity: str = None,
        use_bandit: bool = True
    ) -> OptimizedPrompt:
        """
        최적화된 프롬프트 반환

        Args:
            prompt_type: 프롬프트 유형
            query: 현재 질문
            context: 컨텍스트
            category: 법적 카테고리 필터
            complexity: 복잡도 필터
            use_bandit: 밴딧 기반 선택 사용

        Returns:
            OptimizedPrompt
        """
        # 밴딧 기반 프롬프트 버전 선택
        if use_bandit and self.enable_ab_testing:
            selected_arm = self.bandit.select_arm()
            if selected_arm and selected_arm.startswith(prompt_type):
                version = int(selected_arm.split("_v")[-1]) if "_v" in selected_arm else 1
            else:
                version = None
        else:
            version = None

        # 캐시 확인
        cache_key = f"{prompt_type}_{category or 'all'}_{complexity or 'all'}"
        with self._lock:
            if cache_key in self._prompt_cache:
                cached = self._prompt_cache[cache_key]
                elapsed = (datetime.now() - cached.last_updated).total_seconds()
                if elapsed < 3600:  # 1시간 캐시
                    return cached

        # 기본 프롬프트
        base_prompt = self.BASE_PROMPTS.get(
            prompt_type,
            self.BASE_PROMPTS["contract_analysis"]
        )

        # Few-Shot 예시 선택
        examples = self._select_examples(
            prompt_type, query, category, complexity
        )

        # 성능 메트릭 계산
        metrics = self._calculate_performance_metrics(prompt_type)

        optimized = OptimizedPrompt(
            base_prompt=base_prompt,
            few_shot_examples=examples,
            optimization_score=self._calculate_optimization_score(examples),
            version=version or self._get_current_version(prompt_type),
            performance_metrics=metrics,
            ab_test_id=f"{prompt_type}_v{version}" if version else ""
        )

        with self._lock:
            self._prompt_cache[cache_key] = optimized

        return optimized

    def optimize(
        self,
        prompt_type: str = "contract_analysis",
        force: bool = False,
        create_new_version: bool = True
    ) -> OptimizedPrompt:
        """
        프롬프트 최적화 실행

        Args:
            prompt_type: 프롬프트 유형
            force: 강제 최적화
            create_new_version: 새 버전 생성 여부

        Returns:
            OptimizedPrompt
        """
        # 충분한 피드백 확인
        current_threshold = self.threshold.get_threshold()
        relevant_feedback = [
            f for f in self._feedback_cache
            if f.feedback_score >= current_threshold
        ]

        if len(relevant_feedback) < self.min_examples and not force:
            return self.get_optimized_prompt(prompt_type)

        # 최적 예시 선택
        examples = self._optimize_examples(prompt_type)

        # 새 버전 생성
        current_version = self._get_current_version(prompt_type)
        new_version = current_version + 1 if create_new_version else current_version

        base_prompt = self.BASE_PROMPTS.get(
            prompt_type,
            self.BASE_PROMPTS["contract_analysis"]
        )

        optimized = OptimizedPrompt(
            base_prompt=base_prompt,
            few_shot_examples=examples,
            optimization_score=self._calculate_optimization_score(examples),
            version=new_version,
            performance_metrics=self._calculate_performance_metrics(prompt_type)
        )

        # 버전 히스토리 저장
        with self._lock:
            if prompt_type not in self._prompt_versions:
                self._prompt_versions[prompt_type] = []
            self._prompt_versions[prompt_type].append(optimized)

            # 캐시 무효화
            keys_to_remove = [k for k in self._prompt_cache if k.startswith(prompt_type)]
            for key in keys_to_remove:
                del self._prompt_cache[key]

        # 밴딧에 새 arm 추가
        if create_new_version:
            self.bandit.add_arm(f"{prompt_type}_v{new_version}")

        # 저장
        self._executor.submit(self._save_data)

        return optimized

    def run_ab_test(
        self,
        prompt_type: str,
        test_queries: List[str],
        versions_to_test: List[int] = None
    ) -> Dict[str, Any]:
        """
        A/B 테스트 실행

        Args:
            prompt_type: 프롬프트 유형
            test_queries: 테스트 쿼리들
            versions_to_test: 테스트할 버전들

        Returns:
            테스트 결과
        """
        if not self.enable_ab_testing:
            return {"error": "A/B testing disabled"}

        if versions_to_test is None:
            versions_to_test = [
                p.version for p in self._prompt_versions.get(prompt_type, [])
            ][-3:]  # 최근 3개 버전

        results = {}
        for version in versions_to_test:
            arm_id = f"{prompt_type}_v{version}"
            results[arm_id] = {
                "version": version,
                "trials": 0,
                "avg_score": 0.0,
                "queries_tested": []
            }

            for query in test_queries[:10]:  # 최대 10개
                selected = self.bandit.select_arm()
                if selected == arm_id:
                    results[arm_id]["trials"] += 1
                    results[arm_id]["queries_tested"].append(query)

        # 밴딧 통계 추가
        for arm_id in results:
            if arm_id in self.bandit.arms:
                arm_data = self.bandit.arms[arm_id]
                results[arm_id]["total_trials"] = arm_data["trials"]
                results[arm_id]["avg_score"] = arm_data["avg_score"]
                results[arm_id]["success_rate"] = (
                    arm_data["successes"] / arm_data["trials"]
                    if arm_data["trials"] > 0 else 0
                )

        return {
            "prompt_type": prompt_type,
            "best_version": self.bandit.get_best_arm(),
            "version_results": results,
            "current_threshold": self.threshold.get_threshold(),
        }

    def _determine_quality_grade(
        self,
        score: float,
        source: FeedbackSource,
        was_edited: bool
    ) -> ExampleQuality:
        """품질 등급 결정"""
        # 소스별 신뢰도 가중치
        source_weights = {
            FeedbackSource.EXPERT_REVIEW: 1.2,
            FeedbackSource.USER_EXPLICIT: 1.0,
            FeedbackSource.AB_TEST: 0.9,
            FeedbackSource.USER_IMPLICIT: 0.7,
            FeedbackSource.SYSTEM_AUTO: 0.6,
        }
        weight = source_weights.get(source, 1.0)

        # 수정되었으면 점수 감소
        if was_edited:
            score *= 0.8

        adjusted_score = score * weight

        if adjusted_score >= 0.95:
            return ExampleQuality.EXCELLENT
        elif adjusted_score >= 0.8:
            return ExampleQuality.GOOD
        elif adjusted_score >= 0.6:
            return ExampleQuality.ADEQUATE
        elif adjusted_score >= 0.4:
            return ExampleQuality.MARGINAL
        else:
            return ExampleQuality.POOR

    def _generate_id(self, query: str, response: str) -> str:
        """고유 ID 생성"""
        content = f"{query}{response}{datetime.now().isoformat()}"
        return hashlib.md5(content.encode()).hexdigest()[:12]

    def _get_current_version(self, prompt_type: str) -> int:
        """현재 프롬프트 버전"""
        versions = self._prompt_versions.get(prompt_type, [])
        if versions:
            return max(p.version for p in versions)
        return 1

    def _add_example_candidate(self, record: FeedbackRecord):
        """예시 후보 추가"""
        # 카테고리 추론
        category = self._infer_legal_category(record.query, record.response)
        complexity = self._infer_complexity(record.query)

        example = FewShotExample(
            id=record.id,
            query=record.query,
            response=record.response,
            context=record.context,
            score=record.feedback_score,
            category=record.metadata.get("category", "general"),
            quality_grade=record.quality_grade,
            legal_category=category,
            complexity=complexity
        )

        with self._lock:
            if category not in self._example_cache:
                self._example_cache[category] = []

            self._example_cache[category].append(example)

            # 최대 개수 제한
            if len(self._example_cache[category]) > self.max_examples * 3:
                # 품질순 정렬 후 상위만 유지
                self._example_cache[category].sort(
                    key=lambda x: (
                        x.quality_grade.value,
                        -x.score,
                        -x.success_rate
                    )
                )
                self._example_cache[category] = self._example_cache[category][:self.max_examples * 2]

    def _infer_legal_category(self, query: str, response: str) -> str:
        """법적 카테고리 추론"""
        category_keywords = {
            "임금": ["급여", "임금", "월급", "시급", "최저임금", "수당", "보너스"],
            "근로시간": ["근로시간", "연장근로", "야간근로", "휴일근로", "52시간"],
            "휴일휴가": ["휴일", "휴가", "연차", "주휴", "공휴일"],
            "해고퇴직": ["해고", "퇴직", "퇴직금", "사직", "권고사직"],
            "4대보험": ["보험", "국민연금", "건강보험", "고용보험", "산재"],
            "계약조건": ["계약", "근로계약", "조항", "조건", "기간"],
            "징계": ["징계", "경고", "감봉", "정직", "해임"],
        }

        text = query + " " + response
        scores = {}

        for category, keywords in category_keywords.items():
            score = sum(1 for kw in keywords if kw in text)
            if score > 0:
                scores[category] = score

        if scores:
            return max(scores.keys(), key=lambda k: scores[k])
        return "기타"

    def _infer_complexity(self, query: str) -> str:
        """복잡도 추론"""
        # 질문 길이 기반
        if len(query) < 50:
            return "simple"
        elif len(query) < 150:
            return "medium"
        else:
            return "complex"

    def _select_examples(
        self,
        prompt_type: str,
        query: str,
        category: str = None,
        complexity: str = None
    ) -> List[FewShotExample]:
        """쿼리에 적합한 예시 선택"""
        candidates = []

        with self._lock:
            # 카테고리별 예시 수집
            if category and category in self._example_cache:
                candidates.extend(self._example_cache[category])
            else:
                for cat_examples in self._example_cache.values():
                    candidates.extend(cat_examples)

        if not candidates:
            return []

        # 복잡도 필터링
        if complexity:
            complexity_candidates = [c for c in candidates if c.complexity == complexity]
            if complexity_candidates:
                candidates = complexity_candidates

        # 품질 기준 필터링
        quality_filtered = [
            c for c in candidates
            if c.quality_grade in [ExampleQuality.EXCELLENT, ExampleQuality.GOOD, ExampleQuality.ADEQUATE]
        ]
        if quality_filtered:
            candidates = quality_filtered

        # 유사도 + 품질 + 성공률 기반 정렬
        if query:
            query_keywords = set(query.lower().split())

            def composite_score(ex: FewShotExample) -> float:
                ex_keywords = set(ex.query.lower().split())
                keyword_overlap = len(query_keywords & ex_keywords) / max(len(query_keywords), 1)

                quality_scores = {
                    ExampleQuality.EXCELLENT: 1.0,
                    ExampleQuality.GOOD: 0.8,
                    ExampleQuality.ADEQUATE: 0.6,
                    ExampleQuality.MARGINAL: 0.4,
                    ExampleQuality.POOR: 0.2,
                }
                quality_score = quality_scores.get(ex.quality_grade, 0.5)

                # 복합 점수
                return (
                    keyword_overlap * 0.3 +
                    ex.score * 0.25 +
                    quality_score * 0.25 +
                    ex.success_rate * 0.2
                )

            candidates.sort(key=composite_score, reverse=True)
        else:
            # 품질 + 점수순 정렬
            candidates.sort(
                key=lambda x: (
                    -list(ExampleQuality).index(x.quality_grade),
                    -x.score,
                    -x.success_rate
                )
            )

        # 다양성 확보 (카테고리 분산)
        selected = []
        seen_categories = set()

        for ex in candidates:
            if len(selected) >= self.max_examples:
                break

            # 처음 몇 개는 품질 우선
            if len(selected) < self.min_examples:
                selected.append(ex)
                seen_categories.add(ex.legal_category)
            # 이후는 다양성 고려
            elif ex.legal_category not in seen_categories or len(seen_categories) >= 3:
                selected.append(ex)
                seen_categories.add(ex.legal_category)

        # 사용 카운트 증가
        for ex in selected:
            ex.usage_count += 1
            ex.last_used = datetime.now()

        return selected

    def _optimize_examples(
        self,
        prompt_type: str
    ) -> List[FewShotExample]:
        """최적의 예시 조합 선택"""
        all_examples = []

        with self._lock:
            for cat_examples in self._example_cache.values():
                all_examples.extend(cat_examples)

        if not all_examples:
            return []

        # 품질 등급별 그룹화
        by_quality = defaultdict(list)
        for ex in all_examples:
            by_quality[ex.quality_grade].append(ex)

        # 최적 조합 선택
        selected = []

        # EXCELLENT에서 먼저 선택
        for quality in [ExampleQuality.EXCELLENT, ExampleQuality.GOOD, ExampleQuality.ADEQUATE]:
            available = by_quality.get(quality, [])
            # 성공률 + 점수 기준 정렬
            available.sort(key=lambda x: x.success_rate * 0.5 + x.score * 0.5, reverse=True)

            for ex in available:
                if len(selected) >= self.max_examples:
                    break

                # 카테고리 다양성 확보
                existing_categories = {s.legal_category for s in selected}
                if ex.legal_category not in existing_categories or len(selected) < self.min_examples:
                    selected.append(ex)

        return selected

    def _calculate_optimization_score(
        self,
        examples: List[FewShotExample]
    ) -> float:
        """최적화 점수 계산"""
        if not examples:
            return 0.0

        # 평균 품질 점수
        quality_scores = {
            ExampleQuality.EXCELLENT: 1.0,
            ExampleQuality.GOOD: 0.8,
            ExampleQuality.ADEQUATE: 0.6,
            ExampleQuality.MARGINAL: 0.4,
            ExampleQuality.POOR: 0.2,
        }
        avg_quality = sum(
            quality_scores.get(ex.quality_grade, 0.5) for ex in examples
        ) / len(examples)

        # 평균 성공률
        avg_success = sum(ex.success_rate for ex in examples) / len(examples)

        # 다양성 점수
        categories = set(ex.legal_category for ex in examples)
        diversity = len(categories) / max(len(examples), 1)

        return avg_quality * 0.4 + avg_success * 0.4 + diversity * 0.2

    def _calculate_performance_metrics(
        self,
        prompt_type: str
    ) -> Dict[str, float]:
        """성능 메트릭 계산"""
        relevant_feedback = [
            f for f in self._feedback_cache
            if f.metadata.get("prompt_type") == prompt_type
        ]

        if not relevant_feedback:
            return {}

        recent = relevant_feedback[-100:]

        positive = sum(1 for f in recent if f.feedback_type == "positive")
        negative = sum(1 for f in recent if f.feedback_type == "negative")

        return {
            "total_feedback": len(recent),
            "positive_rate": positive / len(recent) if recent else 0,
            "negative_rate": negative / len(recent) if recent else 0,
            "avg_score": sum(f.feedback_score for f in recent) / len(recent),
            "avg_response_time_ms": sum(f.response_time_ms for f in recent) / len(recent),
            "current_threshold": self.threshold.get_threshold(),
        }

    def _load_data(self):
        """저장된 데이터 로드"""
        # 피드백 로드
        feedback_file = self.storage_path / "feedback.json"
        if feedback_file.exists():
            try:
                with open(feedback_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    for item in data[-1000:]:  # 최근 1000개
                        try:
                            self._feedback_cache.append(FeedbackRecord(
                                id=item["id"],
                                query=item["query"],
                                response=item["response"],
                                context=item.get("context", ""),
                                feedback_type=item.get("feedback_type", "positive"),
                                feedback_score=item.get("feedback_score", 1.0),
                                feedback_source=FeedbackSource(
                                    item.get("feedback_source", "user_explicit")
                                ),
                                quality_grade=ExampleQuality(
                                    item.get("quality_grade", "adequate")
                                ),
                                timestamp=datetime.fromisoformat(item["timestamp"]),
                                metadata=item.get("metadata", {}),
                                prompt_version=item.get("prompt_version", 1),
                            ))
                        except (ValueError, KeyError):
                            continue
            except Exception as e:
                print(f"Failed to load feedback: {e}")

        # 예시 로드
        examples_file = self.storage_path / "examples.json"
        if examples_file.exists():
            try:
                with open(examples_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    for category, examples in data.items():
                        self._example_cache[category] = [
                            FewShotExample(
                                id=ex.get("id", f"ex_{i}"),
                                query=ex["query"],
                                response=ex["response"],
                                context=ex.get("context", ""),
                                score=ex.get("score", 1.0),
                                category=category,
                                quality_grade=ExampleQuality(
                                    ex.get("quality_grade", "adequate")
                                ),
                                usage_count=ex.get("usage_count", 0),
                                success_rate=ex.get("success_rate", 1.0),
                                legal_category=ex.get("legal_category", ""),
                                complexity=ex.get("complexity", "medium"),
                            )
                            for i, ex in enumerate(examples)
                        ]
            except Exception as e:
                print(f"Failed to load examples: {e}")

    def _save_data(self):
        """데이터 저장"""
        # 피드백 저장
        feedback_file = self.storage_path / "feedback.json"
        try:
            with open(feedback_file, "w", encoding="utf-8") as f:
                data = [r.to_dict() for r in self._feedback_cache[-1000:]]
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Failed to save feedback: {e}")

        # 예시 저장
        examples_file = self.storage_path / "examples.json"
        try:
            with open(examples_file, "w", encoding="utf-8") as f:
                data = {
                    category: [ex.to_dict() for ex in examples]
                    for category, examples in self._example_cache.items()
                }
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Failed to save examples: {e}")

    def get_statistics(self) -> Dict[str, Any]:
        """통계 정보 반환"""
        with self._lock:
            positive_count = sum(
                1 for f in self._feedback_cache if f.feedback_type == "positive"
            )
            negative_count = sum(
                1 for f in self._feedback_cache if f.feedback_type == "negative"
            )

            example_count = sum(
                len(examples) for examples in self._example_cache.values()
            )

            quality_distribution = defaultdict(int)
            for examples in self._example_cache.values():
                for ex in examples:
                    quality_distribution[ex.quality_grade.value] += 1

        # 밴딧 통계
        bandit_stats = {}
        for arm_id, data in self.bandit.arms.items():
            bandit_stats[arm_id] = {
                "trials": data["trials"],
                "avg_score": data["avg_score"],
                "success_rate": data["successes"] / data["trials"] if data["trials"] > 0 else 0
            }

        return {
            "total_feedback": len(self._feedback_cache),
            "positive_feedback": positive_count,
            "negative_feedback": negative_count,
            "positive_ratio": positive_count / max(len(self._feedback_cache), 1),
            "total_examples": example_count,
            "categories": list(self._example_cache.keys()),
            "quality_distribution": dict(quality_distribution),
            "current_threshold": self.threshold.get_threshold(),
            "prompt_versions": {
                k: len(v) for k, v in self._prompt_versions.items()
            },
            "bandit_stats": bandit_stats,
            "best_prompt": self.bandit.get_best_arm(),
        }


class SelfEvolvingPipeline:
    """
    자가 진화 파이프라인 (Production Grade)

    분석 실행 후 자동으로 피드백을 수집하고 프롬프트를 개선
    """

    def __init__(
        self,
        optimizer: DSPyOptimizer,
        llm_client: Optional[Any] = None,
        model: str = "gpt-4o",
        auto_optimize_threshold: int = 50
    ):
        self.optimizer = optimizer
        self.model = model
        self.auto_optimize_threshold = auto_optimize_threshold
        self._feedback_count = 0

        if llm_client is None:
            try:
                from openai import OpenAI
                self.llm_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            except ImportError:
                self.llm_client = None
        else:
            self.llm_client = llm_client

    def analyze(
        self,
        query: str,
        context: str = "",
        prompt_type: str = "contract_analysis",
        category: str = None
    ) -> Dict[str, Any]:
        """
        최적화된 프롬프트로 분석 실행

        Args:
            query: 분석 쿼리
            context: 컨텍스트
            prompt_type: 프롬프트 유형
            category: 법적 카테고리

        Returns:
            분석 결과
        """
        import time
        start_time = time.time()

        # 최적화된 프롬프트 가져오기
        optimized = self.optimizer.get_optimized_prompt(
            prompt_type=prompt_type,
            query=query,
            context=context,
            category=category
        )

        full_prompt = optimized.to_full_prompt(query, context)

        # LLM 호출
        if self.llm_client is None:
            return {"error": "LLM 클라이언트 없음"}

        try:
            response = self.llm_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": full_prompt}
                ]
            )

            result_text = response.choices[0].message.content
            response_time = (time.time() - start_time) * 1000

            return {
                "query": query,
                "response": result_text,
                "prompt_version": optimized.version,
                "optimization_score": optimized.optimization_score,
                "few_shot_count": len(optimized.few_shot_examples),
                "response_time_ms": response_time,
                "prompt_id": optimized.ab_test_id,
                "prompt_type": prompt_type,
            }

        except Exception as e:
            return {"error": str(e)}

    def record_result_feedback(
        self,
        analysis_result: Dict[str, Any],
        feedback_type: str = "positive",
        feedback_score: float = None,
        was_edited: bool = False
    ):
        """
        분석 결과에 대한 피드백 기록

        Args:
            analysis_result: analyze() 결과
            feedback_type: 피드백 유형
            feedback_score: 점수
            was_edited: 사용자가 응답 수정했는지
        """
        self.optimizer.record_feedback(
            query=analysis_result.get("query", ""),
            response=analysis_result.get("response", ""),
            feedback_type=feedback_type,
            feedback_score=feedback_score,
            was_edited=was_edited,
            response_time_ms=analysis_result.get("response_time_ms", 0),
            metadata={
                "prompt_version": analysis_result.get("prompt_version", 1),
                "optimization_score": analysis_result.get("optimization_score", 0),
                "prompt_id": analysis_result.get("prompt_id", ""),
                "prompt_type": analysis_result.get("prompt_type", "contract_analysis"),
            }
        )

        self._feedback_count += 1

        # 자동 최적화 트리거
        if self._feedback_count >= self.auto_optimize_threshold:
            prompt_type = analysis_result.get("prompt_type", "contract_analysis")
            self.optimizer.optimize(prompt_type, create_new_version=True)
            self._feedback_count = 0


# ========== 편의 함수 ==========

def create_optimizer(storage_path: str = None) -> DSPyOptimizer:
    """DSPy 최적화기 생성"""
    return DSPyOptimizer(storage_path=storage_path)


def get_optimized_prompt(
    prompt_type: str = "contract_analysis",
    query: str = ""
) -> str:
    """최적화된 프롬프트 반환"""
    optimizer = DSPyOptimizer()
    optimized = optimizer.get_optimized_prompt(prompt_type, query)
    return optimized.to_full_prompt(query)


def create_pipeline(
    storage_path: str = None,
    model: str = "gpt-4o"
) -> SelfEvolvingPipeline:
    """자가 진화 파이프라인 생성"""
    optimizer = DSPyOptimizer(storage_path=storage_path)
    return SelfEvolvingPipeline(optimizer=optimizer, model=model)
