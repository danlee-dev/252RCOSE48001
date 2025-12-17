"""
LLM-as-a-Judge (신뢰도 평가)
- AI 분석 결과의 신뢰도 평가
- 근거 자료와 답변의 일치성 검증
- 사용자에게 신뢰도 점수 제공

Reference: "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena"
"""

import os
import re
import json
import time
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from enum import Enum

from app.core.config import settings
from app.core.token_usage_tracker import record_llm_usage


class JudgmentCategory(Enum):
    """평가 카테고리"""
    ACCURACY = "정확성"           # 법률 해석의 정확성
    CONSISTENCY = "일관성"        # 근거와 결론의 일관성
    COMPLETENESS = "완전성"       # 분석의 완전성
    RELEVANCE = "관련성"          # 질문/계약서와의 관련성
    LEGAL_BASIS = "법적 근거"     # 법적 근거의 타당성


@dataclass
class JudgmentScore:
    """개별 평가 점수"""
    category: JudgmentCategory
    score: float  # 0.0 ~ 1.0
    reasoning: str
    issues: List[str] = field(default_factory=list)


@dataclass
class JudgmentResult:
    """심판 결과"""
    analysis: str
    context: str
    scores: List[JudgmentScore] = field(default_factory=list)
    overall_score: float = 0.0
    confidence_level: str = "Medium"  # High, Medium, Low
    verdict: str = ""
    recommendations: List[str] = field(default_factory=list)

    @property
    def is_reliable(self) -> bool:
        return self.overall_score >= 0.7

    def get_score_summary(self) -> Dict[str, float]:
        return {s.category.value: s.score for s in self.scores}


class LLMJudge:
    """
    LLM-as-a-Judge 구현

    사용법:
        judge = LLMJudge()
        result = judge.evaluate(analysis, context)
        print(f"신뢰도: {result.overall_score:.1%}")
    """

    # 평가 프롬프트
    JUDGE_PROMPT = """당신은 대한민국 노동법 전문가로서 AI 분석 결과의 신뢰도를 평가하는 심판관입니다.
아래 평가 기준표를 엄격히 적용하여 각 항목을 0-10점으로 평가하세요.

[제공된 컨텍스트/근거 자료]
{context}

[AI의 분석 결과]
{analysis}

---
[상세 평가 기준표 (Scoring Rubric)]

1. 정확성 (accuracy)
   - 9-10점: 법령 조항 번호, 판례 인용이 정확하고 해석에 오류 없음
   - 7-8점: 대체로 정확하나 사소한 해석 차이 존재
   - 5-6점: 일부 법령 해석에 오류가 있으나 핵심 결론은 유효
   - 3-4점: 중요한 법령 해석 오류 존재
   - 0-2점: 법령 인용이 틀리거나 해석이 완전히 잘못됨

2. 일관성 (consistency)
   - 9-10점: 근거와 결론이 논리적으로 완벽히 연결됨
   - 7-8점: 대체로 일관되나 일부 논리적 비약 존재
   - 5-6점: 근거와 결론 사이에 간극이 있음
   - 3-4점: 결론이 근거와 맞지 않는 부분이 다수
   - 0-2점: 근거와 결론이 모순됨

3. 완전성 (completeness)
   - 9-10점: 모든 위험 요소를 빠짐없이 분석, 대안까지 제시
   - 7-8점: 주요 위험 요소 분석 완료, 일부 세부사항 누락
   - 5-6점: 핵심 위험만 분석, 부수적 위험 누락
   - 3-4점: 중요한 위험 요소 다수 누락
   - 0-2점: 분석이 피상적이거나 대부분 누락

4. 관련성 (relevance)
   - 9-10점: 계약서 맥락에 완벽히 부합하는 분석
   - 7-8점: 대체로 관련 있으나 일부 불필요한 내용 포함
   - 5-6점: 관련 있는 내용과 없는 내용이 혼재
   - 3-4점: 맥락과 맞지 않는 분석이 다수
   - 0-2점: 계약서 내용과 무관한 분석

5. 법적 근거 (legal_basis)
   - 9-10점: 구체적 법령 조항과 판례를 정확히 인용
   - 7-8점: 법령 인용은 있으나 구체적 조항 번호 일부 누락
   - 5-6점: 일반적 법원칙만 언급, 구체적 근거 부족
   - 3-4점: 법적 근거가 모호하거나 부정확
   - 0-2점: 법적 근거 없이 주관적 의견만 제시

---
[평가 절차 - Chain of Thought]
1. 먼저 AI 분석 내용을 꼼꼼히 읽고 주요 주장을 파악하세요
2. 각 평가 항목별로 위 기준표에 따라 점수를 매기세요
3. 점수의 근거를 구체적으로 작성하세요
4. 발견된 문제점을 issues 배열에 나열하세요
5. 최종 verdict를 결정하세요 (8점 이상: 신뢰, 6-7점: 주의, 5점 이하: 의심)

---
[Few-shot 예시]

예시 1 - 높은 신뢰도 분석:
분석: "근로기준법 제26조에 따르면 해고 예고는 30일 전에 해야 합니다. 본 계약서의 '즉시 해고' 조항은 이에 위반됩니다."
평가: {{"accuracy": {{"score": 9, "reasoning": "근로기준법 제26조 인용이 정확함", "issues": []}}, ...}}

예시 2 - 낮은 신뢰도 분석:
분석: "이 계약은 대체로 괜찮아 보입니다. 특별히 문제될 것은 없습니다."
평가: {{"accuracy": {{"score": 3, "reasoning": "구체적 법령 검토 없이 모호한 결론", "issues": ["법적 근거 없음", "위험 요소 미분석"]}}, ...}}

---
[출력 형식 - JSON]
{{
    "scores": {{
        "accuracy": {{"score": 8, "reasoning": "근거를 구체적으로 작성", "issues": ["발견된 문제점"]}},
        "consistency": {{"score": 7, "reasoning": "근거를 구체적으로 작성", "issues": []}},
        "completeness": {{"score": 8, "reasoning": "근거를 구체적으로 작성", "issues": []}},
        "relevance": {{"score": 9, "reasoning": "근거를 구체적으로 작성", "issues": []}},
        "legal_basis": {{"score": 7, "reasoning": "근거를 구체적으로 작성", "issues": []}}
    }},
    "overall_assessment": "전체 평가 요약 (2-3문장)",
    "verdict": "신뢰/주의/의심 중 하나",
    "recommendations": ["구체적인 개선 권고사항"]
}}"""

    # 팩트 체크 프롬프트
    FACT_CHECK_PROMPT = """당신은 대한민국 노동법 팩트체커입니다. AI 분석에서 사실 오류를 검증하세요.

[AI 분석]
{analysis}

---
[검토 체크리스트]
각 항목을 순서대로 검토하고, 오류 발견 시 기록하세요:

1. 법령 조항 검증
   - 인용된 조항 번호가 실제로 존재하는가?
   - 해당 조항의 내용이 정확히 인용되었는가?
   - 예: "근로기준법 제26조" → 해고예고 규정 (O)
   - 예: "근로기준법 제999조" → 존재하지 않음 (X)

2. 수치/기준 검증 (2024년 기준)
   - 최저임금: 시급 9,860원
   - 주 최대 근로시간: 52시간 (연장근로 포함)
   - 연차휴가: 1년 미만 월 1일, 1년 이상 15일
   - 해고예고: 30일 전
   - 퇴직금: 1년 이상 근속 시 30일분 평균임금

3. 법령 해석 검증
   - 법원/노동부의 공식 해석과 일치하는가?
   - 판례와 모순되지 않는가?

4. 치명적 오류 판단 기준
   - 치명적 오류: 결론을 뒤집을 수 있는 중대한 사실 오류
   - 예: 법령 조항 번호 오류, 핵심 수치 오류
   - 비치명적 오류: 결론에 영향 없는 사소한 오류

---
[Few-shot 예시]

예시 1 - 오류 발견:
분석: "근로기준법 제50조에 따라 주 40시간을 초과하면 즉시 위법입니다."
검토: {{
    "fact_errors": [{{
        "claim": "주 40시간 초과 시 즉시 위법",
        "issue": "연장근로 12시간까지 합법 (주 52시간)",
        "correct_info": "근로기준법 제53조에 따라 주 12시간 연장근로 가능"
    }}],
    "accuracy_score": 5,
    "critical_errors": true
}}

예시 2 - 오류 없음:
분석: "근로기준법 제26조에 따라 해고 30일 전 예고가 필요합니다."
검토: {{
    "fact_errors": [],
    "accuracy_score": 10,
    "critical_errors": false
}}

---
[출력 형식 - JSON]
{{
    "fact_errors": [
        {{"claim": "AI가 주장한 내용", "issue": "오류 내용", "correct_info": "올바른 정보"}}
    ],
    "accuracy_score": 8,
    "critical_errors": false
}}"""

    # 카테고리 매핑
    CATEGORY_MAP = {
        "accuracy": JudgmentCategory.ACCURACY,
        "consistency": JudgmentCategory.CONSISTENCY,
        "completeness": JudgmentCategory.COMPLETENESS,
        "relevance": JudgmentCategory.RELEVANCE,
        "legal_basis": JudgmentCategory.LEGAL_BASIS,
    }

    def __init__(
        self,
        llm_client: Optional[Any] = None,
        model: str = None,
        strict_mode: bool = True,
        contract_id: Optional[str] = None
    ):
        """
        Args:
            llm_client: OpenAI 클라이언트 (legacy, Gemini 사용 시 무시됨)
            model: 심판에 사용할 LLM 모델 (기본값: settings.LLM_JUDGE_MODEL)
            strict_mode: 엄격 모드 (팩트체크 포함)
            contract_id: 계약서 ID (토큰 추적용)
        """
        self.model = model if model else settings.LLM_JUDGE_MODEL
        self.strict_mode = strict_mode
        self.contract_id = contract_id
        self.llm_client = llm_client  # OpenAI fallback

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
                pass

    def evaluate(
        self,
        analysis: str,
        context: str = "",
        query: str = ""
    ) -> JudgmentResult:
        """
        AI 분석 결과 평가

        Args:
            analysis: 평가할 AI 분석 결과
            context: 분석에 사용된 컨텍스트/근거
            query: 원본 질문/요청

        Returns:
            JudgmentResult: 평가 결과
        """
        result = JudgmentResult(
            analysis=analysis,
            context=context
        )

        # 1. 종합 평가
        if self.llm_client is not None:
            scores_data = self._llm_evaluate(analysis, context)
        else:
            scores_data = self._rule_based_evaluate(analysis, context)

        # 2. 점수 파싱
        result.scores = self._parse_scores(scores_data)

        # 3. 전체 점수 계산
        result.overall_score = self._calculate_overall_score(result.scores)

        # 4. 신뢰도 레벨 결정
        result.confidence_level = self._determine_confidence_level(result.overall_score)

        # 5. 판정
        result.verdict = scores_data.get("verdict", self._generate_verdict(result))

        # 6. 권고사항
        result.recommendations = scores_data.get("recommendations", [])

        # 7. 엄격 모드: 팩트 체크
        if self.strict_mode and self.llm_client is not None:
            fact_check = self._fact_check(analysis)
            if fact_check.get("critical_errors"):
                result.overall_score *= 0.7  # 치명적 오류 시 점수 감점
                result.recommendations.append(
                    "팩트 체크에서 치명적 오류가 발견되었습니다. 전문가 검토를 권장합니다."
                )

        return result

    def _llm_evaluate(
        self,
        analysis: str,
        context: str
    ) -> Dict[str, Any]:
        """LLM 기반 평가"""
        if self._gemini_model is None and self.llm_client is None:
            return self._rule_based_evaluate(analysis, context)

        llm_start = time.time()
        try:
            prompt = self.JUDGE_PROMPT.format(
                analysis=analysis,
                context=context or "컨텍스트 없음"
            )

            # Gemini 사용 (우선)
            if self._gemini_model is not None:
                full_prompt = "당신은 법률 AI 분석 결과를 평가하는 공정한 심판관입니다.\n\n" + prompt
                result = self._gemini_model.generate_content(
                    full_prompt,
                    generation_config={
                        "temperature": 0.1,
                        "response_mime_type": "application/json"
                    },
                    safety_settings=self.safety_settings
                )
                llm_duration = (time.time() - llm_start) * 1000

                # 토큰 사용량 기록
                if self.contract_id and hasattr(result, 'usage_metadata'):
                    usage = result.usage_metadata
                    record_llm_usage(
                        contract_id=self.contract_id,
                        module="judge.evaluate",
                        model=self.model,
                        input_tokens=getattr(usage, 'prompt_token_count', 0),
                        output_tokens=getattr(usage, 'candidates_token_count', 0),
                        cached_tokens=getattr(usage, 'cached_content_token_count', 0),
                        duration_ms=llm_duration
                    )

                # JSON 파싱
                result_text = result.text.strip()
                if result_text.startswith("```"):
                    result_text = re.sub(r'^```(?:json)?\s*', '', result_text)
                    result_text = re.sub(r'\s*```$', '', result_text)
                return json.loads(result_text)

            # OpenAI fallback
            else:
                response = self.llm_client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": "당신은 법률 AI 분석 결과를 평가하는 공정한 심판관입니다."
                        },
                        {"role": "user", "content": prompt}
                    ],
                    response_format={"type": "json_object"}
                )

                llm_duration = (time.time() - llm_start) * 1000

                # 토큰 사용량 기록
                if response.usage and self.contract_id:
                    cached = 0
                    if hasattr(response.usage, 'prompt_tokens_details') and response.usage.prompt_tokens_details:
                        cached = getattr(response.usage.prompt_tokens_details, 'cached_tokens', 0) or 0
                    record_llm_usage(
                        contract_id=self.contract_id,
                        module="judge.evaluate",
                        model=self.model,
                        input_tokens=response.usage.prompt_tokens,
                        output_tokens=response.usage.completion_tokens,
                        cached_tokens=cached,
                        duration_ms=llm_duration
                    )

                return json.loads(response.choices[0].message.content)

        except Exception as e:
            print(f"LLM evaluation error: {e}")
            return self._rule_based_evaluate(analysis, context)

    def _rule_based_evaluate(
        self,
        analysis: str,
        context: str
    ) -> Dict[str, Any]:
        """규칙 기반 평가 (폴백)"""
        scores = {}

        # 길이 기반 완전성
        completeness = min(10, len(analysis) / 100)

        # 법률 용어 포함 여부로 관련성
        legal_terms = ["근로기준법", "제", "조", "판례", "법원", "임금", "해고"]
        relevance = sum(2 for term in legal_terms if term in analysis)
        relevance = min(10, relevance)

        # 근거 인용 여부
        legal_basis = 5
        if "근로기준법 제" in analysis or "판례" in analysis:
            legal_basis = 8

        scores = {
            "accuracy": {"score": 5, "reasoning": "규칙 기반 평가", "issues": []},
            "consistency": {"score": 5, "reasoning": "규칙 기반 평가", "issues": []},
            "completeness": {"score": completeness, "reasoning": f"분석 길이: {len(analysis)}자", "issues": []},
            "relevance": {"score": relevance, "reasoning": f"법률 용어 {relevance // 2}개 포함", "issues": []},
            "legal_basis": {"score": legal_basis, "reasoning": "법적 근거 인용 확인", "issues": []},
        }

        return {
            "scores": scores,
            "verdict": "주의",
            "recommendations": ["LLM 기반 상세 평가를 권장합니다."]
        }

    def _fact_check(self, analysis: str) -> Dict[str, Any]:
        """팩트 체크"""
        if self._gemini_model is None and self.llm_client is None:
            return {"fact_errors": [], "accuracy_score": 5, "critical_errors": False}

        llm_start = time.time()
        try:
            prompt = self.FACT_CHECK_PROMPT.format(analysis=analysis)

            # Gemini 사용 (우선)
            if self._gemini_model is not None:
                full_prompt = "당신은 법률 정보 팩트체커입니다.\n\n" + prompt
                result = self._gemini_model.generate_content(
                    full_prompt,
                    generation_config={
                        "temperature": 0.1,
                        "response_mime_type": "application/json"
                    },
                    safety_settings=self.safety_settings
                )
                llm_duration = (time.time() - llm_start) * 1000

                # 토큰 사용량 기록
                if self.contract_id and hasattr(result, 'usage_metadata'):
                    usage = result.usage_metadata
                    record_llm_usage(
                        contract_id=self.contract_id,
                        module="judge.fact_check",
                        model=self.model,
                        input_tokens=getattr(usage, 'prompt_token_count', 0),
                        output_tokens=getattr(usage, 'candidates_token_count', 0),
                        cached_tokens=getattr(usage, 'cached_content_token_count', 0),
                        duration_ms=llm_duration
                    )

                # JSON 파싱
                result_text = result.text.strip()
                if result_text.startswith("```"):
                    result_text = re.sub(r'^```(?:json)?\s*', '', result_text)
                    result_text = re.sub(r'\s*```$', '', result_text)
                return json.loads(result_text)

            # OpenAI fallback
            else:
                response = self.llm_client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": "당신은 법률 정보 팩트체커입니다."
                        },
                        {"role": "user", "content": prompt}
                    ],
                    response_format={"type": "json_object"}
                )

                llm_duration = (time.time() - llm_start) * 1000

                # 토큰 사용량 기록
                if response.usage and self.contract_id:
                    cached = 0
                    if hasattr(response.usage, 'prompt_tokens_details') and response.usage.prompt_tokens_details:
                        cached = getattr(response.usage.prompt_tokens_details, 'cached_tokens', 0) or 0
                    record_llm_usage(
                        contract_id=self.contract_id,
                        module="judge.fact_check",
                        model=self.model,
                        input_tokens=response.usage.prompt_tokens,
                        output_tokens=response.usage.completion_tokens,
                        cached_tokens=cached,
                        duration_ms=llm_duration
                    )

                return json.loads(response.choices[0].message.content)

        except Exception as e:
            print(f"Fact check error: {e}")
            return {"fact_errors": [], "accuracy_score": 5, "critical_errors": False}

    def _parse_scores(self, data: Dict[str, Any]) -> List[JudgmentScore]:
        """점수 파싱"""
        scores = []
        scores_data = data.get("scores", {})

        for key, category in self.CATEGORY_MAP.items():
            if key in scores_data:
                score_info = scores_data[key]
                scores.append(JudgmentScore(
                    category=category,
                    score=score_info.get("score", 5) / 10,  # 0-10 -> 0-1 정규화
                    reasoning=score_info.get("reasoning", ""),
                    issues=score_info.get("issues", [])
                ))
            else:
                scores.append(JudgmentScore(
                    category=category,
                    score=0.5,
                    reasoning="평가 정보 없음",
                    issues=[]
                ))

        return scores

    def _calculate_overall_score(self, scores: List[JudgmentScore]) -> float:
        """전체 점수 계산 (가중 평균)"""
        # 카테고리별 가중치
        weights = {
            JudgmentCategory.ACCURACY: 0.25,
            JudgmentCategory.CONSISTENCY: 0.20,
            JudgmentCategory.COMPLETENESS: 0.15,
            JudgmentCategory.RELEVANCE: 0.15,
            JudgmentCategory.LEGAL_BASIS: 0.25,
        }

        total_weight = 0
        weighted_sum = 0

        for score in scores:
            weight = weights.get(score.category, 0.2)
            weighted_sum += score.score * weight
            total_weight += weight

        return weighted_sum / total_weight if total_weight > 0 else 0.5

    def _determine_confidence_level(self, score: float) -> str:
        """신뢰도 레벨 결정"""
        if score >= 0.8:
            return "High"
        elif score >= 0.6:
            return "Medium"
        else:
            return "Low"

    def _generate_verdict(self, result: JudgmentResult) -> str:
        """판정 생성"""
        if result.overall_score >= 0.8:
            return "신뢰: 분석 결과를 신뢰할 수 있습니다."
        elif result.overall_score >= 0.6:
            return "주의: 분석 결과를 참고하되, 전문가 검토를 권장합니다."
        else:
            return "의심: 분석 결과의 신뢰도가 낮습니다. 전문가 상담을 강력히 권장합니다."

    def get_confidence_badge(self, result: JudgmentResult) -> Dict[str, Any]:
        """신뢰도 배지 정보 반환 (UI 표시용)"""
        badges = {
            "High": {
                "label": "높은 신뢰도",
                "color": "green",
                "icon": "check-circle",
                "description": "AI 분석 결과를 신뢰할 수 있습니다."
            },
            "Medium": {
                "label": "보통 신뢰도",
                "color": "yellow",
                "icon": "alert-circle",
                "description": "참고용으로 활용하되, 전문가 검토를 권장합니다."
            },
            "Low": {
                "label": "낮은 신뢰도",
                "color": "red",
                "icon": "x-circle",
                "description": "전문가 상담을 강력히 권장합니다."
            }
        }

        badge = badges.get(result.confidence_level, badges["Medium"])
        badge["score"] = f"{result.overall_score:.0%}"
        badge["scores_detail"] = result.get_score_summary()

        return badge


class AnalysisValidator:
    """
    분석 결과 검증기

    계약서 분석 파이프라인에 통합
    """

    def __init__(self, judge: LLMJudge):
        self.judge = judge

    def validate_analysis(
        self,
        contract_text: str,
        analysis_result: Dict[str, Any],
        context_docs: List[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        계약서 분석 결과 검증

        Args:
            contract_text: 원본 계약서
            analysis_result: AI 분석 결과
            context_docs: 참고된 문서들

        Returns:
            검증된 분석 결과
        """
        # 분석 텍스트 추출
        if isinstance(analysis_result, dict):
            analysis_text = analysis_result.get("summary", "") + "\n" + \
                           str(analysis_result.get("risk_clauses", []))
        else:
            analysis_text = str(analysis_result)

        # 컨텍스트 구성
        context = contract_text[:2000]  # 계약서 일부
        if context_docs:
            context += "\n\n[참고 문서]\n"
            context += "\n".join([d.get("text", "")[:500] for d in context_docs[:3]])

        # 평가 실행
        judgment = self.judge.evaluate(analysis_text, context)

        return {
            "original_analysis": analysis_result,
            "validation": {
                "overall_score": judgment.overall_score,
                "confidence_level": judgment.confidence_level,
                "verdict": judgment.verdict,
                "scores": judgment.get_score_summary(),
                "recommendations": judgment.recommendations,
                "is_reliable": judgment.is_reliable
            },
            "badge": self.judge.get_confidence_badge(judgment)
        }


# 편의 함수
def create_judge(
    strict_mode: bool = True,
    model: str = "gpt-4o-mini"
) -> LLMJudge:
    """LLMJudge 팩토리 함수"""
    return LLMJudge(model=model, strict_mode=strict_mode)


def evaluate_analysis(
    analysis: str,
    context: str = ""
) -> JudgmentResult:
    """간편 분석 평가"""
    judge = LLMJudge()
    return judge.evaluate(analysis, context)
