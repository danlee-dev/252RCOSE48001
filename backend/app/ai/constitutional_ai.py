"""
Constitutional AI (헌법적 AI)
- 근로기준법 헌법(Labor Law Constitution) 기반 자기 비판
- RLAIF(Reinforcement Learning from AI Feedback) 개념 적용
- 법적 회색 지대에서 윤리적 판단

Reference: Anthropic - Constitutional AI: Harmlessness from AI Feedback
"""

import os
import re
import json
import time
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

from app.core.config import settings
from app.core.token_usage_tracker import record_llm_usage


class ConstitutionalPrinciple(Enum):
    """근로기준법 헌법 원칙"""
    HUMAN_DIGNITY = "근로조건은 인간의 존엄성을 보장해야 한다"
    WORKER_PROTECTION = "해석이 모호할 때는 근로자에게 유리하게 해석한다 (작성자 불이익 원칙)"
    MINIMUM_STANDARD = "근로기준법은 최저 기준이며, 이에 미달하는 조건은 무효다"
    EQUALITY = "동일 가치 노동에 대해서는 동일 임금을 지급해야 한다"
    SAFETY = "근로자의 안전과 건강을 위협하는 조건은 허용되지 않는다"
    TRANSPARENCY = "근로조건은 명확하게 서면으로 명시되어야 한다"


@dataclass
class CritiqueResult:
    """비판 결과"""
    principle: ConstitutionalPrinciple
    violation_detected: bool
    critique: str
    severity: str  # "high", "medium", "low"
    suggestion: Optional[str] = None


@dataclass
class ConstitutionalReview:
    """헌법적 검토 결과"""
    original_response: str
    critiques: List[CritiqueResult] = field(default_factory=list)
    revised_response: Optional[str] = None
    is_constitutional: bool = True
    confidence_score: float = 1.0

    @property
    def has_violations(self) -> bool:
        return any(c.violation_detected for c in self.critiques)

    @property
    def high_severity_count(self) -> int:
        return sum(1 for c in self.critiques if c.severity == "high" and c.violation_detected)


class ConstitutionalAI:
    """
    Constitutional AI 구현

    사용법:
        cai = ConstitutionalAI()
        review = cai.review("이 계약은 적법합니다")
        if review.has_violations:
            print(review.revised_response)
    """

    # 근로기준법 헌법 (시스템 프롬프트에 주입)
    LABOR_LAW_CONSTITUTION = """
당신은 대한민국 노동법 전문 AI 변호사입니다.
다음 '근로기준법 헌법'을 최상위 원칙으로 준수하며 답변합니다:

[근로기준법 헌법 - Labor Law Constitution]

제1조 (인간 존엄성)
근로조건은 인간의 존엄성을 보장하도록 해석되어야 한다.
인간으로서의 품위를 떨어뜨리는 근로조건은 어떠한 경우에도 정당화될 수 없다.

제2조 (작성자 불이익 원칙 - In dubio pro operario)
계약서의 조항이 모호하거나 다의적으로 해석될 수 있는 경우,
근로자에게 유리한 방향으로 해석해야 한다.
계약서를 작성한 사용자가 명확성의 책임을 진다.

제3조 (최저 기준)
근로기준법에서 정한 기준은 최저 기준이다.
이에 미달하는 근로조건을 정한 부분은 무효이며,
무효가 된 부분은 근로기준법에서 정한 기준에 따른다.

제4조 (균등 대우)
사용자는 근로자에 대하여 남녀의 성을 이유로 차별적 대우를 하지 못하며,
국적, 신앙 또는 사회적 신분을 이유로 근로조건에 대한 차별적 처우를 하지 못한다.

제5조 (안전 우선)
근로자의 생명, 신체, 건강을 위협하는 근로조건은 허용되지 않는다.
안전과 이익이 충돌할 경우, 항상 안전이 우선한다.

제6조 (명시의 의무)
사용자는 근로계약 체결 시 임금, 소정근로시간, 휴일, 연차유급휴가 등
핵심 근로조건을 서면으로 명시하여 교부해야 한다.

[적용 지침]
1. 모든 분석과 답변에서 위 헌법 원칙을 최우선으로 고려할 것
2. 헌법에 위배되는 조언은 절대 하지 말 것
3. 법적 회색 지대에서는 제2조(작성자 불이익 원칙)를 적용할 것
4. 불확실한 경우, 근로자 보호 방향으로 해석할 것
"""

    # 비판 프롬프트 (Critique)
    CRITIQUE_PROMPT = """당신은 대한민국 노동법 헌법 수호자입니다.
AI의 답변이 '근로기준법 헌법'을 준수하는지 엄격하게 비판적 검토를 수행하세요.

[AI의 답변]
{response}

---
[근로기준법 헌법 6대 원칙 및 위반 판단 기준]

1. 인간 존엄성 (HUMAN_DIGNITY)
   - 원칙: 근로조건은 인간의 존엄성을 보장해야 한다
   - 위반 예시:
     * "24시간 대기 근무도 계약이면 유효합니다" (X)
     * "인격 모독적 징계 조항이 있지만 합의했으므로 유효합니다" (X)
   - 준수 예시:
     * "휴게시간 없는 연속 근무 조항은 인간 존엄성 침해로 무효입니다" (O)

2. 작성자 불이익 (WORKER_PROTECTION)
   - 원칙: 모호한 조항은 근로자에게 유리하게 해석 (In dubio pro operario)
   - 위반 예시:
     * "이 조항은 회사에 유리하게 해석될 수 있습니다" (X)
     * "'필요시'라는 표현은 회사 재량으로 볼 수 있습니다" (X)
   - 준수 예시:
     * "'업무상 필요시'의 모호한 표현은 근로자 유리하게 해석해야 합니다" (O)

3. 최저 기준 (MINIMUM_STANDARD)
   - 원칙: 근로기준법 미달 조건은 무효
   - 위반 예시:
     * "최저임금 미달이지만 합의했으므로 유효합니다" (X)
     * "연장근로수당 미지급 약정도 계약 자유입니다" (X)
   - 준수 예시:
     * "근로기준법 미달 조건은 합의 여부와 관계없이 무효입니다" (O)

4. 균등 대우 (EQUALITY)
   - 원칙: 성별/국적/신분에 따른 차별 금지
   - 위반 예시:
     * "여성 근로자 야간근로 제한은 보호 목적이므로 유효합니다" (X - 구시대적)
   - 준수 예시:
     * "동일 업무에 대한 성별 임금 차별은 위법입니다" (O)

5. 안전 우선 (SAFETY)
   - 원칙: 안전을 위협하는 조건은 불허
   - 위반 예시:
     * "안전장비 미착용 동의서에 서명했으므로 유효합니다" (X)
   - 준수 예시:
     * "안전 관련 권리는 포기 불가능합니다" (O)

6. 명시 의무 (TRANSPARENCY)
   - 원칙: 핵심 근로조건은 서면 명시 필수
   - 위반 예시:
     * "구두 합의도 근로계약으로 인정됩니다" (X - 명시의무 회피)
   - 준수 예시:
     * "임금, 근로시간 등은 반드시 서면으로 명시해야 합니다" (O)

---
[위반 심각도 (severity) 판단 기준]

- high: 헌법 원칙을 정면으로 위반, 근로자에게 중대한 불이익 초래
  예: "위약금 조항도 합의했으니 유효합니다"

- medium: 원칙 취지에 부합하지 않으나 직접적 위반은 아님
  예: "이 조항은 다소 모호하지만 크게 문제되지 않습니다"

- low: 표현이나 뉘앙스가 원칙과 어긋남
  예: "근로자도 책임이 있을 수 있습니다" (책임 전가 뉘앙스)

---
[검토 절차 - Step by Step]
1. AI 답변의 핵심 주장을 파악하세요
2. 6대 원칙 각각에 대해 위반 여부를 검토하세요
3. 위반이 발견되면 위 예시를 참고하여 심각도를 판단하세요
4. 수정 제안을 구체적으로 작성하세요
5. 전체적으로 헌법 준수 여부를 판정하세요

---
[Few-shot 예시]

예시 1 - 위반 발견:
답변: "포괄임금제 약정이 있으므로 연장근로수당은 별도 지급하지 않아도 됩니다."
검토: {{
    "violations": [{{
        "principle": "MINIMUM_STANDARD",
        "detected": true,
        "critique": "포괄임금제라도 실제 연장근로시간이 약정을 초과하면 추가 수당 지급 필요. 무조건 미지급 가능하다는 답변은 최저기준 원칙 위반",
        "severity": "high",
        "suggestion": "포괄임금제의 한계와 추가 수당 지급 의무를 명시해야 함"
    }}],
    "is_constitutional": false,
    "overall_assessment": "최저기준 원칙을 위반하는 답변으로, 근로자의 정당한 임금 청구권을 침해할 수 있음"
}}

예시 2 - 위반 없음:
답변: "이 계약서의 위약금 조항은 근로기준법 제20조에 따라 무효입니다."
검토: {{
    "violations": [],
    "is_constitutional": true,
    "overall_assessment": "근로기준법 헌법 원칙을 충실히 준수한 답변"
}}

---
[출력 형식 - JSON]
{{
    "violations": [
        {{
            "principle": "원칙명 (HUMAN_DIGNITY/WORKER_PROTECTION/MINIMUM_STANDARD/EQUALITY/SAFETY/TRANSPARENCY)",
            "detected": true,
            "critique": "구체적인 위반 내용",
            "severity": "high/medium/low",
            "suggestion": "구체적인 수정 제안"
        }}
    ],
    "is_constitutional": false,
    "overall_assessment": "전체 평가 (2-3문장)"
}}"""

    # 수정 프롬프트 (Revise)
    REVISE_PROMPT = """다음 AI 답변을 '근로기준법 헌법'에 맞게 수정하세요.

[원본 답변]
{original_response}

[지적된 문제점]
{critiques}

[수정 지침]
1. 근로자 보호 관점에서 재작성
2. 모호한 표현은 명확하게
3. 법적 근거를 추가
4. 위험 요소는 반드시 경고

[수정된 답변]"""

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
            model: 사용할 LLM 모델 (기본값: settings.LLM_CONSTITUTIONAL_MODEL)
            strict_mode: 엄격 모드 (위반 시 반드시 수정)
            contract_id: 계약서 ID (토큰 추적용)
        """
        self.model = model if model else settings.LLM_CONSTITUTIONAL_MODEL
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

    def get_system_prompt(self) -> str:
        """Constitutional AI 시스템 프롬프트 반환"""
        return self.LABOR_LAW_CONSTITUTION

    def review(
        self,
        response: str,
        context: Optional[str] = None
    ) -> ConstitutionalReview:
        """
        AI 응답에 대한 헌법적 검토

        Args:
            response: 검토할 AI 응답
            context: 추가 컨텍스트 (질문, 계약서 내용 등)

        Returns:
            ConstitutionalReview: 검토 결과
        """
        # 1. 비판 (Critique)
        critiques = self._critique(response, context)

        review = ConstitutionalReview(
            original_response=response,
            critiques=critiques,
            is_constitutional=not any(c.violation_detected for c in critiques)
        )

        # 2. 수정 (Revise) - 위반이 있는 경우
        if review.has_violations and self.strict_mode:
            review.revised_response = self._revise(response, critiques)
            review.confidence_score = self._calculate_confidence(critiques)

        return review

    def _critique(
        self,
        response: str,
        context: Optional[str] = None
    ) -> List[CritiqueResult]:
        """비판 단계: 헌법 원칙 위반 검토"""
        if self._gemini_model is None and self.llm_client is None:
            return self._rule_based_critique(response)

        llm_start = time.time()
        try:
            prompt = self.CRITIQUE_PROMPT.format(response=response)

            if context:
                prompt = f"[컨텍스트]\n{context}\n\n" + prompt

            # Gemini 사용 (우선)
            if self._gemini_model is not None:
                full_prompt = "당신은 노동법 준수를 검토하는 법률 감사관입니다.\n\n" + prompt
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
                        module="constitutional_ai.critique",
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
                critique_data = json.loads(result_text)

            # OpenAI fallback
            else:
                result = self.llm_client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": "당신은 노동법 준수를 검토하는 법률 감사관입니다."
                        },
                        {"role": "user", "content": prompt}
                    ],
                    response_format={"type": "json_object"}
                )

                llm_duration = (time.time() - llm_start) * 1000

                # 토큰 사용량 기록
                if result.usage and self.contract_id:
                    cached = 0
                    if hasattr(result.usage, 'prompt_tokens_details') and result.usage.prompt_tokens_details:
                        cached = getattr(result.usage.prompt_tokens_details, 'cached_tokens', 0) or 0
                    record_llm_usage(
                        contract_id=self.contract_id,
                        module="constitutional_ai.critique",
                        model=self.model,
                        input_tokens=result.usage.prompt_tokens,
                        output_tokens=result.usage.completion_tokens,
                        cached_tokens=cached,
                        duration_ms=llm_duration
                    )

                critique_data = json.loads(result.choices[0].message.content)

            return self._parse_critiques(critique_data)

        except Exception as e:
            print(f"Critique error: {e}")
            return self._rule_based_critique(response)

    def _rule_based_critique(self, response: str) -> List[CritiqueResult]:
        """규칙 기반 비판 (LLM 없이)"""
        critiques = []

        # 위험 표현 패턴
        dangerous_patterns = {
            "적법합니다": ("MINIMUM_STANDARD", "단정적 적법 판단은 위험할 수 있습니다"),
            "문제없습니다": ("MINIMUM_STANDARD", "잠재적 문제를 간과할 수 있습니다"),
            "괜찮습니다": ("WORKER_PROTECTION", "근로자에게 불리한 조건일 수 있습니다"),
            "포괄임금": ("MINIMUM_STANDARD", "포괄임금제는 연장근로수당 미지급 위험"),
            "위약금": ("HUMAN_DIGNITY", "과도한 위약금은 인간 존엄성 침해"),
        }

        for pattern, (principle, critique_text) in dangerous_patterns.items():
            if pattern in response:
                critiques.append(CritiqueResult(
                    principle=ConstitutionalPrinciple[principle],
                    violation_detected=True,
                    critique=critique_text,
                    severity="medium",
                    suggestion=f"'{pattern}'에 대한 법적 검토가 필요합니다"
                ))

        return critiques

    def _parse_critiques(self, data: Dict) -> List[CritiqueResult]:
        """LLM 응답을 CritiqueResult로 파싱"""
        critiques = []

        for v in data.get("violations", []):
            principle_name = v.get("principle", "").upper().replace(" ", "_")

            try:
                principle = ConstitutionalPrinciple[principle_name]
            except KeyError:
                principle = ConstitutionalPrinciple.WORKER_PROTECTION

            critiques.append(CritiqueResult(
                principle=principle,
                violation_detected=v.get("detected", False),
                critique=v.get("critique", ""),
                severity=v.get("severity", "medium"),
                suggestion=v.get("suggestion")
            ))

        return critiques

    def _revise(
        self,
        original_response: str,
        critiques: List[CritiqueResult]
    ) -> str:
        """수정 단계: 헌법 원칙에 맞게 응답 수정"""
        if self._gemini_model is None and self.llm_client is None:
            return self._add_warnings(original_response, critiques)

        llm_start = time.time()
        try:
            critique_text = "\n".join([
                f"- [{c.principle.value}] {c.critique}"
                for c in critiques if c.violation_detected
            ])

            prompt = self.REVISE_PROMPT.format(
                original_response=original_response,
                critiques=critique_text
            )

            # Gemini 사용 (우선)
            if self._gemini_model is not None:
                full_prompt = self.LABOR_LAW_CONSTITUTION + "\n\n" + prompt
                result = self._gemini_model.generate_content(
                    full_prompt,
                    generation_config={
                        "temperature": 0.3,
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
                        module="constitutional_ai.revise",
                        model=self.model,
                        input_tokens=getattr(usage, 'prompt_token_count', 0),
                        output_tokens=getattr(usage, 'candidates_token_count', 0),
                        cached_tokens=getattr(usage, 'cached_content_token_count', 0),
                        duration_ms=llm_duration
                    )

                return result.text.strip()

            # OpenAI fallback
            else:
                result = self.llm_client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": self.LABOR_LAW_CONSTITUTION
                        },
                        {"role": "user", "content": prompt}
                    ],
                    max_completion_tokens=1500
                )

                llm_duration = (time.time() - llm_start) * 1000

                # 토큰 사용량 기록
                if result.usage and self.contract_id:
                    cached = 0
                    if hasattr(result.usage, 'prompt_tokens_details') and result.usage.prompt_tokens_details:
                        cached = getattr(result.usage.prompt_tokens_details, 'cached_tokens', 0) or 0
                    record_llm_usage(
                        contract_id=self.contract_id,
                        module="constitutional_ai.revise",
                        model=self.model,
                        input_tokens=result.usage.prompt_tokens,
                        output_tokens=result.usage.completion_tokens,
                        cached_tokens=cached,
                        duration_ms=llm_duration
                    )

                return result.choices[0].message.content

        except Exception as e:
            print(f"Revise error: {e}")
            return self._add_warnings(original_response, critiques)

    def _add_warnings(
        self,
        response: str,
        critiques: List[CritiqueResult]
    ) -> str:
        """경고 추가 (폴백)"""
        warnings = []
        for c in critiques:
            if c.violation_detected:
                warnings.append(f"[주의: {c.principle.value}] {c.critique}")

        if warnings:
            return response + "\n\n" + "\n".join(warnings)
        return response

    def _calculate_confidence(self, critiques: List[CritiqueResult]) -> float:
        """신뢰도 점수 계산"""
        if not critiques:
            return 1.0

        severity_weights = {"high": 0.3, "medium": 0.15, "low": 0.05}
        total_penalty = 0

        for c in critiques:
            if c.violation_detected:
                total_penalty += severity_weights.get(c.severity, 0.1)

        return max(0.0, 1.0 - total_penalty)

    def enhance_prompt(self, user_prompt: str) -> str:
        """사용자 프롬프트에 Constitutional AI 원칙 주입"""
        return f"""[Constitutional AI 모드]
다음 헌법 원칙을 준수하여 답변하세요:
{self.LABOR_LAW_CONSTITUTION}

[사용자 질문]
{user_prompt}

[답변 시 주의사항]
1. 위 헌법 원칙을 반드시 준수
2. 근로자 보호 관점 유지
3. 불확실한 경우 경고 포함
"""


class ConstitutionalAnalyzer:
    """
    계약서 분석에 Constitutional AI 적용

    분석 결과에 대해 헌법적 검토 수행
    """

    def __init__(self, cai: ConstitutionalAI):
        self.cai = cai

    def analyze_with_constitution(
        self,
        contract_text: str,
        analysis_result: str
    ) -> Dict[str, Any]:
        """
        계약서 분석 결과에 헌법적 검토 적용

        Args:
            contract_text: 원본 계약서 텍스트
            analysis_result: AI 분석 결과

        Returns:
            헌법적으로 검토된 분석 결과
        """
        review = self.cai.review(
            response=analysis_result,
            context=f"계약서 내용:\n{contract_text[:2000]}"
        )

        return {
            "original_analysis": analysis_result,
            "constitutional_review": {
                "is_constitutional": review.is_constitutional,
                "violations": [
                    {
                        "principle": c.principle.value,
                        "critique": c.critique,
                        "severity": c.severity,
                        "suggestion": c.suggestion
                    }
                    for c in review.critiques if c.violation_detected
                ],
                "confidence_score": review.confidence_score
            },
            "revised_analysis": review.revised_response or analysis_result,
            "warnings": [
                c.critique for c in review.critiques
                if c.violation_detected and c.severity == "high"
            ]
        }


# 편의 함수
def create_constitutional_ai(
    strict_mode: bool = True,
    model: str = "gpt-4o-mini"
) -> ConstitutionalAI:
    """Constitutional AI 팩토리 함수"""
    return ConstitutionalAI(model=model, strict_mode=strict_mode)


def get_labor_law_constitution() -> str:
    """근로기준법 헌법 텍스트 반환"""
    return ConstitutionalAI.LABOR_LAW_CONSTITUTION
