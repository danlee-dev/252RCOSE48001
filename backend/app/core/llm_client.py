"""
Hybrid LLM Client
Gemini (retrieval/summarization) + OpenAI (reasoning/response) 하이브리드 접근법
"""

from typing import Optional, List, Dict, Any
from enum import Enum
from dataclasses import dataclass

from app.core.config import settings


class LLMTask(Enum):
    """LLM 작업 유형"""
    RETRIEVAL = "retrieval"           # 검색/요약 - Gemini
    SUMMARIZATION = "summarization"   # 문서 요약 - Gemini
    REASONING = "reasoning"           # 법률 추론 - OpenAI
    RESPONSE = "response"             # 응답 생성 - OpenAI
    ANALYSIS = "analysis"             # 분석 - OpenAI


@dataclass
class LLMResponse:
    """LLM 응답"""
    content: str
    model: str
    provider: str
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0


class HybridLLMClient:
    """
    하이브리드 LLM 클라이언트

    - Retrieval/Summarization: Gemini 2.5 Flash Lite (저렴, 빠름)
    - Reasoning/Response: GPT-5 Mini (정확, 지시 이행)
    """

    def __init__(self):
        self._openai_client = None
        self._gemini_model = None

        # 모델 설정
        self.retrieval_model = settings.LLM_RETRIEVAL_MODEL
        self.reasoning_model = settings.LLM_REASONING_MODEL

    @property
    def openai_client(self):
        """OpenAI 클라이언트 (lazy loading)"""
        if self._openai_client is None:
            try:
                from openai import OpenAI
                self._openai_client = OpenAI(api_key=settings.OPENAI_API_KEY)
            except ImportError:
                raise ImportError("openai package not installed")
        return self._openai_client

    @property
    def gemini_model(self):
        """Gemini 모델 (lazy loading)"""
        if self._gemini_model is None:
            try:
                import google.generativeai as genai
                genai.configure(api_key=settings.GEMINI_API_KEY)
                self._gemini_model = genai.GenerativeModel(self.retrieval_model)
            except ImportError:
                raise ImportError("google-generativeai package not installed")
        return self._gemini_model

    def _select_provider(self, task: LLMTask) -> str:
        """작업 유형에 따른 제공자 선택"""
        if task in [LLMTask.RETRIEVAL, LLMTask.SUMMARIZATION]:
            return "gemini"
        return "openai"

    def generate(
        self,
        prompt: str,
        task: LLMTask = LLMTask.REASONING,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2000
    ) -> LLMResponse:
        """
        LLM 응답 생성

        Args:
            prompt: 사용자 프롬프트
            task: 작업 유형 (retrieval/reasoning 등)
            system_prompt: 시스템 프롬프트
            temperature: 온도 파라미터
            max_tokens: 최대 출력 토큰

        Returns:
            LLMResponse
        """
        provider = self._select_provider(task)

        if provider == "gemini":
            return self._generate_gemini(prompt, system_prompt, temperature, max_tokens)
        else:
            return self._generate_openai(prompt, system_prompt, temperature, max_tokens)

    def _generate_gemini(
        self,
        prompt: str,
        system_prompt: Optional[str],
        temperature: float,
        max_tokens: int
    ) -> LLMResponse:
        """Gemini로 생성"""
        try:
            full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt

            generation_config = {
                "temperature": temperature,
                "max_output_tokens": max_tokens,
            }

            response = self.gemini_model.generate_content(
                full_prompt,
                generation_config=generation_config
            )

            # 토큰 수 추정 (Gemini는 직접 제공하지 않음)
            input_tokens = len(full_prompt.split()) * 1.3  # 대략적 추정
            output_tokens = len(response.text.split()) * 1.3

            # 비용 계산 (gemini-2.5-flash-lite 기준)
            cost = (input_tokens / 1_000_000 * 0.10) + (output_tokens / 1_000_000 * 0.40)

            return LLMResponse(
                content=response.text,
                model=self.retrieval_model,
                provider="gemini",
                input_tokens=int(input_tokens),
                output_tokens=int(output_tokens),
                cost_usd=cost
            )

        except Exception as e:
            # Gemini 실패 시 OpenAI로 폴백
            print(f"Gemini failed, falling back to OpenAI: {e}")
            return self._generate_openai(prompt, system_prompt, temperature, max_tokens)

    def _generate_openai(
        self,
        prompt: str,
        system_prompt: Optional[str],
        temperature: float,
        max_tokens: int
    ) -> LLMResponse:
        """OpenAI로 생성"""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        # gpt-5-mini 등 일부 모델은 temperature 커스텀 미지원
        # temperature=1.0만 지원하는 모델은 파라미터 생략
        response = self.openai_client.chat.completions.create(
            model=self.reasoning_model,
            messages=messages,
            max_completion_tokens=max_tokens
        )

        usage = response.usage
        input_tokens = usage.prompt_tokens if usage else 0
        output_tokens = usage.completion_tokens if usage else 0

        # 비용 계산 (gpt-5-mini 기준)
        cost = (input_tokens / 1_000_000 * 0.30) + (output_tokens / 1_000_000 * 1.50)

        return LLMResponse(
            content=response.choices[0].message.content,
            model=self.reasoning_model,
            provider="openai",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost
        )

    def summarize(self, text: str, max_length: int = 500) -> LLMResponse:
        """문서 요약 (Gemini 사용)"""
        prompt = f"""다음 텍스트를 {max_length}자 이내로 요약해주세요.
핵심 내용만 간결하게 정리하세요.

텍스트:
{text}

요약:"""
        return self.generate(prompt, task=LLMTask.SUMMARIZATION, temperature=0.3)

    def analyze_legal(self, contract_text: str, question: str = None) -> LLMResponse:
        """법률 분석 (OpenAI 사용)"""
        system_prompt = """당신은 한국 노동법 전문가입니다.
근로계약서를 분석하여 법적 위험 요소를 찾아내고,
근로기준법, 최저임금법 등 관련 법률에 따른 정확한 판단을 제공합니다."""

        prompt = f"""다음 근로계약서를 분석해주세요.

계약서 내용:
{contract_text}

{f"질문: {question}" if question else "위험 조항과 개선 사항을 분석해주세요."}"""

        return self.generate(
            prompt,
            task=LLMTask.ANALYSIS,
            system_prompt=system_prompt,
            temperature=0.2
        )

    def enhance_query(self, query: str) -> LLMResponse:
        """검색 쿼리 강화 (Gemini 사용) - HyDE"""
        prompt = f"""다음 질문에 대해 법률 전문가가 작성할 것 같은
이상적인 답변을 가상으로 생성해주세요.
이 답변은 관련 법률 조항과 판례를 포함해야 합니다.

질문: {query}

가상 법률 답변:"""
        return self.generate(prompt, task=LLMTask.RETRIEVAL, temperature=0.5)


# 싱글톤 인스턴스
_llm_client: Optional[HybridLLMClient] = None


def get_llm_client() -> HybridLLMClient:
    """LLM 클라이언트 싱글톤 인스턴스 반환"""
    global _llm_client
    if _llm_client is None:
        _llm_client = HybridLLMClient()
    return _llm_client
