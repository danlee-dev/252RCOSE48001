"""
Multi-LLM Evaluator Framework

Supports multiple LLM providers for evaluation:
- OpenAI (GPT-4o, GPT-4o-mini)
- Google Gemini (gemini-2.5-flash, gemini-1.5-pro)
- Anthropic Claude (claude-3-5-sonnet) - optional via env var

Implements LLM-as-Judge pattern with cross-validation between evaluators.

Academic References:
- G-Eval (Liu et al., 2023): LLM-based NLG evaluation
- RAGAS (Es et al., 2023): RAG evaluation framework
- Judging LLM-as-a-Judge (Zheng et al., 2023): MT-Bench evaluation
"""

import os
import json
import asyncio
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
import time
import hashlib


class EvaluatorType(Enum):
    """Available evaluator types"""
    OPENAI = "openai"
    GEMINI = "gemini"
    CLAUDE = "claude"


class EvaluationDimension(Enum):
    """Standard evaluation dimensions (G-Eval inspired)"""
    ACCURACY = "accuracy"           # Factual correctness
    COMPLETENESS = "completeness"   # Coverage of all relevant points
    RELEVANCE = "relevance"         # Relevance to the task
    CONSISTENCY = "consistency"     # Internal logic consistency
    FAITHFULNESS = "faithfulness"   # Grounded in source (no hallucination)
    COHERENCE = "coherence"         # Logical flow and structure
    LEGAL_BASIS = "legal_basis"     # Quality of legal references


@dataclass
class EvaluationScore:
    """Single evaluation score with reasoning"""
    dimension: str
    score: float  # 0.0 - 1.0
    reasoning: str
    evidence: List[str] = field(default_factory=list)
    confidence: float = 1.0


@dataclass
class EvaluationResult:
    """Complete evaluation result from a single evaluator"""
    evaluator_type: str
    evaluator_model: str
    scores: Dict[str, EvaluationScore]
    overall_score: float
    verdict: str  # PASS / FAIL / BORDERLINE
    detailed_feedback: str
    raw_response: str = ""
    latency_ms: float = 0.0
    token_usage: Dict[str, int] = field(default_factory=dict)
    timestamp: str = ""


@dataclass
class CrossValidationResult:
    """Result of cross-validation between multiple evaluators"""
    individual_results: List[EvaluationResult]
    consensus_score: float
    agreement_rate: float  # Inter-rater reliability (0-1)
    disagreement_dimensions: List[str]
    final_verdict: str
    confidence: float


class BaseLLMEvaluator(ABC):
    """Abstract base class for LLM evaluators"""

    def __init__(self, model: str, temperature: float = 0.0):
        self.model = model
        self.temperature = temperature
        self._client = None

    @abstractmethod
    def _init_client(self):
        """Initialize the LLM client"""
        pass

    @abstractmethod
    async def evaluate(
        self,
        content: str,
        reference: Optional[str] = None,
        dimensions: List[EvaluationDimension] = None,
        custom_criteria: Dict[str, str] = None,
        context: Dict[str, Any] = None
    ) -> EvaluationResult:
        """
        Evaluate content using LLM-as-Judge pattern

        Args:
            content: The content to evaluate
            reference: Optional reference/ground truth
            dimensions: Which dimensions to evaluate
            custom_criteria: Custom evaluation criteria
            context: Additional context (e.g., contract text)

        Returns:
            EvaluationResult with scores and feedback
        """
        pass

    def _build_evaluation_prompt(
        self,
        content: str,
        reference: Optional[str],
        dimensions: List[EvaluationDimension],
        custom_criteria: Dict[str, str],
        context: Dict[str, Any]
    ) -> str:
        """Build the evaluation prompt"""

        dimension_descriptions = {
            EvaluationDimension.ACCURACY: "정확성: 제시된 정보가 사실적으로 정확한가?",
            EvaluationDimension.COMPLETENESS: "완전성: 모든 관련 정보가 포함되어 있는가?",
            EvaluationDimension.RELEVANCE: "관련성: 내용이 주어진 맥락에 적절한가?",
            EvaluationDimension.CONSISTENCY: "일관성: 내부적으로 논리적 모순이 없는가?",
            EvaluationDimension.FAITHFULNESS: "충실성: 원본 텍스트에 근거하고 있는가? (환각 없음)",
            EvaluationDimension.COHERENCE: "논리성: 논리적 흐름이 자연스러운가?",
            EvaluationDimension.LEGAL_BASIS: "법적 근거: 인용된 법률/판례가 정확하고 적절한가?"
        }

        dims_text = "\n".join([
            f"- {dim.value}: {dimension_descriptions.get(dim, dim.value)}"
            for dim in dimensions
        ])

        custom_text = ""
        if custom_criteria:
            custom_text = "\n추가 평가 기준:\n" + "\n".join([
                f"- {k}: {v}" for k, v in custom_criteria.items()
            ])

        context_text = ""
        if context:
            if "contract_text" in context:
                context_text = f"\n\n[원본 계약서]\n{context['contract_text'][:3000]}..."
            if "ground_truth" in context:
                context_text += f"\n\n[정답 (Ground Truth)]\n{json.dumps(context['ground_truth'], ensure_ascii=False, indent=2)}"

        reference_text = ""
        if reference:
            reference_text = f"\n\n[참조 정보]\n{reference}"

        prompt = f"""당신은 AI 시스템의 출력물을 평가하는 전문 평가자입니다.
다음 내용을 아래 기준에 따라 엄격하게 평가해주세요.

[평가 대상]
{content}
{reference_text}
{context_text}

[평가 기준]
{dims_text}
{custom_text}

[평가 방법]
각 기준에 대해 0.0 ~ 1.0 점수를 부여하고 그 이유를 설명하세요.
- 0.0 ~ 0.3: 심각한 문제 (FAIL)
- 0.3 ~ 0.6: 개선 필요 (BORDERLINE)
- 0.6 ~ 0.8: 양호 (PASS)
- 0.8 ~ 1.0: 우수 (EXCELLENT)

반드시 아래 JSON 형식으로만 응답하세요:
{{
    "scores": {{
        "<dimension>": {{
            "score": 0.0-1.0,
            "reasoning": "평가 이유",
            "evidence": ["근거1", "근거2"]
        }}
    }},
    "overall_score": 0.0-1.0,
    "verdict": "PASS/FAIL/BORDERLINE",
    "detailed_feedback": "종합 피드백"
}}"""

        return prompt

    def _parse_evaluation_response(
        self,
        response: str,
        dimensions: List[EvaluationDimension]
    ) -> Tuple[Dict[str, EvaluationScore], float, str, str]:
        """Parse LLM response into structured evaluation"""

        try:
            # Try to extract JSON from response
            json_start = response.find("{")
            json_end = response.rfind("}") + 1
            if json_start != -1 and json_end > json_start:
                json_str = response[json_start:json_end]
                data = json.loads(json_str)
            else:
                raise ValueError("No JSON found in response")

            scores = {}
            for dim in dimensions:
                dim_key = dim.value
                if dim_key in data.get("scores", {}):
                    score_data = data["scores"][dim_key]
                    scores[dim_key] = EvaluationScore(
                        dimension=dim_key,
                        score=float(score_data.get("score", 0.5)),
                        reasoning=score_data.get("reasoning", ""),
                        evidence=score_data.get("evidence", [])
                    )
                else:
                    # Default score if dimension not evaluated
                    scores[dim_key] = EvaluationScore(
                        dimension=dim_key,
                        score=0.5,
                        reasoning="Not evaluated",
                        evidence=[]
                    )

            overall = float(data.get("overall_score", 0.5))
            verdict = data.get("verdict", "BORDERLINE")
            feedback = data.get("detailed_feedback", "")

            return scores, overall, verdict, feedback

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            # Return default scores on parse error
            default_scores = {
                dim.value: EvaluationScore(
                    dimension=dim.value,
                    score=0.5,
                    reasoning=f"Parse error: {str(e)}",
                    evidence=[]
                )
                for dim in dimensions
            }
            return default_scores, 0.5, "BORDERLINE", f"Failed to parse response: {str(e)}"


class OpenAIEvaluator(BaseLLMEvaluator):
    """OpenAI-based evaluator (GPT-4o, GPT-4o-mini)"""

    def __init__(self, model: str = "gpt-4o", temperature: float = 0.0):
        super().__init__(model, temperature)
        self._init_client()

    def _init_client(self):
        try:
            from openai import OpenAI
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY not set")
            self._client = OpenAI(api_key=api_key)
        except ImportError:
            raise ImportError("openai package not installed")

    async def evaluate(
        self,
        content: str,
        reference: Optional[str] = None,
        dimensions: List[EvaluationDimension] = None,
        custom_criteria: Dict[str, str] = None,
        context: Dict[str, Any] = None
    ) -> EvaluationResult:

        if dimensions is None:
            dimensions = [EvaluationDimension.ACCURACY, EvaluationDimension.FAITHFULNESS]

        prompt = self._build_evaluation_prompt(
            content, reference, dimensions, custom_criteria or {}, context or {}
        )

        start_time = time.time()

        response = self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            response_format={"type": "json_object"}
        )

        latency = (time.time() - start_time) * 1000
        raw_response = response.choices[0].message.content

        scores, overall, verdict, feedback = self._parse_evaluation_response(
            raw_response, dimensions
        )

        return EvaluationResult(
            evaluator_type=EvaluatorType.OPENAI.value,
            evaluator_model=self.model,
            scores=scores,
            overall_score=overall,
            verdict=verdict,
            detailed_feedback=feedback,
            raw_response=raw_response,
            latency_ms=latency,
            token_usage={
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens
            },
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
        )


class GeminiEvaluator(BaseLLMEvaluator):
    """Google Gemini-based evaluator"""

    def __init__(self, model: str = "gemini-2.5-flash", temperature: float = 0.0):
        super().__init__(model, temperature)
        self._init_client()

    def _init_client(self):
        try:
            import google.generativeai as genai
            api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError("GEMINI_API_KEY or GOOGLE_API_KEY not set")
            genai.configure(api_key=api_key)
            self._client = genai.GenerativeModel(self.model)
        except ImportError:
            raise ImportError("google-generativeai package not installed")

    async def evaluate(
        self,
        content: str,
        reference: Optional[str] = None,
        dimensions: List[EvaluationDimension] = None,
        custom_criteria: Dict[str, str] = None,
        context: Dict[str, Any] = None
    ) -> EvaluationResult:

        if dimensions is None:
            dimensions = [EvaluationDimension.ACCURACY, EvaluationDimension.FAITHFULNESS]

        prompt = self._build_evaluation_prompt(
            content, reference, dimensions, custom_criteria or {}, context or {}
        )

        start_time = time.time()

        response = self._client.generate_content(
            prompt,
            generation_config={
                "temperature": self.temperature,
                "response_mime_type": "application/json"
            }
        )

        latency = (time.time() - start_time) * 1000
        raw_response = response.text

        scores, overall, verdict, feedback = self._parse_evaluation_response(
            raw_response, dimensions
        )

        # Gemini doesn't provide token count in the same way
        token_usage = {}
        if hasattr(response, 'usage_metadata'):
            token_usage = {
                "prompt_tokens": getattr(response.usage_metadata, 'prompt_token_count', 0),
                "completion_tokens": getattr(response.usage_metadata, 'candidates_token_count', 0)
            }

        return EvaluationResult(
            evaluator_type=EvaluatorType.GEMINI.value,
            evaluator_model=self.model,
            scores=scores,
            overall_score=overall,
            verdict=verdict,
            detailed_feedback=feedback,
            raw_response=raw_response,
            latency_ms=latency,
            token_usage=token_usage,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
        )


class ClaudeEvaluator(BaseLLMEvaluator):
    """Anthropic Claude-based evaluator (optional)"""

    def __init__(self, model: str = "claude-sonnet-4-20250514", temperature: float = 0.0):
        super().__init__(model, temperature)
        self._init_client()

    def _init_client(self):
        try:
            import anthropic
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY not set")
            self._client = anthropic.Anthropic(api_key=api_key)
        except ImportError:
            raise ImportError("anthropic package not installed")

    async def evaluate(
        self,
        content: str,
        reference: Optional[str] = None,
        dimensions: List[EvaluationDimension] = None,
        custom_criteria: Dict[str, str] = None,
        context: Dict[str, Any] = None
    ) -> EvaluationResult:

        if dimensions is None:
            dimensions = [EvaluationDimension.ACCURACY, EvaluationDimension.FAITHFULNESS]

        prompt = self._build_evaluation_prompt(
            content, reference, dimensions, custom_criteria or {}, context or {}
        )

        start_time = time.time()

        response = self._client.messages.create(
            model=self.model,
            max_tokens=4096,
            temperature=self.temperature,
            messages=[{"role": "user", "content": prompt}]
        )

        latency = (time.time() - start_time) * 1000
        raw_response = response.content[0].text

        scores, overall, verdict, feedback = self._parse_evaluation_response(
            raw_response, dimensions
        )

        return EvaluationResult(
            evaluator_type=EvaluatorType.CLAUDE.value,
            evaluator_model=self.model,
            scores=scores,
            overall_score=overall,
            verdict=verdict,
            detailed_feedback=feedback,
            raw_response=raw_response,
            latency_ms=latency,
            token_usage={
                "prompt_tokens": response.usage.input_tokens,
                "completion_tokens": response.usage.output_tokens
            },
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
        )


class MultiLLMEvaluator:
    """
    Multi-LLM evaluation with cross-validation

    Uses multiple LLMs to evaluate the same content and calculates
    inter-rater agreement for reliability assessment.
    """

    def __init__(
        self,
        use_openai: bool = True,
        use_gemini: bool = True,
        use_claude: bool = False,  # Optional, requires ANTHROPIC_API_KEY
        openai_model: str = "gpt-4o-mini",
        gemini_model: str = "gemini-2.5-flash",
        claude_model: str = "claude-sonnet-4-20250514"
    ):
        self.evaluators: List[BaseLLMEvaluator] = []

        if use_openai:
            try:
                self.evaluators.append(OpenAIEvaluator(model=openai_model))
            except (ImportError, ValueError) as e:
                print(f"[Warning] OpenAI evaluator not available: {e}")

        if use_gemini:
            try:
                self.evaluators.append(GeminiEvaluator(model=gemini_model))
            except (ImportError, ValueError) as e:
                print(f"[Warning] Gemini evaluator not available: {e}")

        if use_claude:
            try:
                self.evaluators.append(ClaudeEvaluator(model=claude_model))
            except (ImportError, ValueError) as e:
                print(f"[Warning] Claude evaluator not available: {e}")

        if not self.evaluators:
            raise ValueError("No evaluators available. Check API keys.")

    async def evaluate_with_cross_validation(
        self,
        content: str,
        reference: Optional[str] = None,
        dimensions: List[EvaluationDimension] = None,
        custom_criteria: Dict[str, str] = None,
        context: Dict[str, Any] = None
    ) -> CrossValidationResult:
        """
        Evaluate content using multiple LLMs and cross-validate

        Returns:
            CrossValidationResult with consensus score and agreement metrics
        """

        # Run all evaluators
        tasks = [
            evaluator.evaluate(content, reference, dimensions, custom_criteria, context)
            for evaluator in self.evaluators
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out errors
        valid_results = [r for r in results if isinstance(r, EvaluationResult)]

        if not valid_results:
            raise ValueError("All evaluators failed")

        # Calculate consensus
        consensus_score = sum(r.overall_score for r in valid_results) / len(valid_results)

        # Calculate inter-rater agreement (simplified Fleiss' kappa proxy)
        if len(valid_results) >= 2:
            scores = [r.overall_score for r in valid_results]
            score_variance = sum((s - consensus_score) ** 2 for s in scores) / len(scores)
            agreement_rate = max(0, 1 - (score_variance / 0.25))  # Normalize by max variance
        else:
            agreement_rate = 1.0

        # Find disagreement dimensions
        disagreement_dims = []
        if dimensions and len(valid_results) >= 2:
            for dim in dimensions:
                dim_scores = []
                for r in valid_results:
                    if dim.value in r.scores:
                        dim_scores.append(r.scores[dim.value].score)

                if dim_scores and len(dim_scores) >= 2:
                    dim_variance = sum((s - sum(dim_scores)/len(dim_scores)) ** 2 for s in dim_scores) / len(dim_scores)
                    if dim_variance > 0.04:  # Threshold for significant disagreement
                        disagreement_dims.append(dim.value)

        # Determine final verdict
        verdicts = [r.verdict for r in valid_results]
        if verdicts.count("PASS") > len(verdicts) / 2:
            final_verdict = "PASS"
        elif verdicts.count("FAIL") > len(verdicts) / 2:
            final_verdict = "FAIL"
        else:
            final_verdict = "BORDERLINE"

        # Confidence based on agreement
        confidence = agreement_rate * min(1.0, len(valid_results) / 2)

        return CrossValidationResult(
            individual_results=valid_results,
            consensus_score=consensus_score,
            agreement_rate=agreement_rate,
            disagreement_dimensions=disagreement_dims,
            final_verdict=final_verdict,
            confidence=confidence
        )

    async def evaluate_single(
        self,
        content: str,
        evaluator_type: EvaluatorType = EvaluatorType.OPENAI,
        reference: Optional[str] = None,
        dimensions: List[EvaluationDimension] = None,
        custom_criteria: Dict[str, str] = None,
        context: Dict[str, Any] = None
    ) -> EvaluationResult:
        """Use a single specific evaluator"""

        for evaluator in self.evaluators:
            if evaluator_type.value in evaluator.__class__.__name__.lower():
                return await evaluator.evaluate(
                    content, reference, dimensions, custom_criteria, context
                )

        # Fallback to first available
        return await self.evaluators[0].evaluate(
            content, reference, dimensions, custom_criteria, context
        )


def create_evaluator(
    use_cross_validation: bool = True,
    **kwargs
) -> Union[MultiLLMEvaluator, BaseLLMEvaluator]:
    """Factory function to create evaluator"""

    if use_cross_validation:
        return MultiLLMEvaluator(**kwargs)

    # Single evaluator - try in order of preference
    if os.getenv("OPENAI_API_KEY"):
        return OpenAIEvaluator(model=kwargs.get("openai_model", "gpt-4o-mini"))
    elif os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"):
        return GeminiEvaluator(model=kwargs.get("gemini_model", "gemini-2.5-flash"))
    elif os.getenv("ANTHROPIC_API_KEY"):
        return ClaudeEvaluator(model=kwargs.get("claude_model", "claude-sonnet-4-20250514"))
    else:
        raise ValueError("No API keys found for any LLM provider")
