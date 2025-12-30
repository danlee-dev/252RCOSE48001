"""
End-to-End Pipeline Evaluator

Comprehensive evaluation of the entire DocScanner pipeline:
1. Full pipeline execution on test contracts
2. Aggregate metrics from all component evaluators
3. LLM-as-Judge for overall analysis quality
4. Comparison with baseline (pure LLM prompting)

This is the main evaluator that combines all sub-evaluators
and produces the final evaluation report.
"""

import os
import sys
import json
import asyncio
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime
import numpy as np
from scipy import stats

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "backend"))

from evaluation.core.llm_evaluators import (
    MultiLLMEvaluator,
    EvaluationDimension,
    CrossValidationResult,
    OpenAIEvaluator,
    GeminiEvaluator
)
from evaluation.evaluators.clause_extraction_eval import ClauseExtractionEvaluator
from evaluation.evaluators.violation_detection_eval import ViolationDetectionEvaluator
from evaluation.evaluators.legal_citation_eval import LegalCitationEvaluator
from evaluation.evaluators.underpayment_calc_eval import UnderpaymentCalculationEvaluator


@dataclass
class PipelineEvalResult:
    """Result of evaluating a single contract through the pipeline"""
    contract_id: str
    contract_text: str

    # Pipeline outputs
    pipeline_result: Dict[str, Any] = field(default_factory=dict)
    baseline_result: Dict[str, Any] = field(default_factory=dict)

    # Component scores
    clause_extraction_score: float = 0.0
    violation_detection_score: float = 0.0
    legal_citation_score: float = 0.0
    underpayment_accuracy: float = 0.0

    # Overall scores
    pipeline_overall_score: float = 0.0
    baseline_overall_score: float = 0.0

    # Processing metrics
    pipeline_latency_ms: float = 0.0
    baseline_latency_ms: float = 0.0
    pipeline_token_usage: Dict[str, int] = field(default_factory=dict)

    # Ground truth comparison
    ground_truth: Optional[Dict[str, Any]] = None
    ground_truth_match_score: float = 0.0


@dataclass
class EndToEndMetrics:
    """Aggregate metrics for end-to-end evaluation"""
    total_contracts: int = 0

    # Component metrics (averaged)
    avg_clause_extraction_score: float = 0.0
    avg_violation_detection_f1: float = 0.0
    avg_legal_citation_accuracy: float = 0.0
    avg_underpayment_mape: float = 0.0

    # Overall quality
    avg_pipeline_score: float = 0.0
    avg_baseline_score: float = 0.0
    pipeline_vs_baseline_improvement: float = 0.0

    # Statistical significance
    improvement_p_value: float = 1.0
    effect_size_cohens_d: float = 0.0

    # Reliability
    llm_evaluator_agreement: float = 0.0

    # Efficiency
    avg_pipeline_latency_ms: float = 0.0
    avg_baseline_latency_ms: float = 0.0
    avg_token_usage: Dict[str, float] = field(default_factory=dict)

    # Error analysis
    hallucination_rate: float = 0.0
    false_positive_rate: float = 0.0
    false_negative_rate: float = 0.0


# Baseline prompt for pure LLM comparison
BASELINE_ANALYSIS_PROMPT = """당신은 대한민국 노동법 전문 변호사입니다.
다음 근로계약서를 분석하여 법적 위험 요소를 찾아주세요.

[근로계약서]
{contract_text}

다음 JSON 형식으로 응답하세요:
{{
    "risk_level": "High/Medium/Low",
    "risk_score": 0.0-1.0,
    "violations": [
        {{
            "type": "위반 유형",
            "severity": "Critical/High/Medium/Low",
            "description": "위반 설명",
            "legal_basis": "법적 근거",
            "suggestion": "수정 제안"
        }}
    ],
    "annual_underpayment": 연간 예상 체불액 (숫자),
    "analysis_summary": "분석 요약"
}}

반드시 유효한 JSON만 응답하세요.
"""


class BaselineLLMAnalyzer:
    """Pure LLM baseline for comparison"""

    def __init__(self, model: str = "gpt-4o"):
        self.model = model
        self._init_client()

    def _init_client(self):
        try:
            from openai import OpenAI
            self._client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        except Exception:
            self._client = None

    def analyze(self, contract_text: str) -> Dict[str, Any]:
        """Analyze contract with pure LLM prompting"""
        if not self._client:
            return {"error": "OpenAI client not initialized"}

        start_time = time.time()

        try:
            prompt = BASELINE_ANALYSIS_PROMPT.format(contract_text=contract_text[:8000])

            response = self._client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                response_format={"type": "json_object"}
            )

            result = json.loads(response.choices[0].message.content)
            result["processing_time_ms"] = (time.time() - start_time) * 1000
            result["token_usage"] = {
                "prompt": response.usage.prompt_tokens,
                "completion": response.usage.completion_tokens
            }

            return result

        except Exception as e:
            return {
                "error": str(e),
                "risk_level": "Unknown",
                "violations": [],
                "annual_underpayment": 0,
                "processing_time_ms": (time.time() - start_time) * 1000
            }


class EndToEndEvaluator:
    """
    End-to-end pipeline evaluator

    Orchestrates all component evaluators and produces comprehensive results.
    """

    def __init__(
        self,
        output_dir: str = None,
        use_llm_evaluation: bool = True,
        run_baseline_comparison: bool = True
    ):
        self.output_dir = Path(output_dir) if output_dir else Path(__file__).parent.parent / "results" / "end_to_end"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.use_llm_evaluation = use_llm_evaluation
        self.run_baseline_comparison = run_baseline_comparison

        # Initialize component evaluators
        self.clause_evaluator = ClauseExtractionEvaluator(use_llm_evaluation=False)
        self.violation_evaluator = ViolationDetectionEvaluator(use_llm_evaluation=False)
        self.citation_evaluator = LegalCitationEvaluator(use_llm_evaluation=False)
        self.calc_evaluator = UnderpaymentCalculationEvaluator()

        # Initialize LLM evaluator for overall quality
        if use_llm_evaluation:
            try:
                self.llm_evaluator = MultiLLMEvaluator(
                    use_openai=True,
                    use_gemini=True,
                    use_claude=bool(os.getenv("ANTHROPIC_API_KEY"))
                )
            except ValueError:
                self.llm_evaluator = None
        else:
            self.llm_evaluator = None

        # Initialize baseline
        if run_baseline_comparison:
            self.baseline = BaselineLLMAnalyzer()
        else:
            self.baseline = None

        # Initialize DocScanner pipeline (lazy)
        self.pipeline = None

    def _init_pipeline(self):
        """Initialize DocScanner pipeline"""
        if self.pipeline is None:
            try:
                from app.ai.pipeline import AdvancedAIPipeline, PipelineConfig

                config = PipelineConfig(
                    enable_llm_clause_analysis=True,
                    enable_raptor=False,  # Disable for faster evaluation
                    enable_dspy=False
                )
                self.pipeline = AdvancedAIPipeline(config=config)
            except Exception as e:
                print(f"[Warning] Failed to initialize pipeline: {e}")
                self.pipeline = None

    async def evaluate_single_contract(
        self,
        contract_text: str,
        contract_id: str,
        ground_truth: Optional[Dict[str, Any]] = None
    ) -> PipelineEvalResult:
        """Evaluate a single contract through the pipeline"""

        result = PipelineEvalResult(
            contract_id=contract_id,
            contract_text=contract_text,
            ground_truth=ground_truth
        )

        # Run DocScanner pipeline
        self._init_pipeline()
        if self.pipeline:
            try:
                start_time = time.time()
                pipeline_output = self.pipeline.analyze(contract_text, contract_id=contract_id)
                result.pipeline_latency_ms = (time.time() - start_time) * 1000
                result.pipeline_result = pipeline_output.to_dict() if hasattr(pipeline_output, 'to_dict') else pipeline_output
                result.pipeline_token_usage = result.pipeline_result.get("token_usage", {})
            except Exception as e:
                print(f"  Pipeline error: {e}")
                result.pipeline_result = {"error": str(e)}

        # Run baseline if enabled
        if self.baseline and self.run_baseline_comparison:
            start_time = time.time()
            result.baseline_result = self.baseline.analyze(contract_text)
            result.baseline_latency_ms = result.baseline_result.get("processing_time_ms", (time.time() - start_time) * 1000)

        # Evaluate components
        if result.pipeline_result and "error" not in result.pipeline_result:
            # Clause extraction (if available)
            clause_analysis = result.pipeline_result.get("clause_analysis", {})
            extracted_clauses = clause_analysis.get("clauses", [])
            if extracted_clauses:
                clause_result = self.clause_evaluator.evaluate_extraction(
                    contract_text,
                    extracted_clauses,
                    ground_truth.get("clauses") if ground_truth else None,
                    contract_id
                )
                result.clause_extraction_score = clause_result.metrics.f1_score

            # Violation detection
            detected_violations = clause_analysis.get("violations", [])
            if ground_truth and "violations" in ground_truth:
                violation_result = self.violation_evaluator.evaluate_detection(
                    detected_violations,
                    ground_truth["violations"],
                    contract_id
                )
                result.violation_detection_score = violation_result.metrics.f1_score

            # Legal citation
            citation_metrics, _ = self.citation_evaluator.evaluate_citations(
                result.pipeline_result,
                contract_id
            )
            result.legal_citation_score = 1.0 - citation_metrics.hallucination_rate

            # Underpayment accuracy
            if ground_truth and "annual_underpayment" in ground_truth:
                calculated = clause_analysis.get("annual_underpayment", 0)
                expected = ground_truth["annual_underpayment"]
                if expected > 0:
                    result.underpayment_accuracy = 1.0 - min(1.0, abs(calculated - expected) / expected)
                else:
                    result.underpayment_accuracy = 1.0 if calculated == 0 else 0.0

        # LLM evaluation for overall quality
        if self.llm_evaluator and result.pipeline_result:
            try:
                llm_result = await self._evaluate_overall_quality(
                    contract_text,
                    result.pipeline_result,
                    result.baseline_result if self.run_baseline_comparison else None
                )
                result.pipeline_overall_score = llm_result.get("pipeline_score", 0.5)
                result.baseline_overall_score = llm_result.get("baseline_score", 0.5)
            except Exception as e:
                print(f"  LLM evaluation error: {e}")

        # Ground truth comparison
        if ground_truth:
            result.ground_truth_match_score = self._calculate_ground_truth_match(
                result.pipeline_result, ground_truth
            )

        return result

    async def _evaluate_overall_quality(
        self,
        contract_text: str,
        pipeline_result: Dict[str, Any],
        baseline_result: Optional[Dict[str, Any]] = None
    ) -> Dict[str, float]:
        """Use LLM to evaluate overall analysis quality"""

        pipeline_summary = json.dumps({
            "risk_level": pipeline_result.get("risk_level"),
            "violations": pipeline_result.get("clause_analysis", {}).get("violations", [])[:5],
            "summary": pipeline_result.get("analysis_summary", "")[:500]
        }, ensure_ascii=False)

        content = f"""[원본 계약서]
{contract_text[:2000]}...

[DocScanner 분석 결과]
{pipeline_summary}

위 분석 결과의 품질을 평가해주세요."""

        custom_criteria = {
            "analysis_accuracy": "분석이 정확하고 법적으로 타당한가?",
            "comprehensiveness": "모든 주요 위험 요소를 식별했는가?",
            "actionability": "분석 결과를 바탕으로 실제 조치를 취할 수 있는가?",
            "clarity": "분석 결과가 명확하고 이해하기 쉬운가?"
        }

        result = await self.llm_evaluator.evaluate_with_cross_validation(
            content=content,
            dimensions=[
                EvaluationDimension.ACCURACY,
                EvaluationDimension.COMPLETENESS,
                EvaluationDimension.COHERENCE
            ],
            custom_criteria=custom_criteria,
            context={"contract_text": contract_text[:3000]}
        )

        scores = {"pipeline_score": result.consensus_score}

        # Evaluate baseline if available
        if baseline_result and "error" not in baseline_result:
            baseline_content = f"""[원본 계약서]
{contract_text[:2000]}...

[순수 LLM 분석 결과]
{json.dumps(baseline_result, ensure_ascii=False)[:2000]}

위 분석 결과의 품질을 평가해주세요."""

            baseline_eval = await self.llm_evaluator.evaluate_with_cross_validation(
                content=baseline_content,
                dimensions=[EvaluationDimension.ACCURACY, EvaluationDimension.COMPLETENESS],
                custom_criteria=custom_criteria
            )
            scores["baseline_score"] = baseline_eval.consensus_score

        return scores

    def _calculate_ground_truth_match(
        self,
        pipeline_result: Dict[str, Any],
        ground_truth: Dict[str, Any]
    ) -> float:
        """Calculate how well pipeline result matches ground truth"""

        if not pipeline_result or "error" in pipeline_result:
            return 0.0

        scores = []

        # Risk level match
        pipeline_risk = pipeline_result.get("risk_level", "").lower()
        gt_risk = ground_truth.get("risk_level", "").lower()
        if pipeline_risk and gt_risk:
            scores.append(1.0 if pipeline_risk == gt_risk else 0.5 if pipeline_risk[0] == gt_risk[0] else 0.0)

        # Violation count similarity
        pipeline_violations = len(pipeline_result.get("clause_analysis", {}).get("violations", []))
        gt_violations = len(ground_truth.get("violations", []))
        if gt_violations > 0:
            ratio = min(pipeline_violations, gt_violations) / max(pipeline_violations, gt_violations)
            scores.append(ratio)

        # Underpayment similarity
        pipeline_underpay = pipeline_result.get("clause_analysis", {}).get("annual_underpayment", 0)
        gt_underpay = ground_truth.get("annual_underpayment", ground_truth.get("estimated_annual_underpayment", 0))
        if gt_underpay > 0:
            error_rate = abs(pipeline_underpay - gt_underpay) / gt_underpay
            scores.append(max(0, 1.0 - error_rate))
        elif pipeline_underpay == 0:
            scores.append(1.0)

        return np.mean(scores) if scores else 0.5

    async def evaluate_batch(
        self,
        test_cases: List[Dict[str, Any]]
    ) -> Tuple[EndToEndMetrics, List[PipelineEvalResult]]:
        """
        Evaluate multiple contracts

        Args:
            test_cases: List of {contract_text, contract_id, ground_truth?}
        """
        metrics = EndToEndMetrics()
        metrics.total_contracts = len(test_cases)

        results = []
        pipeline_scores = []
        baseline_scores = []
        clause_scores = []
        violation_scores = []
        citation_scores = []
        underpayment_scores = []
        latencies = []
        token_usages = []

        for i, case in enumerate(test_cases):
            contract_id = case.get("contract_id", f"contract_{i}")
            print(f"  Evaluating {i+1}/{len(test_cases)}: {contract_id}")

            result = await self.evaluate_single_contract(
                contract_text=case["contract_text"],
                contract_id=contract_id,
                ground_truth=case.get("ground_truth")
            )
            results.append(result)

            # Collect scores
            if result.pipeline_overall_score > 0:
                pipeline_scores.append(result.pipeline_overall_score)
            if result.baseline_overall_score > 0:
                baseline_scores.append(result.baseline_overall_score)
            if result.clause_extraction_score > 0:
                clause_scores.append(result.clause_extraction_score)
            if result.violation_detection_score > 0:
                violation_scores.append(result.violation_detection_score)
            if result.legal_citation_score > 0:
                citation_scores.append(result.legal_citation_score)
            if result.underpayment_accuracy > 0:
                underpayment_scores.append(result.underpayment_accuracy)

            latencies.append(result.pipeline_latency_ms)
            if result.pipeline_token_usage:
                token_usages.append(result.pipeline_token_usage)

        # Calculate aggregate metrics
        metrics.avg_clause_extraction_score = np.mean(clause_scores) if clause_scores else 0.0
        metrics.avg_violation_detection_f1 = np.mean(violation_scores) if violation_scores else 0.0
        metrics.avg_legal_citation_accuracy = np.mean(citation_scores) if citation_scores else 0.0
        metrics.avg_underpayment_mape = 1.0 - np.mean(underpayment_scores) if underpayment_scores else 0.0

        metrics.avg_pipeline_score = np.mean(pipeline_scores) if pipeline_scores else 0.0
        metrics.avg_baseline_score = np.mean(baseline_scores) if baseline_scores else 0.0

        if metrics.avg_baseline_score > 0:
            metrics.pipeline_vs_baseline_improvement = (
                (metrics.avg_pipeline_score - metrics.avg_baseline_score) / metrics.avg_baseline_score
            )

        # Statistical significance test
        if len(pipeline_scores) >= 3 and len(baseline_scores) >= 3:
            try:
                _, p_value = stats.ttest_rel(pipeline_scores[:len(baseline_scores)], baseline_scores)
                metrics.improvement_p_value = p_value

                # Cohen's d
                diff = np.array(pipeline_scores[:len(baseline_scores)]) - np.array(baseline_scores)
                pooled_std = np.sqrt((np.var(pipeline_scores) + np.var(baseline_scores)) / 2)
                if pooled_std > 0:
                    metrics.effect_size_cohens_d = np.mean(diff) / pooled_std
            except Exception:
                pass

        metrics.avg_pipeline_latency_ms = np.mean(latencies) if latencies else 0.0

        return metrics, results

    def save_results(
        self,
        metrics: EndToEndMetrics,
        results: List[PipelineEvalResult],
        prefix: str = "end_to_end"
    ):
        """Save evaluation results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        serializable = {
            "metrics": asdict(metrics),
            "summary": {
                "total_contracts": metrics.total_contracts,
                "pipeline_score": metrics.avg_pipeline_score,
                "baseline_score": metrics.avg_baseline_score,
                "improvement": metrics.pipeline_vs_baseline_improvement,
                "p_value": metrics.improvement_p_value,
                "effect_size": metrics.effect_size_cohens_d
            },
            "component_scores": {
                "clause_extraction": metrics.avg_clause_extraction_score,
                "violation_detection": metrics.avg_violation_detection_f1,
                "legal_citation": metrics.avg_legal_citation_accuracy,
                "underpayment": 1.0 - metrics.avg_underpayment_mape
            },
            "individual_results": [
                {
                    "contract_id": r.contract_id,
                    "pipeline_score": r.pipeline_overall_score,
                    "baseline_score": r.baseline_overall_score,
                    "ground_truth_match": r.ground_truth_match_score,
                    "latency_ms": r.pipeline_latency_ms
                }
                for r in results
            ]
        }

        json_file = self.output_dir / f"{prefix}_{timestamp}.json"
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(serializable, f, ensure_ascii=False, indent=2)

        report = self._generate_report(metrics, results)
        report_file = self.output_dir / f"{prefix}_report_{timestamp}.md"
        with open(report_file, "w", encoding="utf-8") as f:
            f.write(report)

        print(f"Results saved to {self.output_dir}")
        return json_file, report_file

    def _generate_report(
        self,
        metrics: EndToEndMetrics,
        results: List[PipelineEvalResult]
    ) -> str:
        """Generate comprehensive markdown report"""

        significance = "Yes" if metrics.improvement_p_value < 0.05 else "No"
        effect_interp = self._interpret_effect_size(metrics.effect_size_cohens_d)

        report = f"""# DocScanner End-to-End Evaluation Report

## Executive Summary

| Metric | Value |
|--------|-------|
| Total Contracts Evaluated | {metrics.total_contracts} |
| **Pipeline Overall Score** | **{metrics.avg_pipeline_score:.3f}** |
| Baseline Score | {metrics.avg_baseline_score:.3f} |
| **Improvement over Baseline** | **{metrics.pipeline_vs_baseline_improvement:+.1%}** |

## Statistical Significance

| Test | Value | Interpretation |
|------|-------|----------------|
| p-value | {metrics.improvement_p_value:.4f} | {significance} (p < 0.05) |
| Cohen's d | {metrics.effect_size_cohens_d:.3f} | {effect_interp} effect |

## Component Performance

| Component | Score |
|-----------|-------|
| Clause Extraction | {metrics.avg_clause_extraction_score:.3f} |
| Violation Detection (F1) | {metrics.avg_violation_detection_f1:.3f} |
| Legal Citation Accuracy | {metrics.avg_legal_citation_accuracy:.3f} |
| Underpayment Calculation | {1.0 - metrics.avg_underpayment_mape:.3f} |

## Efficiency

| Metric | Value |
|--------|-------|
| Average Latency | {metrics.avg_pipeline_latency_ms:.0f} ms |
| Baseline Latency | {metrics.avg_baseline_latency_ms:.0f} ms |

## Individual Contract Results

| Contract | Pipeline | Baseline | GT Match | Latency (ms) |
|----------|----------|----------|----------|--------------|
"""
        for r in results[:20]:  # Limit to 20
            report += f"| {r.contract_id} | {r.pipeline_overall_score:.3f} | {r.baseline_overall_score:.3f} | {r.ground_truth_match_score:.3f} | {r.pipeline_latency_ms:.0f} |\n"

        report += f"""
## Key Findings

### Strengths
{"- High overall analysis quality" if metrics.avg_pipeline_score > 0.7 else ""}
{"- Significant improvement over baseline LLM" if metrics.improvement_p_value < 0.05 else ""}
{"- Accurate legal citations" if metrics.avg_legal_citation_accuracy > 0.9 else ""}
{"- Reliable violation detection" if metrics.avg_violation_detection_f1 > 0.7 else ""}

### Areas for Improvement
{"- Legal citation accuracy needs work" if metrics.avg_legal_citation_accuracy < 0.8 else ""}
{"- Violation detection can be improved" if metrics.avg_violation_detection_f1 < 0.6 else ""}
{"- Underpayment calculation precision" if metrics.avg_underpayment_mape > 0.2 else ""}

## Conclusion

{"DocScanner demonstrates statistically significant improvement over pure LLM prompting." if metrics.improvement_p_value < 0.05 else "Further optimization may be needed to show clear advantage over baseline."}

The {effect_interp} effect size ({metrics.effect_size_cohens_d:.2f}) indicates {"substantial" if abs(metrics.effect_size_cohens_d) > 0.5 else "moderate" if abs(metrics.effect_size_cohens_d) > 0.2 else "minimal"} practical significance.

---
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
"""
        return report

    def _interpret_effect_size(self, d: float) -> str:
        """Interpret Cohen's d effect size"""
        d_abs = abs(d)
        if d_abs < 0.2:
            return "negligible"
        elif d_abs < 0.5:
            return "small"
        elif d_abs < 0.8:
            return "medium"
        else:
            return "large"
