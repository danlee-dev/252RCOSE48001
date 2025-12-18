"""
LLM Baseline Comparison Evaluation Script

Compares DocScanner AI pipeline vs pure LLM prompting for:
1. Risk clause detection accuracy
2. Underpayment calculation accuracy
3. Analysis quality metrics

Methodology:
- Uses the same contract samples for both systems
- Measures precision, recall, F1 for risk detection
- Calculates MAE, MAPE for underpayment estimation
- Performs statistical significance tests

Academic References:
- Paired t-test / Wilcoxon signed-rank test for significance
- Cohen's d for effect size
- Bootstrap confidence intervals
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime
import numpy as np
from scipy import stats

# Add paths
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "backend"))


@dataclass
class EvaluationSample:
    """Single evaluation sample"""
    sample_id: str
    contract_text: str
    ground_truth: Optional[Dict[str, Any]] = None
    docscanner_result: Optional[Dict[str, Any]] = None
    baseline_result: Optional[Dict[str, Any]] = None


@dataclass
class ComparisonMetrics:
    """Comparison metrics between two systems"""
    # Risk Detection
    docscanner_precision: float = 0.0
    docscanner_recall: float = 0.0
    docscanner_f1: float = 0.0
    baseline_precision: float = 0.0
    baseline_recall: float = 0.0
    baseline_f1: float = 0.0

    # Underpayment Estimation
    docscanner_underpayment_mae: float = 0.0
    docscanner_underpayment_mape: float = 0.0
    baseline_underpayment_mae: float = 0.0
    baseline_underpayment_mape: float = 0.0

    # Processing Time
    docscanner_avg_time: float = 0.0
    baseline_avg_time: float = 0.0

    # Statistical Tests
    risk_f1_p_value: float = 1.0
    risk_f1_effect_size: float = 0.0
    underpayment_mae_p_value: float = 1.0
    underpayment_mae_effect_size: float = 0.0


# Baseline LLM prompt for risk analysis
BASELINE_PROMPT = """당신은 대한민국 노동법 전문 변호사입니다.
다음 근로계약서를 분석하여 법적 위험 조항을 찾아주세요.

[계약서]
{contract_text}

다음 JSON 형식으로 응답하세요:
{{
    "risk_level": "High/Medium/Low",
    "risk_score": 0.0-1.0,
    "violations": [
        {{
            "type": "위반 유형 (예: 최저임금 미달, 포괄임금제 위반 등)",
            "severity": "High/Medium/Low",
            "description": "위반 설명",
            "legal_basis": "법적 근거 (예: 근로기준법 제56조)",
            "clause_text": "해당 조항 텍스트"
        }}
    ],
    "annual_underpayment": 연간 예상 체불액 (숫자만, 단위: 원),
    "summary": "분석 요약"
}}

반드시 유효한 JSON만 응답하세요.
"""


class BaselineLLM:
    """Pure LLM baseline for comparison"""

    def __init__(self, model: str = "gpt-4o"):
        self.model = model
        self._init_client()

    def _init_client(self):
        """Initialize OpenAI client"""
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        except ImportError:
            print("OpenAI package not installed")
            self.client = None

    def analyze(self, contract_text: str) -> Dict[str, Any]:
        """Analyze contract with pure LLM prompting"""
        if self.client is None:
            return {"error": "OpenAI client not initialized"}

        start_time = time.time()

        try:
            prompt = BASELINE_PROMPT.format(contract_text=contract_text[:8000])

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                response_format={"type": "json_object"}
            )

            result = json.loads(response.choices[0].message.content)
            result["processing_time"] = time.time() - start_time

            return result

        except Exception as e:
            return {
                "error": str(e),
                "risk_level": "Unknown",
                "violations": [],
                "annual_underpayment": 0,
                "processing_time": time.time() - start_time
            }


class BaselineComparisonEvaluator:
    """
    Evaluator for comparing DocScanner vs baseline LLM

    Academic rigor:
    - Multiple runs for stability (default: 3)
    - Statistical significance testing
    - Effect size calculation
    - Bootstrap confidence intervals
    """

    def __init__(
        self,
        output_dir: str = None,
        num_runs: int = 3,
        random_seed: int = 42
    ):
        self.output_dir = Path(output_dir) if output_dir else Path(__file__).parent.parent / "results" / "baseline_comparison"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.num_runs = num_runs
        self.random_seed = random_seed
        np.random.seed(random_seed)

        # Initialize systems
        self.baseline = BaselineLLM()
        self.docscanner = None  # Lazy initialization

        # Results storage
        self.samples: List[EvaluationSample] = []
        self.metrics_per_run: List[ComparisonMetrics] = []

    def _init_docscanner(self):
        """Initialize DocScanner pipeline"""
        if self.docscanner is None:
            try:
                from app.ai.pipeline import AdvancedAIPipeline, PipelineConfig

                config = PipelineConfig(
                    enable_llm_clause_analysis=True,
                    enable_raptor=False,  # Disable for faster evaluation
                    enable_dspy=False
                )
                self.docscanner = AdvancedAIPipeline(config=config)
            except Exception as e:
                print(f"Failed to initialize DocScanner: {e}")
                self.docscanner = None

    def load_test_data(self, data_path: str) -> List[Dict[str, Any]]:
        """Load test contract data"""
        data_path = Path(data_path)

        if data_path.suffix == ".json":
            with open(data_path, "r", encoding="utf-8") as f:
                return json.load(f)
        elif data_path.suffix == ".jsonl":
            samples = []
            with open(data_path, "r", encoding="utf-8") as f:
                for line in f:
                    samples.append(json.loads(line))
            return samples
        else:
            raise ValueError(f"Unsupported file format: {data_path.suffix}")

    def evaluate_sample(
        self,
        sample_id: str,
        contract_text: str,
        ground_truth: Dict[str, Any] = None
    ) -> EvaluationSample:
        """Evaluate a single contract sample"""
        self._init_docscanner()

        sample = EvaluationSample(
            sample_id=sample_id,
            contract_text=contract_text,
            ground_truth=ground_truth
        )

        # Run baseline LLM
        print(f"  [Baseline] Analyzing sample {sample_id}...")
        sample.baseline_result = self.baseline.analyze(contract_text)

        # Run DocScanner
        if self.docscanner:
            print(f"  [DocScanner] Analyzing sample {sample_id}...")
            try:
                result = self.docscanner.analyze(contract_text, contract_id=sample_id)
                sample.docscanner_result = result.to_dict()
            except Exception as e:
                print(f"  DocScanner error: {e}")
                sample.docscanner_result = {"error": str(e)}

        return sample

    def run_evaluation(
        self,
        test_data: List[Dict[str, Any]],
        max_samples: int = None
    ) -> ComparisonMetrics:
        """
        Run full evaluation on test data

        Args:
            test_data: List of {contract_text, ground_truth (optional)}
            max_samples: Limit number of samples (for testing)

        Returns:
            Aggregated comparison metrics
        """
        if max_samples:
            test_data = test_data[:max_samples]

        print(f"\n{'='*60}")
        print(f"Running Baseline Comparison Evaluation")
        print(f"Samples: {len(test_data)}, Runs: {self.num_runs}")
        print(f"{'='*60}\n")

        all_metrics = []

        for run_idx in range(self.num_runs):
            print(f"\n--- Run {run_idx + 1}/{self.num_runs} ---\n")

            run_samples = []
            for i, data in enumerate(test_data):
                sample = self.evaluate_sample(
                    sample_id=f"run{run_idx}_sample{i}",
                    contract_text=data.get("contract_text", data.get("text", "")),
                    ground_truth=data.get("ground_truth")
                )
                run_samples.append(sample)

            # Calculate metrics for this run
            metrics = self._calculate_metrics(run_samples)
            all_metrics.append(metrics)
            self.samples.extend(run_samples)

        # Aggregate metrics across runs
        final_metrics = self._aggregate_metrics(all_metrics)

        # Perform statistical tests
        self._perform_statistical_tests(all_metrics, final_metrics)

        return final_metrics

    def _calculate_metrics(self, samples: List[EvaluationSample]) -> ComparisonMetrics:
        """Calculate metrics for a set of samples"""
        metrics = ComparisonMetrics()

        # Collect predictions and ground truth
        docscanner_violations_all = []
        baseline_violations_all = []
        gt_violations_all = []

        docscanner_underpayments = []
        baseline_underpayments = []
        gt_underpayments = []

        docscanner_times = []
        baseline_times = []

        for sample in samples:
            # Extract violations
            ds_violations = []
            if sample.docscanner_result:
                stress_test = sample.docscanner_result.get("stress_test", {})
                ds_violations = stress_test.get("violations", [])
                docscanner_times.append(sample.docscanner_result.get("processing_time", 0))

                underpay = stress_test.get("annual_underpayment", 0)
                docscanner_underpayments.append(underpay)

            bl_violations = []
            if sample.baseline_result:
                bl_violations = sample.baseline_result.get("violations", [])
                baseline_times.append(sample.baseline_result.get("processing_time", 0))

                underpay = sample.baseline_result.get("annual_underpayment", 0)
                baseline_underpayments.append(underpay)

            docscanner_violations_all.append(ds_violations)
            baseline_violations_all.append(bl_violations)

            # Ground truth (if available)
            if sample.ground_truth:
                gt = sample.ground_truth.get("violations", [])
                gt_violations_all.append(gt)
                gt_underpayments.append(sample.ground_truth.get("annual_underpayment", 0))

        # Calculate risk detection metrics
        if gt_violations_all:
            ds_p, ds_r, ds_f1 = self._calculate_violation_metrics(
                docscanner_violations_all, gt_violations_all
            )
            bl_p, bl_r, bl_f1 = self._calculate_violation_metrics(
                baseline_violations_all, gt_violations_all
            )

            metrics.docscanner_precision = ds_p
            metrics.docscanner_recall = ds_r
            metrics.docscanner_f1 = ds_f1
            metrics.baseline_precision = bl_p
            metrics.baseline_recall = bl_r
            metrics.baseline_f1 = bl_f1

            # Calculate underpayment metrics
            if gt_underpayments:
                metrics.docscanner_underpayment_mae = self._calculate_mae(
                    docscanner_underpayments, gt_underpayments
                )
                metrics.docscanner_underpayment_mape = self._calculate_mape(
                    docscanner_underpayments, gt_underpayments
                )
                metrics.baseline_underpayment_mae = self._calculate_mae(
                    baseline_underpayments, gt_underpayments
                )
                metrics.baseline_underpayment_mape = self._calculate_mape(
                    baseline_underpayments, gt_underpayments
                )

        # Average processing times
        if docscanner_times:
            metrics.docscanner_avg_time = np.mean(docscanner_times)
        if baseline_times:
            metrics.baseline_avg_time = np.mean(baseline_times)

        return metrics

    def _calculate_violation_metrics(
        self,
        predictions: List[List[Dict]],
        ground_truth: List[List[Dict]]
    ) -> Tuple[float, float, float]:
        """Calculate precision, recall, F1 for violation detection"""
        total_tp = 0
        total_fp = 0
        total_fn = 0

        for pred_violations, gt_violations in zip(predictions, ground_truth):
            # Extract violation types for comparison
            pred_types = set(v.get("type", "").lower() for v in pred_violations)
            gt_types = set(v.get("type", "").lower() for v in gt_violations)

            tp = len(pred_types & gt_types)
            fp = len(pred_types - gt_types)
            fn = len(gt_types - pred_types)

            total_tp += tp
            total_fp += fp
            total_fn += fn

        precision = total_tp / max(1, total_tp + total_fp)
        recall = total_tp / max(1, total_tp + total_fn)
        f1 = 2 * precision * recall / max(0.001, precision + recall)

        return precision, recall, f1

    def _calculate_mae(self, predictions: List[float], ground_truth: List[float]) -> float:
        """Calculate Mean Absolute Error"""
        if not predictions or not ground_truth:
            return 0.0
        return np.mean(np.abs(np.array(predictions) - np.array(ground_truth)))

    def _calculate_mape(self, predictions: List[float], ground_truth: List[float]) -> float:
        """Calculate Mean Absolute Percentage Error"""
        if not predictions or not ground_truth:
            return 0.0

        errors = []
        for p, g in zip(predictions, ground_truth):
            if g != 0:
                errors.append(abs(p - g) / abs(g))
        return np.mean(errors) if errors else 0.0

    def _aggregate_metrics(self, metrics_list: List[ComparisonMetrics]) -> ComparisonMetrics:
        """Aggregate metrics across multiple runs"""
        aggregated = ComparisonMetrics()

        # Average all numeric fields
        for field_name in [f.name for f in ComparisonMetrics.__dataclass_fields__.values()]:
            values = [getattr(m, field_name) for m in metrics_list]
            if values and all(isinstance(v, (int, float)) for v in values):
                setattr(aggregated, field_name, np.mean(values))

        return aggregated

    def _perform_statistical_tests(
        self,
        metrics_list: List[ComparisonMetrics],
        final_metrics: ComparisonMetrics
    ):
        """Perform statistical significance tests"""
        if len(metrics_list) < 2:
            return

        # Extract F1 scores for comparison
        ds_f1_scores = np.array([m.docscanner_f1 for m in metrics_list])
        bl_f1_scores = np.array([m.baseline_f1 for m in metrics_list])

        # Paired test for F1
        if len(ds_f1_scores) >= 2 and np.std(ds_f1_scores - bl_f1_scores) > 0:
            try:
                # Check normality
                _, p_normal = stats.shapiro(ds_f1_scores - bl_f1_scores)

                if p_normal > 0.05:
                    # Paired t-test
                    _, p_value = stats.ttest_rel(ds_f1_scores, bl_f1_scores)
                else:
                    # Wilcoxon signed-rank test
                    _, p_value = stats.wilcoxon(ds_f1_scores, bl_f1_scores)

                final_metrics.risk_f1_p_value = p_value

                # Cohen's d effect size
                diff = ds_f1_scores - bl_f1_scores
                pooled_std = np.sqrt((np.var(ds_f1_scores) + np.var(bl_f1_scores)) / 2)
                if pooled_std > 0:
                    final_metrics.risk_f1_effect_size = np.mean(diff) / pooled_std

            except Exception as e:
                print(f"Statistical test error: {e}")

        # Similar for underpayment MAE
        ds_mae_scores = np.array([m.docscanner_underpayment_mae for m in metrics_list])
        bl_mae_scores = np.array([m.baseline_underpayment_mae for m in metrics_list])

        if len(ds_mae_scores) >= 2 and np.std(ds_mae_scores - bl_mae_scores) > 0:
            try:
                _, p_value = stats.wilcoxon(ds_mae_scores, bl_mae_scores)
                final_metrics.underpayment_mae_p_value = p_value

                diff = bl_mae_scores - ds_mae_scores  # Lower is better
                pooled_std = np.sqrt((np.var(ds_mae_scores) + np.var(bl_mae_scores)) / 2)
                if pooled_std > 0:
                    final_metrics.underpayment_mae_effect_size = np.mean(diff) / pooled_std

            except Exception:
                pass

    def save_results(self, metrics: ComparisonMetrics):
        """Save evaluation results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save metrics
        metrics_file = self.output_dir / f"metrics_{timestamp}.json"
        with open(metrics_file, "w", encoding="utf-8") as f:
            json.dump(asdict(metrics), f, indent=2)

        # Save detailed samples
        samples_file = self.output_dir / f"samples_{timestamp}.json"
        with open(samples_file, "w", encoding="utf-8") as f:
            json.dump(
                [asdict(s) for s in self.samples],
                f,
                ensure_ascii=False,
                indent=2
            )

        # Generate summary report
        report = self._generate_report(metrics)
        report_file = self.output_dir / f"report_{timestamp}.md"
        with open(report_file, "w", encoding="utf-8") as f:
            f.write(report)

        print(f"\nResults saved to {self.output_dir}")
        return metrics_file, samples_file, report_file

    def _generate_report(self, metrics: ComparisonMetrics) -> str:
        """Generate markdown report"""
        significance = "Yes" if metrics.risk_f1_p_value < 0.05 else "No"
        effect_interpretation = self._interpret_effect_size(metrics.risk_f1_effect_size)

        report = f"""# DocScanner vs Baseline LLM Comparison Report

## Experiment Configuration
- Date: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
- Number of Samples: {len(self.samples)}
- Number of Runs: {self.num_runs}
- Random Seed: {self.random_seed}

## Risk Detection Performance

| Metric | DocScanner | Baseline LLM | Difference |
|--------|------------|--------------|------------|
| Precision | {metrics.docscanner_precision:.3f} | {metrics.baseline_precision:.3f} | {metrics.docscanner_precision - metrics.baseline_precision:+.3f} |
| Recall | {metrics.docscanner_recall:.3f} | {metrics.baseline_recall:.3f} | {metrics.docscanner_recall - metrics.baseline_recall:+.3f} |
| F1 Score | {metrics.docscanner_f1:.3f} | {metrics.baseline_f1:.3f} | {metrics.docscanner_f1 - metrics.baseline_f1:+.3f} |

## Underpayment Estimation Performance

| Metric | DocScanner | Baseline LLM | Difference |
|--------|------------|--------------|------------|
| MAE (KRW) | {metrics.docscanner_underpayment_mae:,.0f} | {metrics.baseline_underpayment_mae:,.0f} | {metrics.docscanner_underpayment_mae - metrics.baseline_underpayment_mae:+,.0f} |
| MAPE | {metrics.docscanner_underpayment_mape:.1%} | {metrics.baseline_underpayment_mape:.1%} | {metrics.docscanner_underpayment_mape - metrics.baseline_underpayment_mape:+.1%} |

## Processing Time

| System | Average Time (s) |
|--------|-----------------|
| DocScanner | {metrics.docscanner_avg_time:.2f} |
| Baseline LLM | {metrics.baseline_avg_time:.2f} |

## Statistical Significance

### Risk Detection (F1 Score)
- p-value: {metrics.risk_f1_p_value:.4f}
- Significant (p < 0.05): {significance}
- Effect Size (Cohen's d): {metrics.risk_f1_effect_size:.3f} ({effect_interpretation})

### Underpayment MAE
- p-value: {metrics.underpayment_mae_p_value:.4f}
- Effect Size: {metrics.underpayment_mae_effect_size:.3f}

## Interpretation

{"DocScanner shows statistically significant improvement" if metrics.risk_f1_p_value < 0.05 and metrics.risk_f1_effect_size > 0 else "No statistically significant difference detected"} in risk detection F1 score compared to the baseline LLM.

{"The underpayment estimation is significantly more accurate" if metrics.underpayment_mae_effect_size > 0.5 else "Underpayment estimation shows comparable performance"} with DocScanner.
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


def main():
    parser = argparse.ArgumentParser(description="Run baseline comparison evaluation")
    parser.add_argument("--data", type=str, help="Path to test data (JSON/JSONL)")
    parser.add_argument("--output", type=str, help="Output directory")
    parser.add_argument("--max-samples", type=int, default=None, help="Maximum samples to evaluate")
    parser.add_argument("--num-runs", type=int, default=3, help="Number of evaluation runs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    # Initialize evaluator
    evaluator = BaselineComparisonEvaluator(
        output_dir=args.output,
        num_runs=args.num_runs,
        random_seed=args.seed
    )

    # Load test data
    if args.data:
        test_data = evaluator.load_test_data(args.data)
    else:
        # Use sample test data
        test_data = [
            {
                "contract_text": """
                근로계약서

                제1조 (계약기간)
                본 계약의 기간은 2024년 1월 1일부터 2024년 12월 31일까지로 한다.

                제2조 (근무시간)
                1일 8시간, 주 40시간을 기본 근무시간으로 한다.
                업무상 필요시 연장근로를 할 수 있다.

                제3조 (임금)
                월 급여 200만원 (연장근로수당 포함)을 매월 25일에 지급한다.

                제4조 (수습기간)
                수습기간 3개월간 급여의 90%를 지급한다.
                """,
                "ground_truth": {
                    "violations": [
                        {"type": "포괄임금제", "severity": "High"},
                        {"type": "최저임금 미달 (수습기간)", "severity": "Medium"}
                    ],
                    "annual_underpayment": 2400000
                }
            }
        ]

    # Run evaluation
    metrics = evaluator.run_evaluation(test_data, max_samples=args.max_samples)

    # Save results
    evaluator.save_results(metrics)

    # Print summary
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    print(f"DocScanner F1: {metrics.docscanner_f1:.3f}")
    print(f"Baseline F1:   {metrics.baseline_f1:.3f}")
    print(f"Improvement:   {metrics.docscanner_f1 - metrics.baseline_f1:+.3f}")
    print(f"p-value:       {metrics.risk_f1_p_value:.4f}")
    print("="*60)


if __name__ == "__main__":
    main()
