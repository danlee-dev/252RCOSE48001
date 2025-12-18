"""
Violation Detection Evaluator

Evaluates the quality of labor law violation detection:
1. Detection Accuracy: Precision, Recall, F1 vs ground truth
2. Severity Classification: Is severity correctly assigned?
3. Legal Basis Quality: Are cited laws correct and applicable?
4. Description Quality: Is the violation clearly explained?
5. Suggestion Actionability: Are fix suggestions practical?

Uses LLM-as-Judge with cross-validation for subjective quality metrics.

Academic References:
- MT-Bench (Zheng et al., 2023): Multi-turn benchmark
- RAGAS (Es et al., 2023): RAG evaluation
"""

import os
import sys
import json
import asyncio
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Set
from dataclasses import dataclass, field, asdict
from datetime import datetime
from collections import defaultdict
import numpy as np

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "backend"))

from evaluation.core.llm_evaluators import (
    MultiLLMEvaluator,
    EvaluationDimension,
    CrossValidationResult,
    EvaluationResult
)


@dataclass
class ViolationMatch:
    """Match between detected and ground truth violation"""
    detected: Dict[str, Any]
    ground_truth: Dict[str, Any]
    type_match: bool
    severity_match: bool
    similarity_score: float


@dataclass
class ViolationDetectionMetrics:
    """Metrics for violation detection evaluation"""
    # Detection metrics
    total_ground_truth: int = 0
    total_detected: int = 0
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0

    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0

    # Severity metrics
    severity_correct: int = 0
    severity_total: int = 0
    severity_accuracy: float = 0.0

    # Per-violation-type breakdown
    by_type: Dict[str, Dict[str, float]] = field(default_factory=dict)

    # Per-severity breakdown
    by_severity: Dict[str, Dict[str, int]] = field(default_factory=dict)

    # LLM-as-Judge scores (0-1)
    legal_basis_score: float = 0.0
    description_score: float = 0.0
    suggestion_score: float = 0.0
    overall_quality_score: float = 0.0


@dataclass
class ViolationDetectionResult:
    """Result of evaluating violation detection for a single contract"""
    contract_id: str
    metrics: ViolationDetectionMetrics
    matched_violations: List[ViolationMatch]
    false_positives: List[Dict[str, Any]]
    false_negatives: List[Dict[str, Any]]
    llm_evaluation: Optional[CrossValidationResult] = None


class ViolationDetectionEvaluator:
    """
    Evaluator for violation detection quality

    Combines:
    1. Rule-based matching against ground truth
    2. LLM-as-Judge for quality assessment
    3. Cross-validation between multiple LLMs
    """

    # Canonical violation types for normalization
    VIOLATION_TYPE_MAPPING = {
        # Minimum wage related
        "최저임금": "minimum_wage",
        "최저임금 미달": "minimum_wage",
        "최저시급 미달": "minimum_wage",

        # Inclusive wage
        "포괄임금": "inclusive_wage",
        "포괄임금제": "inclusive_wage",
        "포괄임금제 위반": "inclusive_wage",

        # Working hours
        "근로시간": "working_hours",
        "법정근로시간 초과": "working_hours",
        "주52시간": "working_hours",
        "주52시간제 위반": "working_hours",
        "연장근로": "overtime",

        # Overtime pay
        "연장근로수당": "overtime_pay",
        "야간근로수당": "night_pay",
        "휴일근로수당": "holiday_pay",

        # Weekly holiday
        "주휴수당": "weekly_holiday_pay",
        "주휴수당 미지급": "weekly_holiday_pay",

        # Annual leave
        "연차": "annual_leave",
        "연차휴가": "annual_leave",
        "연차유급휴가": "annual_leave",

        # Penalty clause
        "위약금": "penalty",
        "위약금 예정": "penalty",
        "위약금 예정 금지 위반": "penalty",

        # Unfair dismissal
        "부당해고": "unfair_dismissal",
        "해고예고": "dismissal_notice",

        # Social insurance
        "4대보험": "social_insurance",
        "사회보험": "social_insurance",

        # Contract delivery
        "근로계약서 미교부": "contract_delivery",
        "계약서 미교부": "contract_delivery",

        # Wage payment
        "임금 전액 지급": "full_wage_payment",
        "임금 공제": "wage_deduction",

        # Break time
        "휴게시간": "break_time",

        # Other
        "기타": "other"
    }

    SEVERITY_LEVELS = ["critical", "high", "medium", "low"]

    def __init__(
        self,
        output_dir: str = None,
        use_llm_evaluation: bool = True,
        type_match_threshold: float = 0.7
    ):
        self.output_dir = Path(output_dir) if output_dir else Path(__file__).parent.parent / "results" / "violation_detection"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.use_llm_evaluation = use_llm_evaluation
        self.type_match_threshold = type_match_threshold

        if use_llm_evaluation:
            try:
                self.llm_evaluator = MultiLLMEvaluator(
                    use_openai=True,
                    use_gemini=True,
                    use_claude=bool(os.getenv("ANTHROPIC_API_KEY"))
                )
            except ValueError:
                print("[Warning] LLM evaluator not available")
                self.llm_evaluator = None
        else:
            self.llm_evaluator = None

    def _normalize_type(self, violation_type: str) -> str:
        """Normalize violation type to canonical form"""
        type_lower = violation_type.lower().strip()

        # Direct mapping
        for korean, canonical in self.VIOLATION_TYPE_MAPPING.items():
            if korean in type_lower or type_lower in korean:
                return canonical

        # Keyword matching
        if "최저" in type_lower and "임금" in type_lower:
            return "minimum_wage"
        if "포괄" in type_lower:
            return "inclusive_wage"
        if "위약" in type_lower:
            return "penalty"
        if "연장" in type_lower and "수당" in type_lower:
            return "overtime_pay"
        if "주휴" in type_lower:
            return "weekly_holiday_pay"
        if "연차" in type_lower:
            return "annual_leave"
        if "해고" in type_lower:
            return "unfair_dismissal"
        if "보험" in type_lower:
            return "social_insurance"

        return "other"

    def _normalize_severity(self, severity: str) -> str:
        """Normalize severity to standard levels"""
        severity_lower = severity.lower().strip()

        if severity_lower in ["critical", "치명적", "매우 높음"]:
            return "critical"
        elif severity_lower in ["high", "높음", "심각"]:
            return "high"
        elif severity_lower in ["medium", "중간", "보통"]:
            return "medium"
        elif severity_lower in ["low", "낮음", "경미"]:
            return "low"
        else:
            return "medium"  # Default

    def _match_violations(
        self,
        detected: List[Dict[str, Any]],
        ground_truth: List[Dict[str, Any]]
    ) -> Tuple[List[ViolationMatch], List[Dict], List[Dict]]:
        """
        Match detected violations to ground truth

        Returns:
            (matched, false_positives, false_negatives)
        """
        matched = []
        matched_gt_indices = set()
        matched_det_indices = set()

        # Normalize all violations
        det_normalized = [
            (i, self._normalize_type(v.get("type", v.get("violation_type", ""))))
            for i, v in enumerate(detected)
        ]
        gt_normalized = [
            (i, self._normalize_type(v.get("type", v.get("violation_type", ""))))
            for i, v in enumerate(ground_truth)
        ]

        # First pass: exact type matches
        for det_idx, det_type in det_normalized:
            for gt_idx, gt_type in gt_normalized:
                if gt_idx in matched_gt_indices:
                    continue

                if det_type == gt_type:
                    det_v = detected[det_idx]
                    gt_v = ground_truth[gt_idx]

                    det_severity = self._normalize_severity(det_v.get("severity", "medium"))
                    gt_severity = self._normalize_severity(gt_v.get("severity", "medium"))

                    matched.append(ViolationMatch(
                        detected=det_v,
                        ground_truth=gt_v,
                        type_match=True,
                        severity_match=(det_severity == gt_severity),
                        similarity_score=1.0
                    ))

                    matched_gt_indices.add(gt_idx)
                    matched_det_indices.add(det_idx)
                    break

        # Collect unmatched
        false_positives = [detected[i] for i in range(len(detected)) if i not in matched_det_indices]
        false_negatives = [ground_truth[i] for i in range(len(ground_truth)) if i not in matched_gt_indices]

        return matched, false_positives, false_negatives

    def evaluate_detection(
        self,
        detected_violations: List[Dict[str, Any]],
        ground_truth_violations: List[Dict[str, Any]],
        contract_id: str = "unknown"
    ) -> ViolationDetectionResult:
        """
        Evaluate violation detection quality

        Args:
            detected_violations: List of detected violation objects
            ground_truth_violations: List of ground truth violations
            contract_id: Identifier for the contract
        """
        metrics = ViolationDetectionMetrics()

        metrics.total_ground_truth = len(ground_truth_violations)
        metrics.total_detected = len(detected_violations)

        # Match violations
        matched, fps, fns = self._match_violations(detected_violations, ground_truth_violations)

        metrics.true_positives = len(matched)
        metrics.false_positives = len(fps)
        metrics.false_negatives = len(fns)

        # Calculate precision, recall, F1
        if metrics.true_positives + metrics.false_positives > 0:
            metrics.precision = metrics.true_positives / (metrics.true_positives + metrics.false_positives)

        if metrics.true_positives + metrics.false_negatives > 0:
            metrics.recall = metrics.true_positives / (metrics.true_positives + metrics.false_negatives)

        if metrics.precision + metrics.recall > 0:
            metrics.f1_score = 2 * metrics.precision * metrics.recall / (metrics.precision + metrics.recall)

        # Severity accuracy
        metrics.severity_total = len(matched)
        metrics.severity_correct = sum(1 for m in matched if m.severity_match)
        if metrics.severity_total > 0:
            metrics.severity_accuracy = metrics.severity_correct / metrics.severity_total

        # Per-type breakdown
        type_counts = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})

        for m in matched:
            vtype = self._normalize_type(m.ground_truth.get("type", ""))
            type_counts[vtype]["tp"] += 1

        for fp in fps:
            vtype = self._normalize_type(fp.get("type", ""))
            type_counts[vtype]["fp"] += 1

        for fn in fns:
            vtype = self._normalize_type(fn.get("type", ""))
            type_counts[vtype]["fn"] += 1

        for vtype, counts in type_counts.items():
            tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
            p = tp / (tp + fp) if (tp + fp) > 0 else 0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
            metrics.by_type[vtype] = {"precision": p, "recall": r, "f1": f1}

        # Per-severity breakdown
        for severity in self.SEVERITY_LEVELS:
            metrics.by_severity[severity] = {"detected": 0, "ground_truth": 0, "matched": 0}

        for v in detected_violations:
            sev = self._normalize_severity(v.get("severity", "medium"))
            metrics.by_severity[sev]["detected"] += 1

        for v in ground_truth_violations:
            sev = self._normalize_severity(v.get("severity", "medium"))
            metrics.by_severity[sev]["ground_truth"] += 1

        for m in matched:
            sev = self._normalize_severity(m.ground_truth.get("severity", "medium"))
            metrics.by_severity[sev]["matched"] += 1

        return ViolationDetectionResult(
            contract_id=contract_id,
            metrics=metrics,
            matched_violations=matched,
            false_positives=fps,
            false_negatives=fns
        )

    async def evaluate_quality_with_llm(
        self,
        detected_violations: List[Dict[str, Any]],
        contract_text: str,
        contract_id: str = "unknown"
    ) -> Tuple[CrossValidationResult, Dict[str, float]]:
        """
        Use LLM-as-Judge to evaluate violation quality

        Evaluates:
        1. Legal basis correctness
        2. Description clarity
        3. Suggestion practicality
        """
        if not self.llm_evaluator:
            return None, {}

        violations_text = json.dumps(detected_violations, ensure_ascii=False, indent=2)

        content = f"""[탐지된 위반 사항]
{violations_text[:6000]}

위 위반 사항들의 품질을 평가해주세요. 특히:
1. 법적 근거(legal_basis)가 실제로 존재하고 적절하게 인용되었는가?
2. 위반 설명(description)이 명확하고 이해하기 쉬운가?
3. 수정 제안(suggestion)이 실행 가능하고 구체적인가?"""

        custom_criteria = {
            "legal_basis_accuracy": "인용된 법조항이 실제로 존재하고 위반 내용과 관련이 있는가?",
            "description_clarity": "위반 내용이 비전문가도 이해할 수 있게 설명되어 있는가?",
            "suggestion_actionability": "제안된 수정 방법이 구체적이고 실행 가능한가?",
            "severity_appropriateness": "위반의 심각도가 적절하게 분류되어 있는가?"
        }

        result = await self.llm_evaluator.evaluate_with_cross_validation(
            content=content,
            dimensions=[
                EvaluationDimension.ACCURACY,
                EvaluationDimension.LEGAL_BASIS,
                EvaluationDimension.COHERENCE
            ],
            custom_criteria=custom_criteria,
            context={"contract_text": contract_text[:3000]}
        )

        # Extract specific scores from LLM evaluation
        quality_scores = {}
        if result and result.individual_results:
            for eval_result in result.individual_results:
                for dim, score_obj in eval_result.scores.items():
                    if dim not in quality_scores:
                        quality_scores[dim] = []
                    quality_scores[dim].append(score_obj.score)

            # Average across evaluators
            quality_scores = {k: np.mean(v) for k, v in quality_scores.items()}

        return result, quality_scores

    async def evaluate_batch(
        self,
        test_cases: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Evaluate multiple contracts

        Args:
            test_cases: List of {contract_text, detected_violations, ground_truth_violations, contract_id}
        """
        results = []
        aggregated = ViolationDetectionMetrics()
        all_quality_scores = defaultdict(list)

        for i, case in enumerate(test_cases):
            contract_id = case.get("contract_id", f"contract_{i}")
            print(f"  Evaluating {i+1}/{len(test_cases)}: {contract_id}")

            # Rule-based evaluation
            result = self.evaluate_detection(
                detected_violations=case["detected_violations"],
                ground_truth_violations=case["ground_truth_violations"],
                contract_id=contract_id
            )

            # LLM quality evaluation
            if self.llm_evaluator and self.use_llm_evaluation:
                try:
                    llm_result, quality_scores = await self.evaluate_quality_with_llm(
                        case["detected_violations"],
                        case.get("contract_text", ""),
                        contract_id
                    )
                    result.llm_evaluation = llm_result

                    # Update metrics with LLM scores
                    for dim, score in quality_scores.items():
                        all_quality_scores[dim].append(score)

                except Exception as e:
                    print(f"    LLM evaluation failed: {e}")

            results.append(result)

            # Aggregate metrics
            m = result.metrics
            aggregated.total_ground_truth += m.total_ground_truth
            aggregated.total_detected += m.total_detected
            aggregated.true_positives += m.true_positives
            aggregated.false_positives += m.false_positives
            aggregated.false_negatives += m.false_negatives
            aggregated.severity_correct += m.severity_correct
            aggregated.severity_total += m.severity_total

            # Aggregate per-type
            for vtype, scores in m.by_type.items():
                if vtype not in aggregated.by_type:
                    aggregated.by_type[vtype] = {"precision": [], "recall": [], "f1": []}
                aggregated.by_type[vtype]["precision"].append(scores["precision"])
                aggregated.by_type[vtype]["recall"].append(scores["recall"])
                aggregated.by_type[vtype]["f1"].append(scores["f1"])

        # Calculate aggregated metrics
        if aggregated.true_positives + aggregated.false_positives > 0:
            aggregated.precision = aggregated.true_positives / (aggregated.true_positives + aggregated.false_positives)

        if aggregated.true_positives + aggregated.false_negatives > 0:
            aggregated.recall = aggregated.true_positives / (aggregated.true_positives + aggregated.false_negatives)

        if aggregated.precision + aggregated.recall > 0:
            aggregated.f1_score = 2 * aggregated.precision * aggregated.recall / (aggregated.precision + aggregated.recall)

        if aggregated.severity_total > 0:
            aggregated.severity_accuracy = aggregated.severity_correct / aggregated.severity_total

        # Average per-type metrics
        for vtype in aggregated.by_type:
            aggregated.by_type[vtype] = {
                k: np.mean(v) if v else 0.0
                for k, v in aggregated.by_type[vtype].items()
            }

        # Average LLM quality scores
        if all_quality_scores:
            aggregated.legal_basis_score = np.mean(all_quality_scores.get("legal_basis", [0]))
            aggregated.description_score = np.mean(all_quality_scores.get("accuracy", [0]))
            aggregated.suggestion_score = np.mean(all_quality_scores.get("coherence", [0]))
            aggregated.overall_quality_score = np.mean([
                aggregated.legal_basis_score,
                aggregated.description_score,
                aggregated.suggestion_score
            ])

        return {
            "individual_results": results,
            "aggregated_metrics": aggregated,
            "quality_scores": {k: np.mean(v) for k, v in all_quality_scores.items()},
            "summary": self._generate_summary(aggregated, results)
        }

    def _generate_summary(
        self,
        metrics: ViolationDetectionMetrics,
        results: List[ViolationDetectionResult]
    ) -> Dict[str, Any]:
        """Generate evaluation summary"""

        # Collect all FPs and FNs
        all_fps = []
        all_fns = []
        for r in results:
            all_fps.extend(r.false_positives)
            all_fns.extend(r.false_negatives)

        # Most common FP types
        fp_types = defaultdict(int)
        for fp in all_fps:
            fp_types[self._normalize_type(fp.get("type", ""))] += 1

        # Most common FN types
        fn_types = defaultdict(int)
        for fn in all_fns:
            fn_types[self._normalize_type(fn.get("type", ""))] += 1

        return {
            "total_contracts": len(results),
            "detection_f1": metrics.f1_score,
            "detection_precision": metrics.precision,
            "detection_recall": metrics.recall,
            "severity_accuracy": metrics.severity_accuracy,
            "llm_quality_score": metrics.overall_quality_score,
            "total_false_positives": len(all_fps),
            "total_false_negatives": len(all_fns),
            "common_fp_types": dict(sorted(fp_types.items(), key=lambda x: -x[1])[:5]),
            "common_fn_types": dict(sorted(fn_types.items(), key=lambda x: -x[1])[:5]),
            "by_type_f1": {k: v.get("f1", 0) for k, v in metrics.by_type.items()}
        }

    def save_results(self, results: Dict[str, Any], prefix: str = "violation_detection"):
        """Save evaluation results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Convert to serializable format
        def serialize_match(m: ViolationMatch) -> Dict:
            return {
                "detected": m.detected,
                "ground_truth": m.ground_truth,
                "type_match": m.type_match,
                "severity_match": m.severity_match,
                "similarity_score": m.similarity_score
            }

        serializable = {
            "aggregated_metrics": asdict(results["aggregated_metrics"]),
            "quality_scores": results.get("quality_scores", {}),
            "summary": results["summary"],
            "individual_results": [
                {
                    "contract_id": r.contract_id,
                    "metrics": asdict(r.metrics),
                    "matched_count": len(r.matched_violations),
                    "false_positives": r.false_positives[:10],
                    "false_negatives": r.false_negatives[:10],
                    "llm_evaluation": {
                        "consensus_score": r.llm_evaluation.consensus_score,
                        "agreement_rate": r.llm_evaluation.agreement_rate
                    } if r.llm_evaluation else None
                }
                for r in results["individual_results"]
            ]
        }

        json_file = self.output_dir / f"{prefix}_{timestamp}.json"
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(serializable, f, ensure_ascii=False, indent=2)

        report = self._generate_report(results)
        report_file = self.output_dir / f"{prefix}_report_{timestamp}.md"
        with open(report_file, "w", encoding="utf-8") as f:
            f.write(report)

        print(f"Results saved to {self.output_dir}")
        return json_file, report_file

    def _generate_report(self, results: Dict[str, Any]) -> str:
        """Generate markdown report"""
        m = results["aggregated_metrics"]
        s = results["summary"]
        qs = results.get("quality_scores", {})

        # Per-type table
        type_rows = []
        for vtype, scores in sorted(m.by_type.items(), key=lambda x: -x[1].get("f1", 0)):
            type_rows.append(f"| {vtype} | {scores.get('precision', 0):.3f} | {scores.get('recall', 0):.3f} | {scores.get('f1', 0):.3f} |")
        type_table = "\n".join(type_rows) if type_rows else "| (No data) | - | - | - |"

        report = f"""# Violation Detection Evaluation Report

## Summary

| Metric | Value |
|--------|-------|
| Total Contracts | {s['total_contracts']} |
| **F1 Score** | **{m.f1_score:.3f}** |
| Precision | {m.precision:.3f} |
| Recall | {m.recall:.3f} |
| Severity Accuracy | {m.severity_accuracy:.1%} |
| LLM Quality Score | {m.overall_quality_score:.3f} |

## Detection Statistics

| Statistic | Count |
|-----------|-------|
| Total Ground Truth | {m.total_ground_truth} |
| Total Detected | {m.total_detected} |
| True Positives | {m.true_positives} |
| False Positives | {m.false_positives} |
| False Negatives | {m.false_negatives} |

## LLM Quality Assessment

| Dimension | Score |
|-----------|-------|
| Legal Basis | {m.legal_basis_score:.3f} |
| Description | {m.description_score:.3f} |
| Suggestion | {m.suggestion_score:.3f} |

## Performance by Violation Type

| Type | Precision | Recall | F1 |
|------|-----------|--------|-----|
{type_table}

## Error Analysis

### Most Common False Positives (Over-detection)
"""
        for vtype, count in list(s.get("common_fp_types", {}).items())[:5]:
            report += f"- {vtype}: {count} cases\n"

        report += """
### Most Common False Negatives (Missed)
"""
        for vtype, count in list(s.get("common_fn_types", {}).items())[:5]:
            report += f"- {vtype}: {count} cases\n"

        report += f"""
## Interpretation

{"Excellent detection performance." if m.f1_score > 0.8 else "Good detection performance with room for improvement." if m.f1_score > 0.6 else "Detection needs improvement."}

{"High precision indicates reliable detections." if m.precision > 0.8 else "Some false positives are being generated."}

{"High recall indicates comprehensive coverage." if m.recall > 0.8 else "Some violations are being missed."}

---
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
"""
        return report
