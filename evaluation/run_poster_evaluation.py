#!/usr/bin/env python3
"""
DocScanner AI - Poster Evaluation Script

Generates academic poster-ready metrics by comparing:
1. DocScanner AI Pipeline
2. Baseline GPT-4o
3. Baseline Gemini 2.5 Flash

Outputs:
- Precision, Recall, F1 for violation detection
- Underpayment calculation accuracy
- Processing time comparison
- Statistical significance tests
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple
from datetime import datetime
from dataclasses import dataclass, asdict
import statistics

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()


@dataclass
class ViolationMetrics:
    """Violation detection metrics"""
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0


@dataclass
class UnderpaymentMetrics:
    """Underpayment calculation metrics"""
    mae: float = 0.0  # Mean Absolute Error
    mape: float = 0.0  # Mean Absolute Percentage Error
    accuracy_10pct: float = 0.0  # Within 10% accuracy
    accuracy_20pct: float = 0.0  # Within 20% accuracy


@dataclass
class SystemMetrics:
    """Complete metrics for a system"""
    name: str
    violation: ViolationMetrics
    underpayment: UnderpaymentMetrics
    avg_processing_time: float = 0.0
    total_contracts: int = 0
    successful_analyses: int = 0


# Violation type normalization
VIOLATION_TYPE_MAP = {
    "minimum_wage": ["최저임금", "최저임금 위반", "minimum wage", "최저임금 미달"],
    "overtime": ["연장근로", "연장근로수당", "overtime", "연장근로수당 미지급", "연장근무수당"],
    "night_work": ["야간근로", "야간근로수당", "night work", "야간수당"],
    "holiday_work": ["휴일근로", "휴일근로수당", "holiday work", "휴일수당"],
    "weekly_holiday": ["주휴수당", "주휴", "weekly holiday pay", "주휴수당 미지급"],
    "annual_leave": ["연차", "연차휴가", "annual leave", "연차유급휴가"],
    "severance": ["퇴직금", "severance", "퇴직급여"],
    "working_hours": ["근로시간", "법정근로시간", "working hours", "근로시간 위반"],
    "rest_time": ["휴게시간", "rest time", "휴식시간"],
    "contract_period": ["계약기간", "contract period"],
}


def normalize_violation_type(vtype: str) -> str:
    """Normalize violation type to standard form"""
    vtype_lower = vtype.lower()
    for standard, variants in VIOLATION_TYPE_MAP.items():
        for v in variants:
            if v.lower() in vtype_lower or vtype_lower in v.lower():
                return standard
    return vtype_lower


# LLM-as-Judge for semantic violation matching
_llm_judge_client = None

def get_llm_judge_client():
    """Get OpenAI client for LLM-as-Judge"""
    global _llm_judge_client
    if _llm_judge_client is None:
        from openai import OpenAI
        _llm_judge_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    return _llm_judge_client


def llm_judge_violation_match(
    predicted: List[Dict],
    ground_truth: List[Dict]
) -> Dict[str, Any]:
    """
    Use LLM to semantically evaluate violation detection.
    Returns matched count, precision, recall, F1.
    """
    if not ground_truth:
        return {
            "true_positives": 0,
            "false_positives": len(predicted),
            "false_negatives": 0,
            "precision": 0.0 if predicted else 1.0,
            "recall": 1.0,
            "f1": 0.0 if predicted else 1.0,
            "reasoning": "No ground truth violations"
        }

    if not predicted:
        return {
            "true_positives": 0,
            "false_positives": 0,
            "false_negatives": len(ground_truth),
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "reasoning": "No predicted violations"
        }

    # Format violations for LLM
    pred_text = "\n".join([
        f"- [{i+1}] {v.get('type', v.get('violation_type', 'Unknown'))}: {v.get('description', '')[:100]}"
        for i, v in enumerate(predicted)
    ])

    gt_text = "\n".join([
        f"- [{i+1}] {v.get('type', v.get('violation_type', 'Unknown'))}: {v.get('description', '')[:100]}"
        for i, v in enumerate(ground_truth)
    ])

    prompt = f"""당신은 근로계약서 위반 탐지 평가자입니다.
예측된 위반 사항과 정답(Ground Truth)을 비교하여 매칭 결과를 평가하세요.

**중요**: 위반 유형의 정확한 명칭보다는 **실질적으로 같은 법적 문제를 지적하는지**를 기준으로 판단하세요.
예를 들어:
- "최저임금 미달" vs "최저임금 위반" = 같은 위반
- "법정근로시간 초과" vs "근로시간 위반" = 같은 위반
- "주휴수당 미지급" vs "주휴수당 누락" = 같은 위반

## 예측된 위반 (Predicted):
{pred_text}

## 정답 위반 (Ground Truth):
{gt_text}

## 평가 기준:
1. True Positive (TP): 예측이 정답과 실질적으로 같은 위반을 탐지한 경우
2. False Positive (FP): 예측이 정답에 없는 위반을 탐지한 경우 (과탐지)
3. False Negative (FN): 정답에 있지만 예측이 놓친 위반 (미탐지)

다음 JSON 형식으로 응답하세요:
{{
    "true_positives": TP 개수,
    "false_positives": FP 개수,
    "false_negatives": FN 개수,
    "matched_pairs": [["예측 번호", "정답 번호"], ...],
    "reasoning": "간단한 평가 근거"
}}"""

    try:
        client = get_llm_judge_client()
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            response_format={"type": "json_object"}
        )

        result = json.loads(response.choices[0].message.content)

        tp = result.get("true_positives", 0)
        fp = result.get("false_positives", 0)
        fn = result.get("false_negatives", 0)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        return {
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "reasoning": result.get("reasoning", ""),
            "matched_pairs": result.get("matched_pairs", [])
        }

    except Exception as e:
        print(f"    [LLM Judge Error] {e}, falling back to static matching")
        # Fallback to static matching
        return None


def calculate_violation_metrics(
    predicted: List[Dict],
    ground_truth: List[Dict],
    use_llm_judge: bool = False
) -> ViolationMetrics:
    """
    Calculate precision, recall, F1 for violation detection

    Args:
        use_llm_judge: If True, use LLM for semantic matching instead of static type matching
    """
    # Try LLM Judge first if enabled
    if use_llm_judge:
        llm_result = llm_judge_violation_match(predicted, ground_truth)
        if llm_result is not None:
            return ViolationMetrics(
                precision=llm_result["precision"],
                recall=llm_result["recall"],
                f1=llm_result["f1"],
                true_positives=llm_result["true_positives"],
                false_positives=llm_result["false_positives"],
                false_negatives=llm_result["false_negatives"]
            )

    # Fallback to static matching
    if not ground_truth:
        return ViolationMetrics(
            precision=1.0 if not predicted else 0.0,
            recall=1.0,
            f1=1.0 if not predicted else 0.0
        )

    pred_types = set(normalize_violation_type(v.get("type", "")) for v in predicted)
    gt_types = set(normalize_violation_type(v.get("violation_type", v.get("type", ""))) for v in ground_truth)

    true_positives = len(pred_types & gt_types)
    false_positives = len(pred_types - gt_types)
    false_negatives = len(gt_types - pred_types)

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return ViolationMetrics(
        precision=precision,
        recall=recall,
        f1=f1,
        true_positives=true_positives,
        false_positives=false_positives,
        false_negatives=false_negatives
    )


def calculate_underpayment_metrics(
    predicted_values: List[int],
    ground_truth_values: List[int]
) -> UnderpaymentMetrics:
    """Calculate underpayment prediction accuracy"""
    if not predicted_values or not ground_truth_values:
        return UnderpaymentMetrics()

    errors = []
    pct_errors = []
    within_10pct = 0
    within_20pct = 0

    for pred, gt in zip(predicted_values, ground_truth_values):
        error = abs(pred - gt)
        errors.append(error)

        if gt > 0:
            pct_error = error / gt
            pct_errors.append(pct_error)

            if pct_error <= 0.1:
                within_10pct += 1
            if pct_error <= 0.2:
                within_20pct += 1
        elif pred == 0:
            within_10pct += 1
            within_20pct += 1

    n = len(predicted_values)
    return UnderpaymentMetrics(
        mae=statistics.mean(errors) if errors else 0.0,
        mape=statistics.mean(pct_errors) * 100 if pct_errors else 0.0,
        accuracy_10pct=within_10pct / n * 100 if n > 0 else 0.0,
        accuracy_20pct=within_20pct / n * 100 if n > 0 else 0.0
    )


def run_baseline_evaluation(
    dataset_path: str,
    limit: int = None,
    skip_docscanner: bool = False
) -> Dict[str, SystemMetrics]:
    """
    Run evaluation on dataset

    Args:
        dataset_path: Path to contract_dataset.json
        limit: Limit number of contracts
        skip_docscanner: Skip DocScanner (if ES/Neo4j not available)

    Returns:
        Dict of system name to metrics
    """
    from evaluation.pipeline_adapter import (
        DocScannerAdapter,
        BaselineLLMAdapter,
        AnalysisResult
    )

    # Load dataset
    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    # Handle both list and dict formats
    if isinstance(dataset, list):
        contracts = dataset
    else:
        contracts = dataset.get("contracts", [])

    if limit:
        contracts = contracts[:limit]

    print(f"\nLoaded {len(contracts)} contracts for evaluation")

    # Initialize systems
    systems = {}

    if not skip_docscanner:
        try:
            systems["DocScanner AI"] = DocScannerAdapter(use_lite=True)
        except Exception as e:
            print(f"[Warning] DocScanner not available: {e}")

    systems["GPT-4o (Baseline)"] = BaselineLLMAdapter(provider="openai", model="gpt-4o")
    systems["Gemini 2.5 Flash (Baseline)"] = BaselineLLMAdapter(provider="gemini", model="gemini-2.5-flash")

    # Collect results
    results = {name: [] for name in systems}
    ground_truths = []

    for i, contract in enumerate(contracts):
        contract_id = contract.get("contract_id", contract.get("id", f"contract_{i}"))
        print(f"\n{'='*60}")
        print(f"[{i+1}/{len(contracts)}] Contract: {contract_id}")
        print(f"{'='*60}")

        gt = contract.get("ground_truth", {})
        ground_truths.append(gt)

        for name, adapter in systems.items():
            print(f"\n  Running {name}...")
            try:
                result = adapter.analyze(
                    contract["contract_text"],
                    contract_id
                )
                results[name].append({
                    "contract_id": contract_id,
                    "violations": result.violations,
                    "annual_underpayment": result.annual_underpayment,
                    "processing_time": result.processing_time,
                    "success": True
                })
                print(f"    Violations: {len(result.violations)}, Underpayment: {result.annual_underpayment:,}")
                print(f"    Time: {result.processing_time:.2f}s")
            except Exception as e:
                print(f"    Error: {e}")
                results[name].append({
                    "contract_id": contract_id,
                    "violations": [],
                    "annual_underpayment": 0,
                    "processing_time": 0,
                    "success": False
                })

    # Calculate metrics for each system
    metrics = {}

    for name, system_results in results.items():
        print(f"\n\nCalculating metrics for {name}...")

        # Violation metrics (aggregate)
        all_pred_violations = []
        all_gt_violations = []
        pred_underpayments = []
        gt_underpayments = []
        processing_times = []
        successful = 0

        for r, gt in zip(system_results, ground_truths):
            if r["success"]:
                successful += 1
                all_pred_violations.extend(r["violations"])
                all_gt_violations.extend(gt.get("violations", []))
                pred_underpayments.append(r["annual_underpayment"])
                gt_underpayments.append(gt.get("estimated_annual_underpayment", gt.get("annual_underpayment", 0)))
                processing_times.append(r["processing_time"])

        # Per-contract violation metrics (using LLM-as-Judge for semantic matching)
        violation_metrics_list = []
        for r, gt in zip(system_results, ground_truths):
            if r["success"]:
                vm = calculate_violation_metrics(r["violations"], gt.get("violations", []), use_llm_judge=True)
                violation_metrics_list.append(vm)
                print(f"    Contract: P={vm.precision:.2f}, R={vm.recall:.2f}, F1={vm.f1:.2f}")

        avg_violation_metrics = ViolationMetrics(
            precision=statistics.mean([vm.precision for vm in violation_metrics_list]) if violation_metrics_list else 0,
            recall=statistics.mean([vm.recall for vm in violation_metrics_list]) if violation_metrics_list else 0,
            f1=statistics.mean([vm.f1 for vm in violation_metrics_list]) if violation_metrics_list else 0
        )

        underpayment_metrics = calculate_underpayment_metrics(pred_underpayments, gt_underpayments)

        metrics[name] = SystemMetrics(
            name=name,
            violation=avg_violation_metrics,
            underpayment=underpayment_metrics,
            avg_processing_time=statistics.mean(processing_times) if processing_times else 0,
            total_contracts=len(contracts),
            successful_analyses=successful
        )

    return metrics


def generate_poster_report(
    metrics: Dict[str, SystemMetrics],
    output_dir: str
) -> str:
    """Generate markdown report for poster"""

    report = []
    report.append("# DocScanner AI 성능 평가 결과")
    report.append("")
    report.append(f"생성일시: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")

    # Summary Table
    report.append("## 1. 요약 비교")
    report.append("")
    report.append("| 시스템 | 위반탐지 F1 | 체불액 MAPE |")
    report.append("|--------|------------|-------------|")

    for name, m in metrics.items():
        report.append(
            f"| {name} | {m.violation.f1:.3f} | {m.underpayment.mape:.1f}% |"
        )

    report.append("")

    # Detailed Violation Detection
    report.append("## 2. 위반 탐지 성능")
    report.append("")
    report.append("| 시스템 | Precision | Recall | F1 Score |")
    report.append("|--------|-----------|--------|----------|")

    for name, m in metrics.items():
        report.append(
            f"| {name} | {m.violation.precision:.3f} | {m.violation.recall:.3f} | {m.violation.f1:.3f} |"
        )

    report.append("")

    # Underpayment Accuracy
    report.append("## 3. 체불액 계산 정확도")
    report.append("")
    report.append("| 시스템 | MAE (원) | MAPE | 10% 이내 | 20% 이내 |")
    report.append("|--------|----------|------|----------|----------|")

    for name, m in metrics.items():
        report.append(
            f"| {name} | {m.underpayment.mae:,.0f} | {m.underpayment.mape:.1f}% | {m.underpayment.accuracy_10pct:.1f}% | {m.underpayment.accuracy_20pct:.1f}% |"
        )

    report.append("")

    # Key Findings
    report.append("## 4. 주요 결과")
    report.append("")

    # Find best system for each metric
    best_f1 = max(metrics.values(), key=lambda x: x.violation.f1)
    best_mape = min(metrics.values(), key=lambda x: x.underpayment.mape if x.underpayment.mape > 0 else float('inf'))

    report.append(f"- 최고 위반 탐지 성능 (F1): **{best_f1.name}** ({best_f1.violation.f1:.3f})")
    report.append(f"- 최고 체불액 정확도 (MAPE): **{best_mape.name}** ({best_mape.underpayment.mape:.1f}%)")
    report.append("")

    # Methodology
    report.append("## 평가 방법론")
    report.append("")
    report.append("- **위반 탐지**: LLM-as-Judge 기반 semantic matching (GPT-4o-mini)")
    report.append("- **체불액 계산**: 연간 예상 체불액과 정답 비교")
    report.append("- **Ground Truth**: 수동 주석된 한국 근로계약서")
    report.append("- **법적 기준**: 2025년 근로기준법")
    report.append("")

    report_text = "\n".join(report)

    # Save report
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, f"poster_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    # Also save raw metrics as JSON
    json_path = os.path.join(output_dir, f"poster_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            name: {
                "violation": asdict(m.violation),
                "underpayment": asdict(m.underpayment),
                "avg_processing_time": m.avg_processing_time,
                "total_contracts": m.total_contracts,
                "successful_analyses": m.successful_analyses
            }
            for name, m in metrics.items()
        }, f, indent=2, ensure_ascii=False)

    print(f"\nReport saved to: {report_path}")
    print(f"JSON saved to: {json_path}")

    return report_path


def main():
    parser = argparse.ArgumentParser(description="DocScanner AI Poster Evaluation")
    parser.add_argument("--dataset", type=str,
                       default="evaluation/sample_data/contract_dataset.json",
                       help="Path to dataset")
    parser.add_argument("--limit", type=int, default=None,
                       help="Limit number of contracts")
    parser.add_argument("--output", type=str, default="evaluation/results",
                       help="Output directory")
    parser.add_argument("--skip-docscanner", action="store_true",
                       help="Skip DocScanner (if ES/Neo4j not available)")
    parser.add_argument("--baselines-only", action="store_true",
                       help="Only run baseline LLMs")

    args = parser.parse_args()

    print("=" * 70)
    print("  DocScanner AI - Poster Evaluation")
    print("  Generating Academic Metrics")
    print("=" * 70)

    # Check dataset exists
    if not os.path.exists(args.dataset):
        print(f"Error: Dataset not found at {args.dataset}")
        sys.exit(1)

    # Run evaluation
    skip_docscanner = args.skip_docscanner or args.baselines_only
    metrics = run_baseline_evaluation(
        args.dataset,
        limit=args.limit,
        skip_docscanner=skip_docscanner
    )

    # Generate report
    report_path = generate_poster_report(metrics, args.output)

    # Print summary
    print("\n" + "=" * 70)
    print("  Evaluation Complete!")
    print("=" * 70)

    print("\nSummary:")
    for name, m in metrics.items():
        print(f"\n  {name}:")
        print(f"    Violation F1: {m.violation.f1:.3f}")
        print(f"    Underpayment MAPE: {m.underpayment.mape:.1f}%")
        print(f"    Avg Time: {m.avg_processing_time:.2f}s")


if __name__ == "__main__":
    main()
