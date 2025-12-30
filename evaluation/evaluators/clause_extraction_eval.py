"""
Clause Extraction Evaluator

Evaluates the quality of LLM-based clause extraction:
1. Hallucination Detection: Does extracted text exist in original?
2. Completeness: Are all clauses in contract extracted?
3. Position Accuracy: Are character offsets correct for highlighting?
4. Value Extraction: Are numeric values correctly parsed?

Metrics:
- Extraction Recall: % of actual clauses extracted
- Extraction Precision: % of extracted clauses that are real
- Hallucination Rate: % of extracted text not found in original
- Position Accuracy: % of clauses with correct highlighting positions
- Value Accuracy: % of numeric values correctly extracted
"""

import os
import sys
import json
import re
import asyncio
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime
import difflib

# Add paths
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "backend"))

from evaluation.core.llm_evaluators import (
    MultiLLMEvaluator,
    EvaluationDimension,
    CrossValidationResult
)


@dataclass
class ClauseExtractionMetrics:
    """Metrics for clause extraction evaluation"""
    # Basic counts
    total_expected_clauses: int = 0
    total_extracted_clauses: int = 0

    # Extraction quality
    true_positives: int = 0      # Correctly extracted clauses
    false_positives: int = 0     # Hallucinated/wrong clauses
    false_negatives: int = 0     # Missed clauses

    # Calculated metrics
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0

    # Hallucination metrics
    hallucination_count: int = 0
    hallucination_rate: float = 0.0

    # Position accuracy
    position_exact_match: int = 0
    position_fuzzy_match: int = 0  # Within 50 chars
    position_accuracy: float = 0.0

    # Value extraction accuracy
    value_correct: int = 0
    value_total: int = 0
    value_accuracy: float = 0.0

    # Per-clause type breakdown
    by_clause_type: Dict[str, Dict[str, int]] = field(default_factory=dict)


@dataclass
class ClauseExtractionResult:
    """Result of evaluating a single contract's clause extraction"""
    contract_id: str
    metrics: ClauseExtractionMetrics
    hallucinated_clauses: List[Dict[str, Any]]
    missed_clauses: List[Dict[str, Any]]
    position_errors: List[Dict[str, Any]]
    value_errors: List[Dict[str, Any]]
    llm_evaluation: Optional[CrossValidationResult] = None


class ClauseExtractionEvaluator:
    """
    Evaluator for clause extraction quality

    Uses both rule-based checks and LLM-as-Judge for comprehensive evaluation.
    """

    # Standard clause types in Korean labor contracts
    CLAUSE_TYPES = [
        "contract_period",      # 계약기간
        "workplace",            # 근무장소
        "job_description",      # 업무내용
        "working_hours",        # 근로시간
        "break_time",           # 휴게시간
        "work_days",            # 근무일
        "holidays",             # 휴일
        "salary",               # 임금
        "bonus",                # 상여금
        "allowances",           # 수당
        "payment_date",         # 임금지급일
        "annual_leave",         # 연차휴가
        "social_insurance",     # 사회보험
        "severance",            # 퇴직금
        "contract_delivery",    # 계약서 교부
        "penalty",              # 위약금
        "other"                 # 기타
    ]

    def __init__(
        self,
        output_dir: str = None,
        use_llm_evaluation: bool = True,
        fuzzy_match_threshold: float = 0.8
    ):
        self.output_dir = Path(output_dir) if output_dir else Path(__file__).parent.parent / "results" / "clause_extraction"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.use_llm_evaluation = use_llm_evaluation
        self.fuzzy_match_threshold = fuzzy_match_threshold

        if use_llm_evaluation:
            try:
                self.llm_evaluator = MultiLLMEvaluator(
                    use_openai=True,
                    use_gemini=True,
                    use_claude=bool(os.getenv("ANTHROPIC_API_KEY"))
                )
            except ValueError:
                print("[Warning] LLM evaluator not available, using rule-based only")
                self.llm_evaluator = None
        else:
            self.llm_evaluator = None

    def _normalize_text(self, text: str) -> str:
        """Normalize text for comparison"""
        # Remove extra whitespace, newlines
        text = re.sub(r'\s+', ' ', text)
        # Remove common formatting variations
        text = text.strip()
        return text

    def _fuzzy_match(self, text1: str, text2: str) -> float:
        """Calculate fuzzy match ratio between two texts"""
        return difflib.SequenceMatcher(
            None,
            self._normalize_text(text1),
            self._normalize_text(text2)
        ).ratio()

    def _text_exists_in_contract(self, clause_text: str, contract_text: str) -> Tuple[bool, float]:
        """
        Check if extracted clause text exists in original contract

        Returns:
            (exists, similarity_score)
        """
        normalized_clause = self._normalize_text(clause_text)
        normalized_contract = self._normalize_text(contract_text)

        # Exact substring match
        if normalized_clause in normalized_contract:
            return True, 1.0

        # Fuzzy match - slide window over contract
        best_ratio = 0.0
        window_size = len(normalized_clause)

        for i in range(len(normalized_contract) - window_size + 1):
            window = normalized_contract[i:i + window_size]
            ratio = difflib.SequenceMatcher(None, normalized_clause, window).ratio()
            best_ratio = max(best_ratio, ratio)

            if best_ratio >= self.fuzzy_match_threshold:
                return True, best_ratio

        # Also try with smaller window for partial matches
        if window_size > 50:
            for i in range(len(normalized_contract) - 50 + 1):
                window = normalized_contract[i:i + 100]
                ratio = difflib.SequenceMatcher(None, normalized_clause[:100], window).ratio()
                best_ratio = max(best_ratio, ratio)

        return best_ratio >= self.fuzzy_match_threshold, best_ratio

    def _verify_position(
        self,
        clause_text: str,
        position: Dict[str, int],
        contract_text: str
    ) -> Tuple[bool, bool, str]:
        """
        Verify if position (start, end) correctly points to clause text

        Returns:
            (exact_match, fuzzy_match, actual_text_at_position)
        """
        if not position or "start" not in position or "end" not in position:
            return False, False, ""

        start = position.get("start", 0)
        end = position.get("end", 0)

        if start < 0 or end > len(contract_text) or start >= end:
            return False, False, ""

        actual_text = contract_text[start:end]

        # Exact match
        if self._normalize_text(actual_text) == self._normalize_text(clause_text):
            return True, True, actual_text

        # Fuzzy match
        ratio = self._fuzzy_match(actual_text, clause_text)
        return False, ratio >= self.fuzzy_match_threshold, actual_text

    def _extract_numeric_values(self, text: str) -> Dict[str, Any]:
        """Extract numeric values from text for comparison"""
        values = {}

        # Money patterns (Korean Won)
        money_patterns = [
            r'(\d{1,3}(?:,\d{3})*)\s*원',
            r'(\d+)\s*만\s*원',
            r'월\s*(?:급여?|급)?[:\s]*(\d{1,3}(?:,\d{3})*)',
        ]
        for pattern in money_patterns:
            matches = re.findall(pattern, text)
            if matches:
                values['salary'] = matches[0] if isinstance(matches[0], str) else matches[0]

        # Time patterns
        time_pattern = r'(\d{1,2})\s*[시:]\s*(\d{2})?\s*분?'
        times = re.findall(time_pattern, text)
        if times:
            values['times'] = times

        # Hour patterns
        hour_pattern = r'(\d+)\s*시간'
        hours = re.findall(hour_pattern, text)
        if hours:
            values['hours'] = hours

        # Day patterns
        day_pattern = r'(\d+)\s*일'
        days = re.findall(day_pattern, text)
        if days:
            values['days'] = days

        return values

    def evaluate_extraction(
        self,
        contract_text: str,
        extracted_clauses: List[Dict[str, Any]],
        ground_truth_clauses: List[Dict[str, Any]] = None,
        contract_id: str = "unknown"
    ) -> ClauseExtractionResult:
        """
        Evaluate clause extraction quality

        Args:
            contract_text: Original contract text
            extracted_clauses: List of extracted clause objects
            ground_truth_clauses: Optional ground truth for comparison
            contract_id: Identifier for the contract

        Returns:
            ClauseExtractionResult with detailed metrics
        """
        metrics = ClauseExtractionMetrics()
        hallucinated = []
        missed = []
        position_errors = []
        value_errors = []

        metrics.total_extracted_clauses = len(extracted_clauses)

        # 1. Check for hallucinations (extracted text not in original)
        for clause in extracted_clauses:
            clause_text = clause.get("original_text", clause.get("text", ""))
            if not clause_text:
                continue

            exists, similarity = self._text_exists_in_contract(clause_text, contract_text)

            if not exists:
                metrics.hallucination_count += 1
                hallucinated.append({
                    "clause_number": clause.get("clause_number"),
                    "clause_type": clause.get("clause_type"),
                    "extracted_text": clause_text[:200],
                    "similarity_score": similarity
                })

        # 2. Check position accuracy
        for clause in extracted_clauses:
            clause_text = clause.get("original_text", clause.get("text", ""))
            position = clause.get("position", {})

            if clause_text and position:
                exact, fuzzy, actual = self._verify_position(clause_text, position, contract_text)

                if exact:
                    metrics.position_exact_match += 1
                elif fuzzy:
                    metrics.position_fuzzy_match += 1
                else:
                    position_errors.append({
                        "clause_number": clause.get("clause_number"),
                        "expected_text": clause_text[:100],
                        "actual_text_at_position": actual[:100] if actual else "N/A",
                        "position": position
                    })

        # 3. Compare with ground truth if available
        if ground_truth_clauses:
            metrics.total_expected_clauses = len(ground_truth_clauses)

            # Match extracted to ground truth
            matched_gt = set()
            for ext_clause in extracted_clauses:
                ext_text = self._normalize_text(ext_clause.get("original_text", ""))

                for i, gt_clause in enumerate(ground_truth_clauses):
                    if i in matched_gt:
                        continue

                    gt_text = self._normalize_text(gt_clause.get("original_text", gt_clause.get("text", "")))

                    if self._fuzzy_match(ext_text, gt_text) >= self.fuzzy_match_threshold:
                        metrics.true_positives += 1
                        matched_gt.add(i)

                        # Check value extraction
                        ext_values = ext_clause.get("extracted_values", {})
                        gt_values = gt_clause.get("extracted_values", {})

                        for key in gt_values:
                            metrics.value_total += 1
                            if key in ext_values and str(ext_values[key]) == str(gt_values[key]):
                                metrics.value_correct += 1
                            else:
                                value_errors.append({
                                    "clause_number": ext_clause.get("clause_number"),
                                    "field": key,
                                    "expected": gt_values.get(key),
                                    "extracted": ext_values.get(key)
                                })
                        break
                else:
                    # No match found - potential false positive or hallucination
                    if ext_text and len(ext_text) > 10:  # Skip very short extractions
                        metrics.false_positives += 1

            # Find missed clauses
            for i, gt_clause in enumerate(ground_truth_clauses):
                if i not in matched_gt:
                    metrics.false_negatives += 1
                    missed.append({
                        "clause_number": gt_clause.get("clause_number"),
                        "clause_type": gt_clause.get("clause_type"),
                        "text": gt_clause.get("original_text", gt_clause.get("text", ""))[:200]
                    })
        else:
            # Without ground truth, use heuristics
            # Count unique numbered clauses in contract
            clause_numbers = re.findall(r'(?:^|\n)\s*(\d+)\s*[.조]', contract_text)
            metrics.total_expected_clauses = len(set(clause_numbers)) if clause_numbers else metrics.total_extracted_clauses

            metrics.true_positives = metrics.total_extracted_clauses - metrics.hallucination_count
            metrics.false_positives = metrics.hallucination_count

        # 4. Calculate aggregate metrics
        if metrics.total_extracted_clauses > 0:
            metrics.hallucination_rate = metrics.hallucination_count / metrics.total_extracted_clauses

            total_with_position = metrics.position_exact_match + metrics.position_fuzzy_match + len(position_errors)
            if total_with_position > 0:
                metrics.position_accuracy = (metrics.position_exact_match + metrics.position_fuzzy_match) / total_with_position

        if metrics.true_positives + metrics.false_positives > 0:
            metrics.precision = metrics.true_positives / (metrics.true_positives + metrics.false_positives)

        if metrics.true_positives + metrics.false_negatives > 0:
            metrics.recall = metrics.true_positives / (metrics.true_positives + metrics.false_negatives)

        if metrics.precision + metrics.recall > 0:
            metrics.f1_score = 2 * metrics.precision * metrics.recall / (metrics.precision + metrics.recall)

        if metrics.value_total > 0:
            metrics.value_accuracy = metrics.value_correct / metrics.value_total

        return ClauseExtractionResult(
            contract_id=contract_id,
            metrics=metrics,
            hallucinated_clauses=hallucinated,
            missed_clauses=missed,
            position_errors=position_errors,
            value_errors=value_errors
        )

    async def evaluate_with_llm(
        self,
        contract_text: str,
        extracted_clauses: List[Dict[str, Any]],
        contract_id: str = "unknown"
    ) -> CrossValidationResult:
        """
        Use LLM-as-Judge to evaluate extraction quality
        """
        if not self.llm_evaluator:
            return None

        # Format extracted clauses for evaluation
        clauses_text = json.dumps(extracted_clauses, ensure_ascii=False, indent=2)

        content = f"""[추출된 조항 목록]
{clauses_text[:5000]}

위 조항들이 아래 원본 계약서에서 정확하게 추출되었는지 평가해주세요."""

        custom_criteria = {
            "extraction_completeness": "계약서의 모든 조항이 빠짐없이 추출되었는가?",
            "text_fidelity": "추출된 텍스트가 원본과 정확히 일치하는가? (임의 수정/요약 없음)",
            "structure_preservation": "조항 번호, 유형, 계층 구조가 올바르게 파싱되었는가?",
            "value_extraction": "금액, 시간, 날짜 등 핵심 값이 정확히 추출되었는가?"
        }

        return await self.llm_evaluator.evaluate_with_cross_validation(
            content=content,
            dimensions=[
                EvaluationDimension.COMPLETENESS,
                EvaluationDimension.FAITHFULNESS,
                EvaluationDimension.ACCURACY
            ],
            custom_criteria=custom_criteria,
            context={"contract_text": contract_text[:4000]}
        )

    async def evaluate_batch(
        self,
        test_cases: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Evaluate multiple contracts

        Args:
            test_cases: List of {contract_text, extracted_clauses, ground_truth_clauses?, contract_id}
        """
        results = []
        aggregated = ClauseExtractionMetrics()

        for i, case in enumerate(test_cases):
            print(f"  Evaluating {i+1}/{len(test_cases)}: {case.get('contract_id', f'contract_{i}')}")

            result = self.evaluate_extraction(
                contract_text=case["contract_text"],
                extracted_clauses=case["extracted_clauses"],
                ground_truth_clauses=case.get("ground_truth_clauses"),
                contract_id=case.get("contract_id", f"contract_{i}")
            )

            # LLM evaluation if enabled
            if self.llm_evaluator and self.use_llm_evaluation:
                try:
                    llm_result = await self.evaluate_with_llm(
                        case["contract_text"],
                        case["extracted_clauses"],
                        case.get("contract_id", f"contract_{i}")
                    )
                    result.llm_evaluation = llm_result
                except Exception as e:
                    print(f"    LLM evaluation failed: {e}")

            results.append(result)

            # Aggregate metrics
            m = result.metrics
            aggregated.total_expected_clauses += m.total_expected_clauses
            aggregated.total_extracted_clauses += m.total_extracted_clauses
            aggregated.true_positives += m.true_positives
            aggregated.false_positives += m.false_positives
            aggregated.false_negatives += m.false_negatives
            aggregated.hallucination_count += m.hallucination_count
            aggregated.position_exact_match += m.position_exact_match
            aggregated.position_fuzzy_match += m.position_fuzzy_match
            aggregated.value_correct += m.value_correct
            aggregated.value_total += m.value_total

        # Calculate aggregated rates
        if aggregated.total_extracted_clauses > 0:
            aggregated.hallucination_rate = aggregated.hallucination_count / aggregated.total_extracted_clauses

        if aggregated.true_positives + aggregated.false_positives > 0:
            aggregated.precision = aggregated.true_positives / (aggregated.true_positives + aggregated.false_positives)

        if aggregated.true_positives + aggregated.false_negatives > 0:
            aggregated.recall = aggregated.true_positives / (aggregated.true_positives + aggregated.false_negatives)

        if aggregated.precision + aggregated.recall > 0:
            aggregated.f1_score = 2 * aggregated.precision * aggregated.recall / (aggregated.precision + aggregated.recall)

        total_positions = aggregated.position_exact_match + aggregated.position_fuzzy_match
        if aggregated.total_extracted_clauses > 0:
            aggregated.position_accuracy = total_positions / aggregated.total_extracted_clauses

        if aggregated.value_total > 0:
            aggregated.value_accuracy = aggregated.value_correct / aggregated.value_total

        return {
            "individual_results": results,
            "aggregated_metrics": aggregated,
            "summary": self._generate_summary(aggregated, results)
        }

    def _generate_summary(
        self,
        metrics: ClauseExtractionMetrics,
        results: List[ClauseExtractionResult]
    ) -> Dict[str, Any]:
        """Generate evaluation summary"""

        # Collect all hallucinations
        all_hallucinations = []
        for r in results:
            all_hallucinations.extend(r.hallucinated_clauses)

        # LLM consensus if available
        llm_scores = []
        for r in results:
            if r.llm_evaluation:
                llm_scores.append(r.llm_evaluation.consensus_score)

        return {
            "total_contracts": len(results),
            "extraction_f1": metrics.f1_score,
            "extraction_precision": metrics.precision,
            "extraction_recall": metrics.recall,
            "hallucination_rate": metrics.hallucination_rate,
            "position_accuracy": metrics.position_accuracy,
            "value_accuracy": metrics.value_accuracy,
            "llm_consensus_score": sum(llm_scores) / len(llm_scores) if llm_scores else None,
            "total_hallucinations": len(all_hallucinations),
            "hallucination_examples": all_hallucinations[:5]  # Top 5 examples
        }

    def save_results(self, results: Dict[str, Any], prefix: str = "clause_extraction"):
        """Save evaluation results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Convert to serializable format
        serializable = {
            "aggregated_metrics": asdict(results["aggregated_metrics"]),
            "summary": results["summary"],
            "individual_results": [
                {
                    "contract_id": r.contract_id,
                    "metrics": asdict(r.metrics),
                    "hallucinated_clauses": r.hallucinated_clauses,
                    "missed_clauses": r.missed_clauses,
                    "position_errors": r.position_errors[:10],  # Limit
                    "value_errors": r.value_errors[:10],
                    "llm_evaluation": {
                        "consensus_score": r.llm_evaluation.consensus_score,
                        "agreement_rate": r.llm_evaluation.agreement_rate,
                        "final_verdict": r.llm_evaluation.final_verdict
                    } if r.llm_evaluation else None
                }
                for r in results["individual_results"]
            ]
        }

        # Save JSON
        json_file = self.output_dir / f"{prefix}_{timestamp}.json"
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(serializable, f, ensure_ascii=False, indent=2)

        # Save markdown report
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

        report = f"""# Clause Extraction Evaluation Report

## Summary

| Metric | Value |
|--------|-------|
| Total Contracts | {s['total_contracts']} |
| **F1 Score** | **{m.f1_score:.3f}** |
| Precision | {m.precision:.3f} |
| Recall | {m.recall:.3f} |
| **Hallucination Rate** | **{m.hallucination_rate:.1%}** |
| Position Accuracy | {m.position_accuracy:.1%} |
| Value Accuracy | {m.value_accuracy:.1%} |
| LLM Consensus Score | {s['llm_consensus_score']:.3f if s['llm_consensus_score'] else 'N/A'} |

## Extraction Statistics

| Statistic | Count |
|-----------|-------|
| Total Expected Clauses | {m.total_expected_clauses} |
| Total Extracted Clauses | {m.total_extracted_clauses} |
| True Positives | {m.true_positives} |
| False Positives | {m.false_positives} |
| False Negatives (Missed) | {m.false_negatives} |
| Hallucinations | {m.hallucination_count} |

## Hallucination Analysis

Total hallucinated clauses: {s['total_hallucinations']}

### Examples of Hallucinated Content

"""
        for i, h in enumerate(s.get('hallucination_examples', [])[:5]):
            report += f"""
#### Example {i+1}
- Clause Number: {h.get('clause_number', 'N/A')}
- Type: {h.get('clause_type', 'N/A')}
- Similarity Score: {h.get('similarity_score', 0):.2f}
- Extracted Text: "{h.get('extracted_text', '')[:100]}..."
"""

        report += f"""
## Interpretation

{"Excellent extraction quality with minimal hallucinations." if m.hallucination_rate < 0.05 else "Good extraction quality but some hallucinations detected." if m.hallucination_rate < 0.15 else "Significant hallucination issues requiring attention."}

{"High precision indicates reliable extractions." if m.precision > 0.9 else "Precision could be improved."}

{"High recall indicates comprehensive coverage." if m.recall > 0.9 else "Some clauses are being missed."}

---
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
"""
        return report
