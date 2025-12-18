"""
Underpayment Calculation Evaluator

Tests the accuracy of the Neuro-Symbolic underpayment calculator.
This is a critical evaluation because financial calculations must be precise.

Test Cases:
1. Minimum wage violation detection
2. Overtime pay calculation
3. Weekly holiday pay calculation
4. Break time deduction
5. Probation period adjustment
6. Inclusive wage (포괄임금) scenarios

Metrics:
- Mean Absolute Error (MAE) in KRW
- Mean Absolute Percentage Error (MAPE)
- Exact match rate (within 1% tolerance)
- Error distribution analysis
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime
import numpy as np

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "backend"))


# 2025 Korean Labor Law Constants
MINIMUM_WAGE_2025 = 10030  # KRW per hour
STANDARD_MONTHLY_HOURS = 209  # (40 hours * 52 weeks + 8 hours weekly holiday * 52) / 12


@dataclass
class CalculationTestCase:
    """A single test case for underpayment calculation"""
    case_id: str
    description: str

    # Input values (from contract)
    monthly_salary: int
    daily_work_hours: float
    weekly_work_days: int
    break_minutes: int = 60
    has_weekly_holiday_pay: bool = True
    is_inclusive_wage: bool = False
    probation_discount: float = 0.0  # 0.1 = 10% discount

    # Expected outputs
    expected_hourly_wage: float = 0.0
    expected_is_violation: bool = False
    expected_monthly_underpayment: int = 0
    expected_annual_underpayment: int = 0

    # Optional details
    notes: str = ""


@dataclass
class CalculationResult:
    """Result from the calculator"""
    case_id: str
    calculated_hourly_wage: float = 0.0
    calculated_is_violation: bool = False
    calculated_monthly_underpayment: int = 0
    calculated_annual_underpayment: int = 0
    calculation_breakdown: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CalculationEvalMetrics:
    """Metrics for calculation evaluation"""
    total_cases: int = 0

    # Violation detection
    violation_tp: int = 0  # Correctly detected violations
    violation_fp: int = 0  # False alarms
    violation_fn: int = 0  # Missed violations
    violation_tn: int = 0  # Correctly passed clean contracts

    violation_precision: float = 0.0
    violation_recall: float = 0.0
    violation_f1: float = 0.0
    violation_accuracy: float = 0.0

    # Underpayment amount accuracy
    mae_monthly: float = 0.0  # Mean Absolute Error (monthly)
    mae_annual: float = 0.0   # Mean Absolute Error (annual)
    mape: float = 0.0         # Mean Absolute Percentage Error
    rmse: float = 0.0         # Root Mean Squared Error

    # Exact match (within tolerance)
    exact_match_count: int = 0
    exact_match_rate: float = 0.0

    # Hourly wage accuracy
    hourly_wage_mae: float = 0.0
    hourly_wage_mape: float = 0.0

    # Error distribution
    underestimate_count: int = 0
    overestimate_count: int = 0
    max_error: int = 0
    min_error: int = 0


class NeuroSymbolicCalculator:
    """
    Reference implementation of the underpayment calculator

    This mirrors the logic in clause_analyzer.py for testing purposes.
    """

    def __init__(self, minimum_wage: int = MINIMUM_WAGE_2025):
        self.minimum_wage = minimum_wage

    def calculate(
        self,
        monthly_salary: int,
        daily_work_hours: float,
        weekly_work_days: int,
        break_minutes: int = 60,
        has_weekly_holiday_pay: bool = True,
        is_inclusive_wage: bool = False,
        probation_discount: float = 0.0
    ) -> Dict[str, Any]:
        """
        Calculate underpayment based on contract terms

        Returns:
            Dictionary with calculation results
        """
        # Apply probation discount to expected minimum
        effective_salary = monthly_salary
        if probation_discount > 0:
            # During probation, we need to check if discounted salary meets minimum
            effective_salary = monthly_salary / (1 - probation_discount)

        # Calculate actual work hours per week
        work_hours_per_day = daily_work_hours - (break_minutes / 60)
        work_hours_per_week = work_hours_per_day * weekly_work_days

        # Weekly holiday hours (주휴시간)
        weekly_holiday_hours = 0
        if has_weekly_holiday_pay and work_hours_per_week >= 15:
            # Weekly holiday pay = (weekly hours / 40) * 8 hours
            weekly_holiday_hours = min(8, (work_hours_per_week / 40) * 8)

        # Calculate monthly work hours
        # Monthly hours = (weekly work hours + weekly holiday hours) * 52 / 12
        total_weekly_hours = work_hours_per_week + weekly_holiday_hours
        monthly_hours = total_weekly_hours * 52 / 12

        # Calculate overtime hours (if over 40 hours/week)
        base_hours_per_week = min(40, work_hours_per_week)
        overtime_hours_per_week = max(0, work_hours_per_week - 40)

        # Monthly breakdown
        base_monthly_hours = (base_hours_per_week + weekly_holiday_hours) * 52 / 12
        overtime_monthly_hours = overtime_hours_per_week * 52 / 12

        # Calculate legal minimum wage
        # Base pay + overtime pay (1.5x for overtime hours)
        legal_base_pay = base_monthly_hours * self.minimum_wage
        legal_overtime_pay = overtime_monthly_hours * self.minimum_wage * 1.5
        legal_minimum_monthly = legal_base_pay + legal_overtime_pay

        # Calculate actual hourly wage
        actual_hourly_wage = monthly_salary / monthly_hours if monthly_hours > 0 else 0

        # Check for violation
        is_violation = monthly_salary < legal_minimum_monthly

        # Calculate underpayment
        monthly_underpayment = max(0, legal_minimum_monthly - monthly_salary)
        annual_underpayment = monthly_underpayment * 12

        # If inclusive wage, check if overtime is properly compensated
        if is_inclusive_wage and overtime_hours_per_week > 0:
            # Overtime should be paid at 1.5x rate
            # If salary doesn't cover this, it's a violation
            expected_overtime_pay = overtime_monthly_hours * self.minimum_wage * 1.5
            actual_overtime_coverage = monthly_salary - legal_base_pay

            if actual_overtime_coverage < expected_overtime_pay:
                is_violation = True
                monthly_underpayment = max(monthly_underpayment, expected_overtime_pay - actual_overtime_coverage)
                annual_underpayment = monthly_underpayment * 12

        return {
            "actual_hourly_wage": round(actual_hourly_wage),
            "legal_minimum_hourly": self.minimum_wage,
            "is_violation": is_violation,
            "monthly_underpayment": round(monthly_underpayment),
            "annual_underpayment": round(annual_underpayment),
            "calculation_breakdown": {
                "work_hours_per_day": work_hours_per_day,
                "work_hours_per_week": work_hours_per_week,
                "weekly_holiday_hours": weekly_holiday_hours,
                "total_monthly_hours": monthly_hours,
                "base_monthly_hours": base_monthly_hours,
                "overtime_monthly_hours": overtime_monthly_hours,
                "legal_base_pay": legal_base_pay,
                "legal_overtime_pay": legal_overtime_pay,
                "legal_minimum_monthly": legal_minimum_monthly
            }
        }


class UnderpaymentCalculationEvaluator:
    """
    Evaluator for underpayment calculation accuracy
    """

    def __init__(
        self,
        output_dir: str = None,
        tolerance_rate: float = 0.01  # 1% tolerance for exact match
    ):
        self.output_dir = Path(output_dir) if output_dir else Path(__file__).parent.parent / "results" / "underpayment_calc"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.tolerance_rate = tolerance_rate
        self.calculator = NeuroSymbolicCalculator()

    def create_test_cases(self) -> List[CalculationTestCase]:
        """Create comprehensive test cases"""
        cases = []

        # Case 1: Standard contract - no violation
        cases.append(CalculationTestCase(
            case_id="TC001",
            description="Standard 40-hour week, adequate salary",
            monthly_salary=3500000,
            daily_work_hours=8,
            weekly_work_days=5,
            break_minutes=60,
            expected_hourly_wage=16746,
            expected_is_violation=False,
            expected_monthly_underpayment=0,
            expected_annual_underpayment=0
        ))

        # Case 2: Minimum wage violation - clear case
        cases.append(CalculationTestCase(
            case_id="TC002",
            description="Minimum wage violation - salary too low",
            monthly_salary=1800000,
            daily_work_hours=8,
            weekly_work_days=5,
            break_minutes=60,
            expected_hourly_wage=8612,
            expected_is_violation=True,
            expected_monthly_underpayment=296270,
            expected_annual_underpayment=3555240
        ))

        # Case 3: Long hours but adequate total pay
        cases.append(CalculationTestCase(
            case_id="TC003",
            description="Long hours (52 hours/week) with overtime pay",
            monthly_salary=3200000,
            daily_work_hours=10.4,  # 52 hours / 5 days
            weekly_work_days=5,
            break_minutes=60,
            expected_hourly_wage=12190,
            expected_is_violation=True,  # Need to verify overtime is properly paid
            expected_monthly_underpayment=0,
            expected_annual_underpayment=0,
            notes="52시간 근무, 연장근로수당 포함 여부 확인 필요"
        ))

        # Case 4: Part-time worker
        cases.append(CalculationTestCase(
            case_id="TC004",
            description="Part-time worker (15 hours/week)",
            monthly_salary=700000,
            daily_work_hours=5,
            weekly_work_days=3,
            break_minutes=0,
            has_weekly_holiday_pay=True,  # 15시간 이상이므로 주휴수당 대상
            expected_hourly_wage=10030,
            expected_is_violation=False,
            expected_monthly_underpayment=0,
            expected_annual_underpayment=0
        ))

        # Case 5: Under 15 hours - no weekly holiday pay
        cases.append(CalculationTestCase(
            case_id="TC005",
            description="Part-time under 15 hours/week",
            monthly_salary=400000,
            daily_work_hours=4,
            weekly_work_days=3,
            break_minutes=0,
            has_weekly_holiday_pay=False,
            expected_hourly_wage=10256,
            expected_is_violation=False,
            expected_monthly_underpayment=0,
            expected_annual_underpayment=0
        ))

        # Case 6: No break time violation
        cases.append(CalculationTestCase(
            case_id="TC006",
            description="12-hour day with no break (severe violation)",
            monthly_salary=2003290,
            daily_work_hours=12,
            weekly_work_days=6,
            break_minutes=0,
            expected_hourly_wage=6421,
            expected_is_violation=True,
            expected_monthly_underpayment=1500000,  # Approximate
            expected_annual_underpayment=18000000,
            notes="휴게시간 없는 장시간 근무 - CONTRACT_001 참조"
        ))

        # Case 7: Inclusive wage (포괄임금제)
        cases.append(CalculationTestCase(
            case_id="TC007",
            description="Inclusive wage with overtime",
            monthly_salary=2800000,
            daily_work_hours=9,
            weekly_work_days=5,
            break_minutes=60,
            is_inclusive_wage=True,
            expected_hourly_wage=12727,
            expected_is_violation=True,  # 포괄임금은 연장수당 별도 지급 필요
            expected_monthly_underpayment=200000,
            expected_annual_underpayment=2400000,
            notes="포괄임금제 - 연장근로 1시간/일"
        ))

        # Case 8: Probation period (90%)
        cases.append(CalculationTestCase(
            case_id="TC008",
            description="Probation period with 90% salary",
            monthly_salary=1886604,  # 2096227 * 0.9
            daily_work_hours=8,
            weekly_work_days=5,
            break_minutes=60,
            probation_discount=0.1,
            expected_hourly_wage=9027,  # Below minimum but legal during probation
            expected_is_violation=False,  # 수습 중 90%는 합법
            expected_monthly_underpayment=0,
            expected_annual_underpayment=0,
            notes="수습기간 90% 지급 (합법)"
        ))

        # Case 9: Probation with excessive discount (80%)
        cases.append(CalculationTestCase(
            case_id="TC009",
            description="Probation period with 80% salary (illegal)",
            monthly_salary=1676982,  # 2096227 * 0.8
            daily_work_hours=8,
            weekly_work_days=5,
            break_minutes=60,
            probation_discount=0.2,
            expected_hourly_wage=8024,
            expected_is_violation=True,  # 80% 감액은 위법
            expected_monthly_underpayment=419245,
            expected_annual_underpayment=5030940,
            notes="수습기간 80% 지급은 위법"
        ))

        # Case 10: 6-day work week
        cases.append(CalculationTestCase(
            case_id="TC010",
            description="6-day work week, 48 hours",
            monthly_salary=2500000,
            daily_work_hours=8,
            weekly_work_days=6,
            break_minutes=60,
            expected_hourly_wage=10030,  # Barely at minimum
            expected_is_violation=True,  # 48시간 중 8시간은 연장근로
            expected_monthly_underpayment=180000,
            expected_annual_underpayment=2160000,
            notes="주 48시간 - 8시간 연장근로"
        ))

        return cases

    def run_calculation(self, test_case: CalculationTestCase) -> CalculationResult:
        """Run calculation for a single test case"""

        result = self.calculator.calculate(
            monthly_salary=test_case.monthly_salary,
            daily_work_hours=test_case.daily_work_hours,
            weekly_work_days=test_case.weekly_work_days,
            break_minutes=test_case.break_minutes,
            has_weekly_holiday_pay=test_case.has_weekly_holiday_pay,
            is_inclusive_wage=test_case.is_inclusive_wage,
            probation_discount=test_case.probation_discount
        )

        return CalculationResult(
            case_id=test_case.case_id,
            calculated_hourly_wage=result["actual_hourly_wage"],
            calculated_is_violation=result["is_violation"],
            calculated_monthly_underpayment=result["monthly_underpayment"],
            calculated_annual_underpayment=result["annual_underpayment"],
            calculation_breakdown=result["calculation_breakdown"]
        )

    def evaluate(
        self,
        test_cases: List[CalculationTestCase] = None
    ) -> Tuple[CalculationEvalMetrics, List[Dict[str, Any]]]:
        """
        Evaluate calculation accuracy

        Args:
            test_cases: List of test cases (uses default if None)

        Returns:
            (metrics, detailed_results)
        """
        if test_cases is None:
            test_cases = self.create_test_cases()

        metrics = CalculationEvalMetrics()
        metrics.total_cases = len(test_cases)

        detailed_results = []
        monthly_errors = []
        annual_errors = []
        hourly_errors = []

        for tc in test_cases:
            result = self.run_calculation(tc)

            # Compare results
            is_correct_violation = result.calculated_is_violation == tc.expected_is_violation
            monthly_error = result.calculated_monthly_underpayment - tc.expected_monthly_underpayment
            annual_error = result.calculated_annual_underpayment - tc.expected_annual_underpayment
            hourly_error = result.calculated_hourly_wage - tc.expected_hourly_wage

            # Violation detection accuracy
            if tc.expected_is_violation:
                if result.calculated_is_violation:
                    metrics.violation_tp += 1
                else:
                    metrics.violation_fn += 1
            else:
                if result.calculated_is_violation:
                    metrics.violation_fp += 1
                else:
                    metrics.violation_tn += 1

            # Exact match check
            if tc.expected_annual_underpayment > 0:
                percentage_error = abs(annual_error) / tc.expected_annual_underpayment
            else:
                percentage_error = 0 if annual_error == 0 else 1

            is_exact_match = percentage_error <= self.tolerance_rate
            if is_exact_match:
                metrics.exact_match_count += 1

            # Error tracking
            monthly_errors.append(abs(monthly_error))
            annual_errors.append(abs(annual_error))
            hourly_errors.append(abs(hourly_error))

            if annual_error > 0:
                metrics.overestimate_count += 1
            elif annual_error < 0:
                metrics.underestimate_count += 1

            detailed_results.append({
                "case_id": tc.case_id,
                "description": tc.description,
                "expected": {
                    "hourly_wage": tc.expected_hourly_wage,
                    "is_violation": tc.expected_is_violation,
                    "monthly_underpayment": tc.expected_monthly_underpayment,
                    "annual_underpayment": tc.expected_annual_underpayment
                },
                "calculated": {
                    "hourly_wage": result.calculated_hourly_wage,
                    "is_violation": result.calculated_is_violation,
                    "monthly_underpayment": result.calculated_monthly_underpayment,
                    "annual_underpayment": result.calculated_annual_underpayment
                },
                "errors": {
                    "hourly": hourly_error,
                    "monthly": monthly_error,
                    "annual": annual_error
                },
                "is_correct_violation": is_correct_violation,
                "is_exact_match": is_exact_match,
                "breakdown": result.calculation_breakdown,
                "notes": tc.notes
            })

        # Calculate aggregate metrics
        if metrics.violation_tp + metrics.violation_fp > 0:
            metrics.violation_precision = metrics.violation_tp / (metrics.violation_tp + metrics.violation_fp)

        if metrics.violation_tp + metrics.violation_fn > 0:
            metrics.violation_recall = metrics.violation_tp / (metrics.violation_tp + metrics.violation_fn)

        if metrics.violation_precision + metrics.violation_recall > 0:
            metrics.violation_f1 = 2 * metrics.violation_precision * metrics.violation_recall / (metrics.violation_precision + metrics.violation_recall)

        metrics.violation_accuracy = (metrics.violation_tp + metrics.violation_tn) / metrics.total_cases

        metrics.mae_monthly = np.mean(monthly_errors)
        metrics.mae_annual = np.mean(annual_errors)
        metrics.rmse = np.sqrt(np.mean(np.array(annual_errors) ** 2))
        metrics.hourly_wage_mae = np.mean(hourly_errors)

        # MAPE calculation (only for non-zero expected values)
        percentage_errors = []
        for tc, result in zip(test_cases, detailed_results):
            if tc.expected_annual_underpayment > 0:
                pe = abs(result["errors"]["annual"]) / tc.expected_annual_underpayment
                percentage_errors.append(pe)

        if percentage_errors:
            metrics.mape = np.mean(percentage_errors)

        metrics.exact_match_rate = metrics.exact_match_count / metrics.total_cases

        if annual_errors:
            metrics.max_error = max(annual_errors)
            metrics.min_error = min(annual_errors)

        return metrics, detailed_results

    def save_results(
        self,
        metrics: CalculationEvalMetrics,
        detailed_results: List[Dict[str, Any]],
        prefix: str = "underpayment_calc"
    ):
        """Save evaluation results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        results = {
            "metrics": asdict(metrics),
            "detailed_results": detailed_results,
            "summary": {
                "total_cases": metrics.total_cases,
                "violation_f1": metrics.violation_f1,
                "violation_accuracy": metrics.violation_accuracy,
                "mae_annual": metrics.mae_annual,
                "mape": metrics.mape,
                "exact_match_rate": metrics.exact_match_rate
            }
        }

        json_file = self.output_dir / f"{prefix}_{timestamp}.json"
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        report = self._generate_report(metrics, detailed_results)
        report_file = self.output_dir / f"{prefix}_report_{timestamp}.md"
        with open(report_file, "w", encoding="utf-8") as f:
            f.write(report)

        print(f"Results saved to {self.output_dir}")
        return json_file, report_file

    def _generate_report(
        self,
        metrics: CalculationEvalMetrics,
        detailed_results: List[Dict[str, Any]]
    ) -> str:
        """Generate markdown report"""

        report = f"""# Underpayment Calculation Evaluation Report

## Summary

| Metric | Value |
|--------|-------|
| Total Test Cases | {metrics.total_cases} |
| **Violation Detection F1** | **{metrics.violation_f1:.3f}** |
| Violation Precision | {metrics.violation_precision:.3f} |
| Violation Recall | {metrics.violation_recall:.3f} |
| Violation Accuracy | {metrics.violation_accuracy:.1%} |

## Financial Accuracy

| Metric | Value |
|--------|-------|
| **MAE (Annual)** | **{metrics.mae_annual:,.0f} KRW** |
| MAE (Monthly) | {metrics.mae_monthly:,.0f} KRW |
| RMSE | {metrics.rmse:,.0f} KRW |
| **MAPE** | **{metrics.mape:.1%}** |
| Exact Match Rate | {metrics.exact_match_rate:.1%} |

## Hourly Wage Accuracy

| Metric | Value |
|--------|-------|
| Hourly Wage MAE | {metrics.hourly_wage_mae:.0f} KRW |

## Error Distribution

| Statistic | Count/Value |
|-----------|-------------|
| Overestimates | {metrics.overestimate_count} |
| Underestimates | {metrics.underestimate_count} |
| Max Error | {metrics.max_error:,.0f} KRW |

## Detailed Results

| Case | Description | Expected | Calculated | Error | Status |
|------|-------------|----------|------------|-------|--------|
"""
        for r in detailed_results:
            status = "PASS" if r["is_exact_match"] else "FAIL"
            exp = r["expected"]["annual_underpayment"]
            calc = r["calculated"]["annual_underpayment"]
            err = r["errors"]["annual"]

            report += f"| {r['case_id']} | {r['description'][:30]}... | {exp:,} | {calc:,} | {err:+,} | {status} |\n"

        report += f"""
## Test Case Details

"""
        for r in detailed_results:
            report += f"""### {r['case_id']}: {r['description']}

**Expected:**
- Hourly Wage: {r['expected']['hourly_wage']:,} KRW
- Violation: {r['expected']['is_violation']}
- Annual Underpayment: {r['expected']['annual_underpayment']:,} KRW

**Calculated:**
- Hourly Wage: {r['calculated']['hourly_wage']:,} KRW
- Violation: {r['calculated']['is_violation']}
- Annual Underpayment: {r['calculated']['annual_underpayment']:,} KRW

**Breakdown:**
- Work Hours/Day: {r['breakdown'].get('work_hours_per_day', 'N/A')}
- Work Hours/Week: {r['breakdown'].get('work_hours_per_week', 'N/A')}
- Total Monthly Hours: {r['breakdown'].get('total_monthly_hours', 'N/A'):.1f}

{f"**Notes:** {r['notes']}" if r.get('notes') else ""}

---
"""

        report += f"""
## Interpretation

{"Excellent calculation accuracy." if metrics.mape < 0.05 else "Good calculation accuracy with minor errors." if metrics.mape < 0.15 else "Calculation accuracy needs improvement."}

{"Violation detection is highly accurate." if metrics.violation_f1 > 0.9 else "Some violations are being missed or falsely detected."}

---
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Minimum Wage Reference: {MINIMUM_WAGE_2025:,} KRW/hour (2025)
"""
        return report
