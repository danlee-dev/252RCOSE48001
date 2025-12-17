"""
LLM-based Clause Analyzer with CRAG Integration + Neuro-Symbolic Calculation
- LLM 기반 계약서 조항 분할 (Neuro)
- 조항별 CRAG 검색으로 법률 컨텍스트 확보
- LLM이 법률 컨텍스트 기반으로 위반 여부 판단
- Python 기반 정밀 체불액 계산 (Symbolic)

Flow:
1. LLM이 계약서를 조항 단위로 분할 + 값 추출 (Neuro)
2. 각 조항에 대해 CRAG 검색 (Graph + Vector DB)
3. LLM이 CRAG 결과를 컨텍스트로 위반 분석
4. Python으로 체불액 정밀 계산 (Symbolic)
5. 결과 통합
"""

import os
import json
import re
import time
import difflib
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from datetime import datetime

from app.core.config import settings
from app.core.token_usage_tracker import TokenUsageTracker, record_llm_usage


class ClauseType(Enum):
    """조항 유형"""
    WORK_START_DATE = "근로개시일"
    WORKPLACE = "근무장소"
    JOB_DESCRIPTION = "업무내용"
    WORK_HOURS = "근로시간"
    BREAK_TIME = "휴게시간"
    WORK_DAYS = "근무일"
    HOLIDAYS = "휴일"
    SALARY = "임금"
    BONUS = "상여금"
    ALLOWANCES = "수당"
    PAYMENT_DATE = "임금지급일"
    ANNUAL_LEAVE = "연차휴가"
    SOCIAL_INSURANCE = "사회보험"
    CONTRACT_DELIVERY = "계약서교부"
    PENALTY = "위약금"
    TERMINATION = "해지"
    OTHER = "기타"


class ViolationSeverity(Enum):
    """위반 심각도"""
    CRITICAL = "CRITICAL"  # 즉시 시정 필요
    HIGH = "HIGH"          # 심각한 위반
    MEDIUM = "MEDIUM"      # 주의 필요
    LOW = "LOW"            # 경미한 문제
    INFO = "INFO"          # 정보 제공


@dataclass
class ExtractedClause:
    """추출된 조항"""
    clause_number: str              # 조항 번호 (예: "4", "6-1")
    clause_type: ClauseType         # 조항 유형
    title: str                      # 조항 제목
    original_text: str              # 원문 텍스트
    extracted_values: Dict[str, Any] = field(default_factory=dict)  # 추출된 값들
    position: Dict[str, int] = field(default_factory=dict)  # 시작/끝 위치


@dataclass
class ClauseViolation:
    """조항 위반 정보"""
    clause: ExtractedClause
    violation_type: str             # 위반 유형
    severity: ViolationSeverity
    description: str                # 위반 설명
    legal_basis: str                # 법적 근거
    current_value: Any              # 현재 계약서 값
    legal_standard: Any             # 법적 기준 값
    suggestion: str                 # 수정 제안 설명
    suggested_text: str = ""        # 수정된 조항 텍스트 (대체용)
    matched_text: str = ""          # 하이라이팅할 실제 텍스트 (텍스트 기반 매칭용)
    crag_sources: List[str] = field(default_factory=list)  # 참조한 법률 출처
    confidence: float = 0.0         # 판단 신뢰도

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.violation_type,
            "severity": self.severity.value,
            "description": self.description,
            "legal_basis": self.legal_basis,
            "current_value": self.current_value,
            "legal_standard": self.legal_standard,
            "suggestion": self.suggestion,
            "suggested_text": self.suggested_text,
            "matched_text": self.matched_text,  # 하이라이팅용 텍스트
            "clause_number": self.clause.clause_number,
            "clause_title": self.clause.title,
            "original_text": self.clause.original_text,
            "start_index": self.clause.position.get("start", -1),
            "end_index": self.clause.position.get("end", -1),
            "sources": self.crag_sources,
            "confidence": self.confidence
        }


@dataclass
class ClauseAnalysisResult:
    """조항 분석 결과"""
    clauses: List[ExtractedClause] = field(default_factory=list)
    violations: List[ClauseViolation] = field(default_factory=list)
    total_underpayment: int = 0
    annual_underpayment: int = 0
    processing_time: float = 0.0

    @property
    def violation_count(self) -> int:
        return len(self.violations)

    @property
    def high_severity_count(self) -> int:
        return sum(1 for v in self.violations
                   if v.severity in [ViolationSeverity.CRITICAL, ViolationSeverity.HIGH])

    def to_dict(self) -> Dict[str, Any]:
        return {
            "violations": [v.to_dict() for v in self.violations],
            "total_underpayment": self.total_underpayment,
            "annual_underpayment": self.annual_underpayment,
            "clause_count": len(self.clauses),
            "violation_count": self.violation_count,
            "high_severity_count": self.high_severity_count,
            "processing_time": self.processing_time
        }


class NeuroSymbolicCalculator:
    """
    Neuro-Symbolic 체불액 계산기
    - LLM이 추출한 숫자를 기반으로 Python으로 정밀 계산
    - LegalStressTest의 계산 로직을 활용
    """

    # 2025년 법정 기준
    MINIMUM_WAGE_2025 = 10_030              # 시급
    LEGAL_DAILY_HOURS = 8                   # 법정 일일 근로시간
    LEGAL_WEEKLY_HOURS = 40                 # 법정 주간 근로시간
    LEGAL_MONTHLY_HOURS = 209               # 법정 월간 근로시간 (주휴 포함)
    OVERTIME_RATE = Decimal("1.5")          # 연장근로 50% 가산
    NIGHT_RATE = Decimal("1.5")             # 야간근로 50% 가산 (22시~06시)
    BREAK_TIME_4_HOURS = 30                 # 4시간 초과 시 30분 휴게
    BREAK_TIME_8_HOURS = 60                 # 8시간 초과 시 60분 휴게

    @dataclass
    class ContractData:
        """추출된 계약서 데이터"""
        monthly_salary: int = 0             # 월급
        daily_hours: float = 8.0            # 일일 근로시간
        weekly_hours: float = 40.0          # 주간 근로시간
        work_days_per_week: int = 0         # 주간 근무일수 (0 = 미추출)
        break_minutes: int = 60             # 휴게시간 (분)
        start_time: str = "09:00"           # 출근 시간
        end_time: str = "18:00"             # 퇴근 시간
        has_bonus: bool = False             # 상여금 유무
        has_allowances: bool = False        # 수당 유무
        is_inclusive_wage: bool = False     # 포괄임금제 여부

    @dataclass
    class CalculationResult:
        """계산 결과"""
        actual_hourly_wage: int = 0         # 실제 시급
        legal_hourly_wage: int = 0          # 법정 최저시급
        is_minimum_wage_violation: bool = False
        minimum_wage_shortage: int = 0      # 시급 부족분
        monthly_minimum_wage_shortage: int = 0  # 월간 최저임금 부족분
        overtime_hours_weekly: float = 0    # 주간 연장근로시간
        overtime_pay_shortage: int = 0      # 연장근로수당 미지급분 (월)
        break_time_violation: bool = False  # 휴게시간 위반 여부
        total_monthly_underpayment: int = 0 # 총 월간 체불액
        calculation_breakdown: Dict[str, Any] = field(default_factory=dict)

    def extract_contract_data(self, clauses: List['ExtractedClause']) -> 'NeuroSymbolicCalculator.ContractData':
        """
        LLM이 추출한 조항들에서 계산에 필요한 숫자 추출 (Neuro 결과 활용)
        """
        data = self.ContractData()

        for clause in clauses:
            values = clause.extracted_values

            if clause.clause_type == ClauseType.SALARY:
                # 월급 추출 (값이 있을 때만 업데이트 - 여러 임금 조항 중 유효한 값만 사용)
                salary = self._safe_int(values.get("base_salary") or values.get("monthly_salary") or values.get("total_salary", 0))
                if salary > 0 and data.monthly_salary == 0:
                    data.monthly_salary = salary

            elif clause.clause_type == ClauseType.WORK_HOURS:
                # 근로시간 추출
                extracted_daily_hours = self._safe_float(values.get("daily_hours", 0))
                data.weekly_hours = self._safe_float(values.get("weekly_hours", 40.0))
                data.start_time = values.get("start_time", "09:00")
                data.end_time = values.get("end_time", "18:00")

                # 시간대에서 일일 근로시간 계산 (항상 계산해서 검증)
                if data.start_time and data.end_time:
                    calculated_hours = self._calculate_hours_from_time(data.start_time, data.end_time)
                    if calculated_hours > 0:
                        # 계산된 시간이 추출된 시간과 다르면 계산된 값 우선 (더 정확)
                        if extracted_daily_hours > 0 and abs(extracted_daily_hours - calculated_hours) > 1:
                            print(f">>> [HOURS] Correcting daily_hours: {extracted_daily_hours} -> {calculated_hours}")
                        data.daily_hours = calculated_hours
                    elif extracted_daily_hours > 0:
                        data.daily_hours = extracted_daily_hours
                elif extracted_daily_hours > 0:
                    data.daily_hours = extracted_daily_hours

                # 근무일수도 함께 추출 시도 (WORK_HOURS에 포함된 경우)
                work_days = self._safe_int(values.get("work_days_per_week") or values.get("days_per_week", 0))
                if work_days > 0:
                    data.work_days_per_week = work_days

            elif clause.clause_type == ClauseType.BREAK_TIME:
                # 휴게시간 추출
                data.break_minutes = self._safe_int(values.get("break_minutes", 60))

            elif clause.clause_type == ClauseType.WORK_DAYS:
                # 근무일수 추출
                work_days = self._safe_int(values.get("work_days_per_week") or values.get("days_per_week", 0))
                if work_days > 0:
                    data.work_days_per_week = work_days
                else:
                    # 텍스트에서 근무일수 패턴 추출
                    work_days_from_text = self._extract_work_days_from_text(clause.original_text)
                    if work_days_from_text > 0:
                        data.work_days_per_week = work_days_from_text

            elif clause.clause_type == ClauseType.ALLOWANCES:
                # 수당/포괄임금 정보
                data.has_allowances = True
                data.is_inclusive_wage = values.get("is_inclusive", False)

            elif clause.clause_type == ClauseType.BONUS:
                data.has_bonus = values.get("has_bonus", False)

        # 근무일수가 미추출(0)이면 전체 조항 텍스트에서 추출 시도
        if data.work_days_per_week == 0:
            for clause in clauses:
                work_days_from_text = self._extract_work_days_from_text(clause.original_text)
                if work_days_from_text > 0:
                    print(f">>> [WORK_DAYS] Extracted from text: {work_days_from_text} days/week")
                    data.work_days_per_week = work_days_from_text
                    break

        # 여전히 미추출이면 기본값 5일 사용
        if data.work_days_per_week == 0:
            print(f">>> [WORK_DAYS] Using default: 5 days/week")
            data.work_days_per_week = 5

        # 주간 근로시간 계산 (명시되지 않은 경우)
        if data.weekly_hours == 40.0:
            calculated_weekly = data.daily_hours * data.work_days_per_week
            if calculated_weekly != 40.0:
                data.weekly_hours = calculated_weekly

        return data

    def calculate_underpayment(self, data: 'NeuroSymbolicCalculator.ContractData') -> 'NeuroSymbolicCalculator.CalculationResult':
        """
        Symbolic 계산: Python으로 정밀 체불액 계산

        계산 방식:
        1. 법정 임금 = (기본근로 + 주휴수당 + 연장근로수당) * 최저시급
        2. 체불액 = 법정 임금 - 실제 지급 임금
        """
        result = self.CalculationResult()
        result.legal_hourly_wage = self.MINIMUM_WAGE_2025
        minimum_wage = Decimal(str(self.MINIMUM_WAGE_2025))

        if data.monthly_salary <= 0:
            return result

        # 1. 실제 근로시간 계산 (휴게시간 제외)
        break_hours = data.break_minutes / 60.0
        actual_daily_hours = data.daily_hours - break_hours if data.break_minutes > 0 else data.daily_hours
        actual_weekly_hours = actual_daily_hours * data.work_days_per_week
        actual_monthly_hours = Decimal(str(actual_weekly_hours)) * Decimal("4.345")

        # 2. 실제 시급 계산
        actual_hourly = Decimal(str(data.monthly_salary)) / actual_monthly_hours
        result.actual_hourly_wage = int(actual_hourly.quantize(Decimal("1"), rounding=ROUND_HALF_UP))

        # 3. 법정 임금 계산 (최저임금 기준)
        # 3-1. 기본근로 (주 40시간 이내)
        legal_base_hours_weekly = min(actual_weekly_hours, float(self.LEGAL_WEEKLY_HOURS))
        legal_base_hours_monthly = Decimal(str(legal_base_hours_weekly)) * Decimal("4.345")
        legal_base_pay = legal_base_hours_monthly * minimum_wage

        # 3-2. 주휴수당 (주 15시간 이상 근무 시 8시간분)
        weekly_holiday_hours = Decimal("8") if actual_weekly_hours >= 15 else Decimal("0")
        weekly_holiday_pay_monthly = weekly_holiday_hours * Decimal("4.345") * minimum_wage

        # 3-3. 연장근로수당 (주 40시간 초과분, 1.5배)
        overtime_weekly = max(0, actual_weekly_hours - self.LEGAL_WEEKLY_HOURS)
        result.overtime_hours_weekly = overtime_weekly
        overtime_monthly_hours = Decimal(str(overtime_weekly)) * Decimal("4.345")
        overtime_pay = overtime_monthly_hours * minimum_wage * self.OVERTIME_RATE

        # 3-4. 법정 월급 합계
        legal_monthly_salary = legal_base_pay + weekly_holiday_pay_monthly + overtime_pay
        legal_monthly_salary_int = int(legal_monthly_salary.quantize(Decimal("1"), rounding=ROUND_HALF_UP))

        # 4. 체불액 계산
        result.total_monthly_underpayment = max(0, legal_monthly_salary_int - data.monthly_salary)

        # 최저임금 위반 여부 판단
        if actual_hourly < minimum_wage:
            result.is_minimum_wage_violation = True
            shortage_per_hour = minimum_wage - actual_hourly
            result.minimum_wage_shortage = int(shortage_per_hour.quantize(Decimal("1"), rounding=ROUND_HALF_UP))
            # 최저임금 미달분 = 법정임금 - 실제임금 중 기본급+주휴 부분
            result.monthly_minimum_wage_shortage = max(0, int(legal_base_pay + weekly_holiday_pay_monthly) - data.monthly_salary)

        # 연장근로수당 미지급분 (포괄임금제 아닌 경우)
        if overtime_weekly > 0 and not data.is_inclusive_wage:
            result.overtime_pay_shortage = int(overtime_pay.quantize(Decimal("1"), rounding=ROUND_HALF_UP))

        # 5. 휴게시간 위반 검사
        if data.daily_hours > 8 and data.break_minutes < self.BREAK_TIME_8_HOURS:
            result.break_time_violation = True
        elif data.daily_hours > 4 and data.break_minutes < self.BREAK_TIME_4_HOURS:
            result.break_time_violation = True

        # 6. 계산 상세 내역
        result.calculation_breakdown = {
            "input": {
                "monthly_salary": data.monthly_salary,
                "daily_hours": data.daily_hours,
                "break_minutes": data.break_minutes,
                "work_days_per_week": data.work_days_per_week,
            },
            "calculated": {
                "actual_daily_hours": float(actual_daily_hours),
                "actual_weekly_hours": float(actual_weekly_hours),
                "actual_monthly_hours": float(actual_monthly_hours),
                "actual_hourly_wage": result.actual_hourly_wage,
            },
            "legal_standards": {
                "minimum_wage": self.MINIMUM_WAGE_2025,
                "legal_base_hours_monthly": float(legal_base_hours_monthly),
                "weekly_holiday_hours": float(weekly_holiday_hours),
                "overtime_hours_weekly": overtime_weekly,
                "overtime_hours_monthly": float(overtime_monthly_hours),
            },
            "legal_pay": {
                "base_pay": int(legal_base_pay),
                "weekly_holiday_pay": int(weekly_holiday_pay_monthly),
                "overtime_pay": int(overtime_pay),
                "total_legal_salary": legal_monthly_salary_int,
            },
            "shortage": {
                "minimum_wage_shortage_per_hour": result.minimum_wage_shortage,
                "total_monthly_underpayment": result.total_monthly_underpayment,
                "annual_underpayment": result.total_monthly_underpayment * 12,
            }
        }

        return result

    def _calculate_hours_from_time(self, start_time: str, end_time: str) -> float:
        """출퇴근 시간에서 근로시간 계산"""
        try:
            start_parts = start_time.replace("시", ":").replace("분", "").split(":")
            end_parts = end_time.replace("시", ":").replace("분", "").split(":")

            start_hour = int(start_parts[0].strip())
            start_min = int(start_parts[1].strip()) if len(start_parts) > 1 else 0
            end_hour = int(end_parts[0].strip())
            end_min = int(end_parts[1].strip()) if len(end_parts) > 1 else 0

            total_minutes = (end_hour * 60 + end_min) - (start_hour * 60 + start_min)
            return total_minutes / 60.0 if total_minutes > 0 else 0
        except (ValueError, IndexError):
            return 0

    def _extract_work_days_from_text(self, text: str) -> int:
        """텍스트에서 근무일수 패턴 추출"""
        import re
        if not text:
            return 0

        # 패턴 1: "매주 N일 근무" 또는 "주 N일"
        match = re.search(r'(?:매주|주)\s*(\d+)\s*일\s*(?:근무)?', text)
        if match:
            return int(match.group(1))

        # 패턴 2: "N일 근무" (월~토 같은 요일 범위와 함께)
        match = re.search(r'(\d+)\s*일\s*근무', text)
        if match:
            return int(match.group(1))

        # 패턴 3: 요일 범위에서 추론 (월~토 = 6일, 월~금 = 5일)
        day_ranges = {
            r'월\s*[~\-]\s*토': 6,
            r'월\s*[~\-]\s*금': 5,
            r'월\s*[~\-]\s*일': 7,
            r'월요일\s*[~\-]\s*토요일': 6,
            r'월요일\s*[~\-]\s*금요일': 5,
        }
        for pattern, days in day_ranges.items():
            if re.search(pattern, text):
                return days

        return 0

    def _safe_int(self, value: Any) -> int:
        """안전하게 정수 변환"""
        if value is None:
            return 0
        try:
            if isinstance(value, str):
                value = value.replace(",", "").replace("원", "").strip()
            return int(float(value))
        except (ValueError, TypeError):
            return 0

    def _safe_float(self, value: Any) -> float:
        """안전하게 실수 변환"""
        if value is None:
            return 0.0
        try:
            if isinstance(value, str):
                value = value.replace(",", "").strip()
            return float(value)
        except (ValueError, TypeError):
            return 0.0


class LLMClauseAnalyzer:
    """
    LLM 기반 조항 분석기

    사용법:
        analyzer = LLMClauseAnalyzer(crag=crag_instance)
        result = analyzer.analyze(contract_text)
    """

    # 조항 분할 및 값 추출 프롬프트
    CLAUSE_EXTRACTION_PROMPT = """당신은 계약서 분석 전문가입니다.
다음 계약서를 분석하여 각 조항을 구조화된 형식으로 추출하세요.

[계약서]
{contract_text}

[중요 지시사항]
1. 계약서의 모든 조항을 빠짐없이 추출하세요. 하나라도 누락하면 안 됩니다.
2. clause_number는 반드시 계약서 원문에 표기된 조항 번호를 그대로 사용하세요! (예: 원문에 "6. 임금"이면 clause_number는 "6")
3. 하나의 조항에 여러 세부 내용이 있으면 (예: 6. 임금 안에 급여, 지급일, 지급방법이 있을 때) 같은 clause_number로 여러 개 추출하되, clause_type만 다르게 하세요.
4. 조항 유형이 불명확하면 "기타"로 분류하되, 반드시 추출하세요.
5. original_text는 계약서 원문을 글자 그대로 복사하세요. 절대 요약/수정하지 마세요!

[조항 유형 분류 기준]
- 근로시간: 출퇴근 시간, 소정근로시간, 연장근로 관련
- 휴게시간: 휴식시간, 점심시간 관련
- 임금: 기본급, 월급, 시급, 급여 관련
- 수당: 각종 수당, 포괄임금, 제수당 포함 관련
- 연차휴가: 연차, 유급휴가 관련
- 사회보험: 4대보험, 국민연금, 건강보험 관련
- 위약금: 위약금, 손해배상, 벌금, 교육비 반환 관련
- 해지: 계약해지, 해고, 퇴직 관련
- 기타: 위 분류에 해당하지 않는 모든 조항 (반드시 추출!)

[응답 형식 - JSON]
{{
    "clauses": [
        {{
            "clause_number": "1",
            "clause_type": "근로개시일/근무장소/업무내용/근로시간/휴게시간/근무일/휴일/임금/상여금/수당/임금지급일/연차휴가/사회보험/계약서교부/위약금/해지/기타",
            "title": "조항 제목 (없으면 내용 요약)",
            "original_text": "계약서 원문에서 해당 조항 텍스트를 그대로 복사",
            "extracted_values": {{
                "key": "value"
            }}
        }}
    ],
    "contract_metadata": {{
        "employer": "사업주/갑",
        "employee": "근로자/을",
        "contract_date": "계약일",
        "contract_type": "정규직/계약직/기간제/일용직/용역/기타"
    }}
}}

[추출 예시]
- 근로시간: {{"start_time": "09:00", "end_time": "21:00"}}
- 근무일: {{"work_days_per_week": 6}}
- 휴게시간: {{"break_minutes": 60}}
  (주의: "휴게: 없음"이면 break_minutes: 0)
- 임금: {{"base_salary": 3200000, "salary_type": "월급"}}
- 수당: {{"total_allowances": 500000, "meal_allowance": 200000, "is_inclusive": true}}
- 위약금: {{"penalty_amount": 1000000, "penalty_condition": "조기퇴사시"}}

모든 숫자는 정수로 추출하세요.
조항을 하나라도 빠뜨리면 분석에 심각한 오류가 발생합니다!"""

    # 조항별 위반 분석 프롬프트 (법적 기준은 CRAG 검색 결과에서 가져옴)
    VIOLATION_ANALYSIS_PROMPT = """당신은 한국 노동법 전문 변호사입니다.
다음 계약서 조항을 분석하고 법적 위반 여부를 판단하세요.

[분석 대상 조항]
조항 번호: {clause_number}
조항 유형: {clause_type}
원문: {original_text}
추출된 값: {extracted_values}

[관련 법령 - 검색 결과]
{law_context}

[관련 판례 - 검색 결과]
{precedent_context}

[관련 해석례/지침 - 검색 결과]
{interpretation_context}

[위험 패턴 - Graph DB 검색 결과]
{pattern_context}

[분석 지시]
1. 위 법령, 판례, 해석례를 바탕으로 위반 여부 판단
2. 반드시 구체적 법적 근거 (조항 번호)와 함께 설명
3. 관련 판례가 있으면 판례 요지 인용
4. 수정 제안 제시

주요 분석 포인트:
- 근로기준법상 강행규정 위반 여부
- 위약금 예정 금지 (제20조) 위반 여부
- 임금 전액 지급 원칙 (제43조) 위반 여부
- 근로계약서 교부 의무 (제17조) 위반 여부
- 근로기준법에 미달하는 조건의 무효 (제15조)

[응답 형식 - JSON]
{{
    "has_violation": true/false,
    "violations": [
        {{
            "violation_type": "위반 유형을 자연어로 간결하게 작성 (예: 최저임금 미달, 연장근로 초과, 휴게시간 미부여, 위약금 예정 금지 위반, 임금 전액 지급 위반)",
            "severity": "CRITICAL/HIGH/MEDIUM/LOW/INFO",
            "description": "구체적 위반 내용 설명",
            "legal_basis": "법조문만 기재 (예: 근로기준법 제50조)",
            "current_value": "현재 계약서 값",
            "legal_standard": "법적 기준 값 (검색된 법령에서 추출)",
            "suggestion": "구체적인 계약서 수정 방법을 문장으로 작성. 절대 법조문만 적지 마세요!",
            "confidence": 0.0-1.0
        }}
    ],
    "underpayment": {{
        "monthly": 0,
        "annual": 0,
        "calculation": "계산 과정 설명"
    }}
}}

[중요]
- legal_basis와 suggestion은 반드시 다른 내용이어야 합니다!
- legal_basis: 법률명과 조항 번호만 (예: "근로기준법 제20조")
- suggestion: 계약서를 어떻게 수정해야 하는지 구체적 방법 (예: "위약금 조항을 삭제하고, 실제 발생한 손해에 한해 배상 청구 가능하도록 수정")
- description: 반드시 자연어 문장으로 작성! JSON 키-값 형식 (예: "key:value", "field_name:true") 절대 사용 금지. 일반인이 이해할 수 있는 한국어 문장으로 설명하세요. **핵심 키워드나 수치는 마크다운 볼드(**)로 강조**하세요. (예: "**휴게시간 0분**으로 명시되어 있어 **근로기준법 제54조** 위반입니다.")
- current_value: 계약서에 명시된 실제 값을 자연어로 기재 (예: "수습기간 3개월, 4대보험 미가입")

위반이 없으면 violations를 빈 배열로 반환하세요."""

    # 종합 분석 프롬프트 (Cross-clause analysis)
    HOLISTIC_ANALYSIS_PROMPT = """당신은 한국 노동법 전문 변호사입니다.
전체 근로계약서를 종합 분석하여 조항간 연관 분석이 필요한 위반 사항을 찾으세요.

[계약서 요약]
{contract_summary}

[전체 조항 정보]
{all_clauses}

[관련 법령]
{law_context}

[이미 발견된 위반 사항 - 중복 탐지 금지!]
{detected_violations}

위 목록에 이미 포함된 위반 사항은 절대 다시 보고하지 마세요.
종합 분석에서는 개별 조항 분석에서 발견할 수 없는, 여러 조항을 종합해야만 알 수 있는 위반만 보고하세요.

[분석 포인트]
1. 최저임금 위반 여부 (월급여 / 총 근로시간으로 시급 계산)
   - 기본급 + 고정수당 / (월 소정근로시간 + 연장/야간/휴일 근로시간)
   - 2025년 최저시급 기준 적용

2. 주 52시간 초과 여부 (모든 근로시간 합산)

3. 휴일 부족 여부 (주휴일 + 공휴일)

4. 연장근로 가산수당 미지급 여부

5. 포괄임금제 적법성 여부

[응답 형식 - JSON]
{{
    "holistic_violations": [
        {{
            "violation_type": "위반 유형을 자연어로 간결하게 작성 (예: 최저임금 미달, 주 52시간 초과, 포괄임금제 적법성 문제)",
            "severity": "CRITICAL/HIGH/MEDIUM/LOW",
            "description": "구체적 설명을 자연어 문장으로 작성. JSON 키-값 형식 절대 금지!",
            "legal_basis": "법조문만 기재 (예: 근로기준법 제50조)",
            "current_value": "현재 값 (계산 결과)",
            "legal_standard": "법적 기준",
            "suggestion": "구체적인 계약서 수정 방법을 문장으로 작성 (예: '기본급을 3,500,000원 이상으로 인상하고 연장근로수당을 별도 지급하도록 수정'). 절대 법조문만 적지 마세요.",
            "related_clauses": ["관련 조항 번호들"],
            "calculation": "상세 계산 과정"
        }}
    ],
    "minimum_wage_analysis": {{
        "hourly_wage": 0,
        "legal_minimum": 10030,
        "is_violation": true/false,
        "monthly_underpayment": 0,
        "calculation_detail": "계산 과정"
    }}
}}

[중요 주의사항]
- violation_type: 자연어로 간결하게 (예: "최저임금 미달", "주 52시간 초과"). 언더스코어(_) 사용 금지!
- description: 자연어 문장으로 작성. JSON 키-값 형식 (예: "key:value", "field:true") 절대 금지! **핵심 수치와 키워드는 마크다운 볼드(**)로 강조**하세요. (예: "계산된 시급 **8,500원**은 **2025년 최저시급 10,030원**에 미달합니다.")
- current_value: 계약서에 명시된 실제 값을 자연어로 기재 (예: "시급 9,500원", "주 60시간 근무")
- legal_basis: 법률명과 조항 번호만 (예: "근로기준법 제50조")
- suggestion: 계약서를 어떻게 수정해야 하는지 구체적인 방법을 문장으로 작성. 법조문을 복사하면 안 됨! **수정이 필요한 핵심 내용은 볼드로 강조**.
  예시: "**기본급을 3,500,000원 이상으로 인상**하고, **연장근로수당을 별도 지급**하도록 계약서 수정"
"""

    # 텍스트 매칭 프롬프트 (Gemini 2.5 Flash용)
    TEXT_MATCHING_PROMPT = """당신은 계약서 텍스트 분석 전문가입니다.
아래 계약서 전문과 분석된 위반 조항 목록을 보고, 각 위반 조항에 해당하는 계약서 내 **정확한 텍스트 위치**를 찾아주세요.

[계약서 전문]
{contract_text}

[분석된 위반 조항 목록]
{violations_list}

[지시사항]
각 위반 조항에 대해:
1. **exact_text**: 계약서에서 해당 위반과 관련된 정확한 문장/구절을 **글자 그대로** 추출하세요.
   - 이 텍스트는 계약서에서 바로 찾아서 수정안으로 대체할 수 있어야 합니다.
   - 불필요하게 길게 추출하지 마세요. 수정이 필요한 핵심 부분만 추출합니다.
   - 띄어쓰기, 줄바꿈, 특수문자를 정확히 유지하세요.

2. **replacement_text**: 해당 부분을 법적 기준에 맞게 수정한 텍스트를 작성하세요.
   - 원본의 문맥과 형식을 유지하면서 위반 내용만 수정합니다.

[응답 형식 - JSON]
{{
    "matches": [
        {{
            "index": 0,
            "exact_text": "계약서에서 찾은 정확한 원본 텍스트",
            "replacement_text": "법적 기준에 맞게 수정된 텍스트"
        }}
    ]
}}

[중요]
- exact_text는 반드시 계약서 전문에 **그대로** 존재하는 텍스트여야 합니다.
- 요약하거나 수정하지 마세요.
- 찾을 수 없는 경우 해당 인덱스는 응답에서 제외하세요.
"""

    # 동일 조항 위반 병합 프롬프트
    VIOLATION_MERGE_PROMPT = """당신은 한국 노동법 전문 변호사입니다.
동일한 계약서 조항에서 발생한 여러 위반 사항들을 하나로 통합해주세요.

[원본 조항 텍스트]
{original_text}

[위반 사항 목록]
{violations_json}

[통합 지시사항]
1. 여러 위반 사유를 하나의 종합적인 설명으로 통합하세요.
2. 모든 법적 근거를 포함하세요.
3. 수정 제안을 하나로 통합하여, 모든 위반 사항을 해결하는 하나의 수정안을 작성하세요.
4. 수정안(suggested_text)은 원본 조항의 형식을 유지하면서 모든 문제를 해결해야 합니다.
5. 심각도는 가장 높은 것을 사용하세요.

[응답 형식 - JSON]
{{
    "merged_violation": {{
        "violation_type": "통합된 위반 유형 (자연어, 간결하게)",
        "severity": "CRITICAL/HIGH/MEDIUM/LOW (가장 높은 심각도)",
        "description": "모든 위반 사유를 포함한 종합 설명. **핵심 키워드는 볼드**로 강조. 자연어 문장으로 작성.",
        "legal_basis": "모든 관련 법조문 나열 (예: 근로기준법 제54조, 제56조)",
        "current_value": "현재 계약서의 문제점들",
        "legal_standard": "법적 기준들",
        "suggestion": "모든 문제를 해결하는 종합 수정 방법. 구체적으로 작성.",
        "suggested_text": "원본 조항을 대체할 수정된 조항 전문. 모든 위반 사항이 해결된 버전."
    }}
}}

[중요]
- suggested_text는 원본 조항의 문맥과 형식을 유지하면서 모든 위반 사항을 해결해야 합니다.
- 여러 수정안을 하나로 병합하되, 서로 충돌하지 않도록 통합하세요.
"""

    def __init__(
        self,
        crag: Optional[Any] = None,
        llm_client: Optional[Any] = None,
        model: str = None,
        enable_crag: bool = True,
        pipeline_logger: Optional[Any] = None
    ):
        """
        Args:
            crag: GraphGuidedCRAG 인스턴스
            llm_client: OpenAI 클라이언트
            model: 사용할 LLM 모델 (기본값: settings.LLM_REASONING_MODEL)
            enable_crag: CRAG 검색 활성화 여부
            pipeline_logger: PipelineLogger 인스턴스 (상세 로깅용)
        """
        self.crag = crag
        # 모델 기본값: settings에서 가져옴
        self.model = model if model else settings.LLM_CLAUSE_ANALYZER_MODEL
        self.enable_crag = enable_crag
        self.contract_id: Optional[str] = None  # 토큰 추적용
        self.pipeline_logger = pipeline_logger  # 상세 로깅용

        if llm_client is None:
            try:
                from openai import OpenAI
                self.llm_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            except ImportError:
                self.llm_client = None
        else:
            self.llm_client = llm_client

        # Elasticsearch client (for Vector DB search)
        self.es_client = None
        try:
            from elasticsearch import Elasticsearch
            es_host = os.getenv("ELASTICSEARCH_URL", "http://localhost:9200")
            self.es_client = Elasticsearch([es_host])
        except Exception:
            pass

        # Neo4j driver (for Graph DB search) - CRAG에서 가져옴
        self.neo4j_driver = None
        if self.crag and hasattr(self.crag, 'neo4j_driver'):
            self.neo4j_driver = self.crag.neo4j_driver

        # Gemini client for text matching (lazy loading)
        self._gemini_model = None
        self.location_model = settings.LLM_LOCATION_MODEL  # gemini-2.5-flash

        # Gemini safety settings (완전 완화 - 계약서 분석은 합법적 용도)
        self.safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
        ]

        # Neuro-Symbolic 계산기 (LLM 대신 Python으로 정밀 체불액 계산)
        self.neuro_symbolic_calculator = NeuroSymbolicCalculator()

    @property
    def gemini_model(self):
        """Gemini 모델 (lazy loading for text matching)"""
        if self._gemini_model is None:
            try:
                import google.generativeai as genai
                genai.configure(api_key=settings.GEMINI_API_KEY)
                self._gemini_model = genai.GenerativeModel(self.location_model)
            except ImportError:
                print("google-generativeai package not installed")
                self._gemini_model = None
            except Exception as e:
                print(f"Gemini initialization error: {e}")
                self._gemini_model = None
        return self._gemini_model

    def _is_reasoning_model(self) -> bool:
        """reasoning 모델 여부 확인 (temperature 미지원)"""
        reasoning_keywords = ["o1", "o3", "gpt-5"]
        return any(kw in self.model.lower() for kw in reasoning_keywords)

    def _refine_text_matching(
        self,
        contract_text: str,
        violations: List[ClauseViolation]
    ) -> None:
        """
        Gemini 2.5 Flash를 사용하여 위반 조항의 정확한 텍스트 매칭 수행

        Args:
            contract_text: 전체 계약서 텍스트
            violations: 분석된 위반 조항 목록 (in-place 수정)
        """
        if not violations or self.gemini_model is None:
            print(">>> [TEXT_MATCHING] Skipped: no violations or Gemini not available")
            return

        try:
            # 위반 조항 목록 생성 (인덱스 포함)
            violations_list = []
            for i, v in enumerate(violations):
                violations_list.append(
                    f"[{i}] 위반 유형: {v.violation_type}\n"
                    f"    조항 번호: {v.clause.clause_number}\n"
                    f"    설명: {v.description}\n"
                    f"    수정 제안: {v.suggestion}"
                )

            violations_text = "\n\n".join(violations_list)

            # 프롬프트 생성
            prompt = self.TEXT_MATCHING_PROMPT.format(
                contract_text=contract_text[:12000],  # 토큰 제한
                violations_list=violations_text
            )

            # Gemini API 호출
            llm_start = time.time()
            response = self.gemini_model.generate_content(
                prompt,
                generation_config={
                    "temperature": 0.1,
                    "response_mime_type": "application/json"
                },
                safety_settings=self.safety_settings
            )
            llm_duration = (time.time() - llm_start) * 1000

            # 토큰 사용량 기록
            if self.contract_id and hasattr(response, 'usage_metadata'):
                usage = response.usage_metadata
                record_llm_usage(
                    contract_id=self.contract_id,
                    module="clause_analyzer.text_matching",
                    model=self.location_model,
                    input_tokens=getattr(usage, 'prompt_token_count', 0),
                    output_tokens=getattr(usage, 'candidates_token_count', 0),
                    cached_tokens=getattr(usage, 'cached_content_token_count', 0),
                    duration_ms=llm_duration
                )

            # 응답 파싱
            result_text = response.text.strip()
            # JSON 추출 (코드 블록 제거)
            if result_text.startswith("```"):
                result_text = re.sub(r'^```(?:json)?\s*', '', result_text)
                result_text = re.sub(r'\s*```$', '', result_text)

            result = json.loads(result_text)
            matches = result.get("matches", [])

            print(f">>> [TEXT_MATCHING] Found {len(matches)} matches from Gemini")

            # 위반 조항에 매칭 결과 적용
            for match in matches:
                idx = match.get("index")
                exact_text = match.get("exact_text", "")
                replacement_text = match.get("replacement_text", "")

                if idx is None or idx >= len(violations):
                    continue

                # 계약서에서 exact_text가 실제로 존재하는지 검증
                if exact_text and exact_text in contract_text:
                    violations[idx].matched_text = exact_text
                    if replacement_text:
                        violations[idx].suggested_text = replacement_text
                    print(f">>> [TEXT_MATCHING] Matched violation {idx}: '{exact_text[:50]}...'")
                else:
                    # 정확히 일치하지 않으면 fuzzy matching 시도
                    best_match = self._fuzzy_find_text(contract_text, exact_text)
                    if best_match:
                        violations[idx].matched_text = best_match
                        if replacement_text:
                            violations[idx].suggested_text = replacement_text
                        print(f">>> [TEXT_MATCHING] Fuzzy matched violation {idx}: '{best_match[:50]}...'")
                    else:
                        print(f">>> [TEXT_MATCHING] No match found for violation {idx}")

        except json.JSONDecodeError as e:
            print(f">>> [TEXT_MATCHING] JSON parse error: {e}")
        except Exception as e:
            print(f">>> [TEXT_MATCHING] Error: {e}")

    def _fuzzy_find_text(self, contract_text: str, target_text: str, threshold: float = 0.7) -> Optional[str]:
        """
        계약서에서 유사한 텍스트를 찾습니다 (fuzzy matching)

        Args:
            contract_text: 전체 계약서 텍스트
            target_text: 찾으려는 텍스트
            threshold: 최소 유사도 (0-1)

        Returns:
            찾은 텍스트 또는 None
        """
        if not target_text or len(target_text) < 10:
            return None

        target_len = len(target_text)
        best_match = None
        best_ratio = threshold

        # 슬라이딩 윈도우로 유사도 검사
        step = max(1, target_len // 5)
        for i in range(0, len(contract_text) - target_len + 1, step):
            window = contract_text[i:i + target_len]
            ratio = difflib.SequenceMatcher(None, target_text, window).ratio()

            if ratio > best_ratio:
                best_ratio = ratio
                best_match = window

        return best_match

    def _find_text_position(
        self,
        contract_text: str,
        clause_text: str,
        min_ratio: float = 0.5
    ) -> Dict[str, int]:
        """
        계약서 텍스트에서 조항 텍스트의 정확한 위치를 찾습니다.

        Args:
            contract_text: 전체 계약서 텍스트
            clause_text: 찾으려는 조항 텍스트
            min_ratio: 최소 유사도 비율 (fuzzy matching용)

        Returns:
            {"start": 시작 인덱스, "end": 끝 인덱스} 또는 빈 dict
        """
        if not clause_text or not contract_text:
            return {}

        # 마크다운 기호 제거 헬퍼
        def strip_markdown(text: str) -> str:
            text = re.sub(r'^#+\s*', '', text, flags=re.MULTILINE)
            text = re.sub(r'---\s*Page\s*\d+\s*---', '', text)
            text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
            text = re.sub(r'\*([^*]+)\*', r'\1', text)
            return text.strip()

        # 1. 정확한 매칭 시도
        clause_text_clean = clause_text.strip()
        idx = contract_text.find(clause_text_clean)
        if idx != -1:
            return {"start": idx, "end": idx + len(clause_text_clean)}

        # 2. 공백/줄바꿈 정규화 후 매칭
        normalized_contract = re.sub(r'\s+', ' ', contract_text)
        normalized_clause = re.sub(r'\s+', ' ', clause_text_clean)
        idx = normalized_contract.find(normalized_clause)
        if idx != -1:
            ratio = idx / len(normalized_contract)
            approx_start = int(ratio * len(contract_text))
            return {"start": approx_start, "end": approx_start + len(clause_text_clean)}

        # 2.5. 마크다운 제거 후 매칭 (PDF 추출 텍스트에 # 기호가 포함될 수 있음)
        stripped_contract = strip_markdown(contract_text)
        stripped_clause = strip_markdown(clause_text_clean)
        if stripped_clause and len(stripped_clause) > 10:
            idx = stripped_contract.find(stripped_clause)
            if idx != -1:
                ratio = idx / max(1, len(stripped_contract))
                approx_start = int(ratio * len(contract_text))
                return {"start": max(0, approx_start - 5), "end": min(len(contract_text), approx_start + len(clause_text_clean) + 10)}

        # 3. 조항 번호 기반 매칭 (예: "1. 근로개시일", "## 4. 근로시간")
        clause_num_match = re.match(r'^(\d+)\.\s*(.+)', clause_text_clean)
        if clause_num_match:
            num, rest = clause_num_match.groups()
            pattern = rf'#*\s*{num}\.\s*{re.escape(rest[:20])}'
            match = re.search(pattern, contract_text)
            if match:
                start = match.start()
                return {"start": start, "end": min(len(contract_text), start + len(clause_text_clean))}

        # 4. Fuzzy matching으로 가장 유사한 부분 찾기
        clause_len = len(clause_text_clean)
        best_match = {"start": -1, "end": -1, "ratio": 0}

        step = max(1, clause_len // 10)
        for i in range(0, len(contract_text) - clause_len + 1, step):
            window = contract_text[i:i + clause_len]
            ratio = difflib.SequenceMatcher(None, clause_text_clean, window).ratio()
            if ratio > best_match["ratio"]:
                best_match = {"start": i, "end": i + clause_len, "ratio": ratio}

        # 5. 정밀 검색: best match 주변에서 더 정확한 위치 찾기
        if best_match["ratio"] >= min_ratio:
            search_start = max(0, best_match["start"] - step)
            search_end = min(len(contract_text), best_match["end"] + step)

            for i in range(search_start, search_end):
                if i + clause_len > len(contract_text):
                    break
                window = contract_text[i:i + clause_len]
                ratio = difflib.SequenceMatcher(None, clause_text_clean, window).ratio()
                if ratio > best_match["ratio"]:
                    best_match = {"start": i, "end": i + clause_len, "ratio": ratio}

            if best_match["ratio"] >= min_ratio:
                return {"start": best_match["start"], "end": best_match["end"]}

        return {}

    def _search_by_category(
        self,
        query: str,
        doc_type: str,
        limit: int = 3
    ) -> List[Dict[str, Any]]:
        """
        카테고리별 문서 검색 (Vector DB)

        Args:
            query: 검색 쿼리
            doc_type: 문서 타입 (law, precedent, interpretation, manual)
            limit: 검색 결과 수

        Returns:
            검색 결과 리스트
        """
        results = []

        if self.es_client is None:
            return results

        try:
            # Elasticsearch 쿼리 (doc_type 필터링)
            search_body = {
                "query": {
                    "bool": {
                        "must": [
                            {
                                "multi_match": {
                                    "query": query,
                                    "fields": ["text^2", "title", "keywords"],
                                    "type": "best_fields"
                                }
                            }
                        ],
                        "filter": [
                            {
                                "term": {
                                    "doc_type": doc_type
                                }
                            }
                        ]
                    }
                },
                "size": limit,
                "_source": ["text", "source", "doc_type", "title"]
            }

            response = self.es_client.search(
                index="docscanner_chunks",
                body=search_body
            )

            for hit in response.get("hits", {}).get("hits", []):
                source = hit.get("_source", {})
                results.append({
                    "id": hit.get("_id", ""),
                    "text": source.get("text", ""),
                    "source": source.get("source", ""),
                    "doc_type": source.get("doc_type", doc_type),
                    "score": hit.get("_score", 0)
                })

        except Exception as e:
            print(f"Category search error ({doc_type}): {e}")

        return results

    def _get_legal_context_by_categories(
        self,
        query: str
    ) -> Tuple[str, str, str, List[str]]:
        """
        카테고리별로 법률 컨텍스트 검색

        Args:
            query: 검색 쿼리

        Returns:
            Tuple[법령 컨텍스트, 판례 컨텍스트, 해석례 컨텍스트, 출처 목록]
        """
        all_sources = []

        # 1. 법령 검색 (law)
        law_docs = self._search_by_category(query, "law", limit=3)
        if not law_docs and self.crag:
            # CRAG fallback for law
            try:
                crag_result = self.crag.retrieve_and_correct_sync(
                    f"법령 {query}", [], max_graph_hops=1
                )
                law_docs = [
                    {"text": d.text, "source": d.source}
                    for d in crag_result.all_docs[:3]
                    if "법" in d.source or "law" in d.source.lower()
                ]
            except Exception:
                pass

        law_context = self._format_context(law_docs, "법령")
        all_sources.extend([d.get("source", "") for d in law_docs])

        # 2. 판례 검색 (precedent)
        precedent_docs = self._search_by_category(query, "precedent", limit=2)
        if not precedent_docs:
            # source 이름으로 필터링 fallback
            precedent_docs = self._search_by_source_pattern(query, ["판례", "대법원", "precedent"], limit=2)

        precedent_context = self._format_context(precedent_docs, "판례")
        all_sources.extend([d.get("source", "") for d in precedent_docs])

        # 3. 해석례/지침 검색 (interpretation)
        interpretation_docs = self._search_by_category(query, "interpretation", limit=2)
        if not interpretation_docs:
            interpretation_docs = self._search_by_source_pattern(query, ["해석", "지침", "고시", "interpretation"], limit=2)

        interpretation_context = self._format_context(interpretation_docs, "해석례")
        all_sources.extend([d.get("source", "") for d in interpretation_docs])

        return law_context, precedent_context, interpretation_context, all_sources

    def _search_by_source_pattern(
        self,
        query: str,
        patterns: List[str],
        limit: int = 3
    ) -> List[Dict[str, Any]]:
        """source 필드 패턴으로 검색 (fallback)"""
        results = []

        if self.es_client is None:
            return results

        try:
            should_clauses = [
                {"wildcard": {"source": f"*{pattern}*"}}
                for pattern in patterns
            ]

            search_body = {
                "query": {
                    "bool": {
                        "must": [
                            {
                                "multi_match": {
                                    "query": query,
                                    "fields": ["text^2", "title"],
                                    "type": "best_fields"
                                }
                            }
                        ],
                        "should": should_clauses,
                        "minimum_should_match": 1
                    }
                },
                "size": limit,
                "_source": ["text", "source", "doc_type"]
            }

            response = self.es_client.search(
                index="docscanner_chunks",
                body=search_body
            )

            for hit in response.get("hits", {}).get("hits", []):
                source = hit.get("_source", {})
                results.append({
                    "text": source.get("text", ""),
                    "source": source.get("source", ""),
                    "score": hit.get("_score", 0)
                })

        except Exception as e:
            print(f"Source pattern search error: {e}")

        return results

    def _format_context(
        self,
        docs: List[Dict[str, Any]],
        category: str
    ) -> str:
        """검색 결과를 컨텍스트 문자열로 포맷팅"""
        if not docs:
            return f"관련 {category} 검색 결과 없음"

        parts = []
        for i, doc in enumerate(docs, 1):
            source = doc.get("source", "N/A")
            text = doc.get("text", "")[:800]  # 길이 제한
            parts.append(f"[{category} {i}] 출처: {source}\n{text}")

        return "\n\n".join(parts)

    # ========== Graph DB 검색 메서드 (현재 스키마: Document, RiskPattern, Category) ==========

    def _search_graph_risk_patterns(
        self,
        clause_type: str,
        clause_text: str
    ) -> List[Dict[str, Any]]:
        """
        Graph DB에서 위험 패턴 검색

        현재 스키마:
        - RiskPattern (name, explanation, riskLevel, triggers)
        - RiskPattern --[RELATES_TO]--> Document
        - RiskPattern --[IS_A_TYPE_OF]--> ClauseType

        Args:
            clause_type: 조항 유형 (예: "휴게시간", "임금")
            clause_text: 조항 원문

        Returns:
            위험 패턴 및 관련 문서 리스트
        """
        results = []

        if self.neo4j_driver is None:
            return results

        # 위험 키워드 매핑 (triggers 필드와 매칭)
        risk_triggers = {
            "휴게시간": ["휴게", "휴식", "점심"],
            "근로시간": ["근로시간", "연장", "야간", "휴일근로", "52시간"],
            "임금": ["임금", "급여", "최저임금", "시급", "월급", "포괄"],
            "수당": ["수당", "포괄하여", "포함하여", "제수당"],
            "위약금": ["위약금", "손해배상", "벌금", "벌칙", "배상하여야", "반환"],
            "사회보험": ["4대보험", "국민연금", "건강보험", "고용보험", "산재"],
            "연차휴가": ["연차", "휴가", "유급휴일"],
            "해지": ["해고", "해지", "계약해지", "퇴직"],
        }

        triggers = risk_triggers.get(clause_type, [clause_type])

        # 조항 텍스트에서 추가 트리거 추출
        text_triggers = []
        trigger_keywords = ["포괄하여", "포함하여", "위약금", "손해배상", "최저임금", "수습"]
        for kw in trigger_keywords:
            if kw in clause_text:
                text_triggers.append(kw)

        all_triggers = list(set(triggers + text_triggers))

        try:
            with self.neo4j_driver.session() as session:
                # 1. RiskPattern 직접 검색 (triggers 매칭)
                # triggers는 배열 또는 문자열일 수 있음
                query = """
                MATCH (r:RiskPattern)
                WHERE any(trigger IN $triggers WHERE
                    (r.triggers IS NOT NULL AND (
                        any(t IN r.triggers WHERE t CONTAINS trigger)
                        OR toString(r.triggers) CONTAINS trigger
                    ))
                    OR r.name CONTAINS trigger
                )
                OPTIONAL MATCH (r)-[:RELATES_TO]->(d:Document)
                RETURN r.name AS pattern_name,
                       r.explanation AS explanation,
                       r.riskLevel AS risk_level,
                       collect(d.content)[0..1] AS related_docs,
                       collect(d.source)[0..1] AS doc_sources
                LIMIT 3
                """

                result = session.run(query, triggers=all_triggers)
                for record in result:
                    related_doc_text = ""
                    if record['related_docs']:
                        for doc in record['related_docs']:
                            if doc:
                                related_doc_text += doc[:300] + "\n"

                    results.append({
                        "type": "risk_pattern",
                        "text": f"[위험 패턴: {record['pattern_name']}]\n"
                               f"위험도: {record['risk_level']}\n"
                               f"설명: {record['explanation']}\n"
                               f"관련 문서: {related_doc_text[:400] if related_doc_text else '없음'}",
                        "source": f"RiskPattern/{record['pattern_name']}",
                        "score": 0.95
                    })

        except Exception as e:
            print(f"Graph risk pattern search error: {e}")

        return results

    def _search_graph_documents_by_category(
        self,
        clause_type: str,
        keywords: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Graph DB에서 카테고리별 문서 검색

        현재 스키마:
        - Document (id, source, content, category, type)
        - Document --[CATEGORIZED_AS]--> Category
        - Category (name): 근로시간, 임금, 휴일휴가 등

        Args:
            clause_type: 조항 유형
            keywords: 검색 키워드 리스트

        Returns:
            관련 법령/문서 리스트
        """
        results = []

        if self.neo4j_driver is None:
            return results

        # 조항 유형 → 카테고리 매핑
        category_map = {
            "휴게시간": ["근로시간"],
            "근로시간": ["근로시간"],
            "임금": ["임금"],
            "수당": ["임금"],
            "위약금": ["기타", "인사"],
            "연차휴가": ["휴일휴가"],
            "휴일": ["휴일휴가"],
            "사회보험": ["복리후생"],
            "해지": ["인사"],
            "계약서교부": ["채용절차"],
        }

        categories = category_map.get(clause_type, ["기타"])

        try:
            with self.neo4j_driver.session() as session:
                # 카테고리 기반 문서 검색
                query = """
                MATCH (d:Document)-[:CATEGORIZED_AS]->(c:Category)
                WHERE c.name IN $categories
                  AND d.content IS NOT NULL
                  AND (
                    any(kw IN $keywords WHERE d.content CONTAINS kw)
                    OR any(kw IN $keywords WHERE d.source CONTAINS kw)
                  )
                RETURN d.content AS content,
                       d.source AS source,
                       c.name AS category
                LIMIT 3
                """

                result = session.run(query, categories=categories, keywords=keywords)
                for record in result:
                    content = record['content'] or ""
                    results.append({
                        "type": "document",
                        "text": f"[{record['category']}] 출처: {record['source']}\n{content[:500]}",
                        "source": record['source'],
                        "score": 0.85
                    })

        except Exception as e:
            print(f"Graph document search error: {e}")

        return results

    def _search_graph_by_source(
        self,
        clause_type: str,
        keywords: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Graph DB에서 소스별 문서 검색 (법령해석례, 매뉴얼 등)

        Args:
            clause_type: 조항 유형
            keywords: 검색 키워드

        Returns:
            관련 문서 리스트
        """
        results = []

        if self.neo4j_driver is None:
            return results

        try:
            with self.neo4j_driver.session() as session:
                # 키워드 기반 문서 검색
                query = """
                MATCH (d:Document)
                WHERE d.content IS NOT NULL
                  AND any(kw IN $keywords WHERE d.content CONTAINS kw)
                RETURN d.content AS content,
                       d.source AS source,
                       d.category AS category
                ORDER BY size(d.content) DESC
                LIMIT 2
                """

                result = session.run(query, keywords=keywords)
                for record in result:
                    content = record['content'] or ""
                    results.append({
                        "type": "reference",
                        "text": f"[참조문서] 출처: {record['source']}\n{content[:500]}",
                        "source": record['source'],
                        "score": 0.8
                    })

        except Exception as e:
            print(f"Graph source search error: {e}")

        return results

    # ========== 하이브리드 검색 (Vector + Graph) ==========

    def _get_hybrid_legal_context(
        self,
        clause_type: str,
        clause_text: str
    ) -> Tuple[str, str, str, str, List[str]]:
        """
        하이브리드 검색: Vector DB + Graph DB 결합

        현재 스키마:
        - Vector DB (ES): doc_type 기반 필터링 (law, precedent, interpretation)
        - Graph DB (Neo4j): RiskPattern, Document, Category 노드

        Args:
            clause_type: 조항 유형
            clause_text: 조항 원문

        Returns:
            Tuple[법령, 판례, 해석례, 위험패턴, 출처목록]
        """
        all_sources = []
        query = f"{clause_type} {clause_text[:200]}"

        # 키워드 추출
        keywords = [clause_type]
        keyword_map = {
            "시간": ["시간", "근로시간", "휴게"],
            "임금": ["임금", "급여", "월급", "시급"],
            "휴게": ["휴게", "휴식", "점심"],
            "연장": ["연장", "야간", "휴일근로"],
            "위약": ["위약금", "손해배상", "벌금"],
            "포괄": ["포괄", "포함하여", "제수당"],
            "보험": ["4대보험", "사회보험", "국민연금"],
        }
        for key, vals in keyword_map.items():
            if key in clause_text:
                keywords.extend(vals)
        keywords = list(set(keywords))

        # ========== 1. Vector DB 검색 (의미적 유사도) ==========
        law_docs = self._search_by_category(query, "law", limit=2)
        precedent_docs = self._search_by_category(query, "precedent", limit=2)
        interpretation_docs = self._search_by_category(query, "interpretation", limit=2)

        # ========== 2. Graph DB 검색 (구조적 관계) ==========

        # 2-1. 위험 패턴 검색 (RiskPattern 노드)
        graph_risk_patterns = self._search_graph_risk_patterns(clause_type, clause_text)

        # 2-2. 카테고리별 문서 검색 (Document → Category)
        graph_docs = self._search_graph_documents_by_category(clause_type, keywords)

        # 2-3. 키워드 기반 참조 문서 검색
        graph_refs = self._search_graph_by_source(clause_type, keywords)

        # ========== 3. 결과 병합 및 랭킹 ==========

        # 법령: Vector DB + Graph DB 문서 병합
        all_laws = law_docs + graph_docs
        seen_law_sources = set()
        unique_laws = []
        for doc in all_laws:
            source = doc.get("source", "")
            if source and source not in seen_law_sources:
                seen_law_sources.add(source)
                unique_laws.append(doc)
        unique_laws = sorted(unique_laws, key=lambda x: x.get("score", 0), reverse=True)[:3]

        # 판례/참조: Vector DB + Graph 참조문서
        all_precedents = precedent_docs + graph_refs
        seen_precedent = set()
        unique_precedents = []
        for doc in all_precedents:
            source = doc.get("source", "")
            if source and source not in seen_precedent:
                seen_precedent.add(source)
                unique_precedents.append(doc)
        unique_precedents = sorted(unique_precedents, key=lambda x: x.get("score", 0), reverse=True)[:2]

        # 해석례: Vector DB 결과
        unique_interpretations = interpretation_docs[:2]

        # 위험 패턴: Graph DB RiskPattern 노드
        unique_patterns = graph_risk_patterns[:2]

        # ========== 4. 컨텍스트 생성 ==========
        law_context = self._format_context(unique_laws, "법령")
        precedent_context = self._format_context(unique_precedents, "판례/참조")
        interpretation_context = self._format_context(unique_interpretations, "해석례")
        pattern_context = self._format_context(unique_patterns, "위험패턴") if unique_patterns else ""

        # 출처 수집
        for docs in [unique_laws, unique_precedents, unique_interpretations, unique_patterns]:
            all_sources.extend([d.get("source", "") for d in docs if d.get("source")])

        return law_context, precedent_context, interpretation_context, pattern_context, all_sources

    def analyze(self, contract_text: str) -> ClauseAnalysisResult:
        """
        계약서 전체 분석

        Args:
            contract_text: 계약서 텍스트

        Returns:
            ClauseAnalysisResult: 분석 결과
        """
        import time
        from concurrent.futures import ThreadPoolExecutor, as_completed
        start_time = time.time()

        result = ClauseAnalysisResult()

        # 1. LLM으로 조항 분할 및 값 추출
        clauses = self._extract_clauses(contract_text)
        result.clauses = clauses

        # 2. 각 조항별 분석 (병렬 처리)
        total_monthly_underpayment = 0

        # 모든 조항 분석 (LLM이 추출한 조항은 모두 분석 대상)
        clauses_to_analyze = clauses

        # 병렬 처리로 개별 조항 분석
        if clauses_to_analyze:
            with ThreadPoolExecutor(max_workers=min(len(clauses_to_analyze), 5)) as executor:
                future_to_clause = {
                    executor.submit(self._analyze_clause, clause): clause
                    for clause in clauses_to_analyze
                }

                for future in as_completed(future_to_clause):
                    try:
                        violations, underpayment = future.result()
                        result.violations.extend(violations)
                        total_monthly_underpayment += underpayment
                    except Exception as e:
                        print(f"Clause analysis error: {e}")

        # 3. 종합 분석 (Cross-clause analysis) - 순차 처리 필수
        # 최저임금 계산, 주간 근무시간 검증 등 모든 조항 정보 필요
        # 이미 발견된 위반 사항을 전달하여 중복 탐지 방지
        holistic_violations, holistic_underpayment = self._holistic_analysis(
            clauses, contract_text, detected_violations=result.violations
        )
        result.violations.extend(holistic_violations)
        total_monthly_underpayment += holistic_underpayment

        # 3.5. 동일 조항 위반 병합 (같은 조항에서 여러 위반이 발생한 경우 하나로 통합)
        if len(result.violations) > 1:
            result.violations = self._merge_same_clause_violations(result.violations)

        # 4. Neuro-Symbolic 체불액 계산 (LLM 계산 대신 Python으로 정밀 계산)
        # LLM이 추출한 숫자를 기반으로 Python으로 정확하게 계산
        contract_data = self.neuro_symbolic_calculator.extract_contract_data(clauses)
        symbolic_result = self.neuro_symbolic_calculator.calculate_underpayment(contract_data)

        # Symbolic 계산 결과 사용 (LLM 계산 결과는 무시)
        result.total_underpayment = symbolic_result.total_monthly_underpayment
        result.annual_underpayment = symbolic_result.total_monthly_underpayment * 12

        # 계산 상세 내역 로깅
        print(f">>> [NEURO-SYMBOLIC] Calculation breakdown: {symbolic_result.calculation_breakdown}")

        # 5. Gemini 2.5 Flash로 정확한 텍스트 매칭 (하이라이팅용)
        if result.violations:
            self._refine_text_matching(contract_text, result.violations)

        result.processing_time = time.time() - start_time

        return result

    def _extract_clauses(self, contract_text: str) -> List[ExtractedClause]:
        """LLM으로 조항 추출"""
        if self.llm_client is None:
            return self._fallback_extract_clauses(contract_text)

        try:
            prompt = self.CLAUSE_EXTRACTION_PROMPT.format(
                contract_text=contract_text[:8000]
            )

            llm_start = time.time()
            if self._is_reasoning_model():
                response = self.llm_client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_object"}
                )
            else:
                response = self.llm_client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": "당신은 한국 근로계약서 분석 전문가입니다. JSON 형식으로 응답하세요."
                        },
                        {"role": "user", "content": prompt}
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.1
                )
            llm_duration = (time.time() - llm_start) * 1000

            # 토큰 사용량 기록
            if response.usage and self.contract_id:
                # cached_tokens 안전하게 추출 (PromptTokensDetails 객체에서)
                cached = 0
                if hasattr(response.usage, 'prompt_tokens_details') and response.usage.prompt_tokens_details:
                    cached = getattr(response.usage.prompt_tokens_details, 'cached_tokens', 0) or 0
                record_llm_usage(
                    contract_id=self.contract_id,
                    module="clause_analyzer.extract_clauses",
                    model=self.model,
                    input_tokens=response.usage.prompt_tokens,
                    output_tokens=response.usage.completion_tokens,
                    cached_tokens=cached,
                    duration_ms=llm_duration
                )

            response_content = response.choices[0].message.content
            result = json.loads(response_content)

            # 상세 로깅: LLM 프롬프트/응답 전문
            if self.pipeline_logger:
                self.pipeline_logger.log_llm_call(
                    step_name="ClauseExtraction",
                    model=self.model,
                    prompt=prompt,
                    response=response_content,
                    temperature=0.1 if not self._is_reasoning_model() else 0.0,
                    duration_ms=llm_duration,
                    extra={"clause_count": len(result.get("clauses", []))}
                )

            clauses = []

            for c in result.get("clauses", []):
                clause_type = self._map_clause_type(c.get("clause_type", "기타"))
                original_text = c.get("original_text", "")

                # 원본 텍스트에서 조항 위치 계산
                position = self._find_text_position(contract_text, original_text)

                clauses.append(ExtractedClause(
                    clause_number=str(c.get("clause_number", "")),
                    clause_type=clause_type,
                    title=c.get("title", ""),
                    original_text=original_text,
                    extracted_values=c.get("extracted_values", {}),
                    position=position
                ))

            return clauses

        except Exception as e:
            print(f"Clause extraction error: {e}")
            return self._fallback_extract_clauses(contract_text)

    def _map_clause_type(self, type_str: str) -> ClauseType:
        """문자열을 ClauseType으로 변환"""
        type_map = {
            "근로개시일": ClauseType.WORK_START_DATE,
            "근무장소": ClauseType.WORKPLACE,
            "업무내용": ClauseType.JOB_DESCRIPTION,
            "근로시간": ClauseType.WORK_HOURS,
            "휴게시간": ClauseType.BREAK_TIME,
            "근무일": ClauseType.WORK_DAYS,
            "휴일": ClauseType.HOLIDAYS,
            "임금": ClauseType.SALARY,
            "상여금": ClauseType.BONUS,
            "수당": ClauseType.ALLOWANCES,
            "임금지급일": ClauseType.PAYMENT_DATE,
            "연차휴가": ClauseType.ANNUAL_LEAVE,
            "사회보험": ClauseType.SOCIAL_INSURANCE,
            "계약서교부": ClauseType.CONTRACT_DELIVERY,
            "위약금": ClauseType.PENALTY,
            "해지": ClauseType.TERMINATION,
        }
        return type_map.get(type_str, ClauseType.OTHER)

    def _fallback_extract_clauses(self, contract_text: str) -> List[ExtractedClause]:
        """폴백: 정규식 기반 조항 추출"""
        clauses = []

        # 번호 패턴으로 분할
        patterns = [
            r'(\d{1,2})\s*\.\s*([^\n:]+)\s*[:：]\s*([^\n]+(?:\n(?!\d{1,2}\s*\.)[^\n]+)*)',
            r'제\s*(\d+)\s*조\s*\(([^)]+)\)\s*([^\n]+(?:\n(?!제\s*\d+\s*조)[^\n]+)*)',
        ]

        for pattern in patterns:
            matches = re.finditer(pattern, contract_text, re.MULTILINE)
            for match in matches:
                clause_num = match.group(1)
                title = match.group(2).strip()
                content = match.group(3).strip() if len(match.groups()) > 2 else ""

                clause_type = self._infer_clause_type(title)

                # 정규식 매치에서 정확한 위치 추출
                position = {"start": match.start(), "end": match.end()}

                clauses.append(ExtractedClause(
                    clause_number=clause_num,
                    clause_type=clause_type,
                    title=title,
                    original_text=f"{title}: {content}",
                    extracted_values=self._extract_values_regex(content, clause_type),
                    position=position
                ))

            if clauses:
                break

        return clauses

    def _infer_clause_type(self, title: str) -> ClauseType:
        """제목에서 조항 유형 추론"""
        title_lower = title.lower()

        if "근로시간" in title or "소정근로" in title:
            return ClauseType.WORK_HOURS
        elif "휴게" in title:
            return ClauseType.BREAK_TIME
        elif "임금" in title or "급여" in title:
            return ClauseType.SALARY
        elif "수당" in title:
            return ClauseType.ALLOWANCES
        elif "연차" in title or "휴가" in title:
            return ClauseType.ANNUAL_LEAVE
        elif "보험" in title or "4대" in title:
            return ClauseType.SOCIAL_INSURANCE
        elif "위약" in title or "손해배상" in title:
            return ClauseType.PENALTY
        elif "해지" in title or "해고" in title:
            return ClauseType.TERMINATION
        elif "근무일" in title or "휴일" in title:
            return ClauseType.WORK_DAYS
        elif "개시" in title or "시작" in title:
            return ClauseType.WORK_START_DATE
        elif "장소" in title:
            return ClauseType.WORKPLACE
        elif "업무" in title:
            return ClauseType.JOB_DESCRIPTION
        else:
            return ClauseType.OTHER

    def _extract_values_regex(self, content: str, clause_type: ClauseType) -> Dict[str, Any]:
        """정규식으로 값 추출 (폴백)"""
        values = {}

        if clause_type == ClauseType.WORK_HOURS:
            # 시간 추출
            time_match = re.search(
                r'(\d{1,2})\s*시\s*(\d{2})?\s*분?\s*[~\-]\s*(\d{1,2})\s*시\s*(\d{2})?\s*분?',
                content
            )
            if time_match:
                start_h = int(time_match.group(1))
                end_h = int(time_match.group(3))
                values["start_time"] = f"{start_h:02d}:00"
                values["end_time"] = f"{end_h:02d}:00"
                values["daily_hours"] = end_h - start_h

        elif clause_type == ClauseType.BREAK_TIME:
            # 휴게시간 추출
            if "없음" in content or "0분" in content:
                values["break_minutes"] = 0
            else:
                break_match = re.search(r'(\d+)\s*(분|시간)', content)
                if break_match:
                    minutes = int(break_match.group(1))
                    if "시간" in break_match.group(2):
                        minutes *= 60
                    values["break_minutes"] = minutes

        elif clause_type == ClauseType.SALARY:
            # 급여 추출
            salary_match = re.search(r'(\d{1,3}(?:,\d{3})*)\s*원', content)
            if salary_match:
                values["base_salary"] = int(salary_match.group(1).replace(",", ""))

        return values

    def _analyze_clause(
        self,
        clause: ExtractedClause
    ) -> Tuple[List[ClauseViolation], int]:
        """
        개별 조항 분석

        Returns:
            Tuple[List[ClauseViolation], int]: (위반 목록, 월간 체불액)
        """
        violations = []
        monthly_underpayment = 0

        # 1. 하이브리드 검색: Vector DB + Graph DB 결합
        law_context, precedent_context, interpretation_context, pattern_context, crag_sources = \
            self._get_hybrid_legal_context(clause.clause_type.value, clause.original_text)

        # 2. LLM으로 위반 분석
        if self.llm_client is not None:
            try:
                prompt = self.VIOLATION_ANALYSIS_PROMPT.format(
                    clause_number=clause.clause_number,
                    clause_type=clause.clause_type.value,
                    original_text=clause.original_text,
                    extracted_values=json.dumps(clause.extracted_values, ensure_ascii=False),
                    law_context=law_context,
                    precedent_context=precedent_context,
                    interpretation_context=interpretation_context,
                    pattern_context=pattern_context if pattern_context else "관련 위험 패턴 없음"
                )

                llm_start = time.time()
                if self._is_reasoning_model():
                    response = self.llm_client.chat.completions.create(
                        model=self.model,
                        messages=[{"role": "user", "content": prompt}],
                        response_format={"type": "json_object"}
                    )
                else:
                    response = self.llm_client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {
                                "role": "system",
                                "content": "당신은 한국 노동법 전문 변호사입니다. 엄격하게 법적 기준을 적용하여 분석하세요."
                            },
                            {"role": "user", "content": prompt}
                        ],
                        response_format={"type": "json_object"},
                        temperature=0.1
                    )
                llm_duration = (time.time() - llm_start) * 1000

                # 토큰 사용량 기록
                if response.usage and self.contract_id:
                    cached = 0
                    if hasattr(response.usage, 'prompt_tokens_details') and response.usage.prompt_tokens_details:
                        cached = getattr(response.usage.prompt_tokens_details, 'cached_tokens', 0) or 0
                    record_llm_usage(
                        contract_id=self.contract_id,
                        module=f"clause_analyzer.analyze_clause_{clause.clause_type.value}",
                        model=self.model,
                        input_tokens=response.usage.prompt_tokens,
                        output_tokens=response.usage.completion_tokens,
                        cached_tokens=cached,
                        duration_ms=llm_duration
                    )

                response_content = response.choices[0].message.content
                result = json.loads(response_content)

                # 상세 로깅: 조항별 위반 분석 LLM 호출
                if self.pipeline_logger:
                    self.pipeline_logger.log_llm_call(
                        step_name=f"ViolationAnalysis_{clause.clause_number}",
                        model=self.model,
                        prompt=prompt,
                        response=response_content,
                        temperature=0.1 if not self._is_reasoning_model() else 0.0,
                        duration_ms=llm_duration,
                        extra={
                            "clause_type": clause.clause_type.value,
                            "has_violation": result.get("has_violation", False),
                            "violation_count": len(result.get("violations", []))
                        }
                    )
                    # 검색된 법률 컨텍스트도 로깅
                    self.pipeline_logger.log_retrieval(
                        step_name=f"LegalContext_{clause.clause_number}",
                        query=clause.original_text[:200],
                        results=[
                            {"type": "law", "content": law_context},
                            {"type": "precedent", "content": precedent_context},
                            {"type": "interpretation", "content": interpretation_context},
                            {"type": "pattern", "content": pattern_context or "없음"}
                        ],
                        retrieval_type="hybrid",
                        extra={"crag_sources": crag_sources}
                    )

                if result.get("has_violation", False):
                    for v in result.get("violations", []):
                        severity = ViolationSeverity[v.get("severity", "MEDIUM")]
                        violations.append(ClauseViolation(
                            clause=clause,
                            violation_type=v.get("violation_type", ""),
                            severity=severity,
                            description=v.get("description", ""),
                            legal_basis=v.get("legal_basis", ""),
                            current_value=v.get("current_value"),
                            legal_standard=v.get("legal_standard"),
                            suggestion=v.get("suggestion", ""),
                            suggested_text=v.get("suggested_text", ""),
                            crag_sources=crag_sources,
                            confidence=v.get("confidence", 0.8)
                        ))

                # 체불액 추출
                underpayment = result.get("underpayment", {})
                raw_monthly = underpayment.get("monthly") or 0
                monthly_underpayment = int(raw_monthly) if raw_monthly else 0

            except Exception as e:
                print(f"Violation analysis error: {e}")
                # 폴백: 규칙 기반 분석
                fallback_violations = self._rule_based_analysis(clause)
                violations.extend(fallback_violations)

        else:
            # LLM 없으면 규칙 기반
            violations = self._rule_based_analysis(clause)

        return violations, monthly_underpayment

    def _contains_risk_keywords(self, text: str) -> bool:
        """위험 키워드 포함 여부 확인"""
        risk_keywords = [
            "위약금", "손해배상", "벌금", "벌칙", "감봉",
            "미가입", "제외", "적용하지", "포기",
            "야간", "연장", "휴일근로", "초과근무",
            "지급하지", "공제", "차감",
            "해고", "해지", "계약해지",
            "퇴직금", "미지급"
        ]
        return any(kw in text for kw in risk_keywords)

    def _merge_same_clause_violations(
        self,
        violations: List[ClauseViolation]
    ) -> List[ClauseViolation]:
        """
        동일 조항에서 발생한 여러 위반 사항을 하나로 병합

        같은 clause_number를 가진 위반이 2개 이상이면 Gemini Flash로 병합
        """
        if not violations:
            return violations

        # 1. 조항 번호별로 그룹화
        from collections import defaultdict
        clause_groups: Dict[str, List[ClauseViolation]] = defaultdict(list)
        for v in violations:
            clause_groups[v.clause.clause_number].append(v)

        merged_violations = []

        for clause_num, group in clause_groups.items():
            if len(group) == 1:
                # 단일 위반은 그대로 유지
                merged_violations.append(group[0])
            else:
                # 2개 이상이면 병합
                print(f">>> [MERGE] Merging {len(group)} violations for clause {clause_num}")
                merged = self._merge_violations_with_llm(group)
                if merged:
                    merged_violations.append(merged)
                else:
                    # 병합 실패 시 첫 번째 것만 사용
                    merged_violations.append(group[0])

        return merged_violations

    def _merge_violations_with_llm(
        self,
        violations: List[ClauseViolation]
    ) -> Optional[ClauseViolation]:
        """
        Gemini Flash를 사용하여 여러 위반을 하나로 병합
        """
        if not violations:
            return None

        # Gemini 클라이언트 사용 (self.gemini_model - lazy loading)
        if self.gemini_model is None:
            print(f">>> [MERGE] Gemini not available")
            return None

        # 위반 정보를 JSON으로 변환
        violations_data = []
        for v in violations:
            violations_data.append({
                "violation_type": v.violation_type,
                "severity": v.severity.value,
                "description": v.description,
                "legal_basis": v.legal_basis,
                "current_value": v.current_value,
                "legal_standard": v.legal_standard,
                "suggestion": v.suggestion,
                "suggested_text": v.suggested_text
            })

        # 원본 조항 텍스트 (첫 번째 위반에서 가져옴)
        original_clause = violations[0].clause
        original_text = original_clause.original_text

        prompt = self.VIOLATION_MERGE_PROMPT.format(
            original_text=original_text,
            violations_json=json.dumps(violations_data, ensure_ascii=False, indent=2)
        )

        llm_start = time.time()
        try:
            response = self.gemini_model.generate_content(
                prompt,
                generation_config={
                    "temperature": 0.2,
                    "response_mime_type": "application/json"
                },
                safety_settings=self.safety_settings
            )
            llm_duration = (time.time() - llm_start) * 1000

            # 토큰 사용량 기록
            if self.contract_id and hasattr(response, 'usage_metadata'):
                usage = response.usage_metadata
                record_llm_usage(
                    contract_id=self.contract_id,
                    module="clause_analyzer.merge_violations",
                    model=self.location_model,
                    input_tokens=getattr(usage, 'prompt_token_count', 0),
                    output_tokens=getattr(usage, 'candidates_token_count', 0),
                    cached_tokens=getattr(usage, 'cached_content_token_count', 0),
                    duration_ms=llm_duration
                )

            # JSON 파싱
            result_text = response.text.strip()
            if result_text.startswith("```"):
                result_text = re.sub(r'^```(?:json)?\s*', '', result_text)
                result_text = re.sub(r'\s*```$', '', result_text)

            result = json.loads(result_text)
            merged_data = result.get("merged_violation", {})

            # 병합된 위반 객체 생성
            severity = ViolationSeverity[merged_data.get("severity", "HIGH")]

            # crag_sources 통합 (모든 위반의 sources 합침)
            all_sources = []
            for v in violations:
                all_sources.extend(v.crag_sources)
            unique_sources = list(set(all_sources))

            merged_violation = ClauseViolation(
                clause=original_clause,
                violation_type=merged_data.get("violation_type", violations[0].violation_type),
                severity=severity,
                description=merged_data.get("description", ""),
                legal_basis=merged_data.get("legal_basis", ""),
                current_value=merged_data.get("current_value", ""),
                legal_standard=merged_data.get("legal_standard", ""),
                suggestion=merged_data.get("suggestion", ""),
                suggested_text=merged_data.get("suggested_text", ""),
                crag_sources=unique_sources,
                confidence=max(v.confidence for v in violations)  # 가장 높은 신뢰도 사용
            )

            print(f">>> [MERGE] Successfully merged {len(violations)} violations")
            return merged_violation

        except Exception as e:
            print(f">>> [MERGE] LLM merge failed: {e}")
            return None

    def _holistic_analysis(
        self,
        clauses: List[ExtractedClause],
        contract_text: str,
        detected_violations: List[ClauseViolation] = None
    ) -> Tuple[List[ClauseViolation], int]:
        """
        종합 분석 (Cross-clause analysis)
        - 최저임금 계산 (임금 / 근로시간)
        - 주 52시간 초과 여부
        - 포괄임금제 적법성

        Args:
            clauses: 추출된 조항 목록
            contract_text: 계약서 전문
            detected_violations: 이미 발견된 위반 사항 (중복 방지용)

        Returns:
            Tuple[List[ClauseViolation], int]: (위반 목록, 월간 체불액)
        """
        violations = []
        monthly_underpayment = 0

        if self.llm_client is None:
            return violations, monthly_underpayment

        # 1. 조항 정보 요약 생성
        clause_summaries = []
        for c in clauses:
            clause_summaries.append({
                "number": c.clause_number,
                "type": c.clause_type.value,
                "title": c.title,
                "values": c.extracted_values
            })

        all_clauses_json = json.dumps(clause_summaries, ensure_ascii=False, indent=2)

        # 2. 계약서 핵심 정보 요약
        contract_summary = self._extract_contract_summary(clauses)

        # 3. 법령 검색 (최저임금, 근로시간 관련)
        law_docs = self._search_by_category("최저임금 근로시간 주 52시간 포괄임금", "law", limit=3)
        law_context = self._format_context(law_docs, "법령")

        # 3.5. 이미 발견된 위반 사항 요약 (중복 방지용)
        detected_violations_text = "없음"
        if detected_violations:
            violation_summaries = []
            for v in detected_violations:
                violation_summaries.append(
                    f"- 조항 {v.clause.clause_number}: {v.violation_type} ({v.legal_basis})"
                )
            detected_violations_text = "\n".join(violation_summaries)

        # 4. LLM 종합 분석
        try:
            prompt = self.HOLISTIC_ANALYSIS_PROMPT.format(
                contract_summary=contract_summary,
                all_clauses=all_clauses_json,
                law_context=law_context,
                detected_violations=detected_violations_text
            )

            llm_start = time.time()
            if self._is_reasoning_model():
                response = self.llm_client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_object"}
                )
            else:
                response = self.llm_client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": "당신은 한국 노동법 전문 변호사입니다. 계약서 전체를 종합 분석하세요."
                        },
                        {"role": "user", "content": prompt}
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.1
                )
            llm_duration = (time.time() - llm_start) * 1000

            # 토큰 사용량 기록
            if response.usage and self.contract_id:
                cached = 0
                if hasattr(response.usage, 'prompt_tokens_details') and response.usage.prompt_tokens_details:
                    cached = getattr(response.usage.prompt_tokens_details, 'cached_tokens', 0) or 0
                record_llm_usage(
                    contract_id=self.contract_id,
                    module="clause_analyzer.holistic_analysis",
                    model=self.model,
                    input_tokens=response.usage.prompt_tokens,
                    output_tokens=response.usage.completion_tokens,
                    cached_tokens=cached,
                    duration_ms=llm_duration
                )

            response_content = response.choices[0].message.content
            result = json.loads(response_content)

            # 상세 로깅: 종합 분석 LLM 호출
            if self.pipeline_logger:
                self.pipeline_logger.log_llm_call(
                    step_name="HolisticAnalysis",
                    model=self.model,
                    prompt=prompt,
                    response=response_content,
                    temperature=0.1 if not self._is_reasoning_model() else 0.0,
                    duration_ms=llm_duration,
                    extra={
                        "violation_count": len(result.get("holistic_violations", [])),
                        "min_wage_violation": result.get("minimum_wage_analysis", {}).get("is_violation", False)
                    }
                )

            # 종합 위반 사항 처리
            for v in result.get("holistic_violations", []):
                severity = ViolationSeverity[v.get("severity", "MEDIUM")]

                # related_clauses에서 위치 정보 가져오기
                related_clause_numbers = v.get("related_clauses", [])
                position = {}
                original_text_for_clause = v.get("description", "")

                # related_clauses의 첫 번째 조항에서 위치 정보 추출
                if related_clause_numbers:
                    for clause_num in related_clause_numbers:
                        # clauses 리스트에서 해당 조항 찾기
                        matching_clause = next(
                            (c for c in clauses if c.clause_number == str(clause_num)),
                            None
                        )
                        if matching_clause and matching_clause.position:
                            position = matching_clause.position
                            original_text_for_clause = matching_clause.original_text
                            break

                # 위치를 찾지 못한 경우 텍스트 매칭 시도
                if not position and original_text_for_clause:
                    position = self._find_text_position(
                        contract_text,
                        original_text_for_clause,
                        min_ratio=0.4
                    )

                # 종합 분석용 조항 생성 (위치 정보 포함)
                dummy_clause = ExtractedClause(
                    clause_number="종합",
                    clause_type=ClauseType.OTHER,
                    title="종합 분석",
                    original_text=original_text_for_clause,
                    extracted_values={},
                    position=position
                )

                violations.append(ClauseViolation(
                    clause=dummy_clause,
                    violation_type=v.get("violation_type", ""),
                    severity=severity,
                    description=v.get("description", ""),
                    legal_basis=v.get("legal_basis", ""),
                    current_value=v.get("current_value"),
                    legal_standard=v.get("legal_standard"),
                    suggestion=v.get("suggestion", ""),
                    suggested_text=v.get("suggested_text", ""),
                    crag_sources=[d.get("source", "") for d in law_docs],
                    confidence=0.85
                ))

            # 최저임금 분석 결과에서 체불액 추출
            min_wage = result.get("minimum_wage_analysis", {})
            if min_wage.get("is_violation", False):
                raw_underpayment = min_wage.get("monthly_underpayment") or 0
                monthly_underpayment = int(raw_underpayment) if raw_underpayment else 0

        except Exception as e:
            print(f"Holistic analysis error: {e}")

        return violations, monthly_underpayment

    def _extract_contract_summary(self, clauses: List[ExtractedClause]) -> str:
        """조항들에서 핵심 정보 요약 추출"""
        summary_parts = []

        for c in clauses:
            if c.clause_type == ClauseType.SALARY:
                vals = c.extracted_values
                if vals:
                    summary_parts.append(f"임금: {vals}")

            elif c.clause_type == ClauseType.WORK_HOURS:
                vals = c.extracted_values
                if vals:
                    summary_parts.append(f"근로시간: {vals}")

            elif c.clause_type == ClauseType.BREAK_TIME:
                vals = c.extracted_values
                if vals:
                    summary_parts.append(f"휴게시간: {vals}")

            elif c.clause_type == ClauseType.ALLOWANCES:
                vals = c.extracted_values
                if vals:
                    summary_parts.append(f"수당: {vals}")

            elif c.clause_type == ClauseType.WORK_DAYS:
                vals = c.extracted_values
                if vals:
                    summary_parts.append(f"근무일: {vals}")

        if not summary_parts:
            return "조항별 핵심 정보 추출 실패"

        return "\n".join(summary_parts)

    def _rule_based_analysis(self, clause: ExtractedClause) -> List[ClauseViolation]:
        """규칙 기반 폴백 분석"""
        violations = []
        values = clause.extracted_values

        if clause.clause_type == ClauseType.WORK_HOURS:
            daily_hours = values.get("daily_hours", 0)
            if daily_hours > 8:
                excess = daily_hours - 8
                violations.append(ClauseViolation(
                    clause=clause,
                    violation_type="과도한_근로시간",
                    severity=ViolationSeverity.HIGH,
                    description=f"1일 {daily_hours}시간 근무는 법정 8시간을 {excess}시간 초과",
                    legal_basis="근로기준법 제50조",
                    current_value=f"{daily_hours}시간",
                    legal_standard="8시간",
                    suggestion=f"1일 근로시간을 8시간으로 조정 (연장근로는 별도 합의 필요)",
                    confidence=0.9
                ))

        elif clause.clause_type == ClauseType.BREAK_TIME:
            break_minutes = values.get("break_minutes", -1)
            daily_hours = 8  # 기본값

            if break_minutes == 0:
                violations.append(ClauseViolation(
                    clause=clause,
                    violation_type="휴게시간_미부여",
                    severity=ViolationSeverity.HIGH,
                    description="8시간 이상 근무 시 1시간 이상 휴게시간 부여 필수",
                    legal_basis="근로기준법 제54조",
                    current_value="휴게 없음",
                    legal_standard="1시간 이상",
                    suggestion="휴게시간 1시간 이상 명시 (예: 12:00~13:00)",
                    confidence=0.95
                ))

        elif clause.clause_type == ClauseType.PENALTY:
            if "위약금" in clause.original_text or "손해배상" in clause.original_text:
                violations.append(ClauseViolation(
                    clause=clause,
                    violation_type="위약금_예정_금지",
                    severity=ViolationSeverity.HIGH,
                    description="근로계약 불이행에 대한 위약금 예정은 금지됨",
                    legal_basis="근로기준법 제20조",
                    current_value=clause.original_text[:100],
                    legal_standard="위약금 예정 금지",
                    suggestion="해당 조항 삭제 (실손해 배상만 가능)",
                    confidence=0.85
                ))

        elif clause.clause_type == ClauseType.SOCIAL_INSURANCE:
            text = clause.original_text
            if "가입하지" in text or "미가입" in text or "제외" in text:
                violations.append(ClauseViolation(
                    clause=clause,
                    violation_type="사회보험_미가입",
                    severity=ViolationSeverity.HIGH,
                    description="4대 사회보험은 입사일부터 가입 의무 (수습 여부 무관)",
                    legal_basis="고용보험법, 국민연금법, 국민건강보험법, 산업재해보상보험법",
                    current_value="4대보험 미가입",
                    legal_standard="입사일부터 가입",
                    suggestion="입사일부터 4대 사회보험 가입 명시",
                    confidence=0.9
                ))

        return violations


# 위치 매핑 모듈 - Gemini 기반 정확한 하이라이팅 위치 추출 및 수정안 생성
class ViolationLocationMapper:
    """
    분석된 위반 사항들의 정확한 텍스트 위치를 Gemini를 통해 찾고,
    suggestion을 참고하여 suggested_text를 생성하는 모듈.
    분석 파이프라인의 마지막 단계에서 호출됨.
    """

    LOCATION_MAPPING_PROMPT = """당신은 계약서 분석 전문가입니다.
계약서 전문과 분석된 위험 조항 목록이 주어집니다.
각 위험 조항에 해당하는 계약서 내 정확한 텍스트를 추출하고, 수정안을 작성하세요.

[계약서 전문]
{contract_text}

[위험 조항 목록]
{violations_json}

[작업]
각 위험 조항(violation_id로 구분)에 대해:
1. 해당 위반이 발생한 계약서의 정확한 텍스트를 그대로 복사하세요 (matched_text)
2. suggestion을 참고하여 해당 조항을 법적으로 적합하게 수정한 suggested_text를 작성하세요

[중요 규칙]
- matched_text는 계약서에서 해당 위반 조항을 정확히 복사해야 합니다
- 이탤릭(*텍스트*) 및 볼드(**텍스트**) 마커만 제거하고 내용만 추출하세요
  - 예: "*※ 수습기간 3개월*" → "※ 수습기간 3개월"
- 헤딩(#), 번호(1.), 블릿(-), 표(|) 등 다른 마크다운은 그대로 유지하세요
- 텍스트가 너무 길면(500자 이상) 핵심 부분만 추출하세요
- suggested_text는 원본 조항의 형식을 유지하면서 내용만 수정하세요
- 위치를 찾을 수 없으면 해당 violation은 결과에서 제외하세요

[응답 형식 - JSON]
{{
    "mapped_violations": [
        {{
            "violation_id": "원본 violation의 id",
            "matched_text": "계약서에서 해당 위반 조항의 원본 텍스트 (정확히 복사)",
            "suggested_text": "수정된 조항 텍스트 (suggestion을 반영한 법적으로 적합한 버전)"
        }}
    ]
}}"""

    def __init__(self, model: str = None):
        from app.core.config import settings
        self.model = model or settings.LLM_LOCATION_MODEL
        self.api_key = settings.GEMINI_API_KEY

    def _find_text_location(
        self,
        contract_text: str,
        search_text: str,
        fuzzy_threshold: float = 0.7
    ) -> tuple[int, int] | None:
        """
        계약서에서 검색 텍스트의 정확한 위치를 찾습니다.

        1. 정확한 문자열 매칭 시도
        2. 실패 시 공백/줄바꿈 정규화 후 재시도
        3. 실패 시 마크다운 마커 제거 후 재시도
        4. 실패 시 fuzzy matching으로 유사 텍스트 검색

        Args:
            contract_text: 계약서 전문
            search_text: 검색할 텍스트
            fuzzy_threshold: 유사도 임계값 (0.0 ~ 1.0)

        Returns:
            (start_index, end_index) 또는 None
        """
        import re
        from difflib import SequenceMatcher

        if not search_text or not contract_text:
            return None

        # 1. 정확한 매칭 시도
        idx = contract_text.find(search_text)
        if idx != -1:
            return (idx, idx + len(search_text))

        # 2. 공백/줄바꿈 정규화 후 매칭
        def normalize_whitespace(text: str) -> str:
            return re.sub(r'\s+', ' ', text).strip()

        normalized_search = normalize_whitespace(search_text)
        normalized_contract = normalize_whitespace(contract_text)

        normalized_idx = normalized_contract.find(normalized_search)
        if normalized_idx != -1:
            original_pos = self._map_normalized_to_original(
                contract_text, normalized_contract, normalized_idx, len(normalized_search)
            )
            if original_pos:
                return original_pos

        # 3. 마크다운 마커 제거 후 매칭
        def strip_markdown(text: str) -> str:
            # 볼드/이탤릭 마커 제거
            text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
            text = re.sub(r'\*([^*]+)\*', r'\1', text)
            # 리스트 마커 제거
            text = re.sub(r'^\s*[-*+]\s+', '', text, flags=re.MULTILINE)
            text = re.sub(r'^\s*\d+\.\s+', '', text, flags=re.MULTILINE)
            return text.strip()

        clean_search = strip_markdown(normalize_whitespace(search_text))
        clean_contract = strip_markdown(normalize_whitespace(contract_text))

        clean_idx = clean_contract.find(clean_search)
        if clean_idx != -1:
            # 클린 텍스트에서 찾은 위치를 원본 텍스트로 대략적 매핑
            # 비율 기반 추정 (정확하지 않지만 대부분 작동)
            ratio = clean_idx / max(len(clean_contract), 1)
            estimated_start = int(ratio * len(contract_text))

            # 추정 위치 주변에서 유사 텍스트 찾기
            search_radius = min(200, len(contract_text) // 4)
            start_range = max(0, estimated_start - search_radius)
            end_range = min(len(contract_text), estimated_start + len(clean_search) + search_radius)

            # 주변에서 최적 매칭 찾기
            best_local_match = None
            best_local_ratio = 0.6

            for window_size in range(len(clean_search) - 20, len(clean_search) + 50, 5):
                if window_size < 10:
                    continue
                for i in range(start_range, min(end_range, len(contract_text) - window_size + 1)):
                    candidate = contract_text[i:i + window_size]
                    candidate_clean = strip_markdown(normalize_whitespace(candidate))
                    ratio = SequenceMatcher(None, clean_search, candidate_clean).ratio()

                    if ratio > best_local_ratio:
                        best_local_ratio = ratio
                        best_local_match = (i, i + window_size)

            if best_local_match and best_local_ratio > 0.7:
                print(f">>> [LocationMapper] Markdown-stripped match found with {best_local_ratio*100:.1f}% similarity")
                return best_local_match

        # 4. Fuzzy matching (유사 텍스트 검색)
        search_len = len(search_text)
        best_match = None
        best_ratio = fuzzy_threshold

        min_window = max(20, int(search_len * 0.5))
        max_window = min(len(contract_text), int(search_len * 1.5))

        step = max(1, len(contract_text) // 2000)

        for window_size in range(min_window, max_window + 1, max(1, (max_window - min_window) // 10)):
            for i in range(0, len(contract_text) - window_size + 1, step):
                candidate = contract_text[i:i + window_size]
                ratio = SequenceMatcher(None, search_text, candidate).ratio()

                if ratio > best_ratio:
                    best_ratio = ratio
                    best_match = (i, i + window_size)

        if best_match:
            print(f">>> [LocationMapper] Fuzzy match found with {best_ratio*100:.1f}% similarity")
            return best_match

        return None

    def _map_normalized_to_original(
        self,
        original: str,
        normalized: str,
        normalized_start: int,
        normalized_length: int
    ) -> tuple[int, int] | None:
        """
        정규화된 텍스트의 위치를 원본 텍스트 위치로 매핑
        """

        # 원본에서 공백이 아닌 문자들의 위치 추적
        original_positions = []
        for i, char in enumerate(original):
            if not char.isspace() or (original_positions and not original[original_positions[-1]].isspace()):
                original_positions.append(i)

        # 정규화된 위치에서 원본 위치 계산
        normalized_non_space_count = 0
        start_mapped = False
        original_start = 0
        original_end = len(original)

        normalized_char_idx = 0
        for i, char in enumerate(normalized):
            if normalized_char_idx == normalized_start and not start_mapped:
                # 원본에서 해당 위치 찾기
                if normalized_non_space_count < len(original_positions):
                    original_start = original_positions[normalized_non_space_count]
                start_mapped = True

            if normalized_char_idx == normalized_start + normalized_length:
                if normalized_non_space_count < len(original_positions):
                    original_end = original_positions[normalized_non_space_count]
                else:
                    original_end = len(original)
                break

            if not char.isspace():
                normalized_non_space_count += 1
            normalized_char_idx += 1

        if start_mapped:
            return (original_start, original_end)
        return None

    def map_violation_locations(
        self,
        contract_text: str,
        violations: List[Dict[str, Any]],
        contract_id: str = None
    ) -> List[Dict[str, Any]]:
        """
        위반 사항들의 정확한 위치를 Gemini로 매핑하고 수정안 생성

        Args:
            contract_text: 계약서 전문
            violations: 위반 사항 목록 (dict 형태, suggestion 포함)
            contract_id: 토큰 추적용 계약서 ID

        Returns:
            위치와 suggested_text가 추가된 위반 사항 목록
        """
        if not violations or not contract_text:
            return violations

        # Gemini 클라이언트 초기화
        try:
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
        except ImportError:
            print(">>> [LocationMapper] google-generativeai not installed")
            return violations

        # 위반 사항을 간소화된 형태로 변환 (토큰 절약)
        simplified_violations = []
        for i, v in enumerate(violations):
            simplified_violations.append({
                "violation_id": v.get("id", f"v_{i}"),
                "type": v.get("type", ""),
                "clause_number": v.get("clause_number", ""),
                "description": v.get("description", "")[:300],
                "suggestion": v.get("suggestion", ""),  # 수정 제안 포함
                "original_text_hint": v.get("original_text", "")[:200],  # 위치 힌트용
            })

        prompt = self.LOCATION_MAPPING_PROMPT.format(
            contract_text=contract_text,
            violations_json=json.dumps(simplified_violations, ensure_ascii=False, indent=2)
        )

        try:
            import time
            llm_start = time.time()

            # Safety settings (완전 완화 - 계약서 분석은 합법적 용도)
            safety_settings = [
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
            ]

            model = genai.GenerativeModel(
                self.model,
                generation_config={
                    "response_mime_type": "application/json",
                    "temperature": 0.1,
                }
            )

            response = model.generate_content(prompt, safety_settings=safety_settings)
            llm_duration = (time.time() - llm_start) * 1000

            # 토큰 사용량 기록
            if hasattr(response, 'usage_metadata') and contract_id:
                record_llm_usage(
                    contract_id=contract_id,
                    module="violation_location_mapper",
                    model=self.model,
                    input_tokens=response.usage_metadata.prompt_token_count,
                    output_tokens=response.usage_metadata.candidates_token_count,
                    cached_tokens=getattr(response.usage_metadata, 'cached_content_token_count', 0) or 0,
                    duration_ms=llm_duration
                )

            result = json.loads(response.text)
            mapped = result.get("mapped_violations", [])

            # 결과를 원본 violations에 병합
            location_map = {}
            for m in mapped:
                vid = m.get("violation_id")
                if vid:
                    location_map[vid] = m

            # matched_text에서 마크다운 마커 제거 (프론트엔드 하이라이팅 호환성)
            def strip_markdown_markers(text: str) -> str:
                # 볼드/이탤릭 마커 제거
                text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
                text = re.sub(r'\*([^*]+)\*', r'\1', text)
                # 앞뒤 공백 제거
                return text.strip()

            # 원본 violations 업데이트 (Python 기반 텍스트 매칭)
            updated_violations = []
            for i, v in enumerate(violations):
                vid = v.get("id", f"v_{i}")
                if vid in location_map:
                    loc = location_map[vid]
                    matched_text = loc.get("matched_text", "")
                    suggested_text = loc.get("suggested_text", "")

                    # matched_text와 suggested_text 저장
                    if matched_text:
                        # 프론트엔드에서 렌더링된 마크다운에서 검색하므로 마커 제거
                        v["matched_text"] = strip_markdown_markers(matched_text)
                    if suggested_text:
                        v["suggested_text"] = suggested_text

                    # Python 기반 텍스트 위치 검색
                    if matched_text:
                        location = self._find_text_location(contract_text, matched_text)
                        if location:
                            start_idx, end_idx = location
                            coverage = (end_idx - start_idx) / len(contract_text)

                            if coverage <= 0.15:  # 15% 이하만 허용
                                v["start_index"] = start_idx
                                v["end_index"] = end_idx
                                print(f">>> [LocationMapper] Mapped {vid}: {start_idx}-{end_idx} ({coverage*100:.1f}%)")
                            else:
                                print(f">>> [LocationMapper] Skipped {vid}: coverage {coverage*100:.1f}% too large")
                        else:
                            print(f">>> [LocationMapper] Could not locate text for {vid}")
                            print(f"    Searched for: {matched_text[:100]}..." if len(matched_text) > 100 else f"    Searched for: {matched_text}")
                    else:
                        print(f">>> [LocationMapper] No matched_text for {vid}")

                updated_violations.append(v)

            return updated_violations

        except Exception as e:
            print(f">>> [LocationMapper] Error: {e}")
            import traceback
            traceback.print_exc()
            return violations  # 실패 시 원본 반환


# 편의 함수
def analyze_contract_clauses(
    contract_text: str,
    crag: Optional[Any] = None
) -> ClauseAnalysisResult:
    """계약서 조항 분석"""
    analyzer = LLMClauseAnalyzer(crag=crag)
    return analyzer.analyze(contract_text)


def map_violation_locations(
    contract_text: str,
    violations: List[Dict[str, Any]],
    contract_id: str = None
) -> List[Dict[str, Any]]:
    """위반 사항 위치 매핑 (편의 함수)"""
    mapper = ViolationLocationMapper()
    return mapper.map_violation_locations(contract_text, violations, contract_id)
