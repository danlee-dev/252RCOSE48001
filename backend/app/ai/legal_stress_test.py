"""
Legal Stress Test (Neuro-Symbolic AI) - Production Grade
- LLM의 추론 능력(Neuro) + Python 코드의 계산 능력(Symbolic) 결합
- 근로계약서 수치 시뮬레이션
- 최저임금, 연장근로, 야간근로, 휴일근로, 퇴직금, 4대보험, 연차휴가 정밀 계산
- 위반 유형별 체불액 및 법적 제재 산정

Reference: Neuro-Symbolic AI - Combining Neural Networks with Symbolic Reasoning
"""

import re
import json
from typing import Dict, Any, Optional, List, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
from decimal import Decimal, ROUND_HALF_UP, ROUND_DOWN
from enum import Enum
import os
from functools import lru_cache


class ViolationType(Enum):
    """위반 유형"""
    MINIMUM_WAGE = "minimum_wage"              # 최저임금 미달
    OVERTIME_PAY = "overtime_pay"              # 연장근로수당 미지급
    NIGHT_WORK_PAY = "night_work_pay"          # 야간근로수당 미지급
    HOLIDAY_WORK_PAY = "holiday_work_pay"      # 휴일근로수당 미지급
    WEEKLY_HOLIDAY_PAY = "weekly_holiday_pay"  # 주휴수당 미지급
    WORKING_HOURS = "working_hours"            # 근로시간 위반
    SEVERANCE_PAY = "severance_pay"            # 퇴직금 미지급/미달
    SOCIAL_INSURANCE = "social_insurance"      # 4대보험 미가입
    ANNUAL_LEAVE = "annual_leave"              # 연차휴가 미부여
    ILLEGAL_DEDUCTION = "illegal_deduction"    # 불법 공제
    BREAK_TIME = "break_time"                  # 휴게시간 미부여


class RiskLevel(Enum):
    """위험도 수준"""
    CRITICAL = "Critical"   # 형사 처벌 가능, 즉시 시정 필요
    HIGH = "High"           # 노동청 신고 대상
    MEDIUM = "Medium"       # 개선 권고
    LOW = "Low"             # 주의 필요
    INFO = "Info"           # 정보성


class LegalSanction(Enum):
    """법적 제재 유형"""
    CRIMINAL_PENALTY = "criminal_penalty"      # 형사 처벌
    ADMINISTRATIVE_FINE = "administrative_fine" # 과태료
    BACK_PAY = "back_pay"                       # 체불 임금 지급
    PENALTY_INTEREST = "penalty_interest"       # 지연이자
    CIVIL_DAMAGES = "civil_damages"             # 민사 손해배상


@dataclass
class WageInfo:
    """급여 정보 (확장)"""
    base_salary: int = 0                    # 기본급
    allowances: Dict[str, int] = field(default_factory=dict)  # 수당들
    total_salary: int = 0                   # 총 급여
    hourly_wage: Decimal = Decimal("0")     # 시급 (정밀 계산용)
    is_inclusive_wage: bool = False         # 포괄임금제 여부
    inclusive_details: Dict[str, int] = field(default_factory=dict)  # 포괄임금 상세
    payment_date: int = 0                   # 급여 지급일
    payment_method: str = ""                # 지급 방식 (현금/계좌)
    deductions: Dict[str, int] = field(default_factory=dict)  # 공제 항목


@dataclass
class WorkTimeInfo:
    """근로시간 정보 (확장)"""
    daily_hours: float = 8.0                # 일일 근로시간
    weekly_hours: float = 40.0              # 주간 근로시간
    overtime_hours: float = 0.0             # 주간 연장근로시간
    night_hours: float = 0.0                # 주간 야간근로시간 (22시~06시)
    holiday_hours: float = 0.0              # 주간 휴일근로시간
    working_days_per_week: int = 5          # 주간 근무일수
    break_time_minutes: int = 60            # 휴게시간 (분)
    work_start_time: str = "09:00"          # 출근 시간
    work_end_time: str = "18:00"            # 퇴근 시간
    shift_type: str = "day"                 # 근무 형태 (day/night/shift)


@dataclass
class EmploymentInfo:
    """고용 정보"""
    contract_type: str = "regular"          # 정규직/계약직/일용직
    start_date: Optional[date] = None       # 입사일
    end_date: Optional[date] = None         # 종료일 (계약직)
    probation_period: int = 0               # 수습 기간 (개월)
    is_part_time: bool = False              # 단시간 근로자 여부
    employee_count: int = 5                 # 사업장 상시 근로자 수


@dataclass
class SocialInsuranceInfo:
    """4대보험 정보"""
    has_national_pension: bool = True       # 국민연금
    has_health_insurance: bool = True       # 건강보험
    has_employment_insurance: bool = True   # 고용보험
    has_industrial_insurance: bool = True   # 산재보험
    pension_rate: Decimal = Decimal("0.045")     # 국민연금 근로자 부담률
    health_rate: Decimal = Decimal("0.03545")    # 건강보험 근로자 부담률
    longterm_care_rate: Decimal = Decimal("0.1295")  # 장기요양보험률
    employment_rate: Decimal = Decimal("0.009")  # 고용보험 근로자 부담률


@dataclass
class ViolationDetail:
    """위반 상세 정보"""
    violation_type: ViolationType
    risk_level: RiskLevel
    description: str
    legal_basis: str                        # 법적 근거
    current_value: Any                      # 현재 값
    legal_standard: Any                     # 법정 기준
    shortage_amount: int = 0                # 미달/체불 금액
    monthly_shortage: int = 0               # 월간 체불 예상액
    annual_shortage: int = 0                # 연간 체불 예상액
    sanctions: List[LegalSanction] = field(default_factory=list)
    sanction_details: str = ""              # 제재 상세
    remediation: str = ""                   # 시정 방법
    precedent_ref: str = ""                 # 관련 판례
    calculation_breakdown: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StressTestResult:
    """스트레스 테스트 결과 (확장)"""
    wage_info: WageInfo
    work_time_info: WorkTimeInfo
    employment_info: EmploymentInfo = field(default_factory=EmploymentInfo)
    social_insurance_info: SocialInsuranceInfo = field(default_factory=SocialInsuranceInfo)
    violations: List[ViolationDetail] = field(default_factory=list)
    calculations: Dict[str, Any] = field(default_factory=dict)
    total_underpayment: int = 0             # 총 체불 예상액 (월)
    annual_underpayment: int = 0            # 연간 체불 예상액
    three_year_underpayment: int = 0        # 3년간 체불 예상액 (소멸시효)
    risk_level: RiskLevel = RiskLevel.LOW
    risk_score: float = 0.0                 # 0-100
    compliance_score: float = 100.0         # 0-100 (준법 점수)
    summary: str = ""
    detailed_report: str = ""
    simulation_scenarios: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def has_violations(self) -> bool:
        return len(self.violations) > 0

    @property
    def critical_violations(self) -> List[ViolationDetail]:
        return [v for v in self.violations if v.risk_level == RiskLevel.CRITICAL]

    @property
    def high_violations(self) -> List[ViolationDetail]:
        return [v for v in self.violations if v.risk_level == RiskLevel.HIGH]

    def to_dict(self) -> Dict[str, Any]:
        """결과를 딕셔너리로 변환"""
        return {
            "risk_level": self.risk_level.value,
            "risk_score": self.risk_score,
            "compliance_score": self.compliance_score,
            "total_underpayment": self.total_underpayment,
            "annual_underpayment": self.annual_underpayment,
            "three_year_underpayment": self.three_year_underpayment,
            "violations": [
                {
                    "type": v.violation_type.value,
                    "risk_level": v.risk_level.value,
                    "description": v.description,
                    "legal_basis": v.legal_basis,
                    "shortage_amount": v.shortage_amount,
                    "monthly_shortage": v.monthly_shortage,
                    "annual_shortage": v.annual_shortage,
                    "remediation": v.remediation,
                }
                for v in self.violations
            ],
            "summary": self.summary,
            "timestamp": self.timestamp.isoformat(),
        }


class LegalStressTest:
    """
    근로계약서 법적 스트레스 테스트 (Production Grade)

    사용법:
        test = LegalStressTest()
        result = test.run(contract_text)
        print(f"연간 체불 예상액: {result.annual_underpayment:,}원")
        print(f"준법 점수: {result.compliance_score:.1f}점")
    """

    # ========== 2025년 법정 기준 ==========
    MINIMUM_WAGE_2025 = 10_030              # 시급
    MONTHLY_MINIMUM_WAGE_2025 = 2_096_270   # 월 209시간 기준

    # 법정 근로시간
    LEGAL_DAILY_HOURS = 8
    LEGAL_WEEKLY_HOURS = 40
    LEGAL_EXTENDED_LIMIT = 12               # 연장근로 한도 (주)
    LEGAL_TOTAL_WEEKLY_LIMIT = 52           # 주 52시간

    # 가산율 (근로기준법 제56조)
    OVERTIME_RATE = Decimal("1.5")          # 연장근로 50% 가산
    NIGHT_RATE = Decimal("1.5")             # 야간근로 50% 가산 (22시~06시)
    HOLIDAY_RATE = Decimal("1.5")           # 휴일근로 50% 가산 (8시간 이내)
    HOLIDAY_OVER_8_RATE = Decimal("2.0")    # 휴일 8시간 초과 100% 가산
    HOLIDAY_NIGHT_RATE = Decimal("2.0")     # 휴일 야간근로 100% 가산

    # 주휴수당 (주 15시간 이상 근무 시)
    WEEKLY_PAID_HOLIDAY_HOURS = 8
    MIN_WEEKLY_HOURS_FOR_HOLIDAY = 15

    # 휴게시간 기준 (근로기준법 제54조)
    BREAK_TIME_4_HOURS = 30                 # 4시간 초과 시 30분 이상
    BREAK_TIME_8_HOURS = 60                 # 8시간 초과 시 1시간 이상

    # 연차휴가 (근로기준법 제60조)
    ANNUAL_LEAVE_FIRST_YEAR = 11            # 1년 미만 근로자 (월 1일)
    ANNUAL_LEAVE_BASE = 15                  # 1년 이상 근로자 기본
    ANNUAL_LEAVE_MAX = 25                   # 최대 연차

    # 퇴직금 기준 (근로자퇴직급여보장법)
    MIN_SERVICE_FOR_SEVERANCE = 365         # 1년 이상 근무 시 퇴직금 발생

    # 4대보험 요율 (2025년 기준)
    INSURANCE_RATES = {
        "national_pension": {
            "employee": Decimal("0.045"),    # 4.5%
            "employer": Decimal("0.045"),    # 4.5%
            "total": Decimal("0.09"),        # 9%
        },
        "health_insurance": {
            "employee": Decimal("0.03545"),  # 3.545%
            "employer": Decimal("0.03545"),  # 3.545%
            "total": Decimal("0.0709"),      # 7.09%
        },
        "longterm_care": {
            "rate": Decimal("0.1295"),       # 건강보험료의 12.95%
        },
        "employment_insurance": {
            "employee": Decimal("0.009"),    # 0.9%
            "employer_150": Decimal("0.025"), # 150인 미만
            "employer_150_1000": Decimal("0.045"),  # 150~1000인
            "employer_1000": Decimal("0.065"),  # 1000인 이상
        },
        "industrial_insurance": {
            "average_rate": Decimal("0.0186"),  # 업종 평균
        },
    }

    # 지연이자율 (근로기준법 시행령)
    DELAY_INTEREST_RATE = Decimal("0.20")   # 연 20%

    # 벌칙 기준
    CRIMINAL_PENALTY_WAGE = {
        "max_imprisonment": "3년",
        "max_fine": "3000만원",
        "law": "근로기준법 제109조"
    }

    def __init__(
        self,
        minimum_wage: int = None,
        llm_client: Optional[Any] = None,
        model: str = "gpt-4o",
        include_precedents: bool = True
    ):
        """
        Args:
            minimum_wage: 최저임금 (없으면 2025년 기준)
            llm_client: OpenAI 클라이언트 (수치 추출용)
            model: LLM 모델
            include_precedents: 관련 판례 포함 여부
        """
        self.minimum_wage = minimum_wage or self.MINIMUM_WAGE_2025
        self.model = model
        self.include_precedents = include_precedents

        if llm_client is None:
            try:
                from openai import OpenAI
                self.llm_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            except ImportError:
                self.llm_client = None
        else:
            self.llm_client = llm_client

    def run(
        self,
        contract_text: str,
        employment_info: Optional[EmploymentInfo] = None
    ) -> StressTestResult:
        """
        계약서 스트레스 테스트 실행

        Args:
            contract_text: 계약서 텍스트
            employment_info: 고용 정보 (선택)

        Returns:
            StressTestResult: 테스트 결과
        """
        # 1. 정보 추출 (Neuro)
        wage_info = self._extract_wage_info(contract_text)
        work_time_info = self._extract_work_time_info(contract_text)
        emp_info = employment_info or self._extract_employment_info(contract_text)
        social_info = self._extract_social_insurance_info(contract_text)

        # 1.5 실제 근로시간 기준으로 시급 재계산
        # 기본 209시간이 아닌 실제 월간 근로시간으로 계산
        if wage_info.total_salary > 0 and work_time_info.weekly_hours > 0:
            # 월간 근로시간 = 주간 근로시간 * 4.345 (52주/12개월)
            actual_monthly_hours = work_time_info.weekly_hours * Decimal("4.345")
            actual_hourly = Decimal(str(wage_info.total_salary)) / actual_monthly_hours
            wage_info.hourly_wage = actual_hourly

        # 2. 결과 객체 초기화
        result = StressTestResult(
            wage_info=wage_info,
            work_time_info=work_time_info,
            employment_info=emp_info,
            social_insurance_info=social_info
        )

        # 3. 위반 검사 (Symbolic)
        self._check_minimum_wage(result)
        self._check_overtime_pay(result)
        self._check_night_work_pay(result)
        self._check_holiday_work_pay(result)
        self._check_weekly_holiday_pay(result)
        self._check_working_hours(result)
        self._check_break_time(result)
        self._check_severance_pay(result)
        self._check_social_insurance(result)
        self._check_annual_leave(result)
        self._check_illegal_deduction(result)

        # 4. 3년간 체불액 계산 (소멸시효)
        result.three_year_underpayment = result.annual_underpayment * 3

        # 5. 위험도 및 준법 점수 평가
        self._evaluate_risk_level(result)
        self._calculate_compliance_score(result)

        # 6. 시나리오 시뮬레이션
        result.simulation_scenarios = self._simulate_scenarios(result)

        # 7. 요약 및 상세 보고서 생성
        result.summary = self._generate_summary(result)
        result.detailed_report = self._generate_detailed_report(result)

        return result

    # ========== 정보 추출 메서드 ==========

    def _extract_wage_info(self, text: str) -> WageInfo:
        """계약서에서 급여 정보 추출 (확장)"""
        wage_info = WageInfo()

        # 월급/급여 패턴 (확장)
        salary_patterns = [
            r'월\s*급[여]?\s*[:\s]*([0-9,]+)\s*원',
            r'급\s*여\s*[:\s]*([0-9,]+)\s*원',
            r'임\s*금\s*[:\s]*([0-9,]+)\s*원',
            r'월\s*([0-9,]+)\s*원',
            r'([0-9,]+)\s*원\s*/\s*월',
            r'연\s*봉\s*[:\s]*([0-9,]+)\s*만?\s*원',
            r'연\s*([0-9,]+)\s*만\s*원',
        ]

        for pattern in salary_patterns:
            match = re.search(pattern, text)
            if match:
                salary_str = match.group(1).replace(',', '')
                salary = int(salary_str)
                # 연봉인 경우 월급으로 변환
                if '연봉' in text[:match.start()] or '연' in pattern:
                    if salary < 100000:  # 만원 단위
                        salary = salary * 10000
                    wage_info.total_salary = salary // 12
                else:
                    wage_info.total_salary = salary
                break

        # 시급 패턴
        hourly_patterns = [
            r'시\s*급\s*[:\s]*([0-9,]+)\s*원',
            r'시간당\s*([0-9,]+)\s*원',
            r'시급\s*([0-9,]+)\s*원',
        ]

        for pattern in hourly_patterns:
            match = re.search(pattern, text)
            if match:
                hourly_str = match.group(1).replace(',', '')
                wage_info.hourly_wage = Decimal(hourly_str)
                break

        # 기본급 패턴
        base_patterns = [
            r'기본급\s*[:\s]*([0-9,]+)\s*원',
            r'기본\s*급여\s*[:\s]*([0-9,]+)\s*원',
        ]

        for pattern in base_patterns:
            match = re.search(pattern, text)
            if match:
                base_str = match.group(1).replace(',', '')
                wage_info.base_salary = int(base_str)
                break

        # 수당 추출
        allowance_patterns = {
            "식대": r'식[대비]\s*[:\s]*([0-9,]+)\s*원',
            "교통비": r'교통비\s*[:\s]*([0-9,]+)\s*원',
            "직책수당": r'직책\s*수당\s*[:\s]*([0-9,]+)\s*원',
            "자격수당": r'자격\s*수당\s*[:\s]*([0-9,]+)\s*원',
            "연장수당": r'연장\s*[근로]?\s*수당\s*[:\s]*([0-9,]+)\s*원',
            "야간수당": r'야간\s*[근로]?\s*수당\s*[:\s]*([0-9,]+)\s*원',
            "휴일수당": r'휴일\s*[근로]?\s*수당\s*[:\s]*([0-9,]+)\s*원',
        }

        for name, pattern in allowance_patterns.items():
            match = re.search(pattern, text)
            if match:
                amount_str = match.group(1).replace(',', '')
                wage_info.allowances[name] = int(amount_str)

        # 포괄임금제 여부 및 상세
        inclusive_patterns = [
            (r'포괄\s*(임금|급여)', "포괄임금제"),
            (r'(연장|야간|휴일)\s*근로\s*(수당)?\s*포함', "수당 포함"),
            (r'모든\s*수당\s*포함', "전체 포함"),
            (r'일체\s*포함', "일체 포함"),
            (r'고정\s*(OT|연장)', "고정 연장"),
        ]

        for pattern, inclusive_type in inclusive_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                wage_info.is_inclusive_wage = True
                # 포괄임금 상세 추출
                ot_match = re.search(r'연장\s*([0-9]+)\s*시간', text)
                if ot_match:
                    wage_info.inclusive_details["overtime_hours"] = int(ot_match.group(1))
                night_match = re.search(r'야간\s*([0-9]+)\s*시간', text)
                if night_match:
                    wage_info.inclusive_details["night_hours"] = int(night_match.group(1))
                break

        # 공제 항목 추출
        deduction_patterns = {
            "기숙사비": r'기숙사\s*[비료]?\s*[:\s]*([0-9,]+)\s*원',
            "식비공제": r'식[비대]\s*공제\s*[:\s]*([0-9,]+)\s*원',
            "장비비": r'장비\s*[비료]?\s*[:\s]*([0-9,]+)\s*원',
            "유니폼비": r'유니폼\s*[비료]?\s*[:\s]*([0-9,]+)\s*원',
            "교육비": r'교육비\s*공제\s*[:\s]*([0-9,]+)\s*원',
            "위약금": r'위약금\s*[:\s]*([0-9,]+)\s*원',
        }

        for name, pattern in deduction_patterns.items():
            match = re.search(pattern, text)
            if match:
                amount_str = match.group(1).replace(',', '')
                wage_info.deductions[name] = int(amount_str)

        # 시급 계산
        if wage_info.hourly_wage == 0 and wage_info.total_salary > 0:
            wage_info.hourly_wage = Decimal(str(wage_info.total_salary)) / Decimal("209")

        # 기본급이 없으면 총급여로 설정
        if wage_info.base_salary == 0:
            wage_info.base_salary = wage_info.total_salary

        return wage_info

    def _extract_work_time_info(self, text: str) -> WorkTimeInfo:
        """계약서에서 근로시간 정보 추출 (확장)"""
        work_info = WorkTimeInfo()

        # 일일 근로시간 (명시적)
        daily_patterns = [
            r'1\s*일\s*([0-9.]+)\s*시간',
            r'일\s*([0-9.]+)\s*시간\s*근무',
            r'([0-9]+)\s*시간\s*/\s*일',
            r'하루\s*([0-9.]+)\s*시간',
        ]

        for pattern in daily_patterns:
            match = re.search(pattern, text)
            if match:
                work_info.daily_hours = float(match.group(1))
                break

        # 주간 근로시간
        weekly_patterns = [
            r'주\s*([0-9.]+)\s*시간',
            r'1\s*주\s*([0-9.]+)\s*시간',
            r'([0-9]+)\s*시간\s*/\s*주',
            r'주간\s*([0-9.]+)\s*시간',
        ]

        for pattern in weekly_patterns:
            match = re.search(pattern, text)
            if match:
                work_info.weekly_hours = float(match.group(1))
                break

        # 근무일수 (확장된 패턴)
        workday_patterns = [
            (r'주\s*([0-9])\s*일\s*근무', None),
            (r'매주\s*([0-9])\s*일\s*근무', None),
            (r'([0-9])\s*일\s*/\s*주', None),
            (r'월\s*~\s*금', 5),
            (r'월~금', 5),
            (r'월\s*~\s*토', 6),
            (r'월~토', 6),
            (r'월\s*-\s*토', 6),
            (r'월\s*-\s*금', 5),
        ]

        for pattern, days in workday_patterns:
            match = re.search(pattern, text)
            if match:
                if days:
                    work_info.working_days_per_week = days
                else:
                    work_info.working_days_per_week = int(match.group(1))
                break

        # 근무 시간대 (확장된 패턴 - "09시 00분 ~ 21시 00분" 형식 지원)
        time_patterns = [
            # "09시 00분 ~ 21시 00분" 또는 "09시 ~ 21시"
            r'(\d{1,2})\s*시\s*(\d{2})?\s*분?\s*[~\-]\s*(\d{1,2})\s*시\s*(\d{2})?\s*분?',
            # "09:00 ~ 21:00" 또는 "09:00-21:00"
            r'(\d{1,2}):(\d{2})\s*[~\-]\s*(\d{1,2}):(\d{2})',
            # 기본 패턴
            r'(\d{1,2})[:\s]*(\d{2})?\s*[~\-]\s*(\d{1,2})[:\s]*(\d{2})?',
            r'오전\s*(\d+)\s*시\s*[~\-]\s*오후\s*(\d+)\s*시',
        ]

        for pattern in time_patterns:
            match = re.search(pattern, text)
            if match:
                groups = match.groups()
                start_hour = int(groups[0])
                start_min = int(groups[1]) if groups[1] else 0
                end_hour = int(groups[2]) if len(groups) >= 3 and groups[2] else start_hour + 8
                end_min = int(groups[3]) if len(groups) >= 4 and groups[3] else 0

                # 오후 처리
                if "오후" in match.group(0) and end_hour < 12:
                    end_hour += 12

                work_info.work_start_time = f"{start_hour:02d}:{start_min:02d}"
                work_info.work_end_time = f"{end_hour:02d}:{end_min:02d}"

                # 출퇴근 시간에서 일일 근로시간 계산 (명시적 지정 없는 경우)
                if work_info.daily_hours == 8.0:  # 기본값인 경우만 계산
                    total_minutes = (end_hour * 60 + end_min) - (start_hour * 60 + start_min)
                    if total_minutes > 0:
                        work_info.daily_hours = total_minutes / 60.0
                break

        # 야간근로 여부 확인 (22시~06시)
        if work_info.work_end_time:
            end_hour = int(work_info.work_end_time.split(":")[0])
            if end_hour >= 22 or end_hour < 6:
                work_info.shift_type = "night"

        # 휴게시간 (확장된 패턴 - "없음" 감지 포함)
        # 먼저 "휴게 없음" 패턴 확인
        no_break_patterns = [
            r'휴게\s*[:\s]*없음',
            r'휴게시간\s*[:\s]*없음',
            r'휴식\s*[:\s]*없음',
            r'휴게\s*[:\s]*0\s*분',
        ]

        has_no_break = False
        for pattern in no_break_patterns:
            if re.search(pattern, text):
                work_info.break_time_minutes = 0
                has_no_break = True
                break

        # 휴게시간이 명시된 경우
        if not has_no_break:
            break_patterns = [
                r'휴게\s*[시간]*\s*[:\s]*([0-9]+)\s*분',
                r'휴식\s*[시간]*\s*[:\s]*([0-9]+)\s*분',
                r'점심\s*[시간]*\s*[:\s]*([0-9]+)\s*분',
                r'휴게\s*[:\s]*([0-9]+)\s*시간',
            ]

            for pattern in break_patterns:
                match = re.search(pattern, text)
                if match:
                    if '시간' in pattern:
                        work_info.break_time_minutes = int(match.group(1)) * 60
                    else:
                        work_info.break_time_minutes = int(match.group(1))
                    break

        # 주간 근로시간이 명시되지 않은 경우, 일일 근로시간 * 근무일수로 계산
        if work_info.weekly_hours == 40.0:  # 기본값인 경우
            calculated_weekly = work_info.daily_hours * work_info.working_days_per_week
            if calculated_weekly != 40.0:  # 계산 결과가 기본값과 다르면 사용
                work_info.weekly_hours = calculated_weekly

        # 연장근로시간 계산 (주 40시간 초과분)
        if work_info.weekly_hours > self.LEGAL_WEEKLY_HOURS:
            work_info.overtime_hours = work_info.weekly_hours - self.LEGAL_WEEKLY_HOURS

        # 연장/야간/휴일 근로시간 명시
        ot_match = re.search(r'연장\s*[근로]?\s*[:\s]*([0-9.]+)\s*시간', text)
        if ot_match:
            work_info.overtime_hours = float(ot_match.group(1))

        night_match = re.search(r'야간\s*[근로]?\s*[:\s]*([0-9.]+)\s*시간', text)
        if night_match:
            work_info.night_hours = float(night_match.group(1))

        holiday_match = re.search(r'휴일\s*[근로]?\s*[:\s]*([0-9.]+)\s*시간', text)
        if holiday_match:
            work_info.holiday_hours = float(holiday_match.group(1))

        return work_info

    def _extract_employment_info(self, text: str) -> EmploymentInfo:
        """계약서에서 고용 정보 추출"""
        emp_info = EmploymentInfo()

        # 계약 유형
        if re.search(r'정규직|정규\s*근로', text):
            emp_info.contract_type = "regular"
        elif re.search(r'계약직|기간제|촉탁', text):
            emp_info.contract_type = "contract"
        elif re.search(r'일용직|일용\s*근로', text):
            emp_info.contract_type = "daily"
        elif re.search(r'파트타임|단시간|시간제', text):
            emp_info.contract_type = "part_time"
            emp_info.is_part_time = True

        # 수습 기간
        probation_match = re.search(r'수습\s*[기간]*\s*[:\s]*([0-9]+)\s*개?월', text)
        if probation_match:
            emp_info.probation_period = int(probation_match.group(1))

        # 계약 기간
        period_match = re.search(
            r'(\d{4})[.\-/](\d{1,2})[.\-/](\d{1,2})\s*[~\-부터]\s*(\d{4})[.\-/](\d{1,2})[.\-/](\d{1,2})',
            text
        )
        if period_match:
            try:
                emp_info.start_date = date(
                    int(period_match.group(1)),
                    int(period_match.group(2)),
                    int(period_match.group(3))
                )
                emp_info.end_date = date(
                    int(period_match.group(4)),
                    int(period_match.group(5)),
                    int(period_match.group(6))
                )
            except ValueError:
                pass

        return emp_info

    def _extract_social_insurance_info(self, text: str) -> SocialInsuranceInfo:
        """계약서에서 4대보험 정보 추출"""
        social_info = SocialInsuranceInfo()

        # 4대보험 미가입 패턴 (확장)
        no_insurance_patterns = [
            (r'4대\s*보험\s*미가입', "all"),
            (r'4대\s*보험에\s*가입하지\s*않', "all"),
            (r'4대보험에\s*가입하지\s*않', "all"),
            (r'수습\s*기?간.*4대\s*보험.*가입하지\s*않', "all"),
            (r'수습.*4대보험.*미가입', "all"),
            (r'국민연금\s*미가입', "pension"),
            (r'건강보험\s*미가입', "health"),
            (r'고용보험\s*미가입', "employment"),
            (r'산재보험\s*미가입', "industrial"),
            (r'보험\s*없[음이]', "all"),
            (r'사회보험\s*미가입', "all"),
            (r'사회보험에\s*가입하지\s*않', "all"),
        ]

        for pattern, insurance_type in no_insurance_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                if insurance_type == "all":
                    social_info.has_national_pension = False
                    social_info.has_health_insurance = False
                    social_info.has_employment_insurance = False
                    social_info.has_industrial_insurance = False
                elif insurance_type == "pension":
                    social_info.has_national_pension = False
                elif insurance_type == "health":
                    social_info.has_health_insurance = False
                elif insurance_type == "employment":
                    social_info.has_employment_insurance = False
                elif insurance_type == "industrial":
                    social_info.has_industrial_insurance = False

        return social_info

    # ========== 위반 검사 메서드 ==========

    def _check_minimum_wage(self, result: StressTestResult):
        """최저임금 위반 검사 (정밀)"""
        hourly = result.wage_info.hourly_wage

        if hourly == 0:
            return

        # 수습 기간 최저임금 (90%)
        if result.employment_info.probation_period > 0:
            applicable_minimum = Decimal(str(self.minimum_wage)) * Decimal("0.9")
            minimum_label = f"수습 최저임금 ({self.minimum_wage * 0.9:,.0f}원)"
        else:
            applicable_minimum = Decimal(str(self.minimum_wage))
            minimum_label = f"최저임금 ({self.minimum_wage:,}원)"

        if hourly < applicable_minimum:
            shortage = applicable_minimum - hourly
            monthly_hours = Decimal("209")
            monthly_shortage = int(shortage * monthly_hours)
            annual_shortage = monthly_shortage * 12

            violation = ViolationDetail(
                violation_type=ViolationType.MINIMUM_WAGE,
                risk_level=RiskLevel.CRITICAL,
                description=f"시급 {hourly:,.0f}원은 {minimum_label}에 {shortage:,.0f}원 미달",
                legal_basis="근로기준법 제6조, 최저임금법 제6조",
                current_value=float(hourly),
                legal_standard=float(applicable_minimum),
                shortage_amount=int(shortage),
                monthly_shortage=monthly_shortage,
                annual_shortage=annual_shortage,
                sanctions=[
                    LegalSanction.CRIMINAL_PENALTY,
                    LegalSanction.BACK_PAY,
                    LegalSanction.PENALTY_INTEREST
                ],
                sanction_details="3년 이하 징역 또는 2천만원 이하 벌금 (최저임금법 제28조)",
                remediation="최저임금 기준에 맞게 시급 인상 필요",
                precedent_ref="대법원 2019다222796 - 최저임금 미달 임금은 최저임금으로 간주",
                calculation_breakdown={
                    "current_hourly": float(hourly),
                    "minimum_hourly": float(applicable_minimum),
                    "shortage_per_hour": float(shortage),
                    "monthly_hours": 209,
                    "monthly_shortage": monthly_shortage,
                    "annual_shortage": annual_shortage,
                    "three_year_liability": annual_shortage * 3,
                }
            )

            result.violations.append(violation)
            result.calculations["minimum_wage_shortage"] = violation.calculation_breakdown
            result.total_underpayment += monthly_shortage
            result.annual_underpayment += annual_shortage

    def _check_overtime_pay(self, result: StressTestResult):
        """연장근로수당 검사"""
        weekly_hours = result.work_time_info.weekly_hours
        overtime_hours = max(0, weekly_hours - self.LEGAL_WEEKLY_HOURS)

        # 명시된 연장근로시간이 있으면 사용
        if result.work_time_info.overtime_hours > 0:
            overtime_hours = result.work_time_info.overtime_hours

        if overtime_hours <= 0:
            return

        hourly = result.wage_info.hourly_wage or Decimal(str(self.minimum_wage))

        # 포괄임금제 검사
        if result.wage_info.is_inclusive_wage:
            # 포괄임금에 명시된 연장시간 확인
            included_ot = result.wage_info.inclusive_details.get("overtime_hours", 0)

            if included_ot < overtime_hours:
                # 초과 연장근로에 대한 미지급
                unpaid_ot = overtime_hours - included_ot
                overtime_premium = hourly * (self.OVERTIME_RATE - 1)  # 가산분만
                weekly_shortage = int(unpaid_ot * overtime_premium)
                monthly_shortage = weekly_shortage * 4
                annual_shortage = monthly_shortage * 12

                violation = ViolationDetail(
                    violation_type=ViolationType.OVERTIME_PAY,
                    risk_level=RiskLevel.HIGH,
                    description=f"포괄임금 명시 {included_ot}시간 초과 연장근로 {unpaid_ot}시간에 대한 가산수당 미지급 우려",
                    legal_basis="근로기준법 제56조 제1항",
                    current_value=included_ot,
                    legal_standard=overtime_hours,
                    shortage_amount=weekly_shortage,
                    monthly_shortage=monthly_shortage,
                    annual_shortage=annual_shortage,
                    sanctions=[LegalSanction.BACK_PAY, LegalSanction.PENALTY_INTEREST],
                    sanction_details="3년 이하 징역 또는 3천만원 이하 벌금",
                    remediation="실제 연장근로시간에 맞게 가산수당 지급 또는 포괄임금 시간 조정",
                    precedent_ref="대법원 2010다5811 - 포괄임금제가 유효하려면 실제 근로시간과 부합해야 함",
                    calculation_breakdown={
                        "included_overtime_hours": included_ot,
                        "actual_overtime_hours": overtime_hours,
                        "unpaid_overtime_hours": unpaid_ot,
                        "hourly_wage": float(hourly),
                        "overtime_rate": 1.5,
                        "overtime_premium_per_hour": float(overtime_premium),
                        "weekly_shortage": weekly_shortage,
                        "monthly_shortage": monthly_shortage,
                    }
                )

                result.violations.append(violation)
                result.calculations["overtime_pay_shortage"] = violation.calculation_breakdown
                result.total_underpayment += monthly_shortage
                result.annual_underpayment += annual_shortage
        else:
            # 일반 연장근로 - 수당 지급 여부 확인
            if "연장수당" not in result.wage_info.allowances:
                overtime_pay_per_hour = int(hourly * self.OVERTIME_RATE)
                weekly_overtime_pay = int(overtime_hours * overtime_pay_per_hour)
                monthly_overtime_pay = weekly_overtime_pay * 4

                result.calculations["overtime_pay_required"] = {
                    "weekly_overtime_hours": overtime_hours,
                    "overtime_rate": 1.5,
                    "required_weekly_pay": weekly_overtime_pay,
                    "required_monthly_pay": monthly_overtime_pay,
                    "note": "연장근로수당 지급 여부 확인 필요"
                }

    def _check_night_work_pay(self, result: StressTestResult):
        """야간근로수당 검사 (22시~06시)"""
        night_hours = result.work_time_info.night_hours

        # 근무시간대에서 야간근로 추정
        if night_hours == 0 and result.work_time_info.shift_type == "night":
            # 야간 교대근무로 추정되는 경우
            end_hour = int(result.work_time_info.work_end_time.split(":")[0])
            if end_hour >= 22:
                night_hours = min(24 - end_hour + 6, result.work_time_info.daily_hours)
            elif end_hour < 6:
                night_hours = min(end_hour + (24 - 22), result.work_time_info.daily_hours)

        if night_hours <= 0:
            return

        hourly = result.wage_info.hourly_wage or Decimal(str(self.minimum_wage))

        # 포괄임금제에서 야간근로 포함 여부
        if result.wage_info.is_inclusive_wage:
            included_night = result.wage_info.inclusive_details.get("night_hours", 0)
            if included_night >= night_hours:
                return

            unpaid_night = night_hours - included_night
        else:
            if "야간수당" in result.wage_info.allowances:
                return
            unpaid_night = night_hours

        night_premium = hourly * (self.NIGHT_RATE - 1)  # 가산분 50%
        weekly_shortage = int(unpaid_night * night_premium * result.work_time_info.working_days_per_week)
        monthly_shortage = weekly_shortage * 4
        annual_shortage = monthly_shortage * 12

        violation = ViolationDetail(
            violation_type=ViolationType.NIGHT_WORK_PAY,
            risk_level=RiskLevel.HIGH,
            description=f"일 {unpaid_night}시간 야간근로(22시~06시)에 대한 50% 가산수당 미지급 우려",
            legal_basis="근로기준법 제56조 제1항",
            current_value=0,
            legal_standard=float(night_premium * unpaid_night),
            shortage_amount=weekly_shortage,
            monthly_shortage=monthly_shortage,
            annual_shortage=annual_shortage,
            sanctions=[LegalSanction.BACK_PAY, LegalSanction.PENALTY_INTEREST],
            sanction_details="3년 이하 징역 또는 3천만원 이하 벌금",
            remediation="야간근로에 대한 50% 가산수당 지급 필요",
            calculation_breakdown={
                "night_hours_per_day": unpaid_night,
                "working_days_per_week": result.work_time_info.working_days_per_week,
                "hourly_wage": float(hourly),
                "night_premium_rate": 0.5,
                "night_premium_per_hour": float(night_premium),
                "weekly_shortage": weekly_shortage,
                "monthly_shortage": monthly_shortage,
            }
        )

        result.violations.append(violation)
        result.calculations["night_work_pay_shortage"] = violation.calculation_breakdown
        result.total_underpayment += monthly_shortage
        result.annual_underpayment += annual_shortage

    def _check_holiday_work_pay(self, result: StressTestResult):
        """휴일근로수당 검사"""
        holiday_hours = result.work_time_info.holiday_hours

        if holiday_hours <= 0:
            return

        hourly = result.wage_info.hourly_wage or Decimal(str(self.minimum_wage))

        # 8시간 이내/초과 구분
        hours_within_8 = min(holiday_hours, 8)
        hours_over_8 = max(0, holiday_hours - 8)

        # 계산
        pay_within_8 = hourly * self.HOLIDAY_RATE * Decimal(str(hours_within_8))
        pay_over_8 = hourly * self.HOLIDAY_OVER_8_RATE * Decimal(str(hours_over_8))
        weekly_holiday_pay = int(pay_within_8 + pay_over_8)
        monthly_holiday_pay = weekly_holiday_pay * 4

        if result.wage_info.is_inclusive_wage:
            # 포괄임금에 휴일수당 포함 여부 확인
            if "휴일수당" in result.wage_info.allowances:
                return

        violation = ViolationDetail(
            violation_type=ViolationType.HOLIDAY_WORK_PAY,
            risk_level=RiskLevel.HIGH,
            description=f"주 {holiday_hours}시간 휴일근로에 대한 가산수당 미지급 우려",
            legal_basis="근로기준법 제56조 제2항",
            current_value=0,
            legal_standard=weekly_holiday_pay,
            shortage_amount=weekly_holiday_pay,
            monthly_shortage=monthly_holiday_pay,
            annual_shortage=monthly_holiday_pay * 12,
            sanctions=[LegalSanction.BACK_PAY],
            sanction_details="8시간 이내 50% 가산, 8시간 초과 100% 가산",
            remediation="휴일근로에 대한 법정 가산수당 지급 필요",
            calculation_breakdown={
                "holiday_hours_per_week": holiday_hours,
                "hours_within_8": hours_within_8,
                "hours_over_8": hours_over_8,
                "rate_within_8": 1.5,
                "rate_over_8": 2.0,
                "pay_within_8": float(pay_within_8),
                "pay_over_8": float(pay_over_8),
                "weekly_holiday_pay": weekly_holiday_pay,
                "monthly_holiday_pay": monthly_holiday_pay,
            }
        )

        result.violations.append(violation)
        result.calculations["holiday_work_pay"] = violation.calculation_breakdown
        result.total_underpayment += monthly_holiday_pay
        result.annual_underpayment += monthly_holiday_pay * 12

    def _check_weekly_holiday_pay(self, result: StressTestResult):
        """주휴수당 검사"""
        weekly_hours = result.work_time_info.weekly_hours

        if weekly_hours < self.MIN_WEEKLY_HOURS_FOR_HOLIDAY:
            return

        hourly = result.wage_info.hourly_wage or Decimal(str(self.minimum_wage))

        # 시급제이고 주휴수당 별도 미지급 확인
        if result.wage_info.hourly_wage > 0 and result.wage_info.total_salary == 0:
            weekly_holiday_pay = int(hourly * self.WEEKLY_PAID_HOLIDAY_HOURS)
            monthly_holiday_pay = weekly_holiday_pay * 4

            violation = ViolationDetail(
                violation_type=ViolationType.WEEKLY_HOLIDAY_PAY,
                risk_level=RiskLevel.HIGH,
                description=f"주 {weekly_hours}시간 근무자의 주휴수당(8시간) 지급 여부 확인 필요",
                legal_basis="근로기준법 제55조",
                current_value="미확인",
                legal_standard=weekly_holiday_pay,
                shortage_amount=weekly_holiday_pay,
                monthly_shortage=monthly_holiday_pay,
                annual_shortage=monthly_holiday_pay * 12,
                sanctions=[LegalSanction.BACK_PAY],
                remediation="주 15시간 이상 근무 시 주휴수당(유급휴일 8시간분) 지급 필요",
                calculation_breakdown={
                    "weekly_hours": weekly_hours,
                    "hourly_wage": float(hourly),
                    "weekly_holiday_hours": self.WEEKLY_PAID_HOLIDAY_HOURS,
                    "weekly_holiday_pay": weekly_holiday_pay,
                    "monthly_holiday_pay": monthly_holiday_pay,
                    "note": "시급제의 경우 주휴수당 별도 지급 여부 확인 필요"
                }
            )

            result.violations.append(violation)
            result.calculations["weekly_holiday_pay"] = violation.calculation_breakdown

    def _check_working_hours(self, result: StressTestResult):
        """근로시간 위반 검사"""
        weekly = result.work_time_info.weekly_hours
        daily = result.work_time_info.daily_hours

        # 주 52시간 초과
        if weekly > self.LEGAL_TOTAL_WEEKLY_LIMIT:
            excess = weekly - self.LEGAL_TOTAL_WEEKLY_LIMIT

            violation = ViolationDetail(
                violation_type=ViolationType.WORKING_HOURS,
                risk_level=RiskLevel.CRITICAL,
                description=f"주 {weekly}시간 근무는 법정 한도(52시간)를 {excess}시간 초과",
                legal_basis="근로기준법 제53조",
                current_value=weekly,
                legal_standard=self.LEGAL_TOTAL_WEEKLY_LIMIT,
                shortage_amount=0,
                sanctions=[LegalSanction.CRIMINAL_PENALTY, LegalSanction.ADMINISTRATIVE_FINE],
                sanction_details="2년 이하 징역 또는 2천만원 이하 벌금",
                remediation="주 52시간 이내로 근로시간 조정 필요",
                calculation_breakdown={
                    "current_weekly_hours": weekly,
                    "legal_limit": self.LEGAL_TOTAL_WEEKLY_LIMIT,
                    "excess_hours": excess,
                }
            )

            result.violations.append(violation)
            result.calculations["working_hours_violation"] = violation.calculation_breakdown

        # 연장근로 12시간 초과 (합의 없이)
        elif weekly > self.LEGAL_WEEKLY_HOURS:
            overtime = weekly - self.LEGAL_WEEKLY_HOURS
            if overtime > self.LEGAL_EXTENDED_LIMIT:
                violation = ViolationDetail(
                    violation_type=ViolationType.WORKING_HOURS,
                    risk_level=RiskLevel.HIGH,
                    description=f"연장근로 {overtime}시간은 법정 한도({self.LEGAL_EXTENDED_LIMIT}시간)를 초과",
                    legal_basis="근로기준법 제53조 제1항",
                    current_value=overtime,
                    legal_standard=self.LEGAL_EXTENDED_LIMIT,
                    sanctions=[LegalSanction.CRIMINAL_PENALTY],
                    sanction_details="서면 합의 없는 연장근로는 위법",
                    remediation="근로자 대표와 서면 합의 또는 연장근로 축소 필요",
                )
                result.violations.append(violation)

        # 1일 8시간 초과 (점검)
        if daily > self.LEGAL_DAILY_HOURS:
            excess_daily = daily - self.LEGAL_DAILY_HOURS
            result.calculations["daily_overtime"] = {
                "daily_hours": daily,
                "legal_daily_hours": self.LEGAL_DAILY_HOURS,
                "daily_overtime": excess_daily,
                "note": "1일 8시간 초과분은 연장근로로 50% 가산 필요"
            }

    def _check_break_time(self, result: StressTestResult):
        """휴게시간 위반 검사"""
        daily_hours = result.work_time_info.daily_hours
        break_minutes = result.work_time_info.break_time_minutes

        required_break = 0
        if daily_hours > 8:
            required_break = self.BREAK_TIME_8_HOURS
        elif daily_hours > 4:
            required_break = self.BREAK_TIME_4_HOURS

        if required_break > 0 and break_minutes < required_break:
            violation = ViolationDetail(
                violation_type=ViolationType.BREAK_TIME,
                risk_level=RiskLevel.MEDIUM,
                description=f"1일 {daily_hours}시간 근무 시 휴게시간 {required_break}분 이상 필요, 현재 {break_minutes}분",
                legal_basis="근로기준법 제54조",
                current_value=break_minutes,
                legal_standard=required_break,
                sanctions=[LegalSanction.ADMINISTRATIVE_FINE],
                sanction_details="500만원 이하 과태료",
                remediation=f"휴게시간을 {required_break}분 이상으로 조정 필요",
                calculation_breakdown={
                    "daily_hours": daily_hours,
                    "current_break_minutes": break_minutes,
                    "required_break_minutes": required_break,
                    "shortage_minutes": required_break - break_minutes,
                }
            )
            result.violations.append(violation)
            result.calculations["break_time_violation"] = violation.calculation_breakdown

    def _check_severance_pay(self, result: StressTestResult):
        """퇴직금 위반 검사"""
        emp_info = result.employment_info

        # 1년 이상 근무 예정인지 확인
        if emp_info.start_date and emp_info.end_date:
            service_days = (emp_info.end_date - emp_info.start_date).days
        else:
            service_days = 365  # 기본 1년 가정

        if service_days < self.MIN_SERVICE_FOR_SEVERANCE:
            return

        # 퇴직금 계산
        monthly_salary = result.wage_info.total_salary
        if monthly_salary == 0:
            return

        # 1년 기준 퇴직금 = 1개월 평균임금
        # 평균임금 = 최근 3개월 임금 총액 / 3개월 일수
        avg_daily_wage = Decimal(str(monthly_salary * 3)) / Decimal("90")
        severance_per_year = int(avg_daily_wage * 30)  # 30일분

        result.calculations["severance_pay"] = {
            "monthly_salary": monthly_salary,
            "average_daily_wage": float(avg_daily_wage),
            "severance_per_year": severance_per_year,
            "expected_service_days": service_days,
            "severance_total": int(severance_per_year * service_days / 365),
            "legal_basis": "근로자퇴직급여보장법 제8조",
            "note": "1년 이상 계속 근로 시 퇴직금 지급 의무"
        }

        # 4인 이하 사업장 확인
        if emp_info.employee_count < 5:
            result.calculations["severance_pay"]["warning"] = "5인 미만 사업장은 퇴직급여 적용 대상이나 확인 필요"

    def _check_social_insurance(self, result: StressTestResult):
        """4대보험 가입 위반 검사"""
        social = result.social_insurance_info
        emp = result.employment_info

        # 단시간 근로자 기준 (월 60시간, 주 15시간 미만은 일부 예외)
        weekly_hours = result.work_time_info.weekly_hours

        violations_found = []

        if not social.has_national_pension:
            violations_found.append(("국민연금", "국민연금법 제8조", "4.5%"))

        if not social.has_health_insurance:
            violations_found.append(("건강보험", "국민건강보험법 제5조", "3.545%"))

        if not social.has_employment_insurance:
            violations_found.append(("고용보험", "고용보험법 제8조", "0.9%"))

        if not social.has_industrial_insurance:
            violations_found.append(("산재보험", "산업재해보상보험법 제6조", "사업주 전액 부담"))

        for insurance_name, legal_basis, rate in violations_found:
            # 예외 사항 확인
            is_exception = False
            if weekly_hours < 15 and insurance_name in ["국민연금", "건강보험"]:
                is_exception = True  # 주 15시간 미만은 적용 제외 가능

            if is_exception:
                continue

            violation = ViolationDetail(
                violation_type=ViolationType.SOCIAL_INSURANCE,
                risk_level=RiskLevel.HIGH,
                description=f"{insurance_name} 미가입 (사용자 의무 위반)",
                legal_basis=legal_basis,
                current_value="미가입",
                legal_standard="가입 필수",
                sanctions=[LegalSanction.ADMINISTRATIVE_FINE, LegalSanction.BACK_PAY],
                sanction_details=f"과태료 및 소급 가입 시 사용자 부담분 ({rate}) 추징",
                remediation=f"{insurance_name} 즉시 가입 필요",
            )
            result.violations.append(violation)

        # 4대보험 비용 계산
        if result.wage_info.total_salary > 0:
            result.calculations["social_insurance_cost"] = self._calculate_social_insurance_cost(
                result.wage_info.total_salary, emp.employee_count
            )

    def _calculate_social_insurance_cost(
        self,
        monthly_salary: int,
        employee_count: int = 10
    ) -> Dict[str, Any]:
        """4대보험 비용 계산"""
        salary = Decimal(str(monthly_salary))
        rates = self.INSURANCE_RATES

        # 국민연금
        pension_employee = int(salary * rates["national_pension"]["employee"])
        pension_employer = int(salary * rates["national_pension"]["employer"])

        # 건강보험
        health_employee = int(salary * rates["health_insurance"]["employee"])
        health_employer = int(salary * rates["health_insurance"]["employer"])

        # 장기요양보험 (건강보험료의 12.95%)
        longterm_employee = int(Decimal(str(health_employee)) * rates["longterm_care"]["rate"])
        longterm_employer = int(Decimal(str(health_employer)) * rates["longterm_care"]["rate"])

        # 고용보험
        employment_employee = int(salary * rates["employment_insurance"]["employee"])
        if employee_count < 150:
            employment_employer = int(salary * rates["employment_insurance"]["employer_150"])
        elif employee_count < 1000:
            employment_employer = int(salary * rates["employment_insurance"]["employer_150_1000"])
        else:
            employment_employer = int(salary * rates["employment_insurance"]["employer_1000"])

        # 산재보험 (사업주 전액)
        industrial_employer = int(salary * rates["industrial_insurance"]["average_rate"])

        total_employee = pension_employee + health_employee + longterm_employee + employment_employee
        total_employer = pension_employer + health_employer + longterm_employer + employment_employer + industrial_employer

        return {
            "monthly_salary": monthly_salary,
            "employee_deduction": {
                "national_pension": pension_employee,
                "health_insurance": health_employee,
                "longterm_care": longterm_employee,
                "employment_insurance": employment_employee,
                "total": total_employee,
            },
            "employer_contribution": {
                "national_pension": pension_employer,
                "health_insurance": health_employer,
                "longterm_care": longterm_employer,
                "employment_insurance": employment_employer,
                "industrial_insurance": industrial_employer,
                "total": total_employer,
            },
            "net_salary": monthly_salary - total_employee,
            "total_labor_cost": monthly_salary + total_employer,
        }

    def _check_annual_leave(self, result: StressTestResult):
        """연차휴가 위반 검사"""
        emp = result.employment_info

        if emp.start_date is None:
            return

        # 근속 연수 계산
        today = date.today()
        if emp.end_date and emp.end_date < today:
            reference_date = emp.end_date
        else:
            reference_date = today

        service_days = (reference_date - emp.start_date).days
        service_years = service_days / 365

        # 연차휴가 일수 계산
        if service_years < 1:
            # 1년 미만: 1개월 개근 시 1일
            months_worked = service_days // 30
            annual_leave = min(months_worked, 11)
            leave_type = "입사 1년 미만 연차"
        else:
            # 1년 이상: 15일 + 2년마다 1일 추가 (최대 25일)
            years_completed = int(service_years)
            additional_days = (years_completed - 1) // 2
            annual_leave = min(self.ANNUAL_LEAVE_BASE + additional_days, self.ANNUAL_LEAVE_MAX)
            leave_type = f"{years_completed}년차 연차"

        # 연차수당 계산
        hourly = result.wage_info.hourly_wage or Decimal(str(self.minimum_wage))
        daily_wage = int(hourly * 8)
        annual_leave_value = annual_leave * daily_wage

        result.calculations["annual_leave"] = {
            "service_days": service_days,
            "service_years": round(service_years, 1),
            "leave_type": leave_type,
            "annual_leave_days": annual_leave,
            "daily_wage": daily_wage,
            "annual_leave_value": annual_leave_value,
            "legal_basis": "근로기준법 제60조",
            "note": "미사용 연차는 연차수당으로 지급 필요"
        }

    def _check_illegal_deduction(self, result: StressTestResult):
        """불법 공제 검사"""
        deductions = result.wage_info.deductions

        illegal_deductions = []

        # 위약금, 손해배상 예정 공제
        if "위약금" in deductions:
            illegal_deductions.append({
                "name": "위약금",
                "amount": deductions["위약금"],
                "legal_basis": "근로기준법 제20조",
                "description": "근로 불이행 위약금 예정 금지"
            })

        # 교육비 강제 공제 (퇴사 시 환급 조건)
        if "교육비" in deductions:
            illegal_deductions.append({
                "name": "교육비 강제 공제",
                "amount": deductions["교육비"],
                "legal_basis": "근로기준법 제20조",
                "description": "의무 교육비 공제는 위약금 예정에 해당할 수 있음"
            })

        # 장비/유니폼 등 업무용품 비용 공제
        for item in ["장비비", "유니폼비"]:
            if item in deductions:
                illegal_deductions.append({
                    "name": item,
                    "amount": deductions[item],
                    "legal_basis": "근로기준법 제43조",
                    "description": "업무 수행에 필요한 비용은 사용자 부담 원칙"
                })

        for illegal in illegal_deductions:
            violation = ViolationDetail(
                violation_type=ViolationType.ILLEGAL_DEDUCTION,
                risk_level=RiskLevel.HIGH,
                description=f"불법 공제: {illegal['name']} {illegal['amount']:,}원",
                legal_basis=illegal['legal_basis'],
                current_value=illegal['amount'],
                legal_standard=0,
                shortage_amount=illegal['amount'],
                monthly_shortage=illegal['amount'],
                annual_shortage=illegal['amount'] * 12,
                sanctions=[LegalSanction.BACK_PAY, LegalSanction.CRIMINAL_PENALTY],
                sanction_details="500만원 이하 벌금 (근로기준법 제114조)",
                remediation="불법 공제 금액 전액 환급 필요",
            )
            result.violations.append(violation)
            result.total_underpayment += illegal['amount']
            result.annual_underpayment += illegal['amount'] * 12

    # ========== 평가 및 보고서 메서드 ==========

    def _evaluate_risk_level(self, result: StressTestResult):
        """위험도 평가"""
        critical_count = len(result.critical_violations)
        high_count = len(result.high_violations)
        medium_count = sum(1 for v in result.violations if v.risk_level == RiskLevel.MEDIUM)

        # 점수 계산 (100점 만점에서 감점)
        risk_score = 0
        risk_score += critical_count * 30
        risk_score += high_count * 15
        risk_score += medium_count * 5
        risk_score = min(100, risk_score)

        result.risk_score = risk_score

        # 금액 기준 추가 평가
        if critical_count > 0 or result.annual_underpayment >= 5_000_000:
            result.risk_level = RiskLevel.CRITICAL
        elif high_count >= 2 or result.annual_underpayment >= 2_000_000:
            result.risk_level = RiskLevel.HIGH
        elif high_count >= 1 or medium_count >= 3 or result.annual_underpayment >= 1_000_000:
            result.risk_level = RiskLevel.MEDIUM
        else:
            result.risk_level = RiskLevel.LOW

    def _calculate_compliance_score(self, result: StressTestResult):
        """준법 점수 계산 (0-100)"""
        # 기본 100점에서 감점
        score = 100.0

        for v in result.violations:
            if v.risk_level == RiskLevel.CRITICAL:
                score -= 25
            elif v.risk_level == RiskLevel.HIGH:
                score -= 15
            elif v.risk_level == RiskLevel.MEDIUM:
                score -= 8
            elif v.risk_level == RiskLevel.LOW:
                score -= 3

        result.compliance_score = max(0, score)

    def _simulate_scenarios(
        self,
        result: StressTestResult,
        months: int = 12
    ) -> Dict[str, Any]:
        """시나리오 시뮬레이션"""
        scenarios = {
            "current": {
                "description": "현재 계약 조건 유지",
                "monthly_risk": result.total_underpayment,
                "annual_risk": result.annual_underpayment,
                "three_year_risk": result.three_year_underpayment,
                "potential_penalty": self._estimate_penalty(result),
            }
        }

        # 완전 준법 시나리오
        total_additional_cost = result.total_underpayment
        scenarios["full_compliance"] = {
            "description": "모든 위반 사항 시정",
            "additional_monthly_cost": total_additional_cost,
            "additional_annual_cost": result.annual_underpayment,
            "risk_eliminated": result.annual_underpayment + self._estimate_penalty(result),
            "roi": "법적 리스크 해소, 근로자 만족도 향상"
        }

        # 우선순위 시정 시나리오
        critical_cost = sum(v.monthly_shortage for v in result.critical_violations)
        scenarios["priority_compliance"] = {
            "description": "Critical 위반만 우선 시정",
            "additional_monthly_cost": critical_cost,
            "remaining_risks": len(result.high_violations),
            "recommendation": "최저임금, 근로시간 위반 우선 해결"
        }

        return scenarios

    def _estimate_penalty(self, result: StressTestResult) -> int:
        """예상 제재금 추정"""
        penalty = 0

        for v in result.violations:
            if LegalSanction.CRIMINAL_PENALTY in v.sanctions:
                penalty += 10_000_000  # 형사 처벌 리스크
            if LegalSanction.ADMINISTRATIVE_FINE in v.sanctions:
                penalty += 5_000_000   # 과태료
            if LegalSanction.PENALTY_INTEREST in v.sanctions:
                # 지연이자 (체불액의 20%/년)
                penalty += int(v.annual_shortage * float(self.DELAY_INTEREST_RATE))

        return penalty

    def _generate_summary(self, result: StressTestResult) -> str:
        """결과 요약 생성"""
        lines = [
            "=== 근로계약서 법적 스트레스 테스트 결과 ===",
            "",
            f"위험도: {result.risk_level.value} (점수: {result.risk_score:.0f}/100)",
            f"준법 점수: {result.compliance_score:.1f}/100",
            f"위반 항목: {len(result.violations)}건 (Critical: {len(result.critical_violations)}, High: {len(result.high_violations)})",
            "",
            f"월간 체불 예상액: {result.total_underpayment:,}원",
            f"연간 체불 예상액: {result.annual_underpayment:,}원",
            f"3년간 체불 예상액: {result.three_year_underpayment:,}원 (소멸시효)",
            "",
        ]

        if result.violations:
            lines.append("[위반 사항]")
            for i, v in enumerate(result.violations, 1):
                lines.append(f"{i}. [{v.risk_level.value}] {v.violation_type.value}")
                lines.append(f"   {v.description}")
                if v.monthly_shortage > 0:
                    lines.append(f"   월 체불액: {v.monthly_shortage:,}원")
            lines.append("")

        return "\n".join(lines)

    def _generate_detailed_report(self, result: StressTestResult) -> str:
        """상세 보고서 생성"""
        lines = [self._generate_summary(result)]

        lines.append("\n[상세 분석]")

        # 급여 분석
        wage = result.wage_info
        lines.append(f"\n1. 급여 분석")
        lines.append(f"   - 총 급여: {wage.total_salary:,}원")
        lines.append(f"   - 시급: {wage.hourly_wage:,.0f}원")
        lines.append(f"   - 포괄임금제: {'적용' if wage.is_inclusive_wage else '미적용'}")

        # 근로시간 분석
        work = result.work_time_info
        lines.append(f"\n2. 근로시간 분석")
        lines.append(f"   - 주간 근로시간: {work.weekly_hours}시간")
        lines.append(f"   - 일일 근로시간: {work.daily_hours}시간")
        lines.append(f"   - 근무형태: {work.shift_type}")

        # 4대보험 비용
        if "social_insurance_cost" in result.calculations:
            ins = result.calculations["social_insurance_cost"]
            lines.append(f"\n3. 4대보험 비용")
            lines.append(f"   - 근로자 공제액: {ins['employee_deduction']['total']:,}원/월")
            lines.append(f"   - 사용자 부담액: {ins['employer_contribution']['total']:,}원/월")
            lines.append(f"   - 실수령액: {ins['net_salary']:,}원")

        # 퇴직금
        if "severance_pay" in result.calculations:
            sev = result.calculations["severance_pay"]
            lines.append(f"\n4. 퇴직금 예상")
            lines.append(f"   - 1년 기준 퇴직금: {sev['severance_per_year']:,}원")

        # 시정 권고
        lines.append("\n[시정 권고사항]")
        for i, v in enumerate(result.violations, 1):
            lines.append(f"{i}. {v.remediation}")
            lines.append(f"   근거: {v.legal_basis}")

        return "\n".join(lines)


# ========== 편의 함수 ==========

def run_stress_test(contract_text: str) -> StressTestResult:
    """간편 스트레스 테스트 실행"""
    test = LegalStressTest()
    return test.run(contract_text)


def calculate_minimum_wage_compliance(
    monthly_salary: int,
    weekly_hours: float = 40.0
) -> Dict[str, Any]:
    """최저임금 준수 여부 계산"""
    test = LegalStressTest()

    # 월 근로시간 계산 (주휴시간 포함)
    monthly_hours = Decimal(str((weekly_hours + 8) * (365 / 7 / 12)))
    hourly_wage = Decimal(str(monthly_salary)) / monthly_hours

    is_compliant = hourly_wage >= test.minimum_wage
    shortage = max(Decimal("0"), Decimal(str(test.minimum_wage)) - hourly_wage)

    return {
        "monthly_salary": monthly_salary,
        "hourly_wage": int(hourly_wage),
        "minimum_wage": test.minimum_wage,
        "is_compliant": is_compliant,
        "hourly_shortage": int(shortage),
        "monthly_shortage": int(shortage * monthly_hours),
        "annual_shortage": int(shortage * monthly_hours * 12)
    }


def calculate_overtime_cost(
    hourly_wage: int,
    overtime_hours: float,
    night_hours: float = 0,
    holiday_hours: float = 0
) -> Dict[str, Any]:
    """초과근로 비용 계산"""
    wage = Decimal(str(hourly_wage))

    overtime_pay = wage * Decimal("1.5") * Decimal(str(overtime_hours))
    night_pay = wage * Decimal("1.5") * Decimal(str(night_hours))

    # 휴일근로 (8시간 이내 1.5배, 초과 2배)
    holiday_within_8 = min(holiday_hours, 8)
    holiday_over_8 = max(0, holiday_hours - 8)
    holiday_pay = (wage * Decimal("1.5") * Decimal(str(holiday_within_8)) +
                   wage * Decimal("2.0") * Decimal(str(holiday_over_8)))

    total = overtime_pay + night_pay + holiday_pay

    return {
        "hourly_wage": hourly_wage,
        "overtime_hours": overtime_hours,
        "overtime_pay": int(overtime_pay),
        "night_hours": night_hours,
        "night_pay": int(night_pay),
        "holiday_hours": holiday_hours,
        "holiday_pay": int(holiday_pay),
        "total_premium_pay": int(total),
        "total_with_base": int(total + wage * Decimal(str(overtime_hours + night_hours + holiday_hours))),
    }


def calculate_severance_pay(
    monthly_salary: int,
    service_days: int
) -> Dict[str, Any]:
    """퇴직금 계산"""
    if service_days < 365:
        return {
            "eligible": False,
            "reason": "1년 미만 근무",
            "service_days": service_days,
        }

    avg_daily_wage = Decimal(str(monthly_salary * 3)) / Decimal("90")
    severance = int(avg_daily_wage * 30 * service_days / 365)

    return {
        "eligible": True,
        "monthly_salary": monthly_salary,
        "service_days": service_days,
        "service_years": round(service_days / 365, 2),
        "average_daily_wage": int(avg_daily_wage),
        "severance_pay": severance,
        "legal_basis": "근로자퇴직급여보장법 제8조",
    }
