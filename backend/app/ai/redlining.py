"""
Generative Redlining (자동 수정 제안)
- 독소 조항을 법적으로 안전한 문장으로 재작성
- Git Diff 스타일 시각화 (Red/Blue)
- 수정 사유 및 법적 근거 제공

Reference: Contract Redlining, Legal Tech AI
"""

import os
import re
import json
import time
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from enum import Enum
from difflib import SequenceMatcher, unified_diff

from app.core.config import settings
from app.core.token_usage_tracker import record_llm_usage


class ChangeType(Enum):
    """변경 유형"""
    DELETE = "delete"      # 삭제 (위험 조항)
    INSERT = "insert"      # 추가 (필수 조항)
    MODIFY = "modify"      # 수정 (개선)
    KEEP = "keep"          # 유지


@dataclass
class RedlineChange:
    """개별 수정 사항"""
    change_type: ChangeType
    original_text: str
    revised_text: str
    reason: str
    legal_basis: str = ""
    severity: str = "Medium"  # High, Medium, Low
    position: Dict[str, int] = field(default_factory=dict)  # start, end

    @property
    def diff_html(self) -> str:
        """HTML 형식의 Diff 출력"""
        if self.change_type == ChangeType.DELETE:
            return f'<del class="redline-delete">{self.original_text}</del>'
        elif self.change_type == ChangeType.INSERT:
            return f'<ins class="redline-insert">{self.revised_text}</ins>'
        elif self.change_type == ChangeType.MODIFY:
            return (
                f'<del class="redline-delete">{self.original_text}</del>'
                f'<ins class="redline-insert">{self.revised_text}</ins>'
            )
        else:
            return self.original_text


@dataclass
class RedlineResult:
    """Redlining 결과"""
    original_text: str
    revised_text: str
    changes: List[RedlineChange] = field(default_factory=list)
    summary: str = ""
    risk_reduction: float = 0.0  # 위험도 감소율

    @property
    def change_count(self) -> int:
        return len(self.changes)

    @property
    def high_risk_changes(self) -> List[RedlineChange]:
        return [c for c in self.changes if c.severity == "High"]

    def get_unified_diff(self) -> str:
        """Unified Diff 형식 출력"""
        original_lines = self.original_text.splitlines(keepends=True)
        revised_lines = self.revised_text.splitlines(keepends=True)

        diff = unified_diff(
            original_lines,
            revised_lines,
            fromfile="원본 계약서",
            tofile="수정 제안"
        )
        return "".join(diff)

    def get_html_diff(self) -> str:
        """HTML Diff 출력"""
        html_parts = []
        for change in self.changes:
            html_parts.append(change.diff_html)
        return "".join(html_parts)


class GenerativeRedlining:
    """
    Generative Redlining 구현

    사용법:
        redliner = GenerativeRedlining()
        result = redliner.redline(contract_text)
        print(result.get_unified_diff())
    """

    # 수정 프롬프트
    REDLINE_PROMPT = """당신은 근로계약서 수정 전문 변호사입니다.
다음 계약서 조항을 검토하고 법적 문제가 있는 부분을 수정하세요.

[원본 조항]
{clause}

[수정 지침]
1. 근로기준법에 위반되는 내용은 법에 맞게 수정
2. 모호하거나 불명확한 표현은 명확하게 수정
3. 근로자에게 불리한 조항은 균형있게 수정
4. 누락된 필수 내용이 있으면 추가

[출력 형식 - JSON]
{{
    "changes": [
        {{
            "change_type": "DELETE/INSERT/MODIFY/KEEP",
            "original": "원본 텍스트",
            "revised": "수정된 텍스트",
            "reason": "수정 사유",
            "legal_basis": "법적 근거 (예: 근로기준법 제17조)",
            "severity": "High/Medium/Low"
        }}
    ],
    "revised_full_text": "전체 수정된 조항",
    "summary": "수정 요약"
}}"""

    # 위험 패턴 및 수정 제안 (확장)
    RISK_PATTERNS = {
        "포괄임금": {
            "pattern": r"(포괄\s*(임금|급여)|모든\s*수당\s*포함|연장\s*근로\s*수당?\s*포함)",
            "severity": "High",
            "suggestion": "기본급과 각종 수당(연장근로수당, 야간근로수당, 휴일근로수당)을 별도로 명시",
            "legal_basis": "근로기준법 제56조"
        },
        "일방적_해지": {
            "pattern": r"(회사|사용자|갑)\s*(가|은)\s*(언제든지|일방적으로|즉시)\s*(해지|해고|계약\s*종료)",
            "severity": "High",
            "suggestion": "해고는 정당한 사유가 있어야 하며, 30일 전 예고 또는 30일분 통상임금 지급",
            "legal_basis": "근로기준법 제23조, 제26조"
        },
        "위약금": {
            "pattern": r"(위약금|교육비\s*반환|연수비\s*반환|채용\s*비용).*\d+\s*(만\s*)?원",
            "severity": "High",
            "suggestion": "근로계약 불이행에 대한 위약금 예정 금지. 실손해 배상만 가능",
            "legal_basis": "근로기준법 제20조"
        },
        "손해배상_임금공제": {
            "pattern": r"(손해|배상|변상).*임금.*공제|임금.*공제.*(손해|배상)|전액\s*공제",
            "severity": "High",
            "suggestion": "임금은 전액 지급 원칙. 손해배상은 별도 절차 필요하며 임금에서 일방 공제 불가",
            "legal_basis": "근로기준법 제43조"
        },
        "휴게시간_미부여": {
            "pattern": r"휴게\s*[:\s]*(없음|0\s*분)|휴식\s*시간\s*없",
            "severity": "High",
            "suggestion": "4시간 초과 근무 시 30분, 8시간 초과 시 1시간 이상 휴게시간 부여 필수",
            "legal_basis": "근로기준법 제54조"
        },
        "공휴일_연차대체": {
            "pattern": r"공휴일.*(연차|휴가).*대체|공휴일.*(연차|휴가).*간주|공휴일.*(연차|휴가).*사용.*것으로",
            "severity": "High",
            "suggestion": "관공서 공휴일은 유급휴일. 근로자 동의 없이 연차로 대체 불가",
            "legal_basis": "근로기준법 제55조 제2항"
        },
        "연차휴가_미발생": {
            "pattern": r"1\s*년\s*미만.*(연차|휴가).*(발생하지|없|미발생)|연차.*(휴가)?.*발생하지\s*않",
            "severity": "High",
            "suggestion": "1년 미만 근로자도 1개월 개근 시 1일의 유급휴가 발생",
            "legal_basis": "근로기준법 제60조 제2항"
        },
        "수습_4대보험_미가입": {
            "pattern": r"수습.*4대\s*보험.*가입하지|수습.*보험.*미가입|수습\s*기간.*보험.*제외",
            "severity": "High",
            "suggestion": "4대 사회보험은 입사일부터 가입 의무. 수습 여부와 무관",
            "legal_basis": "고용보험법, 국민연금법, 국민건강보험법"
        },
        "계약서_미교부": {
            "pattern": r"(계약서|근로계약).*보관.*요청.*열람|계약서.*교부하지|사본.*제공.*않",
            "severity": "High",
            "suggestion": "근로계약서는 작성 즉시 근로자에게 1부 교부 필수",
            "legal_basis": "근로기준법 제17조"
        },
        "법령_우선순위_위반": {
            "pattern": r"(근로기준법|법령).*보다.*(회사|내규|취업규칙).*우선|(회사|내규|취업규칙).*우선.*적용",
            "severity": "High",
            "suggestion": "근로기준법에 미달하는 근로조건은 무효이며, 법 기준 적용",
            "legal_basis": "근로기준법 제15조"
        },
        "경쟁금지": {
            "pattern": r"(경쟁\s*금지|동종\s*업체\s*취업\s*금지|전직\s*금지)",
            "severity": "Medium",
            "suggestion": "경쟁금지 조항은 합리적 범위(기간, 지역, 대상)로 제한 필요",
            "legal_basis": "판례법리"
        },
        "퇴직_제한": {
            "pattern": r"(퇴직|사직)\s*(은|을)?\s*\d+\s*(개월|일)\s*(전|이전).*통보",
            "severity": "Medium",
            "suggestion": "민법상 해지통고 기간은 1개월. 과도한 제한은 무효",
            "legal_basis": "민법 제660조"
        },
        "과도한_근무시간": {
            "pattern": r"(주|매주)\s*[67]\s*일\s*근무|[67]\s*일\s*/\s*주|매일\s*근무|주\s*(52|53|54|55|60|70)\s*시간|월\s*~\s*일\s*근무",
            "severity": "High",
            "suggestion": "주 52시간(연장 포함) 초과 근로 금지. 주휴일 1일 이상 보장 필요",
            "legal_basis": "근로기준법 제50조, 제53조, 제55조"
        },
        "야간근로_미성년": {
            "pattern": r"(18세\s*미만|미성년|청소년).*야간|야간.*(18세\s*미만|미성년|청소년)",
            "severity": "High",
            "suggestion": "18세 미만 근로자는 야간근로(22시~06시) 및 휴일근로 금지",
            "legal_basis": "근로기준법 제70조"
        },
    }

    def __init__(
        self,
        llm_client: Optional[Any] = None,
        model: str = None,
        contract_id: Optional[str] = None
    ):
        """
        Args:
            llm_client: OpenAI 클라이언트
            model: 사용할 LLM 모델 (기본값: settings.LLM_REDLINER_MODEL)
            contract_id: 계약서 ID (토큰 추적용)
        """
        self.model = model if model else settings.LLM_REDLINER_MODEL
        self.contract_id = contract_id

        if llm_client is None:
            try:
                from openai import OpenAI
                self.llm_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            except ImportError:
                self.llm_client = None
        else:
            self.llm_client = llm_client

    def redline(
        self,
        contract_text: str,
        clause_by_clause: bool = True
    ) -> RedlineResult:
        """
        계약서 Redlining 수행

        Args:
            contract_text: 계약서 텍스트
            clause_by_clause: 조항별 분석 여부

        Returns:
            RedlineResult: Redlining 결과
        """
        result = RedlineResult(
            original_text=contract_text,
            revised_text=contract_text
        )

        if clause_by_clause:
            # 조항별 분석
            clauses = self._split_clauses(contract_text)
            all_changes = []
            revised_parts = []

            for clause in clauses:
                clause_result = self._redline_clause(clause)
                all_changes.extend(clause_result.changes)
                revised_parts.append(clause_result.revised_text)

            result.changes = all_changes
            result.revised_text = "\n\n".join(revised_parts)
        else:
            # 전체 분석
            result = self._redline_full(contract_text)

        # 위험도 감소율 계산
        result.risk_reduction = self._calculate_risk_reduction(result)

        # 요약 생성
        result.summary = self._generate_summary(result)

        return result

    def _split_clauses(self, text: str) -> List[str]:
        """조항별 분할 (다양한 형식 지원)"""
        # 여러 형식 지원: 제1조, 1., 제1항, - 등
        patterns = [
            r'(?=제\s*\d+\s*조)',           # 제1조, 제 2 조
            r'(?=\n\s*\d{1,2}\s*\.\s)',     # 1. , 2.
            r'(?=\n\s*\d{1,2}\s*\)\s)',     # 1) , 2)
            r'(?=\n\s*제\s*\d+\s*항)',      # 제1항
        ]

        clauses = [text]
        for pattern in patterns:
            new_clauses = []
            for clause in clauses:
                parts = re.split(pattern, clause)
                new_clauses.extend([p.strip() for p in parts if p.strip()])
            if len(new_clauses) > len(clauses):
                clauses = new_clauses
                break

        # 분할이 안 되면 줄바꿈 기준으로 의미 있는 단락 추출
        if len(clauses) <= 1:
            paragraphs = text.split('\n\n')
            clauses = [p.strip() for p in paragraphs if p.strip() and len(p.strip()) > 20]

        return clauses if clauses else [text]

    def _redline_clause(self, clause: str) -> RedlineResult:
        """개별 조항 Redlining"""
        result = RedlineResult(
            original_text=clause,
            revised_text=clause
        )

        # 1. 규칙 기반 위험 탐지
        rule_changes = self._apply_rules(clause)
        result.changes.extend(rule_changes)

        # 2. LLM 기반 분석 (규칙으로 탐지 안 된 경우)
        if not rule_changes and self.llm_client is not None:
            llm_result = self._llm_redline(clause)
            result.changes.extend(llm_result.get("changes", []))
            result.revised_text = llm_result.get("revised_full_text", clause)
        elif rule_changes:
            # 규칙 기반 수정 적용
            result.revised_text = self._apply_changes(clause, rule_changes)

        return result

    def _apply_rules(self, text: str) -> List[RedlineChange]:
        """규칙 기반 위험 탐지 및 수정 제안"""
        changes = []

        for pattern_name, pattern_info in self.RISK_PATTERNS.items():
            matches = re.finditer(pattern_info["pattern"], text, re.IGNORECASE)

            for match in matches:
                changes.append(RedlineChange(
                    change_type=ChangeType.MODIFY,
                    original_text=match.group(0),
                    revised_text=f"[수정 필요: {pattern_info['suggestion']}]",
                    reason=f"{pattern_name} 위험 패턴 탐지",
                    legal_basis=pattern_info["legal_basis"],
                    severity=pattern_info["severity"],
                    position={"start": match.start(), "end": match.end()}
                ))

        return changes

    def _llm_redline(self, clause: str) -> Dict[str, Any]:
        """LLM 기반 Redlining"""
        llm_start = time.time()
        try:
            prompt = self.REDLINE_PROMPT.format(clause=clause)

            response = self.llm_client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "당신은 근로계약서 수정 전문 변호사입니다."
                    },
                    {"role": "user", "content": prompt}
                ],

                response_format={"type": "json_object"}
            )

            llm_duration = (time.time() - llm_start) * 1000

            # 토큰 사용량 기록
            if response.usage and self.contract_id:
                cached = 0
                if hasattr(response.usage, 'prompt_tokens_details') and response.usage.prompt_tokens_details:
                    cached = getattr(response.usage.prompt_tokens_details, 'cached_tokens', 0) or 0
                record_llm_usage(
                    contract_id=self.contract_id,
                    module="redlining.redline",
                    model=self.model,
                    input_tokens=response.usage.prompt_tokens,
                    output_tokens=response.usage.completion_tokens,
                    cached_tokens=cached,
                    duration_ms=llm_duration
                )

            result = json.loads(response.choices[0].message.content)

            # 변경사항 파싱
            parsed_changes = []
            for c in result.get("changes", []):
                change_type = ChangeType[c.get("change_type", "KEEP")]
                parsed_changes.append(RedlineChange(
                    change_type=change_type,
                    original_text=c.get("original", ""),
                    revised_text=c.get("revised", ""),
                    reason=c.get("reason", ""),
                    legal_basis=c.get("legal_basis", ""),
                    severity=c.get("severity", "Medium")
                ))

            return {
                "changes": parsed_changes,
                "revised_full_text": result.get("revised_full_text", clause),
                "summary": result.get("summary", "")
            }

        except Exception as e:
            print(f"LLM redline error: {e}")
            return {"changes": [], "revised_full_text": clause}

    def _redline_full(self, text: str) -> RedlineResult:
        """전체 계약서 Redlining"""
        result = RedlineResult(
            original_text=text,
            revised_text=text
        )

        # 규칙 기반 분석
        result.changes = self._apply_rules(text)

        # LLM 분석
        if self.llm_client is not None:
            llm_result = self._llm_redline(text)
            result.changes.extend(llm_result.get("changes", []))
            result.revised_text = llm_result.get("revised_full_text", text)

        return result

    def _apply_changes(
        self,
        text: str,
        changes: List[RedlineChange]
    ) -> str:
        """변경사항 적용"""
        # 위치 기준 역순 정렬 (뒤에서부터 수정)
        sorted_changes = sorted(
            [c for c in changes if c.position],
            key=lambda x: x.position.get("start", 0),
            reverse=True
        )

        result = text
        for change in sorted_changes:
            start = change.position.get("start", 0)
            end = change.position.get("end", len(text))
            result = result[:start] + change.revised_text + result[end:]

        return result

    def _calculate_risk_reduction(self, result: RedlineResult) -> float:
        """위험도 감소율 계산"""
        if not result.changes:
            return 0.0

        severity_weights = {"High": 3, "Medium": 2, "Low": 1}
        total_risk = sum(
            severity_weights.get(c.severity, 1)
            for c in result.changes
        )
        max_risk = len(result.changes) * 3

        return total_risk / max_risk if max_risk > 0 else 0.0

    def _generate_summary(self, result: RedlineResult) -> str:
        """수정 요약 생성"""
        high_count = len([c for c in result.changes if c.severity == "High"])
        medium_count = len([c for c in result.changes if c.severity == "Medium"])
        low_count = len([c for c in result.changes if c.severity == "Low"])

        lines = [
            f"총 {result.change_count}개 수정 제안",
            f"- 높은 위험: {high_count}건",
            f"- 중간 위험: {medium_count}건",
            f"- 낮은 위험: {low_count}건",
            f"위험도 감소 예상: {result.risk_reduction:.0%}"
        ]

        return "\n".join(lines)

    def get_diff_view(
        self,
        result: RedlineResult,
        format: str = "html"
    ) -> str:
        """
        Diff 뷰 생성

        Args:
            result: Redlining 결과
            format: 출력 형식 (html, unified, side_by_side)

        Returns:
            Diff 문자열
        """
        if format == "html":
            return self._generate_html_diff(result)
        elif format == "unified":
            return result.get_unified_diff()
        else:
            return self._generate_side_by_side(result)

    def _generate_html_diff(self, result: RedlineResult) -> str:
        """HTML Diff 생성"""
        html = ['<div class="redline-diff">']

        for change in result.changes:
            html.append('<div class="redline-change">')
            html.append(f'<div class="severity-badge severity-{change.severity.lower()}">'
                       f'{change.severity}</div>')
            html.append('<div class="change-content">')

            if change.change_type == ChangeType.DELETE:
                html.append(f'<del>{change.original_text}</del>')
            elif change.change_type == ChangeType.INSERT:
                html.append(f'<ins>{change.revised_text}</ins>')
            elif change.change_type == ChangeType.MODIFY:
                html.append(f'<del>{change.original_text}</del>')
                html.append('<span class="arrow"> -> </span>')
                html.append(f'<ins>{change.revised_text}</ins>')

            html.append('</div>')
            html.append(f'<div class="reason">{change.reason}</div>')
            if change.legal_basis:
                html.append(f'<div class="legal-basis">근거: {change.legal_basis}</div>')
            html.append('</div>')

        html.append('</div>')
        return "\n".join(html)

    def _generate_side_by_side(self, result: RedlineResult) -> str:
        """Side-by-side Diff 생성"""
        original_lines = result.original_text.splitlines()
        revised_lines = result.revised_text.splitlines()

        lines = []
        lines.append("원본                              | 수정안")
        lines.append("-" * 70)

        matcher = SequenceMatcher(None, original_lines, revised_lines)

        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == "equal":
                for line in original_lines[i1:i2]:
                    lines.append(f"{line[:30]:30} | {line[:30]}")
            elif tag == "delete":
                for line in original_lines[i1:i2]:
                    lines.append(f"- {line[:30]:30} |")
            elif tag == "insert":
                for line in revised_lines[j1:j2]:
                    lines.append(f"{' ':32} | + {line[:30]}")
            elif tag == "replace":
                for o_line, r_line in zip(original_lines[i1:i2], revised_lines[j1:j2]):
                    lines.append(f"- {o_line[:30]:30} | + {r_line[:30]}")

        return "\n".join(lines)


# 편의 함수
def redline_contract(contract_text: str) -> RedlineResult:
    """간편 계약서 Redlining"""
    redliner = GenerativeRedlining()
    return redliner.redline(contract_text)


def get_redline_html(contract_text: str) -> str:
    """HTML Diff 생성"""
    redliner = GenerativeRedlining()
    result = redliner.redline(contract_text)
    return redliner.get_diff_view(result, format="html")
