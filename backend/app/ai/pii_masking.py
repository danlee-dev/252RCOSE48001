"""
PII Masking Pipeline (Privacy-Preserving)
- 개인정보 자동 탐지 및 마스킹
- 주민등록번호, 전화번호, 이메일, 계좌번호, 주소 등 비식별화
- LLM 전송 전 개인정보 유출 원천 차단
"""

import re
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum


class PIIType(Enum):
    """개인정보 유형"""
    RRN = "주민등록번호"
    PHONE = "전화번호"
    EMAIL = "이메일"
    ACCOUNT = "계좌번호"
    ADDRESS = "주소"
    NAME = "이름"
    CARD = "카드번호"
    PASSPORT = "여권번호"
    DRIVER_LICENSE = "운전면허번호"
    COMPANY_REG = "사업자등록번호"


@dataclass
class PIIMatch:
    """탐지된 개인정보"""
    pii_type: PIIType
    original: str
    masked: str
    start: int
    end: int


@dataclass
class MaskingResult:
    """마스킹 결과"""
    masked_text: str
    original_text: str
    pii_matches: List[PIIMatch] = field(default_factory=list)
    pii_map: Dict[str, str] = field(default_factory=dict)  # masked -> original (복원용)

    @property
    def pii_count(self) -> int:
        return len(self.pii_matches)

    @property
    def has_pii(self) -> bool:
        return self.pii_count > 0

    def get_summary(self) -> Dict[str, int]:
        """유형별 PII 개수 요약"""
        summary = {}
        for match in self.pii_matches:
            pii_type = match.pii_type.value
            summary[pii_type] = summary.get(pii_type, 0) + 1
        return summary


class PIIMasker:
    """
    개인정보 마스킹 파이프라인

    사용법:
        masker = PIIMasker()
        result = masker.mask(text)
        print(result.masked_text)  # 마스킹된 텍스트
        print(result.pii_count)    # 탐지된 개인정보 수
    """

    def __init__(self, mask_char: str = "*", preserve_format: bool = True):
        """
        Args:
            mask_char: 마스킹에 사용할 문자
            preserve_format: 형식 보존 여부 (예: 010-****-1234)
        """
        self.mask_char = mask_char
        self.preserve_format = preserve_format
        self._compile_patterns()

    def _compile_patterns(self):
        """정규표현식 패턴 컴파일"""
        self.patterns = {
            # 주민등록번호: 123456-1234567 또는 1234561234567
            PIIType.RRN: re.compile(
                r'\b(\d{6})[-\s]?([1-4]\d{6})\b'
            ),

            # 전화번호: 010-1234-5678, 02-123-4567, 031-1234-5678
            PIIType.PHONE: re.compile(
                r'\b(0\d{1,2})[-.\s]?(\d{3,4})[-.\s]?(\d{4})\b'
            ),

            # 이메일
            PIIType.EMAIL: re.compile(
                r'\b([a-zA-Z0-9._%+-]+)@([a-zA-Z0-9.-]+\.[a-zA-Z]{2,})\b'
            ),

            # 계좌번호: 다양한 형식 (10-16자리 숫자)
            PIIType.ACCOUNT: re.compile(
                r'\b(\d{2,6})[-\s]?(\d{2,6})[-\s]?(\d{2,6})[-\s]?(\d{0,6})\b'
            ),

            # 카드번호: 1234-5678-1234-5678
            PIIType.CARD: re.compile(
                r'\b(\d{4})[-\s]?(\d{4})[-\s]?(\d{4})[-\s]?(\d{4})\b'
            ),

            # 사업자등록번호: 123-12-12345
            PIIType.COMPANY_REG: re.compile(
                r'\b(\d{3})[-\s]?(\d{2})[-\s]?(\d{5})\b'
            ),

            # 여권번호: M12345678
            PIIType.PASSPORT: re.compile(
                r'\b([A-Z]{1,2})(\d{7,8})\b'
            ),

            # 운전면허번호: 12-12-123456-12
            PIIType.DRIVER_LICENSE: re.compile(
                r'\b(\d{2})[-\s]?(\d{2})[-\s]?(\d{6})[-\s]?(\d{2})\b'
            ),
        }

        # 주소 패턴 (시/도, 구/군, 동/읍/면 등)
        self.address_pattern = re.compile(
            r'(서울|부산|대구|인천|광주|대전|울산|세종|경기|강원|충북|충남|전북|전남|경북|경남|제주)'
            r'[시도]?\s*'
            r'[가-힣]+[시군구]\s*'
            r'[가-힣]+[동읍면로길]\s*'
            r'[\d-]+\s*'
            r'[가-힣\d\s,-]*'
        )

        # 한국 이름 패턴 (성 + 이름)
        self.korean_name_pattern = re.compile(
            r'\b([김이박최정강조윤장임한오서신권황안송류홍전고문손양배조백허유남심노하곽성차주우구신임나전민유진지엄채원천방공강현함변염양'
            r'복사판선설길연위표명기반왕금옥육인맹제모장남탁국여진어은편구용'
            r'])[가-힣]{1,3}\b'
        )

    def mask(self, text: str) -> MaskingResult:
        """
        텍스트에서 개인정보를 탐지하고 마스킹

        Args:
            text: 원본 텍스트

        Returns:
            MaskingResult: 마스킹 결과
        """
        if not text:
            return MaskingResult(masked_text="", original_text="", pii_matches=[], pii_map={})

        masked_text = text
        pii_matches: List[PIIMatch] = []
        pii_map: Dict[str, str] = {}

        # 1. 패턴 기반 PII 탐지 및 마스킹
        for pii_type, pattern in self.patterns.items():
            masked_text, matches = self._mask_pattern(
                masked_text, pattern, pii_type, pii_map
            )
            pii_matches.extend(matches)

        # 2. 주소 마스킹
        masked_text, address_matches = self._mask_addresses(masked_text, pii_map)
        pii_matches.extend(address_matches)

        # 3. 이름 마스킹 (context 기반)
        masked_text, name_matches = self._mask_names(masked_text, pii_map)
        pii_matches.extend(name_matches)

        return MaskingResult(
            masked_text=masked_text,
            original_text=text,
            pii_matches=pii_matches,
            pii_map=pii_map
        )

    def _mask_pattern(
        self,
        text: str,
        pattern: re.Pattern,
        pii_type: PIIType,
        pii_map: Dict[str, str]
    ) -> Tuple[str, List[PIIMatch]]:
        """패턴 기반 마스킹"""
        matches = []

        def replace_func(match):
            original = match.group(0)
            masked = self._generate_mask(original, pii_type)

            pii_map[masked] = original
            matches.append(PIIMatch(
                pii_type=pii_type,
                original=original,
                masked=masked,
                start=match.start(),
                end=match.end()
            ))

            return masked

        masked_text = pattern.sub(replace_func, text)
        return masked_text, matches

    def _mask_addresses(
        self,
        text: str,
        pii_map: Dict[str, str]
    ) -> Tuple[str, List[PIIMatch]]:
        """주소 마스킹"""
        matches = []

        def replace_func(match):
            original = match.group(0)
            # 시/도만 남기고 나머지 마스킹
            city = match.group(1)
            masked = f"{city} <ADDR_MASKED>"

            pii_map[masked] = original
            matches.append(PIIMatch(
                pii_type=PIIType.ADDRESS,
                original=original,
                masked=masked,
                start=match.start(),
                end=match.end()
            ))

            return masked

        masked_text = self.address_pattern.sub(replace_func, text)
        return masked_text, matches

    def _mask_names(
        self,
        text: str,
        pii_map: Dict[str, str]
    ) -> Tuple[str, List[PIIMatch]]:
        """
        이름 마스킹 (문맥 기반)
        - '갑', '을', '병' 등 계약서 용어는 제외
        - '본인', '당사자' 앞뒤의 이름만 마스킹
        """
        matches = []

        # 계약서에서 이름이 나올 수 있는 문맥 패턴
        name_context_patterns = [
            r'(성\s*명\s*[:\s]*)([김이박최정강조윤장임한오서신권황안송류홍][가-힣]{1,3})',
            r'(이\s*름\s*[:\s]*)([김이박최정강조윤장임한오서신권황안송류홍][가-힣]{1,3})',
            r'(근\s*로\s*자\s*[:\s]*)([김이박최정강조윤장임한오서신권황안송류홍][가-힣]{1,3})',
            r'(사\s*용\s*자\s*[:\s]*)([김이박최정강조윤장임한오서신권황안송류홍][가-힣]{1,3})',
            r'(대\s*표\s*자?\s*[:\s]*)([김이박최정강조윤장임한오서신권황안송류홍][가-힣]{1,3})',
            r'(["\']?\s*)([김이박최정강조윤장임한오서신권황안송류홍][가-힣]{1,3})(\s*["\']?\s*\(?\s*이하)',
        ]

        for pattern_str in name_context_patterns:
            pattern = re.compile(pattern_str)

            def make_replace_func(p):
                def replace_func(match):
                    groups = match.groups()
                    if len(groups) >= 2:
                        prefix = groups[0]
                        name = groups[1]
                        suffix = groups[2] if len(groups) > 2 else ""

                        # 성만 남기고 마스킹
                        masked_name = name[0] + self.mask_char * (len(name) - 1)
                        masked = f"{prefix}{masked_name}{suffix}"

                        pii_map[masked_name] = name
                        matches.append(PIIMatch(
                            pii_type=PIIType.NAME,
                            original=name,
                            masked=masked_name,
                            start=match.start(),
                            end=match.end()
                        ))

                        return masked
                    return match.group(0)
                return replace_func

            text = pattern.sub(make_replace_func(pattern), text)

        return text, matches

    def _generate_mask(self, original: str, pii_type: PIIType) -> str:
        """마스킹 문자열 생성"""
        if self.preserve_format:
            return self._preserve_format_mask(original, pii_type)
        else:
            return f"<{pii_type.value}_MASKED>"

    def _preserve_format_mask(self, original: str, pii_type: PIIType) -> str:
        """형식을 보존하면서 마스킹"""
        if pii_type == PIIType.RRN:
            # 주민등록번호: 앞 6자리만 표시
            if '-' in original:
                return original[:7] + self.mask_char * 7
            return original[:6] + self.mask_char * 7

        elif pii_type == PIIType.PHONE:
            # 전화번호: 중간 4자리 마스킹
            parts = re.split(r'[-.\s]', original)
            if len(parts) >= 3:
                return f"{parts[0]}-{self.mask_char * len(parts[1])}-{parts[2]}"
            return self.mask_char * len(original)

        elif pii_type == PIIType.EMAIL:
            # 이메일: @ 앞부분 일부 마스킹
            parts = original.split('@')
            if len(parts) == 2:
                local = parts[0]
                if len(local) > 3:
                    masked_local = local[:2] + self.mask_char * (len(local) - 2)
                else:
                    masked_local = self.mask_char * len(local)
                return f"{masked_local}@{parts[1]}"
            return self.mask_char * len(original)

        elif pii_type == PIIType.ACCOUNT:
            # 계좌번호: 앞 4자리, 뒤 4자리만 표시
            digits = re.sub(r'[-\s]', '', original)
            if len(digits) >= 8:
                return digits[:4] + self.mask_char * (len(digits) - 8) + digits[-4:]
            return self.mask_char * len(original)

        elif pii_type == PIIType.CARD:
            # 카드번호: 앞 4자리, 뒤 4자리만 표시
            return original[:4] + "-" + self.mask_char * 4 + "-" + self.mask_char * 4 + "-" + original[-4:]

        elif pii_type == PIIType.COMPANY_REG:
            # 사업자등록번호: 전체 마스킹
            return f"<사업자등록번호>"

        else:
            return f"<{pii_type.value}>"

    def unmask(self, masked_text: str, pii_map: Dict[str, str]) -> str:
        """
        마스킹된 텍스트를 원본으로 복원

        Args:
            masked_text: 마스킹된 텍스트
            pii_map: 마스킹 매핑 (masked -> original)

        Returns:
            복원된 원본 텍스트
        """
        result = masked_text
        for masked, original in pii_map.items():
            result = result.replace(masked, original)
        return result

    def get_pii_report(self, result: MaskingResult) -> str:
        """PII 탐지 보고서 생성"""
        if not result.has_pii:
            return "개인정보가 탐지되지 않았습니다."

        lines = [
            "=== 개인정보 탐지 보고서 ===",
            f"총 탐지된 개인정보: {result.pii_count}건",
            "",
            "[유형별 통계]"
        ]

        for pii_type, count in result.get_summary().items():
            lines.append(f"  - {pii_type}: {count}건")

        lines.append("")
        lines.append("[상세 목록]")

        for i, match in enumerate(result.pii_matches, 1):
            lines.append(f"  {i}. [{match.pii_type.value}] {match.original} -> {match.masked}")

        return "\n".join(lines)


# 편의 함수
def mask_pii(text: str, preserve_format: bool = True) -> MaskingResult:
    """
    간편하게 PII 마스킹 수행

    Args:
        text: 원본 텍스트
        preserve_format: 형식 보존 여부

    Returns:
        MaskingResult
    """
    masker = PIIMasker(preserve_format=preserve_format)
    return masker.mask(text)
