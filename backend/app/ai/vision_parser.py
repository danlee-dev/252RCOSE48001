"""
Multimodal Parsing (Vision RAG)
- VLM(Vision Language Model)을 활용한 계약서 이미지 분석
- 표, 체크박스, 복잡한 레이아웃 구조화
- OCR 한계 극복

Provider: Gemini 2.5 Flash-Lite (기본) / GPT-4o (옵션)
"""

import os
import base64
import json
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from pathlib import Path
import io
from enum import Enum


class VisionProvider(Enum):
    """Vision API 제공자"""
    GEMINI = "gemini"
    OPENAI = "openai"


@dataclass
class ParsedElement:
    """파싱된 요소"""
    element_type: str  # table, checkbox, signature, text, header
    content: str
    position: Dict[str, int] = field(default_factory=dict)  # x, y, width, height
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class VisionParseResult:
    """Vision 파싱 결과"""
    raw_text: str
    structured_markdown: str
    elements: List[ParsedElement] = field(default_factory=list)
    tables: List[Dict[str, Any]] = field(default_factory=list)
    checkboxes: List[Dict[str, Any]] = field(default_factory=list)
    signatures: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def has_tables(self) -> bool:
        return len(self.tables) > 0

    @property
    def has_checkboxes(self) -> bool:
        return len(self.checkboxes) > 0


class VisionParser:
    """
    Vision 기반 문서 파서

    기본: Gemini 2.5 Flash-Lite (저렴, 빠름)
    옵션: GPT-4o (고품질)

    사용법:
        parser = VisionParser()  # Gemini 사용
        parser = VisionParser(provider=VisionProvider.OPENAI)  # GPT-4o 사용
        result = parser.parse_image(image_path)
        print(result.structured_markdown)
    """

    # 구조화 프롬프트
    STRUCTURE_PROMPT = """이 계약서 이미지를 분석하여 마크다운 형식으로 구조화하세요.

다음 요소들을 식별하고 추출하세요:
1. 제목과 헤더
2. 조항 (제1조, 제2조 등) 또는 번호 목록 (1., 2., 3. 등)
3. 표 (임금 구성, 근로시간 등) - 마크다운 테이블로 변환
4. 체크박스 (체크 여부 표시: [x] 또는 [ ])
5. 서명란
6. 날짜
7. 핵심 수치 (금액, 시간 등)

출력 형식:
```markdown
# 계약서 제목

## 제1조 (조항명)
조항 내용...

| 항목 | 내용 |
|------|------|
| 기본급 | 2,000,000원 |

[x] 동의함
[ ] 동의하지 않음

---
서명: _______________
날짜: 2024년 01월 01일
```

중요:
- 모든 숫자와 금액은 정확하게 추출하세요.
- 이미지에 있는 모든 텍스트를 빠짐없이 추출하세요.
- 한글을 정확하게 인식하세요."""

    # 표 추출 프롬프트
    TABLE_EXTRACTION_PROMPT = """이 이미지에서 표(테이블)를 찾아 JSON 형식으로 추출하세요.

각 표에 대해:
1. 표 제목/설명
2. 열 헤더
3. 행 데이터
4. 표의 맥락 (임금 구성표, 근로시간표 등)

출력 형식 (JSON만 출력):
{
    "tables": [
        {
            "title": "표 제목",
            "context": "표의 맥락 설명",
            "headers": ["열1", "열2", "열3"],
            "rows": [
                ["값1", "값2", "값3"],
                ["값4", "값5", "값6"]
            ]
        }
    ]
}"""

    # OCR 전용 프롬프트 (텍스트 추출에 집중)
    OCR_PROMPT = """이 이미지에서 모든 텍스트를 정확하게 추출하세요.

요구사항:
1. 이미지에 있는 모든 텍스트를 빠짐없이 추출
2. 원본 레이아웃과 순서를 최대한 유지
3. 표는 | 구분자로 표현
4. 체크박스는 [x] 또는 [ ]로 표현
5. 숫자와 금액은 정확하게 추출

텍스트만 출력하세요 (설명 없이):"""

    def __init__(
        self,
        provider: VisionProvider = VisionProvider.GEMINI,
        model: Optional[str] = None,
        max_image_size: int = 4096
    ):
        """
        Args:
            provider: Vision API 제공자 (GEMINI 또는 OPENAI)
            model: 사용할 모델 (None이면 기본값 사용)
            max_image_size: 최대 이미지 크기 (픽셀)
        """
        self.provider = provider
        self.max_image_size = max_image_size

        # 모델 설정
        if model:
            self.model = model
        elif provider == VisionProvider.GEMINI:
            self.model = "gemini-2.5-flash-lite"
        else:
            self.model = "gpt-4o"

        # 클라이언트 초기화
        self._gemini_model = None
        self._openai_client = None

    @property
    def gemini_model(self):
        """Gemini 모델 (lazy loading)"""
        if self._gemini_model is None:
            try:
                import google.generativeai as genai
                genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
                self._gemini_model = genai.GenerativeModel(self.model)
            except ImportError:
                raise ImportError("google-generativeai package not installed")
        return self._gemini_model

    @property
    def openai_client(self):
        """OpenAI 클라이언트 (lazy loading)"""
        if self._openai_client is None:
            try:
                from openai import OpenAI
                self._openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            except ImportError:
                raise ImportError("openai package not installed")
        return self._openai_client

    def parse_image(
        self,
        image_source: str,
        extract_tables: bool = True,
        extract_checkboxes: bool = True
    ) -> VisionParseResult:
        """
        이미지에서 문서 구조 파싱

        Args:
            image_source: 이미지 경로 또는 base64 문자열
            extract_tables: 표 추출 여부
            extract_checkboxes: 체크박스 추출 여부

        Returns:
            VisionParseResult: 파싱 결과
        """
        # 이미지 로드
        image_data = self._load_image(image_source)

        if image_data is None:
            return VisionParseResult(
                raw_text="이미지 로드 실패",
                structured_markdown="",
                metadata={"error": "이미지 로드 실패"}
            )

        result = VisionParseResult(
            raw_text="",
            structured_markdown=""
        )

        # 1. 구조화된 마크다운 추출
        structured = self._extract_structure(image_data)
        result.structured_markdown = structured
        result.raw_text = self._markdown_to_text(structured)

        # 2. 표 추출
        if extract_tables:
            result.tables = self._extract_tables(image_data)

        # 3. 체크박스 추출
        if extract_checkboxes:
            result.checkboxes = self._extract_checkboxes(structured)

        # 4. 서명란 추출
        result.signatures = self._extract_signatures(structured)

        # 5. 요소 분류
        result.elements = self._classify_elements(structured)

        return result

    def extract_text_only(self, image_source: str) -> str:
        """
        이미지에서 텍스트만 추출 (OCR 전용)

        Args:
            image_source: 이미지 경로 또는 base64 문자열

        Returns:
            추출된 텍스트
        """
        image_data = self._load_image(image_source)
        if image_data is None:
            return ""

        if self.provider == VisionProvider.GEMINI:
            return self._gemini_ocr(image_data)
        else:
            return self._openai_ocr(image_data)

    def parse_pdf_pages(
        self,
        pdf_path: str,
        pages: List[int] = None,
        ocr_only: bool = False
    ) -> List[VisionParseResult]:
        """
        PDF 페이지들을 이미지로 변환하여 파싱

        Args:
            pdf_path: PDF 파일 경로
            pages: 파싱할 페이지 번호 (None이면 전체)
            ocr_only: True면 텍스트 추출만 수행

        Returns:
            페이지별 파싱 결과
        """
        try:
            import pypdfium2 as pdfium

            pdf = pdfium.PdfDocument(pdf_path)
            results = []

            page_indices = pages if pages is not None else range(len(pdf))

            for page_idx in page_indices:
                if page_idx >= len(pdf):
                    continue

                page = pdf[page_idx]
                # 고해상도로 렌더링
                bitmap = page.render(scale=2.0)
                pil_image = bitmap.to_pil()

                # 이미지를 base64로 변환
                buffer = io.BytesIO()
                pil_image.save(buffer, format="PNG")
                image_base64 = base64.b64encode(buffer.getvalue()).decode()
                image_data = f"data:image/png;base64,{image_base64}"

                if ocr_only:
                    # OCR만 수행
                    text = self.extract_text_only(image_data)
                    result = VisionParseResult(
                        raw_text=text,
                        structured_markdown=text,
                        metadata={"page_number": page_idx + 1}
                    )
                else:
                    # 전체 파싱
                    result = self.parse_image(image_data)
                    result.metadata["page_number"] = page_idx + 1

                results.append(result)

            return results

        except ImportError:
            print("pypdfium2 not installed. Run: pip install pypdfium2")
            return []
        except Exception as e:
            print(f"PDF parsing error: {e}")
            return []

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """
        PDF에서 전체 텍스트 추출 (OCR)

        Args:
            pdf_path: PDF 파일 경로

        Returns:
            전체 텍스트
        """
        results = self.parse_pdf_pages(pdf_path, ocr_only=True)

        texts = []
        for result in results:
            page_num = result.metadata.get("page_number", "?")
            texts.append(f"\n--- Page {page_num} ---\n")
            texts.append(result.raw_text)

        return "\n".join(texts)

    def _load_image(self, source: str) -> Optional[str]:
        """이미지 로드 및 base64 변환"""
        try:
            if source.startswith("data:image"):
                # 이미 base64
                return source

            elif os.path.isfile(source):
                # 파일 경로
                with open(source, "rb") as f:
                    image_data = f.read()

                # 이미지 포맷 확인
                ext = Path(source).suffix.lower()
                mime_type = {
                    ".png": "image/png",
                    ".jpg": "image/jpeg",
                    ".jpeg": "image/jpeg",
                    ".gif": "image/gif",
                    ".webp": "image/webp"
                }.get(ext, "image/png")

                base64_data = base64.b64encode(image_data).decode()
                return f"data:{mime_type};base64,{base64_data}"

            else:
                # URL로 가정
                return source

        except Exception as e:
            print(f"Image load error: {e}")
            return None

    def _extract_structure(self, image_data: str) -> str:
        """이미지에서 구조화된 마크다운 추출"""
        if self.provider == VisionProvider.GEMINI:
            return self._gemini_extract_structure(image_data)
        else:
            return self._openai_extract_structure(image_data)

    def _gemini_extract_structure(self, image_data: str) -> str:
        """Gemini로 구조 추출"""
        try:
            import PIL.Image

            # base64 디코딩
            if image_data.startswith("data:"):
                # data:image/png;base64,... 형식
                base64_str = image_data.split(",", 1)[1]
            else:
                base64_str = image_data

            image_bytes = base64.b64decode(base64_str)
            image = PIL.Image.open(io.BytesIO(image_bytes))

            response = self.gemini_model.generate_content(
                [self.STRUCTURE_PROMPT, image],
                generation_config={
                    "max_output_tokens": 4000,
                }
            )

            return response.text

        except Exception as e:
            print(f"Gemini structure extraction error: {e}")
            return f"[구조 추출 실패: {e}]"

    def _openai_extract_structure(self, image_data: str) -> str:
        """OpenAI로 구조 추출"""
        try:
            response = self.openai_client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "당신은 법률 문서 분석 전문가입니다. 계약서 이미지를 정확하게 구조화합니다."
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": self.STRUCTURE_PROMPT},
                            {
                                "type": "image_url",
                                "image_url": {"url": image_data}
                            }
                        ]
                    }
                ],
                max_completion_tokens=4000
            )

            return response.choices[0].message.content

        except Exception as e:
            print(f"OpenAI structure extraction error: {e}")
            return f"[구조 추출 실패: {e}]"

    def _gemini_ocr(self, image_data: str) -> str:
        """Gemini OCR"""
        try:
            import PIL.Image

            if image_data.startswith("data:"):
                base64_str = image_data.split(",", 1)[1]
            else:
                base64_str = image_data

            image_bytes = base64.b64decode(base64_str)
            image = PIL.Image.open(io.BytesIO(image_bytes))

            response = self.gemini_model.generate_content(
                [self.OCR_PROMPT, image],
                generation_config={
                    "max_output_tokens": 4000,
                }
            )

            return response.text

        except Exception as e:
            print(f"Gemini OCR error: {e}")
            return ""

    def _openai_ocr(self, image_data: str) -> str:
        """OpenAI OCR"""
        try:
            response = self.openai_client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": self.OCR_PROMPT},
                            {
                                "type": "image_url",
                                "image_url": {"url": image_data}
                            }
                        ]
                    }
                ],
                max_completion_tokens=4000
            )

            return response.choices[0].message.content

        except Exception as e:
            print(f"OpenAI OCR error: {e}")
            return ""

    def _extract_tables(self, image_data: str) -> List[Dict[str, Any]]:
        """이미지에서 표 추출"""
        if self.provider == VisionProvider.GEMINI:
            return self._gemini_extract_tables(image_data)
        else:
            return self._openai_extract_tables(image_data)

    def _gemini_extract_tables(self, image_data: str) -> List[Dict[str, Any]]:
        """Gemini로 표 추출"""
        try:
            import PIL.Image

            if image_data.startswith("data:"):
                base64_str = image_data.split(",", 1)[1]
            else:
                base64_str = image_data

            image_bytes = base64.b64decode(base64_str)
            image = PIL.Image.open(io.BytesIO(image_bytes))

            response = self.gemini_model.generate_content(
                [self.TABLE_EXTRACTION_PROMPT, image],
                generation_config={
                    "max_output_tokens": 2000,
                }
            )

            # JSON 파싱
            text = response.text
            # ```json ... ``` 제거
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0]
            elif "```" in text:
                text = text.split("```")[1].split("```")[0]

            result = json.loads(text)
            return result.get("tables", [])

        except Exception as e:
            print(f"Gemini table extraction error: {e}")
            return []

    def _openai_extract_tables(self, image_data: str) -> List[Dict[str, Any]]:
        """OpenAI로 표 추출"""
        try:
            response = self.openai_client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": self.TABLE_EXTRACTION_PROMPT},
                            {
                                "type": "image_url",
                                "image_url": {"url": image_data}
                            }
                        ]
                    }
                ],
                response_format={"type": "json_object"}
            )

            result = json.loads(response.choices[0].message.content)
            return result.get("tables", [])

        except Exception as e:
            print(f"OpenAI table extraction error: {e}")
            return []

    def _extract_checkboxes(self, markdown: str) -> List[Dict[str, Any]]:
        """마크다운에서 체크박스 추출"""
        import re
        checkboxes = []

        # [x] 또는 [ ] 패턴
        pattern = r'\[(x| )\]\s*(.+?)(?:\n|$)'
        matches = re.findall(pattern, markdown, re.IGNORECASE)

        for i, (checked, label) in enumerate(matches):
            checkboxes.append({
                "id": f"checkbox_{i}",
                "checked": checked.lower() == 'x',
                "label": label.strip()
            })

        return checkboxes

    def _extract_signatures(self, markdown: str) -> List[Dict[str, Any]]:
        """마크다운에서 서명란 추출"""
        import re
        signatures = []

        # 서명 패턴
        patterns = [
            r'서명\s*[:\s]*([_\-]+)',
            r'(갑|을|병|근로자|사용자)\s*[:\s]*([_\-]+)',
            r'날짜\s*[:\s]*([\d년월일\.\s]+)',
        ]

        for pattern in patterns:
            matches = re.findall(pattern, markdown)
            for match in matches:
                if isinstance(match, tuple):
                    signatures.append({
                        "type": match[0] if len(match) > 1 else "서명",
                        "value": match[-1]
                    })
                else:
                    signatures.append({
                        "type": "서명",
                        "value": match
                    })

        return signatures

    def _classify_elements(self, markdown: str) -> List[ParsedElement]:
        """마크다운 요소 분류"""
        import re
        elements = []

        # 헤더
        headers = re.findall(r'^(#{1,6})\s+(.+)$', markdown, re.MULTILINE)
        for level, text in headers:
            elements.append(ParsedElement(
                element_type="header",
                content=text,
                metadata={"level": len(level)}
            ))

        # 조항
        articles = re.findall(r'(제\s*\d+\s*조[^\n]*)\n([^#]*?)(?=제\s*\d+\s*조|$)', markdown)
        for title, content in articles:
            elements.append(ParsedElement(
                element_type="article",
                content=f"{title}\n{content.strip()}",
                metadata={"title": title}
            ))

        # 표
        tables = re.findall(r'(\|.+\|[\s\S]*?\|.+\|)', markdown)
        for table in tables:
            elements.append(ParsedElement(
                element_type="table",
                content=table
            ))

        return elements

    def _markdown_to_text(self, markdown: str) -> str:
        """마크다운을 일반 텍스트로 변환"""
        import re

        text = markdown

        # 헤더 제거
        text = re.sub(r'^#+\s*', '', text, flags=re.MULTILINE)

        # 테이블 구분자 제거
        text = re.sub(r'\|[-:]+\|', '', text)
        text = re.sub(r'\|', ' ', text)

        # 체크박스 정리
        text = re.sub(r'\[x\]', '[체크됨]', text, flags=re.IGNORECASE)
        text = re.sub(r'\[ \]', '[미체크]', text)

        # 빈 줄 정리
        text = re.sub(r'\n{3,}', '\n\n', text)

        return text.strip()


class ContractVisionAnalyzer:
    """
    계약서 전용 Vision 분석기

    계약서 특화 분석 기능 제공
    """

    def __init__(self, parser: VisionParser = None):
        self.parser = parser or VisionParser()

    def analyze_wage_table(
        self,
        image_source: str
    ) -> Dict[str, Any]:
        """
        임금 구성표 분석

        Args:
            image_source: 이미지 소스

        Returns:
            임금 정보 구조화
        """
        result = self.parser.parse_image(image_source, extract_tables=True)

        wage_info = {
            "base_salary": 0,
            "allowances": {},
            "total": 0,
            "raw_tables": result.tables
        }

        # 임금 관련 표 찾기
        for table in result.tables:
            if any(kw in str(table).lower() for kw in ["임금", "급여", "수당", "기본급"]):
                wage_info = self._parse_wage_table(table)
                break

        return wage_info

    def _parse_wage_table(self, table: Dict[str, Any]) -> Dict[str, Any]:
        """임금 표 파싱"""
        import re

        wage_info = {
            "base_salary": 0,
            "allowances": {},
            "total": 0
        }

        headers = table.get("headers", [])
        rows = table.get("rows", [])

        for row in rows:
            if len(row) < 2:
                continue

            item_name = row[0]
            amount_str = row[1] if len(row) > 1 else ""

            # 금액 추출
            amount_match = re.search(r'[\d,]+', str(amount_str))
            if amount_match:
                amount = int(amount_match.group().replace(',', ''))

                if "기본급" in item_name or "기본" in item_name:
                    wage_info["base_salary"] = amount
                elif "합계" in item_name or "총" in item_name:
                    wage_info["total"] = amount
                else:
                    wage_info["allowances"][item_name] = amount

        # 합계가 없으면 계산
        if wage_info["total"] == 0:
            wage_info["total"] = wage_info["base_salary"] + sum(wage_info["allowances"].values())

        return wage_info

    def extract_contract_parties(
        self,
        image_source: str
    ) -> Dict[str, Any]:
        """
        계약 당사자 정보 추출

        Args:
            image_source: 이미지 소스

        Returns:
            당사자 정보
        """
        result = self.parser.parse_image(image_source)

        parties = {
            "employer": {},
            "employee": {},
            "signatures": result.signatures
        }

        # 마크다운에서 당사자 정보 추출
        import re

        # 사용자(갑) 정보
        employer_patterns = [
            r'사용자[:\s]+(.+?)(?:\n|$)',
            r'갑[:\s]+(.+?)(?:\n|$)',
            r'회사명[:\s]+(.+?)(?:\n|$)',
        ]

        for pattern in employer_patterns:
            match = re.search(pattern, result.structured_markdown)
            if match:
                parties["employer"]["name"] = match.group(1).strip()
                break

        # 근로자(을) 정보
        employee_patterns = [
            r'근로자[:\s]+(.+?)(?:\n|$)',
            r'을[:\s]+(.+?)(?:\n|$)',
            r'성명[:\s]+(.+?)(?:\n|$)',
        ]

        for pattern in employee_patterns:
            match = re.search(pattern, result.structured_markdown)
            if match:
                parties["employee"]["name"] = match.group(1).strip()
                break

        return parties


# 편의 함수
def parse_contract_image(image_path: str, provider: VisionProvider = VisionProvider.GEMINI) -> VisionParseResult:
    """간편 계약서 이미지 파싱"""
    parser = VisionParser(provider=provider)
    return parser.parse_image(image_path)


def extract_text_from_pdf(pdf_path: str, provider: VisionProvider = VisionProvider.GEMINI) -> str:
    """PDF에서 텍스트 추출 (Vision OCR)"""
    parser = VisionParser(provider=provider)
    return parser.extract_text_from_pdf(pdf_path)


def extract_tables_from_image(image_path: str, provider: VisionProvider = VisionProvider.GEMINI) -> List[Dict[str, Any]]:
    """이미지에서 표 추출"""
    parser = VisionParser(provider=provider)
    result = parser.parse_image(image_path, extract_tables=True)
    return result.tables
