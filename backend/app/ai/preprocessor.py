import pdfplumber
import re
from typing import List

class ContractPreprocessor:
    """
    계약서 PDF 전처리 클래스
    - PDF 텍스트 추출 (pdfplumber)
    - 조항 단위 청킹 (Regex)
    """

    def extract_text(self, pdf_path: str) -> str:
        """PDF 파일 경로를 받아 텍스트 전체를 추출"""
        text = ''
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages, start=1):
                    page_text = page.extract_text()
                    if page_text:
                        # 페이지 구분자 추가 (나중에 필요할 수 있음)
                        text += f"\n--- Page {page_num} ---\n"
                        text += page_text
        except Exception as e:
            print(f"Error extracting text from {pdf_path}: {e}")
            return ""
        
        return text.strip()

    def chunk_text(self, text: str) -> List[str]:
        """
        추출된 텍스트를 조항(Article) 단위로 분할하여 리스트로 반환
        """
        # 1. 헤더(--- Page X ---) 제거 (청킹 방해 요소)
        clean_text = re.sub(r'\n--- Page \d+ ---\n', '\n', text)
        
        # 2. 조항 패턴으로 분할 (제1조, 제2조 ... 또는 제 1 조)
        # 기존 2_chunk.py의 로직 활용
        articles = re.split(r'\n(제\s*\d+\s*조)', clean_text)
        
        chunks = []
        current_article_title = ""
        
        for i, part in enumerate(articles):
            if i == 0:
                # 서두 부분 (제목 등)
                if part.strip():
                    chunks.append(part.strip())
                continue

            if i % 2 == 1:
                # 조항 번호 (예: 제1조)
                current_article_title = part.strip()
            else:
                # 조항 내용
                content = part.strip()
                if content:
                    # "제1조" + "내용" 합쳐서 하나의 청크로 저장
                    full_chunk = f"{current_article_title}\n{content}"
                    chunks.append(full_chunk)
        
        # 만약 조항 패턴이 없어서 chunks가 비어있다면 (일반 문서 등), 문단 단위로라도 나눔
        if not chunks:
            chunks = [p.strip() for p in clean_text.split('\n\n') if p.strip()]
            
        return chunks