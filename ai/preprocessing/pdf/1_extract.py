import pdfplumber
from pathlib import Path
import json
from datetime import datetime
from typing import Dict, List, Optional


class PDFExtractor:
    """PDF 파일에서 텍스트를 추출하는 클래스"""

    def __init__(self, input_dir: str, output_dir: str):
        """
        Args:
            input_dir: PDF 파일이 있는 디렉토리
            output_dir: 추출된 텍스트를 저장할 디렉토리
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)

        # 출력 디렉토리 생성
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def extract_text(self, pdf_path: Path) -> str:
        """PDF 파일에서 텍스트 추출

        Args:
            pdf_path: PDF 파일 경로

        Returns:
            추출된 텍스트
        """
        text = ''
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages, start=1):
                    page_text = page.extract_text()
                    if page_text:
                        text += f"\n--- Page {page_num} ---\n"
                        text += page_text
        except Exception as e:
            print(f"Error extracting text from {pdf_path.name}: {e}")

        return text.strip()

    def get_page_count(self, pdf_path: Path) -> int:
        """PDF 페이지 수 반환

        Args:
            pdf_path: PDF 파일 경로

        Returns:
            페이지 수
        """
        try:
            with pdfplumber.open(pdf_path) as pdf:
                return len(pdf.pages)
        except Exception as e:
            print(f"Error getting page count from {pdf_path.name}: {e}")
            return 0

    def process_pdf(self, pdf_path: Path) -> Optional[Dict]:
        """PDF 파일 처리하고 JSON으로 저장

        Args:
            pdf_path: PDF 파일 경로

        Returns:
            처리 결과 딕셔너리
        """
        print(f"Processing: {pdf_path.name}")

        # 텍스트 추출
        text = self.extract_text(pdf_path)

        if not text:
            print(f"Warning: No text extracted from {pdf_path.name}")
            return None

        # 메타데이터 생성
        output_data = {
            'source': str(pdf_path),
            'filename': pdf_path.name,
            'extracted_at': datetime.now().isoformat(),
            'page_count': self.get_page_count(pdf_path),
            'text_length': len(text),
            'text': text
        }

        # JSON 파일로 저장
        output_file = self.output_dir / f"{pdf_path.stem}.json"
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2)
            print(f"Saved: {output_file.name}")
        except Exception as e:
            print(f"Error saving {output_file.name}: {e}")
            return None

        return output_data

    def process_all(self) -> List[Dict]:
        """디렉토리 내 모든 PDF 파일 처리

        Returns:
            처리 결과 리스트
        """
        pdf_files = list(self.input_dir.glob('**/*.pdf'))

        if not pdf_files:
            print(f"No PDF files found in {self.input_dir}")
            return []

        print(f"Found {len(pdf_files)} PDF files")
        print("-" * 50)

        results = []
        for pdf_file in pdf_files:
            result = self.process_pdf(pdf_file)
            if result:
                results.append(result)

        print("-" * 50)
        print(f"Successfully processed {len(results)}/{len(pdf_files)} files")

        return results


def main():
    """메인 실행 함수"""
    # 기본 경로 설정
    base_dir = Path(__file__).parent.parent.parent
    input_dir = base_dir / 'data' / 'raw' / 'documents' / 'standard_contracts'
    output_dir = base_dir / 'data' / 'processed' / 'documents' / 'standard_contracts'

    # PDF 추출 실행
    extractor = PDFExtractor(
        input_dir=str(input_dir),
        output_dir=str(output_dir)
    )

    results = extractor.process_all()

    # 요약 출력
    if results:
        print("\n=== Extraction Summary ===")
        for result in results:
            print(f"- {result['filename']}: {result['page_count']} pages, {result['text_length']} characters")


if __name__ == "__main__":
    main()
