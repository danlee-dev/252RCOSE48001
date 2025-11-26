import json
import re
from pathlib import Path
from typing import List, Dict
import uuid


class DocumentChunker:
    def __init__(self, input_dir: str, output_dir: str):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def chunk_all_documents(self):
        """모든 JSON 문서를 청킹 (standard_contract 제외)"""
        all_chunks = []

        files = {
            # "개정 표준근로계약서(2025년, 배포).json": self.chunk_standard_contract,  # 별도 체크리스트로 관리
            "_25년 채용절차의 공정화에 관한 법률 업무 매뉴얼.json": self.chunk_hiring_manual,
            "개정 표준취업규칙(2025년, 배포).json": self.chunk_employment_rules,
            "2025년 적용 최저임금 안내.json": self.chunk_minimum_wage_guide,
            "★채용절차의 공정화에 관한 법률 리플릿.json": self.chunk_hiring_leaflet
        }

        for filename, chunker_func in files.items():
            filepath = self.input_dir / filename
            print(f"\n파일 경로 확인: {filepath}")
            print(f"파일 존재 여부: {filepath.exists()}")
            if filepath.exists():
                print(f"처리 중: {filename}")
                try:
                    chunks = chunker_func(filepath)
                    all_chunks.extend(chunks)
                    print(f"생성된 청크: {len(chunks)}개")
                except Exception as e:
                    print(f"에러 발생: {e}")
                    import traceback
                    traceback.print_exc()
            else:
                print(f"파일 없음: {filename}")
                # 실제 파일 목록 출력
                print(f"디렉토리 내 파일: {list(self.input_dir.glob('*.json'))}")

        # 통합 청크 저장
        output_file = self.output_dir / "all_chunks.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_chunks, f, ensure_ascii=False, indent=2)

        print(f"\n총 {len(all_chunks)}개 청크 생성 완료")
        print(f"저장 위치: {output_file}")

        # 메타데이터 요약
        self._save_metadata(all_chunks)

        return all_chunks

    def chunk_standard_contract(self, filepath: Path) -> List[Dict]:
        """표준근로계약서: 계약서 타입별 + 조항별 분할"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        chunks = []
        text = data['text']
        pages = text.split('--- Page')

        for page in pages[1:]:  # 첫 번째는 빈 문자열
            page_num = re.search(r'(\d+) ---', page)
            if not page_num:
                continue

            page_content = page.split('---', 1)[1].strip()

            # 계약서 타입 추출
            contract_type = self._extract_contract_type(page_content)

            # 조항 분할 (1., 2., ... 패턴)
            sections = re.split(r'\n(\d+)\.\s+', page_content)

            current_section = ""
            for i, part in enumerate(sections):
                if i == 0:
                    # 헤더 부분
                    if part.strip():
                        chunks.append({
                            "chunk_id": str(uuid.uuid4()),
                            "doc_type": "standard_contract",
                            "contract_type": contract_type,
                            "section": "헤더",
                            "clause_number": None,
                            "content": part.strip(),
                            "source": filepath.name,
                            "page": int(page_num.group(1)),
                            "is_mandatory": True,
                            "category": "계약서양식"
                        })
                elif i % 2 == 1:
                    # 조항 번호
                    current_section = part
                else:
                    # 조항 내용
                    section_title = self._extract_section_title(part)
                    category = self._categorize_section(section_title, part)

                    chunks.append({
                        "chunk_id": str(uuid.uuid4()),
                        "doc_type": "standard_contract",
                        "contract_type": contract_type,
                        "section": section_title,
                        "clause_number": current_section,
                        "content": part.strip(),
                        "source": filepath.name,
                        "page": int(page_num.group(1)),
                        "is_mandatory": True,
                        "category": category,
                        "keywords": self._extract_keywords(part)
                    })

        return chunks

    def chunk_hiring_manual(self, filepath: Path) -> List[Dict]:
        """채용절차 법률 매뉴얼: 장/절 단위 분할 (제목+내용 병합, 최소 길이 필터링)"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        chunks = []
        text = data['text']
        pages = text.split('--- Page')

        current_chapter = ""
        current_section = ""
        accumulated_content = []
        section_start_page = None

        for page in pages[1:]:
            page_num = re.search(r'(\d+) ---', page)
            if not page_num:
                continue

            page_content = page.split('---', 1)[1].strip()
            page_number = int(page_num.group(1))

            # 장 추출 (제1장, 제2장 등)
            chapter_match = re.search(r'제(\d+)장\s+(.+)', page_content)
            if chapter_match:
                current_chapter = f"제{chapter_match.group(1)}장 {chapter_match.group(2)}"

            lines = page_content.split('\n')

            for line in lines:
                # 새로운 주요 섹션 시작 (1., 2., 3., ...)
                if re.match(r'^\d+\.\s+[가-힣]', line):
                    # 이전 섹션 저장 (내용이 충분한 경우만)
                    if accumulated_content and current_section:
                        content = '\n'.join(accumulated_content).strip()
                        if len(content) >= 50:  # 최소 50자 이상
                            chunks.append(self._create_manual_chunk(
                                current_chapter, current_section, "",
                                content, filepath.name, section_start_page or page_number
                            ))

                    # 새 섹션 시작
                    current_section = line.strip()
                    accumulated_content = [line]  # 제목도 포함
                    section_start_page = page_number
                else:
                    # 내용 누적
                    accumulated_content.append(line)

        # 마지막 섹션 저장
        if accumulated_content and current_section:
            content = '\n'.join(accumulated_content).strip()
            if len(content) >= 50:
                chunks.append(self._create_manual_chunk(
                    current_chapter, current_section, "",
                    content, filepath.name, section_start_page or page_number
                ))

        return chunks

    def chunk_employment_rules(self, filepath: Path) -> List[Dict]:
        """표준취업규칙: 조문 단위 분할"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        chunks = []
        text = data['text']

        # 조문 패턴으로 분할 (제1조, 제2조 등)
        articles = re.split(r'\n(제\s*\d+\s*조)', text)

        current_article = ""
        for i, part in enumerate(articles):
            if i == 0:
                continue  # 헤더 부분

            if i % 2 == 1:
                current_article = part.strip()
            else:
                # 조문 제목 추출
                title_match = re.search(r'^\s*\((.+?)\)', part)
                title = title_match.group(1) if title_match else ""

                category = self._categorize_employment_rule(title, part)

                chunks.append({
                    "chunk_id": str(uuid.uuid4()),
                    "doc_type": "employment_rules",
                    "article": current_article,
                    "title": title,
                    "content": part.strip(),
                    "source": filepath.name,
                    "category": category,
                    "keywords": self._extract_keywords(part),
                    "is_mandatory": True
                })

        return chunks

    def chunk_minimum_wage_guide(self, filepath: Path) -> List[Dict]:
        """최저임금 안내: 주제별 분할"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        chunks = []
        text = data['text']

        # 질문 패턴으로 분할
        topics = re.split(r'\n(.+?나요\?)', text)

        # 기본 정보 청크 (최저임금액)
        intro = text.split('나요?')[0] if '나요?' in text else text[:500]
        chunks.append({
            "chunk_id": str(uuid.uuid4()),
            "doc_type": "guide",
            "sub_type": "minimum_wage",
            "topic": "최저임금액",
            "year": "2025",
            "content": intro,
            "source": filepath.name,
            "category": "임금",
            "keywords": ["최저임금", "10030원", "2025년"]
        })

        # Q&A 형식 청크
        current_question = ""
        for i, part in enumerate(topics):
            if i == 0:
                continue

            if i % 2 == 1:
                current_question = part.strip()
            else:
                chunks.append({
                    "chunk_id": str(uuid.uuid4()),
                    "doc_type": "guide",
                    "sub_type": "minimum_wage",
                    "topic": current_question,
                    "year": "2025",
                    "content": part.strip(),
                    "source": filepath.name,
                    "category": "임금",
                    "keywords": self._extract_keywords(current_question + " " + part)
                })

        return chunks

    def chunk_hiring_leaflet(self, filepath: Path) -> List[Dict]:
        """채용절차 리플릿: 단계별 분할"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        chunks = []
        text = data['text']
        pages = text.split('--- Page')

        # 주요 키워드로 섹션 분할
        sections = {
            "채용광고": ["거짓 채용광고", "채용광고 내용"],
            "지원서 접수": ["입사지원서", "개인정보"],
            "채용심사": ["채용심사비용", "채용과정"],
            "채용확정": ["채용여부 고지", "채용서류 반환"]
        }

        for page in pages[1:]:
            page_content = page.split('---', 1)[1].strip()

            for stage, keywords in sections.items():
                for keyword in keywords:
                    if keyword in page_content:
                        # 해당 키워드 주변 문맥 추출
                        context = self._extract_context(page_content, keyword, window=300)

                        # 위반 사항 및 처벌 추출
                        penalty = self._extract_penalty(context)

                        chunks.append({
                            "chunk_id": str(uuid.uuid4()),
                            "doc_type": "leaflet",
                            "sub_type": "hiring_law",
                            "stage": stage,
                            "topic": keyword,
                            "content": context,
                            "penalty": penalty,
                            "source": filepath.name,
                            "category": "채용절차",
                            "keywords": [keyword, stage]
                        })

        return chunks

    # Helper methods
    def _extract_contract_type(self, content: str) -> str:
        """계약서 타입 추출"""
        if "기간의 정함이 없는" in content:
            return "정규직"
        elif "기간의 정함이 있는" in content:
            return "기간제"
        elif "연소근로자" in content or "18세 미만" in content:
            return "연소근로자"
        elif "건설일용" in content:
            return "건설일용"
        elif "단시간" in content:
            return "단시간근로자"
        return "기타"

    def _extract_section_title(self, content: str) -> str:
        """조항 제목 추출"""
        first_line = content.split('\n')[0].split(':')[0].strip()
        return first_line[:20]  # 최대 20자

    def _categorize_section(self, title: str, content: str) -> str:
        """조항 카테고리 분류"""
        keywords = {
            "근로시간": ["근로시간", "근무시간", "소정근로"],
            "임금": ["임금", "급여", "상여금", "수당"],
            "휴가": ["휴가", "휴일", "연차"],
            "사회보험": ["사회보험", "4대보험", "고용보험", "산재보험"],
            "근로계약": ["계약", "계약서", "교부"],
            "근무장소": ["근무장소", "사업장"],
            "업무내용": ["업무", "직무"]
        }

        text = title + " " + content
        for category, kws in keywords.items():
            if any(kw in text for kw in kws):
                return category
        return "기타"

    def _categorize_employment_rule(self, title: str, content: str) -> str:
        """취업규칙 카테고리 분류"""
        keywords = {
            "총칙": ["목적", "적용범위", "정의"],
            "근로시간": ["근로시간", "연장근로", "야간근로", "휴게"],
            "휴일휴가": ["휴일", "휴가", "연차", "경조사"],
            "임금": ["임금", "급여", "상여금", "수당", "퇴직금"],
            "인사": ["채용", "승진", "전보", "휴직", "퇴직"],
            "상벌": ["포상", "징계", "해고"],
            "안전보건": ["안전", "보건", "재해"],
            "복리후생": ["복리", "후생", "교육"]
        }

        text = title + " " + content
        for category, kws in keywords.items():
            if any(kw in text for kw in kws):
                return category
        return "기타"

    def _extract_keywords(self, text: str, max_keywords: int = 5) -> List[str]:
        """중요 키워드 추출 (간단한 규칙 기반)"""
        keywords = []

        # 법령 키워드
        law_pattern = r'[가-힣]+법\s*제?\s*\d+조'
        keywords.extend(re.findall(law_pattern, text)[:3])

        # 중요 단어 (명사구)
        important_words = [
            "근로시간", "임금", "휴가", "휴일", "연차", "상여금", "퇴직금",
            "사회보험", "고용보험", "산재보험", "국민연금", "건강보험",
            "최저임금", "연장근로", "야간근로", "휴게시간", "주휴일",
            "채용", "해고", "징계", "휴직", "퇴직"
        ]

        for word in important_words:
            if word in text and word not in keywords:
                keywords.append(word)
                if len(keywords) >= max_keywords:
                    break

        return keywords

    def _create_manual_chunk(self, chapter: str, section: str, subsection: str,
                            content: str, source: str, page: int) -> Dict:
        """매뉴얼 청크 생성"""
        return {
            "chunk_id": str(uuid.uuid4()),
            "doc_type": "manual",
            "sub_type": "hiring_law",
            "chapter": chapter,
            "section": section,
            "subsection": subsection,
            "content": content.strip(),
            "source": source,
            "page": page,
            "category": "채용절차",
            "keywords": self._extract_keywords(content)
        }

    def _extract_context(self, text: str, keyword: str, window: int = 300) -> str:
        """키워드 주변 문맥 추출"""
        idx = text.find(keyword)
        if idx == -1:
            return ""

        start = max(0, idx - window)
        end = min(len(text), idx + len(keyword) + window)
        return text[start:end].strip()

    def _extract_penalty(self, text: str) -> str:
        """처벌 규정 추출"""
        penalty_patterns = [
            r'(\d+년\s*이하\s*징역)',
            r'(\d+만원\s*이하\s*(?:과태료|벌금))',
            r'(시정명령)',
        ]

        penalties = []
        for pattern in penalty_patterns:
            matches = re.findall(pattern, text)
            penalties.extend(matches)

        return ", ".join(penalties) if penalties else ""

    def _save_metadata(self, chunks: List[Dict]):
        """메타데이터 요약 저장"""
        metadata = {
            "total_chunks": len(chunks),
            "doc_types": {},
            "categories": {},
            "sources": {}
        }

        for chunk in chunks:
            doc_type = chunk.get("doc_type", "unknown")
            category = chunk.get("category", "unknown")
            source = chunk.get("source", "unknown")

            metadata["doc_types"][doc_type] = metadata["doc_types"].get(doc_type, 0) + 1
            metadata["categories"][category] = metadata["categories"].get(category, 0) + 1
            metadata["sources"][source] = metadata["sources"].get(source, 0) + 1

        output_file = self.output_dir / "metadata.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

        print(f"\n메타데이터 저장: {output_file}")


if __name__ == "__main__":
    # 프로젝트 루트 기준 절대 경로
    from pathlib import Path
    project_root = Path(__file__).resolve().parent.parent.parent.parent
    input_dir = project_root / "ai" / "data" / "processed" / "documents" / "standard_contracts"
    output_dir = project_root / "ai" / "data" / "processed" / "chunks"

    print(f"프로젝트 루트: {project_root}")
    print(f"입력 디렉토리: {input_dir}")
    print(f"출력 디렉토리: {output_dir}")

    chunker = DocumentChunker(str(input_dir), str(output_dir))
    chunks = chunker.chunk_all_documents()

    print("\n=== 청킹 완료 ===")
    print(f"총 청크 수: {len(chunks)}")
