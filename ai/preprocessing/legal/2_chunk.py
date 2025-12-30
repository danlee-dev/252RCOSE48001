"""
법률 데이터 청킹 스크립트

수집한 법률 데이터를 Elasticsearch RAG에 적합한 청크로 분할합니다.
구조 기반 의미 청킹(Structure-aware Semantic Chunking) 적용

기존 PDF 청크(674개)는 유지하고, 법률 데이터만 별도 청킹합니다.
"""

import json
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime
import uuid


class LegalDataChunker:
    """법률 데이터 청킹 처리기 (Elasticsearch 최적화)"""

    def __init__(self, max_chunk_length: int = 1000, min_chunk_length: int = 150):
        """
        Args:
            max_chunk_length: 최대 청크 길이 (기본 1000자 - 법률 문서용)
            min_chunk_length: 최소 청크 길이 (기본 150자)
        """
        self.max_chunk_length = max_chunk_length
        self.min_chunk_length = min_chunk_length

    def _split_long_text(self, text: str, max_length: int) -> List[str]:
        """긴 텍스트를 문장 단위로 분할"""
        if len(text) <= max_length:
            return [text]

        chunks = []
        sentences = text.replace('. ', '.\n').replace('。', '。\n').split('\n')
        current_chunk = ""

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            if len(current_chunk) + len(sentence) <= max_length:
                current_chunk += sentence + " "
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + " "

        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks

    def chunk_interpretation(self, item: Dict) -> List[Dict]:
        """법령해석례 청킹 (Elasticsearch 최적화)"""
        chunks = []
        base_metadata = {
            'chunk_id': str(uuid.uuid4()),
            'source_type': 'legal',
            'doc_type': 'interpretation',
            'source_id': item.get('법령해석례일련번호', ''),
            'title': item.get('안건명', ''),
            'case_number': item.get('안건번호', ''),
            'date': item.get('해석일자', ''),
            'reply_org': item.get('회신기관명', '법제처'),
            'search_keyword': item.get('검색키워드', ''),
            'collected_at': item.get('수집일시', ''),
        }

        # Chunk 1: 질의 (안건명 + 질의요지)
        question_text = f"[법령해석례 질의]\n\n안건: {item.get('안건명', '')}\n\n질의요지:\n{item.get('질의요지', '')}"
        if len(question_text) >= self.min_chunk_length:
            chunks.append({
                **base_metadata,
                'chunk_id': str(uuid.uuid4()),
                'chunk_type': 'question',
                'chunk_index': 0,
                'total_chunks': -1,  # 나중에 업데이트
                'content': question_text.strip(),
            })

        # Chunk 2: 회답
        answer = item.get('회답', '')
        if answer and len(answer) >= self.min_chunk_length:
            answer_chunks = self._split_long_text(answer, self.max_chunk_length)
            for i, chunk_text in enumerate(answer_chunks):
                chunks.append({
                    **base_metadata,
                    'chunk_id': str(uuid.uuid4()),
                    'chunk_type': 'answer',
                    'chunk_index': i,
                    'total_chunks': len(answer_chunks),
                    'content': f"[법령해석례 회답]\n\n{chunk_text}",
                })

        # Chunk 3: 이유 (법리 설명)
        reason = item.get('이유', '')
        if reason and len(reason) >= self.min_chunk_length:
            reason_chunks = self._split_long_text(reason, self.max_chunk_length)
            for i, chunk_text in enumerate(reason_chunks):
                chunks.append({
                    **base_metadata,
                    'chunk_id': str(uuid.uuid4()),
                    'chunk_type': 'reason',
                    'chunk_index': i,
                    'total_chunks': len(reason_chunks),
                    'content': f"[법령해석례 이유]\n\n{chunk_text}",
                })

        # total_chunks 업데이트
        for chunk in chunks:
            if chunk['chunk_type'] == 'question':
                chunk['total_chunks'] = len(chunks)

        return chunks

    def chunk_precedent(self, item: Dict) -> List[Dict]:
        """판례 청킹 (Elasticsearch 최적화)"""
        chunks = []
        base_metadata = {
            'chunk_id': str(uuid.uuid4()),
            'source_type': 'legal',
            'doc_type': 'precedent',
            'source_id': item.get('판례일련번호', ''),
            'title': item.get('사건명', ''),
            'case_number': item.get('사건번호', ''),
            'court': item.get('법원명', '대법원'),
            'case_type': item.get('사건종류명', ''),
            'date': item.get('선고일자', ''),
            'search_keyword': item.get('검색키워드', ''),
            'collected_at': item.get('수집일시', ''),
        }

        # Chunk 1: 사건 기본정보 + 판시사항
        case_info = f"[판례 사건정보]\n\n사건명: {item.get('사건명', '')}\n사건번호: {item.get('사건번호', '')}\n법원: {item.get('법원명', '')}\n선고일: {item.get('선고일자', '')}\n\n판시사항:\n{item.get('판시사항', '')}"
        if len(case_info) >= self.min_chunk_length:
            chunks.append({
                **base_metadata,
                'chunk_id': str(uuid.uuid4()),
                'chunk_type': 'case_info',
                'chunk_index': 0,
                'total_chunks': -1,
                'content': case_info.strip(),
            })

        # Chunk 2: 판결요지 (핵심 법리)
        summary = item.get('판결요지', '')
        if summary and len(summary) >= self.min_chunk_length:
            summary_chunks = self._split_long_text(summary, self.max_chunk_length)
            for i, chunk_text in enumerate(summary_chunks):
                chunks.append({
                    **base_metadata,
                    'chunk_id': str(uuid.uuid4()),
                    'chunk_type': 'summary',
                    'chunk_index': i,
                    'total_chunks': len(summary_chunks),
                    'content': f"[판결요지]\n\n{chunk_text}",
                })

        # Chunk 3: 판례내용 (상세 판결문)
        content = item.get('판례내용', '')
        if content and len(content) >= self.min_chunk_length:
            # 판례내용은 매우 길므로 800자 단위로 분할
            content_chunks = self._split_long_text(content, 800)
            for i, chunk_text in enumerate(content_chunks):
                chunks.append({
                    **base_metadata,
                    'chunk_id': str(uuid.uuid4()),
                    'chunk_type': 'content',
                    'chunk_index': i,
                    'total_chunks': len(content_chunks),
                    'content': f"[판례내용]\n\n{chunk_text}",
                })

        # Chunk 4: 참조조문
        references = item.get('참조조문', '')
        if references and len(references) >= self.min_chunk_length:
            chunks.append({
                **base_metadata,
                'chunk_id': str(uuid.uuid4()),
                'chunk_type': 'references',
                'chunk_index': 0,
                'total_chunks': 1,
                'content': f"[참조조문]\n\n{references}",
            })

        # total_chunks 업데이트
        for chunk in chunks:
            if chunk['chunk_type'] == 'case_info':
                chunk['total_chunks'] = len(chunks)

        return chunks

    def chunk_labor_ministry(self, item: Dict) -> List[Dict]:
        """고용노동부 법령해설 청킹 (Elasticsearch 최적화)"""
        chunks = []
        base_metadata = {
            'chunk_id': str(uuid.uuid4()),
            'source_type': 'legal',
            'doc_type': 'labor_ministry',
            'source_id': item.get('법령해석일련번호', ''),
            'title': item.get('안건명', ''),
            'case_number': item.get('안건번호', ''),
            'date': item.get('해석일자', ''),
            'reply_org': '고용노동부',
            'search_keyword': item.get('검색키워드', ''),
            'collected_at': item.get('수집일시', ''),
        }

        # Chunk 1: 질의 (안건명 + 질의요지)
        question_text = f"[고용노동부 법령해설 질의]\n\n안건: {item.get('안건명', '')}\n\n질의요지:\n{item.get('질의요지', '')}"
        if len(question_text) >= self.min_chunk_length:
            chunks.append({
                **base_metadata,
                'chunk_id': str(uuid.uuid4()),
                'chunk_type': 'question',
                'chunk_index': 0,
                'total_chunks': -1,
                'content': question_text.strip(),
            })

        # Chunk 2: 회답 (실무 해석)
        answer = item.get('회답', '')
        if answer and len(answer) >= self.min_chunk_length:
            answer_chunks = self._split_long_text(answer, self.max_chunk_length)
            for i, chunk_text in enumerate(answer_chunks):
                chunks.append({
                    **base_metadata,
                    'chunk_id': str(uuid.uuid4()),
                    'chunk_type': 'answer',
                    'chunk_index': i,
                    'total_chunks': len(answer_chunks),
                    'content': f"[고용노동부 회답]\n\n{chunk_text}",
                })

        # Chunk 3: 이유 (있는 경우)
        reason = item.get('이유', '')
        if reason and len(reason) >= self.min_chunk_length:
            reason_chunks = self._split_long_text(reason, self.max_chunk_length)
            for i, chunk_text in enumerate(reason_chunks):
                chunks.append({
                    **base_metadata,
                    'chunk_id': str(uuid.uuid4()),
                    'chunk_type': 'reason',
                    'chunk_index': i,
                    'total_chunks': len(reason_chunks),
                    'content': f"[이유]\n\n{chunk_text}",
                })

        # Chunk 4: 관련법령 (있는 경우)
        related_laws = item.get('관련법령', '')
        if related_laws and len(related_laws) >= self.min_chunk_length:
            chunks.append({
                **base_metadata,
                'chunk_id': str(uuid.uuid4()),
                'chunk_type': 'related_laws',
                'chunk_index': 0,
                'total_chunks': 1,
                'content': f"[관련법령]\n\n{related_laws}",
            })

        # total_chunks 업데이트
        for chunk in chunks:
            if chunk['chunk_type'] == 'question':
                chunk['total_chunks'] = len(chunks)

        return chunks

    def process_file(self, input_file: Path, source_type: str) -> List[Dict]:
        """파일 단위 청킹 처리"""
        print(f"\n처리 중: {input_file.name}")

        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        all_chunks = []
        chunk_methods = {
            'interpretation': self.chunk_interpretation,
            'precedent': self.chunk_precedent,
            'labor_ministry': self.chunk_labor_ministry,
        }

        chunk_method = chunk_methods.get(source_type)
        if not chunk_method:
            print(f"  오류: 알 수 없는 source_type: {source_type}")
            return []

        for item in data:
            chunks = chunk_method(item)
            all_chunks.extend(chunks)

        print(f"  원본: {len(data)}건")
        print(f"  청크: {len(all_chunks)}개")
        print(f"  비율: {len(all_chunks) / len(data):.1f}x")

        return all_chunks


def main():
    """메인 실행 함수 - 법률 데이터만 청킹 (기존 PDF 청크는 유지)"""
    print("\n" + "="*70)
    print("법률 데이터 청킹 시작 (Elasticsearch용)")
    print("="*70)
    print("기존 PDF 청크 (674개)는 유지됩니다.")
    print("법률 데이터만 별도로 청킹하여 legal_chunks.json에 저장합니다.")
    print("="*70)

    # 경로 설정
    project_root = Path(__file__).parent.parent.parent
    input_dir = project_root / "data" / "raw" / "api"
    output_dir = project_root / "data" / "processed" / "chunks"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 청커 초기화 (법률 문서는 더 긴 청크 허용)
    chunker = LegalDataChunker(max_chunk_length=1000, min_chunk_length=150)

    # 처리할 파일 자동 탐색 (날짜 무관)
    files_to_process = []
    file_patterns = [
        ("interpretations_*.json", "interpretation"),
        ("precedents_*.json", "precedent"),
        ("labor_ministry_*.json", "labor_ministry"),
    ]

    for pattern, source_type in file_patterns:
        matching_files = sorted(input_dir.glob(pattern), reverse=True)
        if matching_files:
            files_to_process.append((matching_files[0].name, source_type))
            print(f"발견: {matching_files[0].name}")

    if not files_to_process:
        print("오류: raw/api 폴더에 처리할 파일이 없습니다.")
        return

    today = datetime.now().strftime('%Y%m%d')

    all_chunks = []
    stats = {
        'interpretation': 0,
        'precedent': 0,
        'labor_ministry': 0
    }
    doc_counts = {}

    for filename, source_type in files_to_process:
        input_file = input_dir / filename
        if not input_file.exists():
            print(f"\n건너뜀: {filename} (파일 없음)")
            continue

        chunks = chunker.process_file(input_file, source_type)
        all_chunks.extend(chunks)
        stats[source_type] = len(chunks)

        # 원본 문서 개수 계산
        with open(input_file, 'r', encoding='utf-8') as f:
            doc_counts[source_type] = len(json.load(f))

    # 통합 청크 저장
    output_file = output_dir / f"legal_chunks_{today}.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)

    file_size = output_file.stat().st_size / 1024 / 1024

    # 메타데이터 저장
    metadata = {
        'created_at': datetime.now().isoformat(),
        'total_chunks': len(all_chunks),
        'total_docs': sum(doc_counts.values()),
        'doc_counts': doc_counts,
        'chunk_counts': stats,
        'avg_chunks_per_doc': {
            k: round(stats[k] / doc_counts[k], 1) if doc_counts.get(k, 0) > 0 else 0
            for k in stats.keys()
        },
        'chunking_config': {
            'max_chunk_length': chunker.max_chunk_length,
            'min_chunk_length': chunker.min_chunk_length,
        }
    }

    metadata_file = output_dir / f"legal_chunks_metadata_{today}.json"
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    # 요약 통계
    print("\n" + "="*70)
    print("청킹 완료 요약")
    print("="*70)
    print(f"법령해석례:         {doc_counts.get('interpretation', 0):>4}건 → {stats['interpretation']:>6}개 청크 (평균 {metadata['avg_chunks_per_doc']['interpretation']:.1f}x)")
    print(f"판례:               {doc_counts.get('precedent', 0):>4}건 → {stats['precedent']:>6}개 청크 (평균 {metadata['avg_chunks_per_doc']['precedent']:.1f}x)")
    print(f"고용노동부:         {doc_counts.get('labor_ministry', 0):>4}건 → {stats['labor_ministry']:>6}개 청크 (평균 {metadata['avg_chunks_per_doc']['labor_ministry']:.1f}x)")
    print("-" * 70)
    print(f"총 원본 문서:       {sum(doc_counts.values()):>4}건")
    print(f"총 청크 수:         {len(all_chunks):>6}개")
    print(f"평균 증가율:        {len(all_chunks) / sum(doc_counts.values()):.1f}x")
    print("="*70)
    print(f"저장 파일:          {output_file.name}")
    print(f"파일 크기:          {file_size:.2f}MB")
    print(f"메타데이터:         {metadata_file.name}")
    print("="*70)
    print("\n다음 단계: Elasticsearch 인덱싱")
    print("- 기존 PDF 청크 (674개)와 함께 Elasticsearch에 저장")
    print("- KURE-v1 모델로 임베딩 생성")
    print("- 하이브리드 검색 설정 (BM25 + Vector)")
    print("="*70)


if __name__ == "__main__":
    main()
