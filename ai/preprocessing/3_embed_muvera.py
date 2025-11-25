"""
MUVERA 기반 법률 데이터 임베딩 생성 스크립트

Multi-Vector Retrieval을 사용하여 문서를 여러 문장으로 분할하고,
각 문장을 임베딩한 후 FDE(Fixed Dimensional Encodings)로 압축합니다.
"""

import json
import re
import logging
import gc
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from datetime import datetime

from fde_generator import (
    FixedDimensionalEncodingConfig,
    EncodingType,
    ProjectionType,
    generate_document_fde,
    generate_query_fde,
    generate_document_fde_batch
)

logging.basicConfig(level=logging.INFO)


class SentenceSplitter:
    """한국어 문장 분할기"""

    @staticmethod
    def split_sentences(text: str, min_length: int = 10) -> List[str]:
        """
        텍스트를 문장 단위로 분할

        Args:
            text: 입력 텍스트
            min_length: 최소 문장 길이 (너무 짧은 문장 필터링)

        Returns:
            문장 리스트
        """
        # 기본 문장 분할 (마침표, 느낌표, 물음표)
        sentences = re.split(r'[.!?]\s+', text)

        # 빈 문장 및 너무 짧은 문장 제거
        sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) >= min_length]

        # 문장이 없으면 원본 텍스트를 하나의 문장으로
        if not sentences:
            sentences = [text]

        return sentences


class MuveraLegalEmbedder:
    """MUVERA 기반 법률 데이터 임베딩 생성기"""

    def __init__(
        self,
        model_name: str = "nlpai-lab/KURE-v1",
        batch_size: int = 8,
        chunk_batch_size: int = 500,
        fde_config: FixedDimensionalEncodingConfig = None
    ):
        """
        Args:
            model_name: 임베딩 모델
            batch_size: 임베딩 배치 크기 (메모리 절약: 8)
            chunk_batch_size: 청크 처리 배치 크기 (메모리 절약: 500)
            fde_config: FDE 설정 (None이면 기본값 사용)
        """
        print(f"\n임베딩 모델 로딩 중: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.model.max_seq_length = 512
        self.batch_size = batch_size
        self.chunk_batch_size = chunk_batch_size

        embedding_dim = self.model.get_sentence_embedding_dimension()
        print(f"모델 로딩 완료")
        print(f"  - 임베딩 차원: {embedding_dim}")
        print(f"  - 최대 시퀀스 길이: {self.model.max_seq_length}")
        print(f"  - 임베딩 배치 크기: {self.batch_size}")
        print(f"  - 청크 배치 크기: {self.chunk_batch_size}")

        # FDE 설정
        if fde_config is None:
            # 메모리 효율적인 FDE 설정
            self.fde_config = FixedDimensionalEncodingConfig(
                dimension=embedding_dim,  # KURE-v1: 1024
                num_repetitions=1,  # 메모리 절약: 5 -> 3
                num_simhash_projections=3,  # 메모리 절약: 5 -> 4 (16 파티션)
                seed=42,
                encoding_type=EncodingType.AVERAGE,
                projection_type=ProjectionType.DEFAULT_IDENTITY,
                fill_empty_partitions=True,
                final_projection_dimension=1024  # 최종 차원 (Elasticsearch와 호환)
            )
        else:
            self.fde_config = fde_config

        print(f"\nFDE 설정:")
        print(f"  - Repetitions: {self.fde_config.num_repetitions}")
        print(f"  - Partitions: {2**self.fde_config.num_simhash_projections}")
        print(f"  - Final dimension: {self.fde_config.final_projection_dimension}")

        self.sentence_splitter = SentenceSplitter()

    def _split_chunk_into_sentences(self, chunk: Dict) -> List[str]:
        """청크를 문장으로 분할"""
        content = chunk['content']

        # 문서 타입 정보 추가 (첫 문장에만)
        doc_type_map = {
            # Legal data types
            'interpretation': '법령해석례',
            'precedent': '판례',
            'labor_ministry': '고용노동부 법령해설',
            # PDF data types
            'manual': '업무 매뉴얼',
            'standard_contract': '표준근로계약서',
            'law': '법령'
        }
        doc_type_kr = doc_type_map.get(chunk.get('doc_type', ''), '')

        sentences = self.sentence_splitter.split_sentences(content)

        # 첫 문장에 문서 타입 추가
        if sentences and doc_type_kr:
            sentences[0] = f"[{doc_type_kr}] {sentences[0]}"

        return sentences

    def _embed_sentences(self, sentences: List[str]) -> np.ndarray:
        """문장들을 임베딩 (multi-vector)"""
        if not sentences:
            return np.array([]).reshape(0, self.model.get_sentence_embedding_dimension())

        embeddings = self.model.encode(
            sentences,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        return embeddings

    def embed_legal_chunks(
        self,
        chunks_file: Path,
        output_dir: Path
    ) -> Tuple[List[Dict], np.ndarray]:
        """
        MUVERA를 사용한 법률 청크 임베딩 생성

        Args:
            chunks_file: 청크 JSON 파일
            output_dir: 출력 디렉토리

        Returns:
            (chunks, fde_embeddings) 튜플
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        print("\n" + "="*70)
        print("MUVERA 기반 법률 데이터 임베딩 생성")
        print("="*70)

        # 청크 로드
        print(f"\n청크 파일 로딩: {chunks_file.name}")
        with open(chunks_file, 'r', encoding='utf-8') as f:
            chunks = json.load(f)

        print(f"총 {len(chunks):,}개 청크 로딩 완료")

        # 문서 타입별 통계
        doc_type_counts = {}
        for chunk in chunks:
            doc_type = chunk.get('doc_type', 'unknown')
            doc_type_counts[doc_type] = doc_type_counts.get(doc_type, 0) + 1

        print("\n문서 타입별 청크 수:")
        for doc_type, count in sorted(doc_type_counts.items()):
            print(f"  - {doc_type}: {count:,}개")

        # Step 1 & 2: 배치 단위로 임베딩 생성 및 FDE 압축 (메모리 절약)
        print(f"\nStep 1 & 2: 배치 단위로 임베딩 생성 및 FDE 압축 중...")
        print(f"  - 배치 크기: {self.chunk_batch_size}개 청크")
        print(f"  - 총 배치 수: {(len(chunks) + self.chunk_batch_size - 1) // self.chunk_batch_size}")

        all_fde_embeddings = []
        all_sentence_counts = []

        num_batches = (len(chunks) + self.chunk_batch_size - 1) // self.chunk_batch_size

        for batch_idx in range(num_batches):
            start_idx = batch_idx * self.chunk_batch_size
            end_idx = min(start_idx + self.chunk_batch_size, len(chunks))
            batch_chunks = chunks[start_idx:end_idx]

            print(f"\n배치 {batch_idx + 1}/{num_batches} 처리 중 ({start_idx}~{end_idx-1})...")

            # 배치 내 청크들을 문장으로 분할하고 임베딩
            batch_multi_vectors = []
            batch_sentence_counts = []

            for chunk in tqdm(batch_chunks, desc=f"배치 {batch_idx + 1} 임베딩", leave=False):
                sentences = self._split_chunk_into_sentences(chunk)
                batch_sentence_counts.append(len(sentences))
                sentence_embeddings = self._embed_sentences(sentences)
                batch_multi_vectors.append(sentence_embeddings)

            # 배치를 FDE로 압축
            batch_fde = generate_document_fde_batch(
                batch_multi_vectors,
                self.fde_config
            )

            all_fde_embeddings.append(batch_fde)
            all_sentence_counts.extend(batch_sentence_counts)

            # 메모리 정리
            del batch_multi_vectors
            del batch_fde
            gc.collect()

            print(f"배치 {batch_idx + 1} 완료 ({len(batch_chunks)}개 청크)")

        # 모든 배치의 FDE를 하나로 합침
        print("\n모든 배치 결과 병합 중...")
        fde_embeddings = np.vstack(all_fde_embeddings)

        # 메모리 정리
        del all_fde_embeddings
        gc.collect()

        avg_sentences = np.mean(all_sentence_counts)
        sentence_counts = all_sentence_counts

        print(f"\nFDE 압축 완료: {fde_embeddings.shape}")
        print(f"  - 청크 수: {fde_embeddings.shape[0]:,}")
        print(f"  - FDE 차원: {fde_embeddings.shape[1]}")
        print(f"  - 평균 문장 수/청크: {avg_sentences:.1f}")
        print(f"  - 최소 문장 수: {min(sentence_counts)}")
        print(f"  - 최대 문장 수: {max(sentence_counts)}")

        # 청크에 FDE 임베딩 추가
        print("\n청크에 FDE 임베딩 추가 중...")
        for i, chunk in enumerate(tqdm(chunks, desc="임베딩 추가")):
            chunk['embedding'] = fde_embeddings[i].tolist()
            chunk['embedding_type'] = 'muvera_fde'
            chunk['num_sentences'] = sentence_counts[i]

        # 저장
        timestamp = datetime.now().strftime('%Y%m%d')

        # 1. 임베딩 포함 전체 청크 저장
        output_file = output_dir / f"all_chunks_with_muvera_embeddings_{timestamp}.json"
        print(f"\n임베딩 포함 청크 저장 중: {output_file.name}")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)

        file_size = output_file.stat().st_size / 1024 / 1024
        print(f"  - 파일 크기: {file_size:.2f}MB")

        # 2. FDE 임베딩만 numpy 저장
        embeddings_file = output_dir / f"all_muvera_embeddings_{timestamp}.npy"
        np.save(embeddings_file, fde_embeddings)
        print(f"\nFDE 임베딩 numpy 저장: {embeddings_file.name}")

        # 3. 메타데이터 저장
        metadata = {
            'created_at': datetime.now().isoformat(),
            'total_chunks': len(chunks),
            'embedding_dim': int(fde_embeddings.shape[1]),
            'embedding_type': 'muvera_fde',
            'model_name': 'nlpai-lab/KURE-v1',
            'fde_config': {
                'num_repetitions': self.fde_config.num_repetitions,
                'num_simhash_projections': self.fde_config.num_simhash_projections,
                'num_partitions': 2**self.fde_config.num_simhash_projections,
                'final_dimension': self.fde_config.final_projection_dimension
            },
            'avg_sentences_per_chunk': float(avg_sentences),
            'min_sentences': int(min(sentence_counts)),
            'max_sentences': int(max(sentence_counts)),
            'doc_type_counts': doc_type_counts,
            'files': {
                'chunks_with_embeddings': output_file.name,
                'embeddings_numpy': embeddings_file.name
            }
        }

        metadata_file = output_dir / f"all_muvera_embeddings_metadata_{timestamp}.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

        print(f"\n메타데이터 저장: {metadata_file.name}")

        # 요약
        print("\n" + "="*70)
        print("MUVERA 임베딩 완료 요약")
        print("="*70)
        print(f"총 청크 수:         {metadata['total_chunks']:,}개")
        print(f"FDE 차원:           {metadata['embedding_dim']}")
        print(f"평균 문장 수:       {metadata['avg_sentences_per_chunk']:.1f}")
        print(f"모델:               {metadata['model_name']}")
        print("-" * 70)
        print(f"JSON 파일:          {output_file.name} ({file_size:.2f}MB)")
        print(f"NumPy 파일:         {embeddings_file.name}")
        print(f"메타데이터:         {metadata_file.name}")
        print("="*70)

        return chunks, fde_embeddings

    def encode_query(self, query: str) -> np.ndarray:
        """
        쿼리를 MUVERA FDE로 인코딩

        Args:
            query: 검색 쿼리

        Returns:
            FDE 벡터
        """
        # 쿼리를 문장으로 분할
        sentences = self.sentence_splitter.split_sentences(query)

        # 각 문장 임베딩
        sentence_embeddings = self._embed_sentences(sentences)

        # FDE로 압축 (쿼리는 SUM aggregation)
        query_fde = generate_query_fde(sentence_embeddings, self.fde_config)

        return query_fde


def main():
    """메인 실행 함수"""
    # 경로 설정
    project_root = Path(__file__).parent.parent

    # 청크 파일 경로
    chunks_dir = project_root / "data" / "processed" / "chunks"
    output_dir = project_root / "data" / "processed" / "embeddings"

    if not chunks_dir.exists():
        print(f"오류: 청크 디렉토리가 없습니다: {chunks_dir}")
        return

    print("\n" + "="*70)
    print("MUVERA 임베딩 파이프라인 (PDF + Legal 통합)")
    print("="*70)

    # 1. Legal 청크 파일 찾기
    legal_files = sorted([f for f in chunks_dir.glob("legal_chunks_*.json")
                         if "metadata" not in f.name], reverse=True)

    # 2. PDF 청크 파일 찾기 (all_chunks.json)
    pdf_file = chunks_dir / "all_chunks.json"

    all_chunks = []

    # Legal 데이터 로드
    if legal_files:
        legal_file = legal_files[0]
        print(f"\n1. Legal 청크 로드: {legal_file.name}")
        with open(legal_file, 'r', encoding='utf-8') as f:
            legal_chunks = json.load(f)
        print(f"   - Legal 청크 수: {len(legal_chunks):,}개")
        all_chunks.extend(legal_chunks)
    else:
        print("\n경고: Legal 청크 파일을 찾을 수 없습니다.")

    # PDF 데이터 로드
    if pdf_file.exists():
        print(f"\n2. PDF 청크 로드: {pdf_file.name}")
        with open(pdf_file, 'r', encoding='utf-8') as f:
            pdf_chunks = json.load(f)
        print(f"   - PDF 청크 수: {len(pdf_chunks):,}개")
        all_chunks.extend(pdf_chunks)
    else:
        print("\n경고: PDF 청크 파일(all_chunks.json)을 찾을 수 없습니다.")

    if not all_chunks:
        print("\n오류: 임베딩할 청크가 없습니다.")
        print("먼저 legal/2_chunk.py 또는 pdf/2_chunk.py를 실행하세요.")
        return

    print(f"\n총 청크 수: {len(all_chunks):,}개")

    # 임시 통합 청크 파일 생성
    merged_file = chunks_dir / "merged_chunks_temp.json"
    print(f"\n3. 통합 청크 파일 생성: {merged_file.name}")
    with open(merged_file, 'w', encoding='utf-8') as f:
        json.dump(all_chunks, f, ensure_ascii=False)

    # MUVERA 임베더 초기화 (메모리 효율적 설정)
    print("\n4. MUVERA 임베더 초기화 중...")
    embedder = MuveraLegalEmbedder(
        model_name="nlpai-lab/KURE-v1",
        batch_size=2,           # 메모리 절약
        chunk_batch_size=50    # 배치 단위 처리
    )

    # 임베딩 생성
    print("\n5. MUVERA 임베딩 생성 시작...")
    chunks, fde_embeddings = embedder.embed_legal_chunks(merged_file, output_dir)

    # 임시 파일 삭제
    merged_file.unlink()
    print(f"\n임시 파일 삭제: {merged_file.name}")

    print("\n" + "="*70)
    print("MUVERA 임베딩 생성 완료!")
    print("="*70)
    print("다음 단계: Elasticsearch 인덱싱 (4_index.py)")
    print("="*70)


if __name__ == "__main__":
    main()
