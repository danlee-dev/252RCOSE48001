import os
import re
from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer
import numpy as np

from fde_generator import (
    FixedDimensionalEncodingConfig,
    EncodingType,
    ProjectionType,
    generate_query_fde
)

# --- 설정 ---
# 1. 쿼리 임베딩 모델 로드
print("임베딩 모델 로드 중... (KURE-v1)")
MODEL_NAME = "nlpai-lab/KURE-v1"
try:
    model = SentenceTransformer(MODEL_NAME)
    model.max_seq_length = 512
except Exception as e:
    print(f"모델 로드 실패! '{MODEL_NAME}'이 맞는지 확인하세요. (오류: {e})")
    print("인터넷 연결을 확인하거나, 모델 이름을 다시 확인하세요.")
    exit()

# 2. FDE 설정 (임베딩 생성 시와 동일하게 설정 - 메모리 효율 버전)
embedding_dim = model.get_sentence_embedding_dimension()
fde_config = FixedDimensionalEncodingConfig(
    dimension=embedding_dim,  # KURE-v1: 1024
    num_repetitions=3,        # 메모리 절약: 5 -> 3
    num_simhash_projections=4,  # 메모리 절약: 5 -> 4 (16 파티션)
    seed=42,
    encoding_type=EncodingType.DEFAULT_SUM,  # 쿼리는 SUM
    projection_type=ProjectionType.DEFAULT_IDENTITY,
    final_projection_dimension=1024
)

print(f"FDE 설정 로드 완료")
print(f"  - Final dimension: {fde_config.final_projection_dimension}")

# 3. Elasticsearch 연결
es = Elasticsearch("http://localhost:9200")

# 4. 인덱스 이름
INDEX_NAME = "docscanner_chunks"

# 5. 문서 타입 매핑
DOC_TYPE_MAP = {
    'interpretation': '법령해석례',
    'precedent': '판례',
    'labor_ministry': '고용노동부',
    'manual': '매뉴얼(PDF)',
    'employment_rules': '취업규칙(PDF)',
    'guide': '안내서(PDF)',
    'leaflet': '리플릿(PDF)',
    'question': '질의(법률)',
    'answer': '답변(법률)',
}
# -----------------


def split_sentences(text: str, min_length: int = 10):
    """텍스트를 문장으로 분할"""
    sentences = re.split(r'[.!?]\s+', text)
    sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) >= min_length]
    if not sentences:
        sentences = [text]
    return sentences


def encode_query_with_fde(query: str):
    """쿼리를 MUVERA FDE로 인코딩"""
    # 1. 문장 분할
    sentences = split_sentences(query)

    # 2. 각 문장 임베딩
    sentence_embeddings = model.encode(
        sentences,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    # 3. FDE로 압축
    query_fde = generate_query_fde(sentence_embeddings, fde_config)

    return query_fde


def search_es(query: str, top_k: int = 5, filter_source: str = None):
    """Elasticsearch에 MUVERA FDE 벡터 검색 실행"""

    # 1. 쿼리를 FDE로 변환
    print("\n쿼리를 FDE로 인코딩 중...")
    try:
        query_vector = encode_query_with_fde(query).tolist()
    except Exception as e:
        print(f"쿼리 인코딩 실패: {e}")
        return []

    print("Elasticsearch로 검색 실행 중...")

    # 2. Elasticsearch KNN 쿼리
    knn_query = {
        "field": "embedding",
        "query_vector": query_vector,
        "k": top_k,
        "num_candidates": 100
    }

    # 3. 필터 추가
    if filter_source:
        knn_query["filter"] = {
            "term": {
                "source": filter_source
            }
        }
        print(f"(필터 적용: source == '{filter_source}')")

    try:
        response = es.search(
            index=INDEX_NAME,
            knn=knn_query,
            size=top_k,
            request_timeout=30
        )
        return response['hits']['hits']

    except Exception as e:
        print(f"ES 검색 중 오류 발생: {e}")
        return []


def main():
    """대화형 검색 메인 루프"""
    if not es.indices.exists(index=INDEX_NAME):
        print(f"오류: '{INDEX_NAME}' 인덱스를 찾을 수 없습니다.")
        print("4_index.py를 먼저 실행했는지 확인하세요.")
        return

    print("\n" + "="*70)
    print("MUVERA 기반 Elasticsearch 검색 테스트")
    print("="*70)
    print("\n사용법:")
    print("  - 검색: 쿼리 입력")
    print("  - 필터: @필터명 추가 (예: 최저임금 @precedent)")
    print("  - 상위 결과: #숫자 추가 (예: 연차 #10)")
    print("  - 종료: 'q' 또는 'exit' 입력")
    print("\n사용 가능한 필터:")
    print("  - @precedent (판례만)")
    print("  - @interpretation (법령해석례만)")
    print("  - @manual (매뉴얼만)")
    print("="*70 + "\n")

    while True:
        try:
            user_input = input("검색 쿼리: ").strip()

            if user_input.lower() in ['q', 'exit', 'quit', '종료']:
                print("\n테스트를 종료합니다.")
                break

            if not user_input:
                continue

            # --- 파라미터 파싱 ---
            query = user_input
            filter_source = None
            top_k = 5

            # 1. @ 필터 파싱
            if '@' in query:
                parts = query.split('@')
                query = parts[0].strip()
                filter_str = parts[1].strip().split()[0]
                filter_source = filter_str

            # 2. # top_k 파싱
            if '#' in query:
                parts = query.split('#')
                query = parts[0].strip()
                try:
                    top_k = int(parts[1].strip().split()[0])
                    top_k = min(max(top_k, 1), 20)
                except ValueError:
                    print("잘못된 숫자 형식. 기본 5개로 검색합니다.")
                    top_k = 5

            if not query:
                print("검색어가 없습니다.")
                continue

            # --- ES 검색 실행 ---
            results = search_es(query, top_k=top_k, filter_source=filter_source)

            if not results:
                print("\n-> 검색 결과가 없습니다.")
                print("="*70)
                continue

            # --- 결과 출력 ---
            print(f"\n--- '{query}' 검색 결과 (Top {len(results)}) ---")
            for i, hit in enumerate(results):
                source_data = hit['_source']
                source_type = source_data.get('source', 'unknown')

                print(f"\n[{i+1}] 유사도: {hit['_score']:.4f}")
                print(f"  (출처 타입: {DOC_TYPE_MAP.get(source_type, source_type)})")

                # 내용 미리보기
                content = source_data.get('text', '').replace('\n', ' ')[:200]
                print(f"  내용: {content}...")
                print("-"*20)
            print("="*70)

        except KeyboardInterrupt:
            print("\n테스트를 종료합니다.")
            break
        except Exception as e:
            print(f"!!! 전체 루프 오류 발생: {e}")


if __name__ == "__main__":
    main()
