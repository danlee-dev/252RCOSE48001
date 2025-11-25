"""
Elasticsearch 인덱스 초기화 스크립트

기존 docscanner_chunks 인덱스를 삭제하고 새로 생성합니다.
"""

from elasticsearch import Elasticsearch

# Elasticsearch 연결
es = Elasticsearch("http://localhost:9200")

# 인덱스 이름
INDEX_NAME = "docscanner_chunks"


def reset_index():
    """인덱스 삭제 및 재생성"""

    print(f"\n{'='*60}")
    print(f"Elasticsearch 인덱스 초기화: {INDEX_NAME}")
    print(f"{'='*60}\n")

    # 1. 기존 인덱스 존재 여부 확인
    if es.indices.exists(index=INDEX_NAME):
        print(f"기존 '{INDEX_NAME}' 인덱스 발견")

        # 문서 수 확인
        try:
            count_response = es.count(index=INDEX_NAME)
            doc_count = count_response['count']
            print(f"  - 현재 문서 수: {doc_count:,}개")
        except Exception as e:
            print(f"  - 문서 수 조회 실패: {e}")

        # 삭제
        print(f"\n'{INDEX_NAME}' 인덱스 삭제 중...")
        es.indices.delete(index=INDEX_NAME)
        print("삭제 완료")
    else:
        print(f"'{INDEX_NAME}' 인덱스가 존재하지 않음")

    # 2. 새 인덱스 생성
    print(f"\n'{INDEX_NAME}' 인덱스 생성 중...")

    index_settings = {
        "mappings": {
            "properties": {
                "text": {
                    "type": "text",
                    "analyzer": "nori"  # 한국어 형태소 분석기
                },
                "embedding": {
                    "type": "dense_vector",
                    "dims": 1024  # MUVERA FDE 벡터 차원
                },
                "source": {
                    "type": "keyword"  # 필터링용 출처
                }
            }
        }
    }

    es.indices.create(index=INDEX_NAME, body=index_settings)
    print("인덱스 생성 완료")

    print(f"\n{'='*60}")
    print("초기화 완료!")
    print(f"{'='*60}\n")
    print("다음 단계: 4_index.py 실행하여 데이터 인덱싱")
    print()


if __name__ == "__main__":
    try:
        reset_index()
    except Exception as e:
        print(f"\n오류 발생: {e}")
        import traceback
        traceback.print_exc()
