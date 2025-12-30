"""
MUVERA 기반 Elasticsearch 검색 테스트 스크립트

MUVERA (Multi-Vector Retrieval with FDE) 임베딩을 사용한
Elasticsearch 검색 기능을 대화형으로 테스트합니다.
"""

import sys
import re
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
# scripts/ -> preprocessing/ -> ai/ -> docscanner-ai/
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer
import numpy as np
import json

# FDE 생성기 import
from ai.preprocessing.fde_generator import (
    FixedDimensionalEncodingConfig,
    EncodingType,
    ProjectionType,
    generate_query_fde
)


class SentenceSplitter:
    """한국어 문장 분할기"""

    @staticmethod
    def split_sentences(text: str, min_length: int = 10):
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


class MuveraSearchTester:
    """MUVERA 기반 검색 테스트"""

    def __init__(
        self,
        es_url: str = "http://localhost:9200",
        index_name: str = "docscanner_chunks",
        model_name: str = "nlpai-lab/KURE-v1"
    ):
        """
        Args:
            es_url: Elasticsearch URL
            index_name: 인덱스 이름
            model_name: 임베딩 모델
        """
        print("초기화 중...")

        # Elasticsearch 연결
        self.es = Elasticsearch(es_url)
        self.index_name = index_name

        # 임베딩 모델 로드
        print(f"임베딩 모델 로딩: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.model.max_seq_length = 512

        # FDE 설정 (쿼리용이므로 fill_empty_partitions=False)
        embedding_dim = self.model.get_sentence_embedding_dimension()
        self.fde_config = FixedDimensionalEncodingConfig(
            dimension=embedding_dim,
            num_repetitions=1,
            num_simhash_projections=3,
            seed=42,
            encoding_type=EncodingType.AVERAGE,
            projection_type=ProjectionType.DEFAULT_IDENTITY,
            fill_empty_partitions=False,  # 쿼리는 fill_empty_partitions 미지원
            final_projection_dimension=1024
        )

        self.sentence_splitter = SentenceSplitter()

        print("초기화 완료")
        print(f"  - ES 연결: {es_url}")
        print(f"  - 인덱스: {index_name}")
        print(f"  - 모델: {model_name}")
        print(f"  - 임베딩 차원: {embedding_dim}")
        print()

    def encode_query(self, query: str) -> np.ndarray:
        """쿼리를 MUVERA FDE로 인코딩"""
        # 쿼리를 문장으로 분할
        sentences = self.sentence_splitter.split_sentences(query)

        # 각 문장 임베딩
        sentence_embeddings = self.model.encode(
            sentences,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        # FDE로 압축 (쿼리는 SUM aggregation)
        query_fde = generate_query_fde(sentence_embeddings, self.fde_config)

        return query_fde

    def search(
        self,
        query: str,
        top_k: int = 5,
        source_filter: str = None,
        use_hybrid: bool = True
    ) -> dict:
        """
        MUVERA 검색 수행

        Args:
            query: 검색 쿼리
            top_k: 반환할 결과 수
            source_filter: 출처 필터 (예: "precedent", "interpretation")
            use_hybrid: BM25 + 벡터 하이브리드 검색 사용 여부

        Returns:
            검색 결과
        """
        # 쿼리 임베딩
        query_vector = self.encode_query(query)

        # Elasticsearch 쿼리 구성
        if use_hybrid:
            # 하이브리드 검색 (BM25 + 벡터)
            es_query = {
                "query": {
                    "bool": {
                        "should": [
                            {
                                "match": {
                                    "text": {
                                        "query": query,
                                        "boost": 0.3  # BM25 가중치
                                    }
                                }
                            }
                        ],
                        "filter": []
                    }
                },
                "knn": {
                    "field": "embedding",
                    "query_vector": query_vector.tolist(),
                    "k": top_k * 2,  # 더 많이 가져온 후 재순위
                    "num_candidates": 100,
                    "boost": 0.7  # 벡터 검색 가중치
                },
                "size": top_k
            }
        else:
            # 벡터 검색만
            es_query = {
                "knn": {
                    "field": "embedding",
                    "query_vector": query_vector.tolist(),
                    "k": top_k,
                    "num_candidates": 100
                },
                "size": top_k
            }

        # 출처 필터 추가
        if source_filter and use_hybrid:
            es_query["query"]["bool"]["filter"].append({
                "term": {"source": source_filter}
            })
        elif source_filter and not use_hybrid:
            # 벡터 검색 only일 때는 filter 추가
            es_query["query"] = {
                "bool": {
                    "filter": [
                        {"term": {"source": source_filter}}
                    ]
                }
            }

        # 검색 실행
        response = self.es.options(request_timeout=30).search(
            index=self.index_name,
            body=es_query
        )

        return response

    def print_results(self, response: dict, query: str):
        """검색 결과 출력"""
        hits = response['hits']['hits']

        print(f"\n검색 쿼리: '{query}'")
        print(f"총 {len(hits)}개 결과")
        print("=" * 80)

        for i, hit in enumerate(hits, 1):
            source = hit['_source']
            score = hit['_score']

            print(f"\n[{i}] 유사도: {score:.4f}")
            print(f"출처: {source.get('source', 'unknown')}")

            # 내용 출력 (최대 200자)
            content = source.get('text', '')
            if len(content) > 200:
                content = content[:200] + "..."
            print(f"내용: {content}")
            print("-" * 80)

    def get_index_stats(self):
        """인덱스 통계 출력"""
        try:
            count_response = self.es.count(index=self.index_name)
            total_docs = count_response['count']

            # source별 통계
            agg_response = self.es.search(
                index=self.index_name,
                body={
                    "size": 0,
                    "aggs": {
                        "sources": {
                            "terms": {
                                "field": "source",
                                "size": 20
                            }
                        }
                    }
                }
            )

            print("\n인덱스 통계")
            print("=" * 80)
            print(f"총 문서 수: {total_docs:,}개")
            print("\n출처별 문서 수:")

            for bucket in agg_response['aggregations']['sources']['buckets']:
                source = bucket['key']
                count = bucket['doc_count']
                print(f"  - {source}: {count:,}개")

            print("=" * 80)

        except Exception as e:
            print(f"통계 조회 오류: {e}")

    def interactive_search(self):
        """대화형 검색 모드"""
        print("\nMUVERA 검색 테스트 (종료: q 또는 exit)")
        print("\n사용 가능한 필터:")
        print("  - @precedent (판례)")
        print("  - @labor_ministry (고용노동부)")
        print("  - @interpretation (법령해석례)")
        print("  - @manual (업무 매뉴얼)")
        print("  - @standard_contract (표준근로계약서)")
        print("\n예시: 근로시간은 하루 몇 시간? @precedent")
        print("=" * 80)

        while True:
            try:
                user_input = input("\n검색어 입력: ").strip()

                if user_input.lower() in ['q', 'exit', 'quit']:
                    print("종료합니다.")
                    break

                if not user_input:
                    continue

                # 필터 파싱
                source_filter = None
                if '@' in user_input:
                    parts = user_input.split('@')
                    query = parts[0].strip()
                    filter_value = parts[1].strip()
                    source_filter = filter_value
                else:
                    query = user_input

                # 검색 수행
                response = self.search(
                    query=query,
                    top_k=5,
                    source_filter=source_filter,
                    use_hybrid=True
                )

                # 결과 출력
                self.print_results(response, query)

            except KeyboardInterrupt:
                print("\n\n종료합니다.")
                break
            except Exception as e:
                print(f"\n오류 발생: {e}")


def main():
    """메인 함수"""
    # 테스터 초기화
    tester = MuveraSearchTester(
        es_url="http://localhost:9200",
        index_name="docscanner_chunks",
        model_name="nlpai-lab/KURE-v1"
    )

    # 인덱스 통계 출력
    tester.get_index_stats()

    # 대화형 검색 시작
    tester.interactive_search()


if __name__ == "__main__":
    main()
