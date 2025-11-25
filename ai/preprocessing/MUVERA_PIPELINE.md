# MUVERA 기반 Elasticsearch 구축 파이프라인

MUVERA (Multi-Vector Retrieval with FDE) 임베딩을 사용한 Elasticsearch 검색 시스템 구축 가이드

## 개요

이 파이프라인은 PDF 문서와 Legal API 데이터를 통합하여 MUVERA 임베딩을 생성하고 Elasticsearch에 인덱싱합니다.

## 파일 구조

```
ai/preprocessing/
├── 3_embed_muvera.py          # MUVERA 임베딩 생성 (PDF + Legal 통합)
├── 4_index.py                  # Elasticsearch 인덱싱
├── fde_generator.py            # FDE 구현
├── legal/
│   ├── 1_collect.py           # Legal 데이터 수집
│   └── 2_chunk.py             # Legal 데이터 청킹
├── pdf/
│   ├── 1_extract.py           # PDF 추출
│   └── 2_chunk.py             # PDF 청킹
└── scripts/
    ├── reset_es_index.py      # ES 인덱스 초기화
    └── test_muvera_search.py  # 검색 테스트
```

## 실행 순서

### 1. 데이터 수집 (이미 완료된 경우 생략)

```bash
cd ai/preprocessing/legal
python 1_collect.py
```

**출력**:
- `ai/data/raw/api/interpretations_*.json`
- `ai/data/raw/api/precedents_*.json`
- `ai/data/raw/api/labor_ministry_*.json`

### 2. 데이터 청킹 (이미 완료된 경우 생략)

```bash
# Legal 데이터 청킹
cd ai/preprocessing/legal
python 2_chunk.py

# PDF 데이터 청킹 (필요 시)
cd ai/preprocessing/pdf
python 1_extract.py  # PDF 텍스트 추출
python 2_chunk.py    # 청킹
```

**출력**:
- `ai/data/processed/chunks/legal_chunks_20251124.json` (14,549개)
- `ai/data/processed/chunks/all_chunks.json` (674개, PDF)

### 3. MUVERA 임베딩 생성

```bash
cd ai/preprocessing
python 3_embed_muvera.py
```

**동작**:
1. Legal 청크 로드 (`legal_chunks_*.json`)
2. PDF 청크 로드 (`all_chunks.json`)
3. 통합 (총 15,223개)
4. 각 청크를 문장으로 분할
5. 각 문장 임베딩 (KURE-v1)
6. FDE 압축 (1024차원)

**출력**:
- `ai/data/processed/embeddings/all_chunks_with_muvera_embeddings_20251125.json`
- `ai/data/processed/embeddings/all_muvera_embeddings_20251125.npy`
- `ai/data/processed/embeddings/all_muvera_embeddings_metadata_20251125.json`

**예상 시간**: 약 30-60분 (데이터 크기에 따라 다름)

### 4. Elasticsearch 인덱스 초기화

```bash
cd ai/preprocessing/scripts
python reset_es_index.py
```

**동작**:
- 기존 `docscanner_chunks` 인덱스 삭제
- 새 인덱스 생성 (nori 분석기, 1024차원 벡터)

### 5. Elasticsearch 인덱싱

```bash
cd ai/preprocessing
python 4_index.py
```

**동작**:
1. 최신 MUVERA 임베딩 파일 로드
2. Elasticsearch bulk API로 인덱싱 (500개씩)

**출력**:
```
성공: 15,223개
실패: 0개
```

### 6. 검색 테스트

```bash
cd ai/preprocessing/scripts
python test_muvera_search.py
```

**기능**:
- 대화형 검색 인터페이스
- 출처 필터링 지원
- BM25 + 벡터 하이브리드 검색

**사용 예시**:
```
검색어 입력: 근로시간은 하루 몇 시간?
검색어 입력: 최저임금 @precedent
검색어 입력: 연차 계산 @labor_ministry
```

## 필터 사용법

검색 시 `@필터명`을 추가하여 특정 출처만 검색:

- `@precedent` - 판례만 (10,576개)
- `@interpretation` - 법령해석례만 (589개)
- `@labor_ministry` - 고용노동부만 (3,384개)
- `@manual` - 업무 매뉴얼만
- `@standard_contract` - 표준근로계약서만

## 데이터 통계

**전체**: 15,223개
- Legal 데이터: 14,549개
  - 판례: 10,576개
  - 고용노동부: 3,384개
  - 법령해석례: 589개
- PDF 데이터: 674개

## MUVERA 설정

**FDE 구성**:
- Repetitions: 1
- SimHash Projections: 3 (8 파티션)
- Final Dimension: 1024
- Encoding: AVERAGE (문서), SUM (쿼리)

**임베딩 모델**:
- 모델: nlpai-lab/KURE-v1
- 차원: 1024
- 최대 시퀀스 길이: 512

## 트러블슈팅

### Elasticsearch 연결 오류
```bash
# Elasticsearch 상태 확인
docker ps | grep elasticsearch

# 재시작
docker restart docscanner-elasticsearch
```

### Nori 플러그인 오류
```bash
# Nori 플러그인 설치
docker exec docscanner-elasticsearch bin/elasticsearch-plugin install analysis-nori
docker restart docscanner-elasticsearch
```

### 디스크 부족
```bash
# 디스크 사용량 확인
df -h

# Docker 정리
docker system prune -f
```

### 메모리 부족
`3_embed_muvera.py`에서 배치 크기 조정:
```python
embedder = MuveraLegalEmbedder(
    batch_size=1,           # 2 → 1로 감소
    chunk_batch_size=25     # 50 → 25로 감소
)
```

## 다음 단계

1. 그래프 데이터베이스 구축 (선택):
   ```bash
   python 5_build_graph.py
   python 6_create_relationships.py
   python 7_seed_ontology.py
   ```

2. Backend API 통합
3. 검색 성능 최적화
4. 프로덕션 배포

## 참고 자료

- [MUVERA Paper](https://arxiv.org/abs/2405.17800)
- [KURE-v1 Model](https://huggingface.co/nlpai-lab/KURE-v1)
- [Elasticsearch Guide](https://www.elastic.co/guide/en/elasticsearch/reference/current/index.html)
