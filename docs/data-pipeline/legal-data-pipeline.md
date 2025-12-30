# 법률 데이터 파이프라인

## 개요

근로계약서 분석 RAG 시스템을 위한 법률 데이터 수집, 청킹, 임베딩 파이프라인을 설명합니다.

### 목적

근로계약서의 법적 유효성을 검증하기 위해 한국 법률 데이터를 수집하고 RAG 시스템에 활용할 수 있도록 전처리합니다.

### 전체 워크플로우

```
1. 데이터 수집 (legal/1_collect.py)
   국가법령정보 Open API → JSON (2,931건)

2. 데이터 청킹 (legal/2_chunk.py)
   JSON → 의미 단위 청크 (14,549개)

3. 임베딩 생성 (legal/3_embed.py)
   청크 → KURE-v1 임베딩 (1024차원)

4. Elasticsearch 인덱싱 (다음 단계)
   임베딩 → 하이브리드 검색 인덱스
```

### 데이터 소스

- **법령해석례**: 법제처 공식 법령 해석
- **판례**: 대법원 등 실제 판결문
- **고용노동부 법령해설**: 고용노동부 실무 해석


## 1. 데이터 수집

### 스크립트

`ai/preprocessing/legal/1_collect.py`

### 사용 방법

```bash
# .env 파일에 API 키 설정
LEGAL_API_USER_ID=your_api_key

# 실행
cd ai/preprocessing/legal
python 1_collect.py
```

### 수집 키워드 (13개)

```python
keywords = [
    "근로계약", "근로기준법", "최저임금", "퇴직금", "해고", "임금",
    "연차", "근로시간", "휴게시간", "휴일", "연장근로", "통상임금", "수습기간"
]
```

### 수집 결과

| 데이터 타입 | 원본 건수 | 파일 크기 | 파일명 |
|------------|----------|----------|--------|
| 법령해석례 | 135건 | 1.1MB | `interpretations_20251027.json` |
| 판례 | 969건 | 16MB | `precedents_20251027.json` |
| 고용노동부 | 1,827건 | 5MB | `labor_ministry_20251027.json` |
| **합계** | **2,931건** | **22.1MB** | - |

### 데이터 구조

**법령해석례:**
```json
{
  "법령해석례일련번호": "329605",
  "안건명": "...",
  "안건번호": "16-0454",
  "질의요지": "...",
  "회답": "...",
  "이유": "...",
  "해석일자": "2016.11.23",
  "검색키워드": "근로계약"
}
```

**판례:**
```json
{
  "판례일련번호": "240951",
  "사건명": "...",
  "사건번호": "2020도16541",
  "판시사항": "...",
  "판결요지": "...",
  "판례내용": "...",
  "참조조문": "...",
  "법원명": "대법원",
  "선고일자": "2024.06.27"
}
```

**고용노동부:**
```json
{
  "법령해석일련번호": "13036",
  "안건명": "...",
  "안건번호": "근기 68207-695",
  "질의요지": "...",
  "회답": "...",
  "이유": "...",
  "관련법령": "...",
  "해석일자": "1999.11.23"
}
```


## 2. 데이터 청킹

### 스크립트

`ai/preprocessing/legal/2_chunk.py`

### 사용 방법

```bash
cd ai/preprocessing/legal
python 2_chunk.py
```

### 청킹 전략

#### 구조 기반 의미 청킹 (Structure-aware Semantic Chunking)

법률 문서의 의미 구조를 유지하면서 청킹합니다.

**법령해석례:**
```
청크 1: 안건명 + 질의요지 (500자)
청크 2: 회답 (500-1000자, 분할 가능)
청크 3: 이유 (500-1000자, 분할 가능)
```

**판례:**
```
청크 1: 사건정보 + 판시사항 (500자)
청크 2: 판결요지 (1000자, 분할 가능)
청크 3: 판례내용 (800자 단위 분할)
청크 4: 참조조문 (있는 경우)
```

**고용노동부:**
```
청크 1: 안건명 + 질의요지 (500자)
청크 2: 회답 (500-1000자, 분할 가능)
청크 3: 이유 (500-1000자, 분할 가능)
청크 4: 관련법령 (있는 경우)
```

### 청킹 파라미터

```python
max_chunk_length = 1000  # 최대 청크 길이 (법률 문서용)
min_chunk_length = 150   # 최소 청크 길이
```

### 청킹 결과

| 데이터 타입 | 원본 문서 | 생성 청크 | 증가율 |
|------------|----------|----------|--------|
| 법령해석례 | 135건 | 589개 | 4.4x |
| 판례 | 969건 | 10,576개 | 10.9x |
| 고용노동부 | 1,827건 | 3,384개 | 1.9x |
| **합계** | **2,931건** | **14,549개** | **5.0x** |

**출력 파일:**
- `legal_chunks_20251027.json` (28MB)
- `legal_chunks_metadata_20251027.json`

### 청크 메타데이터

```json
{
  "chunk_id": "uuid",
  "source_type": "legal",
  "doc_type": "interpretation|precedent|labor_ministry",
  "source_id": "329605",
  "title": "사건명/안건명",
  "case_number": "사건번호",
  "court": "법원명",
  "date": "날짜",
  "chunk_type": "question|answer|reason|case_info|summary|content|references",
  "chunk_index": 0,
  "total_chunks": 4,
  "search_keyword": "근로계약",
  "content": "실제 텍스트 내용"
}
```


## 3. 임베딩 생성

### 스크립트

`ai/preprocessing/legal/3_embed.py`

### 사용 방법

```bash
cd ai/preprocessing/legal
python 3_embed.py
```

### 임베딩 모델

**KURE-v1 (Korean Legal Document Embedding)**
- 모델: `nlpai-lab/KURE-v1`
- 임베딩 차원: 1024
- 최대 시퀀스 길이: 512 tokens
- 특화 분야: 한국어 법률 문서

### 임베딩 설정

```python
model = SentenceTransformer("nlpai-lab/KURE-v1")
model.max_seq_length = 512
batch_size = 16
normalize_embeddings = True  # 코사인 유사도 최적화
```

### 텍스트 전처리

청크 content에 메타데이터를 추가하여 검색 성능 향상:

```python
# 문서 타입 추가
text = f"[{doc_type_kr}] {content}"

# 검색 키워드 추가
text = f"{text}\n관련 키워드: {search_keyword}"
```

### 임베딩 결과

| 항목 | 값 |
|------|-----|
| 총 청크 수 | 14,549개 |
| 임베딩 차원 | 1024 |
| 정규화 | True |
| 소요 시간 | 약 25분 |
| JSON 파일 크기 | 429.02MB |
| NumPy 파일 크기 | 56.83MB |

**출력 파일:**
- `legal_chunks_with_embeddings_20251027.json` (429MB) - 임베딩 포함 전체 청크
- `legal_embeddings_20251027.npy` (57MB) - 임베딩만 NumPy 배열
- `legal_embeddings_metadata_20251027.json` - 메타데이터

### 검색 성능 테스트

테스트 쿼리로 유사도 검색 검증:

| 쿼리 | 필터 | 유사도 | 결과 문서 타입 |
|------|------|--------|---------------|
| "근로계약서에 수습기간 6개월 쓰면 문제 있나요?" | None | 0.70 | 고용노동부/법령해석례 |
| "최저임금 위반 시 벌칙은?" | labor_ministry | 0.67 | 고용노동부 |
| "기간제 근로계약 갱신 거절 가능한가요?" | precedent | 0.75 | 판례 |
| "연차 휴가 미사용 수당은?" | interpretation | 0.62 | 법령해석례 |


## 결과물

### 디렉토리 구조

```
docscanner-ai/
└── ai/
    └── data/
        ├── raw/api/
        │   ├── interpretations_20251027.json (1.1MB)
        │   ├── precedents_20251027.json (16MB)
        │   └── labor_ministry_20251027.json (5MB)
        │
        └── processed/
            ├── chunks/
            │   ├── all_chunks.json (1.1MB)              ← 기존 PDF 청크 (674개)
            │   ├── legal_chunks_20251027.json (28MB)    ← 법률 청크 (14,549개)
            │   └── legal_chunks_metadata_20251027.json
            │
            └── embeddings/
                ├── chunks_with_embeddings.json (20MB)                       ← 기존 PDF 임베딩 (674개)
                ├── legal_chunks_with_embeddings_20251027.json (429MB)      ← 법률 임베딩 (14,549개)
                ├── legal_embeddings_20251027.npy (57MB)
                └── legal_embeddings_metadata_20251027.json
```

### 통합 통계

| 구분 | PDF 문서 | 법률 데이터 | 합계 |
|------|---------|------------|------|
| 원본 문서 | 4개 | 2,931건 | 2,935개 |
| 청크 수 | 674개 | 14,549개 | **15,223개** |
| 임베딩 차원 | 1024 | 1024 | 1024 |
| 모델 | KURE-v1 | KURE-v1 | KURE-v1 |


## 다음 단계

### Elasticsearch 인덱싱

**1. Elasticsearch 스키마 설계**

```json
{
  "mappings": {
    "properties": {
      "chunk_id": { "type": "keyword" },
      "source_type": { "type": "keyword" },
      "doc_type": { "type": "keyword" },
      "content": { "type": "text", "analyzer": "nori" },
      "embedding": { "type": "dense_vector", "dims": 1024 },
      "title": { "type": "text" },
      "case_number": { "type": "keyword" },
      "date": { "type": "date" }
    }
  }
}
```

**2. 하이브리드 검색 구현**

```python
# BM25 (키워드 검색)
keyword_query = {
    "match": {
        "content": user_query
    }
}

# Vector (의미 검색)
vector_query = {
    "knn": {
        "field": "embedding",
        "query_vector": query_embedding,
        "k": 50
    }
}

# 하이브리드 (BM25 + Vector)
hybrid_search = {
    "query": {
        "bool": {
            "should": [
                keyword_query,
                vector_query
            ]
        }
    }
}
```

**3. FastAPI 검색 API**

```python
@app.post("/search")
async def search_legal_docs(
    query: str,
    doc_type: Optional[str] = None,
    top_k: int = 5
):
    # 쿼리 임베딩
    query_embedding = embedder.encode(query)

    # Elasticsearch 하이브리드 검색
    results = es.search(
        index="legal_docs",
        body=hybrid_search
    )

    return results
```

### 예상 성능

- **검색 속도**: < 100ms (Elasticsearch)
- **정확도**: 60-70% (현재 데이터 기준)
- **확장성**: 최대 100,000+ 청크 지원

### 개선 방향

**데이터 확장:**
- 추가 키워드 (15개): 육아휴직, 출산휴가, 직장내괴롭힘, 특수고용직 등
- 목표: 5,000-6,000건 → 20,000-25,000 청크

**검색 성능 향상:**
- Re-ranking 모델 추가
- 필터링 최적화 (날짜, 법원, 문서 타입)
- 쿼리 확장 (동의어, 관련어)


## 참고 자료

### API 문서

- [국가법령정보 Open API](http://www.law.go.kr/DRF/lawService.do?OC=)
- [KURE-v1 모델](https://huggingface.co/nlpai-lab/KURE-v1)

### 관련 문서

- `docs/legal-data-types.md` - 데이터 타입 상세 설명
- `docs/legal-data-collection.md` - 데이터 수집 가이드

### 스크립트

- `ai/preprocessing/legal/1_collect.py` - 데이터 수집
- `ai/preprocessing/legal/2_chunk.py` - 데이터 청킹
- `ai/preprocessing/legal/3_embed.py` - 임베딩 생성


## 문제 해결

### 데이터 수집 오류

**문제:** API 응답 없음
- `.env` 파일에 `LEGAL_API_USER_ID` 확인
- API 키 유효성 검증

**문제:** 노동위원회 JSON 파싱 오류
- 노동위원회 API는 HTML 반환 → 사용 불가 (코드에서 비활성화됨)

### 청킹 오류

**문제:** 파일을 찾을 수 없음
- `legal/1_collect.py`를 먼저 실행했는지 확인
- `ai/data/raw/api/` 디렉토리에 파일이 있는지 확인

### 임베딩 오류

**문제:** 메모리 부족
- `batch_size`를 16 → 8로 줄이기
- M3 Mac 기준 16GB 메모리에서 batch_size=16 권장

**문제:** CUDA 오류
- KURE-v1은 CPU에서도 작동 (약 25-30분 소요)
- GPU 사용 시 더 빠름 (약 5-10분)
