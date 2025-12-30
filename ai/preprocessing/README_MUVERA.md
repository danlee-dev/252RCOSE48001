# MUVERA 기반 법률 문서 검색 시스템 구축 가이드

## 현재 진행 상황

- [x] 프로젝트 의존성 통합 (`requirements.txt`)
- [x] 법률 데이터 전처리 및 청크 생성
- [x] MUVERA FDE 임베딩 생성 (메모리 최적화)
- [x] Elasticsearch 인덱싱
- [x] 벡터 검색 시스템 구축

---

## 0. 필수 요구사항 (Prerequisite)

- **Git:** 최신 브랜치(`feature/muvera`) Pull 받기
- **Docker Desktop:** 실행 중일 것
- **Python:** 3.10 이상
- **메모리:** 최소 8GB RAM (16GB 권장)

---

## 1. 환경 설정 (통합 의존성 설치)

프로젝트 루트(`docscanner-ai`)에서 의존성 설치:

```bash
# 가상환경 활성화 상태에서 실행
pip install -r requirements.txt
```

### 주요 패키지
- `sentence-transformers`: KURE-v1 임베딩 모델
- `elasticsearch`: 벡터 검색 엔진
- `numpy`: 수치 연산
- `tqdm`: 진행률 표시

---

## 2. Elasticsearch 실행 (Docker Desktop)

### (1) Docker Desktop 실행
- Docker Desktop 앱을 실행합니다.

### (2) Elasticsearch 컨테이너 실행

**터미널에서 실행:**

```bash
docker run -d \
  --name elasticsearch \
  -p 9200:9200 \
  -e "discovery.type=single-node" \
  -e "xpack.security.enabled=false" \
  -e "ES_JAVA_OPTS=-Xms2g -Xmx2g" \
  docker.elastic.co/elasticsearch/elasticsearch:8.11.0
```

**또는 Docker Desktop에서:**
1. Images 탭에서 `elasticsearch:8.11.0` 검색 및 Pull
2. 이미지 실행 시 다음 설정:
   - Container name: `elasticsearch`
   - Ports: `9200:9200`
   - Environment variables:
     - `discovery.type=single-node`
     - `xpack.security.enabled=false`
     - `ES_JAVA_OPTS=-Xms2g -Xmx2g`

### (3) 연결 확인

1분 후 터미널에서 확인:

```bash
curl http://localhost:9200
```

**정상 응답 예시:**
```json
{
  "name" : "...",
  "cluster_name" : "docker-cluster",
  "version" : {
    "number" : "8.11.0"
  }
}
```

Docker Desktop의 Containers 탭에서 `elasticsearch` 컨테이너가 **녹색(Running)** 상태인지 확인.

---

## 3. MUVERA 임베딩 파이프라인 실행

데이터 임베딩부터 검색까지 **순서대로** 스크립트를 실행해야 함.

### 경로 이동

```bash
cd ai/preprocessing
```

### (1) MUVERA 임베딩 생성 (Embeddings)

법률 청크를 Multi-Vector Retrieval + FDE로 임베딩 생성.
**메모리 최적화:** 배치 처리 + 자동 메모리 정리

```bash
python 3_embed_muvera.py
```

**예상 소요 시간:** 청크 수에 따라 10-30분
**출력 파일:** `data/processed/embeddings/legal_chunks_with_muvera_embeddings_{날짜}.json`

**주요 설정 (메모리 절약):**
- 임베딩 배치: 8
- 청크 배치: 500개씩
- FDE repetitions: 3
- FDE projections: 16 파티션

### (2) Elasticsearch 인덱싱 (Indexing)

생성된 임베딩을 Elasticsearch에 인덱싱.

```bash
python 4_index.py
```

**예상 소요 시간:** 2-5분
**인덱스 이름:** `docscanner_chunks`

### (3) 검색 테스트 (Search Test)

대화형 검색 인터페이스 실행.

```bash
python test_es_search_muvera.py
```

**사용법:**
- 검색: 쿼리 입력 (예: `최저임금`)
- 필터: `@필터명` 추가 (예: `최저임금 @precedent`)
- 상위 결과: `#숫자` 추가 (예: `연차 #10`)
- 종료: `q` 또는 `exit`

**필터 옵션:**
- `@precedent`: 판례만
- `@interpretation`: 법령해석례만
- `@labor_ministry`: 고용노동부만

---

## 4. 결과 검증 (Verification)

### 테스트 쿼리

검색 인터페이스에서 아래 쿼리를 입력하여 **검색 품질** 확인:

```
최저임금
```

**성공 기준:**
- Top 5 결과가 반환됨
- 유사도 점수가 0.5 이상
- 최저임금 관련 법령/판례가 검색됨

### 추가 테스트 쿼리

```
연차휴가
퇴직금
부당해고
근로시간
임금 체불
```

---

## 5. 트러블슈팅 (Troubleshooting)

### (1) 메모리 부족 (Memory Error)

**증상:** `3_embed_muvera.py` 실행 시 메모리 에러 또는 시스템 느려짐

**해결 방법 1:** 배치 크기 감소

`3_embed_muvera.py` 파일의 `main()` 함수 수정:

```python
embedder = MuveraLegalEmbedder(
    model_name="nlpai-lab/KURE-v1",
    batch_size=4,           # 8 -> 4
    chunk_batch_size=250    # 500 -> 250
)
```

**해결 방법 2:** FDE 설정 간소화

추가로 메모리가 부족하면 `__init__` 메서드에서:

```python
num_repetitions=2,          # 3 -> 2
num_simhash_projections=3,  # 4 -> 3
```

### (2) Elasticsearch 연결 실패

**증상:** `Connection refused` 또는 `No connection`

**해결 방법:**

1. Docker Desktop에서 `elasticsearch` 컨테이너 상태 확인
2. 컨테이너가 중지되어 있으면 Start 버튼 클릭
3. 컨테이너가 Running이면 Restart 버튼 클릭
4. 1분 후 연결 확인:

```bash
curl http://localhost:9200
```

### (3) 청크 파일이 없음

**증상:** `legal_chunks_*.json 파일을 찾을 수 없습니다`

**해결 방법:**

청크 생성 스크립트 먼저 실행:

```bash
python 2_chunk.py
```

### (4) 임베딩 파일을 찾을 수 없음

**증상:** `4_index.py` 실행 시 파일을 찾을 수 없음

**해결 방법:**

3번 단계부터 다시 실행:

```bash
python 3_embed_muvera.py
```

### (5) Docker 포트 충돌

**증상:** `port is already allocated` 또는 `address already in use`

**해결 방법:**

1. Docker Desktop에서 기존 `elasticsearch` 컨테이너 삭제
2. 9200 포트를 사용 중인 다른 프로세스 종료
3. Elasticsearch 컨테이너 재실행

---

## 6. 시스템 아키텍처

```
법률 문서 (JSON)
    ↓
[2_chunk.py] 청크 생성
    ↓
legal_chunks_*.json
    ↓
[3_embed_muvera.py] MUVERA FDE 임베딩
    ↓
legal_chunks_with_muvera_embeddings_*.json
    ↓
[4_index.py] Elasticsearch 인덱싱
    ↓
docscanner_chunks (인덱스)
    ↓
[test_es_search_muvera.py] 벡터 검색
```

---

## 7. 핵심 기술 스택

- **MUVERA (Multi-Vector Retrieval):** 문서를 여러 문장으로 분할하여 임베딩
- **FDE (Fixed Dimensional Encoding):** Multi-vector를 single-vector로 압축
- **KURE-v1:** 한국어 법률 도메인 특화 임베딩 모델 (1024차원)
- **Elasticsearch:** KNN 벡터 검색 엔진
- **메모리 최적화:** 배치 처리 + 자동 garbage collection

---

## 8. 참고 자료

### FDE 설정 의미

```python
num_repetitions=3           # FDE 반복 횟수 (높을수록 정확도↑, 메모리↑)
num_simhash_projections=4   # 2^4 = 16 파티션 (높을수록 메모리↑)
final_projection_dimension=1024  # 최종 벡터 차원
```

### 벡터 검색 과정

1. 쿼리를 KURE-v1로 임베딩 (1024차원)
2. 문장 분할 및 multi-vector 생성
3. FDE로 single-vector 압축 (SUM aggregation)
4. Elasticsearch KNN 검색 (코사인 유사도)
5. Top-K 결과 반환

---

## 문의 및 이슈

- 담당자: 이성민 (2023320132)
- GitHub: https://github.com/danlee-dev
