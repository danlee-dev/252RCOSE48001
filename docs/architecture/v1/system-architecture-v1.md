# DocScanner AI System Architecture (V1)

근로계약서 법적 위험 분석 시스템의 전체 아키텍처 문서.

## 목차

1. [시스템 개요](#시스템-개요)
2. [하이브리드 LLM 구성](#하이브리드-llm-구성)
3. [파이프라인 상세 흐름](#파이프라인-상세-흐름)
4. [하이브리드 검색 시스템](#하이브리드-검색-시스템)
5. [조항 분석 시스템](#조항-분석-시스템)
6. [성능 최적화](#성능-최적화)
7. [설정 및 환경 변수](#설정-및-환경-변수)

---

## 시스템 개요

DocScanner AI는 한국 근로계약서를 분석하여 법적 위험을 탐지하는 시스템이다.

### 핵심 아키텍처

```
[사용자]
    |
    v
[Frontend (Next.js)] --> [API (FastAPI)]
                              |
                              v
                    [Celery Worker (AI Pipeline)]
                              |
            +-----------------+-----------------+
            |                 |                 |
            v                 v                 v
      [Vector DB]       [Graph DB]        [LLM APIs]
    (Elasticsearch)      (Neo4j)      (OpenAI, Gemini)
```

### 핵심 설계 원칙

| 원칙 | 설명 |
|------|------|
| Hybrid AI | 규칙 기반(Symbolic) + LLM 기반(Neural) 결합 |
| Self-Correcting | CRAG를 통한 검색 품질 자동 보정 |
| Explainable | 모든 판단에 법적 근거 및 추론 과정 시각화 |
| Constitutional | 노동법 원칙에 따른 검증 레이어 적용 |

---

## 하이브리드 LLM 구성

### 모델별 역할 분담

```
+-------------------+-------------------------+------------------------+
|      역할          |         모델             |         용도            |
+-------------------+-------------------------+------------------------+
| Retrieval/Summary | gemini-2.5-flash-lite   | Vision OCR, 문서 요약    |
| HyDE Generation   | gpt-4o                  | 가상 문서 생성 (HyDE)     |
| Reasoning/Analysis| gpt-5-mini              | 법률 추론, 조항 분석      |
+-------------------+-------------------------+------------------------+
```

### 선택 근거

1. **Gemini (gemini-2.5-flash-lite)**
   - 빠른 응답 속도, 저렴한 비용
   - Vision 기능 지원 (PDF/이미지 OCR)
   - 간단한 텍스트 추출, 요약 작업

2. **GPT-4o (HyDE 전용)**
   - Temperature 파라미터 지원 (가변 창의성 필요)
   - 가상 법률 문서 생성에 적합
   - 검색 정확도 향상을 위한 가상 문서 품질 중요

3. **GPT-5-mini (Reasoning)**
   - 복잡한 법률 추론 능력
   - 구조화된 JSON 출력 안정성
   - 법령/판례 해석 정확성
   - 주의: temperature 파라미터 미지원 (자동 감지하여 생략)

### 환경 변수 설정

```bash
# .env
LLM_RETRIEVAL_MODEL=gemini-2.5-flash-lite  # Vision/요약
LLM_REASONING_MODEL=gpt-5-mini             # 법률 추론
LLM_HYDE_MODEL=gpt-4o                      # HyDE 생성
```

### Reasoning Model 자동 감지

```python
def _is_reasoning_model(self) -> bool:
    """reasoning 모델 여부 확인 (temperature 미지원)"""
    reasoning_keywords = ["o1", "o3", "gpt-5"]
    return any(kw in self.model.lower() for kw in reasoning_keywords)
```

적용 모듈: HyDE, RAPTOR, CRAG, ClauseAnalyzer

---

## 파이프라인 상세 흐름

### Pipeline V2 전체 흐름

```
[계약서 업로드]
       |
       v
+----------------------+
| 1. PII Masking       |  개인정보 비식별화 (정규식)
| (~100ms)             |
+----------------------+
       |
       v
+----------------------+
| 2. Text Chunking     |  의미 단위 분할
| (~50ms)              |
+----------------------+
       |
       v
+----------------------+     +-------------------+
| 3. HyDE              | --> | 4. CRAG           |
| (~56-71s, gpt-4o)    |     | Vector+Graph 검색  |
+----------------------+     +-------------------+
       |
       v
+----------------------+
| 5. Clause Analysis   |  LLM 기반 조항 분석 (병렬 처리)
| (~180-255s)          |  - 조항 분할
|                      |  - 개별 조항 분석 (ThreadPool)
|                      |  - 종합 분석 (순차)
+----------------------+
       |
       v
+----------------------+
| 6. RAPTOR            |  계층적 요약 트리
| (~30-60s)            |
+----------------------+
       |
       v
+----------------------+
| 7. Constitutional AI |  노동법 원칙 검증
| (~78s)               |
+----------------------+
       |
       v
+----------------------+
| 8. Judge             |  신뢰도 평가
| (~65-98s)            |
+----------------------+
       |
       v
[최종 분석 결과]
(~8분 총 소요)
```

### 단계별 상세 설명

#### 1. PII Masking (개인정보 비식별화)

**파일**: `backend/app/ai/pii_masking.py`

정규표현식 기반으로 개인정보를 탐지하고 가역적으로 마스킹.

```python
PATTERNS = {
    "주민등록번호": r"\d{6}-[1-4]\d{6}",
    "전화번호": r"01[0-9]-\d{3,4}-\d{4}",
    "이메일": r"[\w.-]+@[\w.-]+\.\w+",
    "계좌번호": r"\d{3}-\d{2,6}-\d{2,6}",
}
```

#### 2. HyDE (Hypothetical Document Embeddings)

**파일**: `backend/app/ai/hyde.py`

**모델**: gpt-4o (temperature 지원 필요)

사용자 쿼리로부터 가상의 이상적인 답변 문서를 생성하여 검색 정확도 향상.

```
[기존 방식]
쿼리: "최저임금 위반" -> 쿼리 임베딩 -> 검색

[HyDE 방식]
쿼리: "최저임금 위반"
  -> LLM이 가상 문서 생성:
     "근로기준법 제6조에 따르면 최저임금은 시간당 10,030원이며..."
  -> 가상 문서 임베딩 -> 검색 (더 정확한 결과)
```

#### 3. Clause Analysis (조항 분석)

**파일**: `backend/app/ai/clause_analyzer.py`

핵심 분석 단계. 계약서를 조항 단위로 분할하고 각 조항별 위반 여부 판단.

**처리 흐름**:
```
계약서 텍스트
       |
       v
[1. 조항 추출] --> LLM이 구조화된 JSON으로 분할
       |
       v
[2. 개별 조항 분석] --> 병렬 처리 (ThreadPoolExecutor, max 5 workers)
       |                 - 하이브리드 검색 (Vector + Graph DB)
       |                 - LLM 위반 분석
       v
[3. 종합 분석] --> 순차 처리 (모든 조항 정보 필요)
                   - 최저임금 계산 (총 임금 / 총 근로시간)
                   - 주 52시간 초과 검증
                   - 포괄임금제 적법성
```

#### 4. RAPTOR (계층적 요약)

**파일**: `backend/app/ai/raptor.py`

문서를 계층적 요약 트리로 구조화.

```
                [루트 요약]
               "주요 법적 위험 3건"
                     |
         +-----------+-----------+
         |                       |
   [임금 관련 요약]         [근로시간 요약]
         |                       |
    +----+----+            +----+----+
    |         |            |         |
 [청크1]   [청크2]       [청크3]   [청크4]
```

#### 5. Constitutional AI (노동법 원칙 검증)

**파일**: `backend/app/ai/constitutional_ai.py`

분석 결과가 노동법 기본 원칙을 위반하지 않는지 검증.

```python
class ConstitutionalPrinciple(Enum):
    HUMAN_DIGNITY = "근로조건은 인간의 존엄성을 보장해야 한다"
    WORKER_PROTECTION = "해석이 모호할 때는 근로자에게 유리하게 해석한다"
    MINIMUM_STANDARD = "근로기준법은 최저 기준이며, 미달 조건은 무효"
    EQUALITY = "동일 가치 노동에 대해서는 동일 임금 지급"
    SAFETY = "근로자의 안전과 건강을 위협하는 조건 불허"
    TRANSPARENCY = "근로조건은 명확하게 서면으로 명시"
```

#### 6. LLM-as-a-Judge (신뢰도 평가)

**파일**: `backend/app/ai/judge.py`

분석 결과의 품질을 다각도로 평가.

```python
class JudgmentCategory(Enum):
    ACCURACY = "정확성"       # 법률 해석의 정확성
    CONSISTENCY = "일관성"    # 근거와 결론의 일관성
    COMPLETENESS = "완전성"   # 분석의 완전성
    RELEVANCE = "관련성"      # 질문/계약서와의 관련성
    LEGAL_BASIS = "법적 근거" # 법적 근거의 타당성
```

---

## 하이브리드 검색 시스템

### Vector DB + Graph DB 통합

```
+------------------+        +------------------+
|   Vector DB      |        |   Graph DB       |
| (Elasticsearch)  |        |    (Neo4j)       |
+------------------+        +------------------+
| - 의미적 유사도    |        | - 구조적 관계      |
| - doc_type 필터   |        | - RiskPattern    |
| - BM25 + 벡터     |        | - Category       |
+------------------+        +------------------+
        |                          |
        +------------+-------------+
                     |
                     v
              [하이브리드 랭킹]
              - Vector score 70%
              - Graph centrality 30%
```

### 카테고리별 검색 전략

| 카테고리 | Vector DB (doc_type) | Graph DB 노드 |
|----------|---------------------|---------------|
| 법령 | law | Document (CATEGORIZED_AS Category) |
| 판례 | precedent | Document (source 패턴 매칭) |
| 해석례 | interpretation | Document (keywords 매칭) |
| 위험패턴 | - | RiskPattern (triggers 매칭) |

### Graph DB 스키마

```
(RiskPattern) --[RELATES_TO]--> (Document)
(RiskPattern) --[IS_A_TYPE_OF]--> (ClauseType)
(Document) --[CATEGORIZED_AS]--> (Category)

RiskPattern: name, explanation, riskLevel, triggers
Document: id, source, content, category, type
Category: name (근로시간, 임금, 휴일휴가 등)
```

### 위험 패턴 매칭

```python
# 조항 유형별 트리거 키워드
risk_triggers = {
    "휴게시간": ["휴게", "휴식", "점심"],
    "근로시간": ["근로시간", "연장", "야간", "휴일근로", "52시간"],
    "임금": ["임금", "급여", "최저임금", "시급", "월급", "포괄"],
    "수당": ["수당", "포괄하여", "포함하여", "제수당"],
    "위약금": ["위약금", "손해배상", "벌금", "벌칙", "배상하여야", "반환"],
}
```

---

## 조항 분석 시스템

### 분석 프롬프트 구조

#### 1. 조항 추출 프롬프트

계약서를 구조화된 JSON으로 분할:

```json
{
    "clauses": [
        {
            "clause_number": "1",
            "clause_type": "근로시간",
            "title": "근로시간",
            "original_text": "원문 텍스트",
            "extracted_values": {
                "start_time": "09:00",
                "end_time": "18:00",
                "daily_hours": 8
            }
        }
    ]
}
```

#### 2. 위반 분석 프롬프트

조항별 법률 컨텍스트 기반 분석:

```
[분석 대상 조항]
조항 번호, 유형, 원문, 추출된 값

[관련 법령 - 검색 결과]
Vector DB + Graph DB 검색 결과

[관련 판례 - 검색 결과]
판례 검색 결과

[위험 패턴 - Graph DB]
RiskPattern 노드 매칭 결과
```

#### 3. 출력 필드 설명

```json
{
    "violations": [
        {
            "violation_type": "위반 유형 (예: 최저임금_미달)",
            "severity": "CRITICAL/HIGH/MEDIUM/LOW/INFO",
            "description": "구체적 위반 내용",
            "legal_basis": "법률명과 조항 번호만 (예: 근로기준법 제50조)",
            "current_value": "현재 계약서 값",
            "legal_standard": "법적 기준 값",
            "suggestion": "구체적 수정 방법 (예: '1일 근로시간을 8시간으로 변경')"
        }
    ]
}
```

### 위반 심각도 기준

| 레벨 | 설명 | 예시 |
|------|------|------|
| CRITICAL | 즉시 시정 필요 | 위약금 예정, 강제노동 |
| HIGH | 심각한 위반 | 최저임금 미달, 주 52시간 초과 |
| MEDIUM | 주의 필요 | 휴게시간 부족, 연차 미명시 |
| LOW | 경미한 문제 | 계약서 교부 지연 |
| INFO | 정보 제공 | 개선 권고 사항 |

---

## 성능 최적화

### 병렬 처리 구현

**파일**: `backend/app/ai/clause_analyzer.py` (Lines 843-918)

```python
def analyze(self, contract_text: str) -> ClauseAnalysisResult:
    from concurrent.futures import ThreadPoolExecutor, as_completed

    # 1. 조항 추출 (순차)
    clauses = self._extract_clauses(contract_text)

    # 2. 개별 조항 분석 (병렬 처리)
    with ThreadPoolExecutor(max_workers=min(len(clauses_to_analyze), 5)) as executor:
        future_to_clause = {
            executor.submit(self._analyze_clause, clause): clause
            for clause in clauses_to_analyze
        }

        for future in as_completed(future_to_clause):
            violations, underpayment = future.result()
            result.violations.extend(violations)

    # 3. 종합 분석 (순차 - 모든 조항 정보 필요)
    holistic_violations, holistic_underpayment = self._holistic_analysis(
        clauses, contract_text
    )
```

### 병렬 처리 대상

| 단계 | 처리 방식 | 이유 |
|------|----------|------|
| 조항 추출 | 순차 | 전체 문서 한 번에 처리 |
| 개별 조항 분석 | 병렬 (max 5) | 각 조항 독립적으로 분석 가능 |
| 종합 분석 | 순차 | 모든 조항 정보 필요 (최저임금 계산 등) |

### 예상 성능 개선

**Before (순차 처리)**:
- 3개 조항 분석: 약 180초 (60초 x 3)

**After (병렬 처리)**:
- 3개 조항 분석: 약 60초 (가장 오래 걸리는 조항 시간)
- 개선율: 약 3배

### HyDE 모델 최적화

**Before**: gpt-5-mini (reasoning 모델, 71초)
- Temperature 미지원
- 가변적 창의성 불가

**After**: gpt-4o (56-60초)
- Temperature 지원 (0.5 사용)
- 다양한 관점의 가상 문서 생성 가능

---

## 설정 및 환경 변수

### 환경 변수 전체 목록

```bash
# .env

# === LLM API Keys ===
OPENAI_API_KEY=sk-proj-...
GEMINI_API_KEY=AIzaSy...

# === LLM Model Selection (하이브리드) ===
LLM_RETRIEVAL_MODEL=gemini-2.5-flash-lite  # Vision/요약
LLM_REASONING_MODEL=gpt-5-mini             # 법률 추론
LLM_HYDE_MODEL=gpt-4o                      # HyDE 생성

# === Neo4j Graph DB ===
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=DocScannerGraphPass!2025

# === Elasticsearch Vector DB ===
ES_URL=http://localhost:9200

# === PostgreSQL ===
POSTGRES_HOST=localhost
POSTGRES_PORT=5435
POSTGRES_DB=docscanner_db

# === Redis/Celery ===
REDIS_HOST=localhost
REDIS_PORT=6379
CELERY_BROKER_URL=redis://localhost:6379/0
```

### 파이프라인 설정 클래스

**파일**: `backend/app/ai/pipeline.py` (PipelineConfig)

```python
@dataclass
class PipelineConfig:
    # V2: LLM 기반 조항 분석 (권장)
    enable_llm_clause_analysis: bool = True

    # 공통
    enable_pii_masking: bool = True
    enable_raptor: bool = True
    enable_constitutional_ai: bool = True
    enable_judge: bool = True
    enable_reasoning_trace: bool = True
    enable_dspy: bool = False  # 기본 비활성화

    # Legacy (폴백용)
    enable_hyde: bool = True
    enable_crag: bool = True
    enable_stress_test: bool = False  # LLM 분석이 대체
    enable_redlining: bool = False    # LLM 분석이 대체

    # 모델 설정
    retrieval_model: str = settings.LLM_RETRIEVAL_MODEL
    reasoning_model: str = settings.LLM_REASONING_MODEL
    hyde_model: str = settings.LLM_HYDE_MODEL
    embedding_model: str = "nlpai-lab/KURE-v1"

    # 검색 설정
    search_top_k: int = 5
    crag_max_hops: int = 2
```

### 관련 파일 위치

| 파일 | 설명 |
|------|------|
| `/.env` | 환경 변수 |
| `/backend/app/core/config.py` | 설정 클래스 |
| `/backend/app/core/llm_client.py` | 하이브리드 LLM 클라이언트 |
| `/backend/app/ai/pipeline.py` | 파이프라인 설정 및 실행 |
| `/backend/app/ai/clause_analyzer.py` | 조항 분석기 (병렬 처리) |
| `/backend/app/ai/hyde.py` | HyDE 생성기 |
| `/backend/app/ai/crag.py` | CRAG 검색기 |

---

## 참고 사항

### 모델 변경 시 체크리스트

1. `.env` 파일 환경 변수 수정
2. Celery worker 재시작 필요 (`celery -A app.core.celery_app worker -l info`)
3. 모델별 API 키 확인 (OPENAI_API_KEY, GEMINI_API_KEY)
4. 토큰 제한 및 비용 고려
5. Reasoning 모델은 temperature 미지원 (자동 처리됨)

### 트러블슈팅

| 문제 | 원인 | 해결 |
|------|------|------|
| HyDE 느림 | Reasoning 모델 사용 | LLM_HYDE_MODEL=gpt-4o 설정 |
| 법적 근거/수정 제안 동일 | 프롬프트 모호 | 프롬프트에 필드별 명확한 지시 추가 |
| 조항 분석 느림 | 순차 처리 | ThreadPoolExecutor 병렬 처리 적용 |
| Graph DB 연결 실패 | 스키마 불일치 | 현재 스키마 기준으로 쿼리 수정 |

---

*Document Version: 1.0*
*Last Updated: 2024-12*
