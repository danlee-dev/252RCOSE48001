# DocScanner AI v1 개발 현황

이 문서는 DocScanner AI v1의 전체 기능, 적용된 기술, 구현 방식을 상세히 설명합니다.

---

## 목차

1. [시스템 개요](#시스템-개요)
2. [AI 분석 파이프라인](#ai-분석-파이프라인)
3. [핵심 기술 상세](#핵심-기술-상세)
4. [LangGraph 채팅 에이전트](#langgraph-채팅-에이전트)
5. [프론트엔드 분석 페이지](#프론트엔드-분석-페이지)
6. [데이터베이스 아키텍처](#데이터베이스-아키텍처)
7. [기술 스택 요약](#기술-스택-요약)

---

## 시스템 개요

DocScanner AI는 근로계약서를 AI로 분석하여 법적 위험 조항을 자동으로 탐지하고, 사용자에게 법적 근거와 함께 개선안을 제시하는 시스템입니다.

### v1 핵심 목표

1. **정확한 위반 탐지**: LLM 기반 조항 분석으로 정규식 한계 극복
2. **법적 근거 제시**: Vector DB + Graph DB 하이브리드 검색으로 관련 법령/판례 제시
3. **실시간 채팅 지원**: LangGraph 에이전트로 계약서 관련 질문에 실시간 응답
4. **신뢰할 수 있는 분석**: Constitutional AI, LLM-as-a-Judge로 분석 품질 보장

### 시스템 아키텍처

```
[Frontend (Next.js)]
       |
       v
[FastAPI Backend]
       |
       +---> [AI Pipeline] ---> [LLM APIs (OpenAI, Gemini)]
       |           |
       |           +---> [Elasticsearch (Vector DB)]
       |           |
       |           +---> [Neo4j (Graph DB)]
       |
       +---> [LangGraph Agent] ---> [Tavily (Web Search)]
       |
       +---> [PostgreSQL] ---> [Contract Storage]
       |
       +---> [Redis] ---> [Celery Task Queue]
```

---

## AI 분석 파이프라인

### 파이프라인 실행 흐름

계약서 분석은 12단계의 순차적 파이프라인으로 진행됩니다.

```
1. PII Masking (개인정보 마스킹)ㄱㅏㄴ도
       |
2. Document Parsing (문서 텍스트 추출)
       |
3. Text Chunking (텍스트 분할)
       |
4. HyDE (가상 문서 생성)
       |
5. CRAG (품질 인식 검색)
       |
6. LLM Clause Analysis (조항별 위반 분석)
       |
7. RAPTOR (계층적 요약 인덱싱)
       |
8. Legal Stress Test (정량적 법적 검증)
       |
9. Generative Redlining (수정안 생성)
       |
10. Constitutional AI (헌법적 원칙 검증)
       |
11. LLM-as-a-Judge (분석 품질 평가)
       |
12. Reasoning Trace (추론 과정 시각화)
```

### 모듈별 상세 설명

#### 1. PII Masking (개인정보 보호)

**파일**: `backend/app/ai/pii_masking.py`

**목적**: LLM에 전송하기 전 개인정보를 마스킹하여 프라이버시 보호

**지원하는 PII 유형** (10종):
- 주민등록번호 (RRN)
- 전화번호
- 이메일
- 은행 계좌번호
- 주소
- 한국인 이름
- 신용카드 번호
- 여권 번호
- 운전면허 번호
- 사업자등록번호

**마스킹 예시**:
```
원본: 홍길동 (010-1234-5678, 서울시 강남구 테헤란로 123)
마스킹: [이름_1] ([전화번호_1], 서울시 [주소_1])
```

**구현 방식**:
```python
# 형식 보존 마스킹 (Format-Preserving Masking)
RRN: "123456-1234567" -> "123456-****567"  # 앞 6자리 + 마지막 1자리 유지
Phone: "010-1234-5678" -> "010-****-5678"  # 가운데 4자리 마스킹
Email: "user@example.com" -> "us****@example.com"
```

---

#### 2. Document Parsing (문서 파싱)

**파일**: `backend/app/ai/document_parser.py`, `backend/app/ai/vision_parser.py`

**목적**: PDF, HWP, DOCX, 이미지 등 다양한 형식의 계약서에서 텍스트 추출

**지원 형식**:
| 형식 | 파서 | 라이브러리 |
|------|------|------------|
| PDF | PDFParser | pdfplumber |
| HWP/HWPX | HWPParser | olefile |
| DOCX | DOCXParser | python-docx |
| TXT | TextParser | chardet (인코딩 자동 감지) |
| 이미지 | VisionParser | Gemini Vision API |

**Fallback 전략**:
```
Native Parsing 시도
       |
       v
추출 텍스트 < 100자?
       |
       +-- Yes --> Vision API (OCR) 사용
       |
       +-- No --> 추출 완료
```

**Vision Parser 특징**:
- Gemini 2.5 Flash-Lite 사용 (빠르고 저렴)
- 테이블, 체크박스, 서명 영역 자동 인식
- 마크다운 형식으로 구조화된 출력

---

#### 3. LLM Clause Analyzer (조항 분석기)

**파일**: `backend/app/ai/clause_analyzer.py`

**목적**: 계약서의 각 조항을 LLM으로 분석하여 법적 위반 사항 탐지

**2단계 분석 아키텍처**:

```
=== Phase 1: 병렬 조항 분석 (ThreadPoolExecutor) ===

[조항 추출] --> [조항 1] [조항 2] [조항 3] [조항 4] ...
                   |         |         |         |
                   v         v         v         v
              [Thread 1] [Thread 2] [Thread 3] [Thread 4]
                   |         |         |         |
                   +----+----+----+----+
                        |
                        v
               [개별 위반 사항 수집]

=== Phase 2: 순차적 종합 분석 (Cross-Clause) ===

[전체 조항 정보 통합]
       |
       v
[최저임금 계산: 기본급 / 총 근로시간]
       |
       v
[주 52시간 초과 검증: 모든 근로시간 합산]
       |
       v
[포괄임금제 적법성 검토]
       |
       v
[종합 위반 사항 + 체불액 산출]
```

**Phase 1: 병렬 조항 분석**

```python
# clause_analyzer.py:899-913
with ThreadPoolExecutor(max_workers=min(len(clauses_to_analyze), 5)) as executor:
    future_to_clause = {
        executor.submit(self._analyze_clause, clause): clause
        for clause in clauses_to_analyze
    }
    for future in as_completed(future_to_clause):
        violations, underpayment = future.result()
        result.violations.extend(violations)
```

- 최대 5개 쓰레드로 병렬 처리
- 각 조항은 독립적으로 Vector DB + Graph DB 검색 수행
- 개별 위반 사항 탐지 (휴게시간 미부여, 위약금 조항 등)

**Phase 2: 순차적 종합 분석 (Holistic Analysis)**

병렬 분석이 완료된 후, 전체 조항 정보를 종합하여 조항 간 연관 분석이 필요한 위반 사항을 탐지합니다.

```python
# clause_analyzer.py:917-921
holistic_violations, holistic_underpayment = self._holistic_analysis(
    clauses, contract_text
)
```

**종합 분석 항목**:

| 분석 항목 | 필요한 조항 정보 | 계산 방식 |
|----------|----------------|----------|
| 최저임금 위반 | 임금 + 근로시간 | (기본급 + 고정수당) / (소정근로시간 + 연장근로시간) |
| 주 52시간 초과 | 근로시간 + 연장근로 | 소정근로 40시간 + 연장근로 12시간 한도 |
| 포괄임금제 적법성 | 임금 + 수당 + 근로시간 | 연장/야간/휴일수당이 실제 계산액과 일치하는지 |
| 연장근로 가산수당 | 임금 + 근로시간 | 연장근로시간 x 통상시급 x 1.5 |

**왜 2단계로 나누는가?**

| 분석 유형 | 병렬 처리 가능 | 이유 |
|----------|--------------|------|
| 휴게시간 미부여 | O | 해당 조항만 보면 판단 가능 |
| 위약금 조항 | O | 해당 조항만 보면 판단 가능 |
| 4대보험 미가입 | O | 해당 조항만 보면 판단 가능 |
| **최저임금 위반** | **X** | 임금 조항 + 근로시간 조항 모두 필요 |
| **52시간 초과** | **X** | 모든 근로시간 정보 합산 필요 |
| **포괄임금제** | **X** | 임금, 수당, 근로시간 정보 모두 필요 |

**분석 과정 (상세)**:
```
1. 조항 추출 (LLM 기반 세그멘테이션)
       |
2. [병렬] 조항별 하이브리드 검색 (Vector + Graph)
       |
3. [병렬] 법적 맥락과 함께 LLM 분석
       |
4. [순차] 전체 조항 종합 분석 (Cross-clause)
       |
5. 위반 사항 + 법적 근거 + 개선안 + 체불액 생성
```

**출력 데이터 구조**:
```python
@dataclass
class ClauseViolation:
    violation_type: str      # 위반 유형 (예: "최저임금 미달")
    severity: str            # HIGH, MEDIUM, LOW
    description: str         # 상세 설명
    legal_basis: str         # 법적 근거 (예: "근로기준법 제50조")
    suggestion: str          # 개선안
    sources: list[str]       # CRAG 검색 출처
    original_text: str       # 원본 조항 텍스트
```

**하이브리드 검색**:
```python
# Vector DB (Elasticsearch) - 의미적 유사도 검색
law_context = search_vector_db(query="근로시간 8시간 초과", doc_type="law")

# Graph DB (Neo4j) - 관계 기반 확장
risk_patterns = search_graph_db(clause_type="근로시간", keywords=["연장", "초과"])
```

---

#### 4. Legal Stress Test (법적 스트레스 테스트)

**파일**: `backend/app/ai/legal_stress_test.py`

**목적**: Neuro-Symbolic AI 방식으로 정량적 법적 검증 수행

**Neuro-Symbolic 접근법**:
```
Neuro (LLM): 계약서에서 급여, 근로시간 등 정보 추출
       |
       v
Symbolic (Python): 정확한 수치 계산 및 법적 기준 비교
       |
       v
결과: 위반 여부 + 미지급 금액 산출
```

**검증 항목** (11종):
1. 최저임금 위반
2. 연장근로수당 미지급
3. 야간근로수당 미지급
4. 휴일근로수당 미지급
5. 주휴수당 미지급
6. 연차휴가 미부여
7. 퇴직금 미지급
8. 4대보험 미가입
9. 52시간 초과
10. 휴게시간 미부여
11. 위약금/손해배상 조항

**2025년 법적 기준** (실제 코드):

```python
# legal_stress_test.py
MINIMUM_WAGE_2025 = 10_030              # 시급
MONTHLY_MINIMUM_WAGE_2025 = 2_096_270   # 월 209시간 기준
LEGAL_DAILY_HOURS = 8
LEGAL_WEEKLY_HOURS = 40
LEGAL_EXTENDED_LIMIT = 12               # 연장근로 한도 (주)
LEGAL_TOTAL_WEEKLY_LIMIT = 52           # 주 52시간
OVERTIME_RATE = Decimal("1.5")          # 연장근로 50% 가산
NIGHT_RATE = Decimal("1.5")             # 야간근로 50% 가산
HOLIDAY_RATE = Decimal("1.5")           # 휴일근로 50% 가산
```

**계산 예시**:

```python
# LLM이 추출한 정보
base_salary = 3_000_000  # 월 급여
daily_hours = 9          # 일 근로시간
monthly_days = 22        # 월 근로일수

# Python 계산 (Neuro-Symbolic)
total_hours = daily_hours * monthly_days  # 198시간
hourly_wage = base_salary / total_hours   # 15,151원/시간

# 2025년 최저임금: 10,030원
is_violation = hourly_wage < 10_030  # False (위반 아님)

# 연장근로 (8시간 초과분)
overtime_hours = (daily_hours - 8) * monthly_days  # 22시간
overtime_pay_required = overtime_hours * hourly_wage * 1.5  # 499,983원
```

**왜 Neuro-Symbolic인가?**

| 접근법 | 장점 | 단점 |
|--------|------|------|
| Pure LLM | 자연어 이해력 | 수치 계산 오류 가능 |
| Pure Rule-based | 정확한 계산 | 자연어 파싱 어려움 |
| **Neuro-Symbolic** | 둘의 장점 결합 | 구현 복잡도 증가 |

---

## 핵심 기술 상세

### HyDE (Hypothetical Document Embeddings)

**파일**: `backend/app/ai/hyde.py`

**문제 상황**:
```
사용자 쿼리: "야근 수당 안 줘도 되나요?"
       |
       v
일반 검색: "야근", "수당" 키워드 매칭
       |
       v
결과: 관련 없는 문서도 다수 포함
```

**HyDE 해결책**:
```
사용자 쿼리: "야근 수당 안 줘도 되나요?"
       |
       v
LLM이 가상 문서 생성:
"근로기준법 제56조에 따르면 연장근로에 대해서는 통상임금의
50% 이상을 가산하여 지급해야 합니다. 야간근로(오후 10시~
오전 6시)의 경우에도 50% 가산이 필요합니다..."
       |
       v
가상 문서로 임베딩 검색
       |
       v
결과: 실제 법령 조항과 높은 유사도
```

**구현 전략** (4가지):

| 전략 | 설명 | 사용 상황 |
|------|------|----------|
| SINGLE | 단일 가상 문서 생성 | 단순 키워드 질문 |
| ENSEMBLE | 3-5개 문서를 다른 temperature로 생성 후 평균 | 일반적 법률 질문 |
| ADAPTIVE | 쿼리 복잡도에 따라 동적 전략 선택 | 복합적 법률 해석 |
| MULTI_PERSPECTIVE | 3가지 관점에서 생성 | 전문가 수준 분석 |

**Multi-Perspective 관점**:
- 법률 전문가 관점
- 판례 분석가 관점
- 행정해석 담당자 관점

**쿼리 복잡도 분류** (실제 코드):

```python
class QueryComplexity(Enum):
    SIMPLE = "simple"           # 단순 키워드 질문 (예: "최저임금이 얼마야?")
    MODERATE = "moderate"       # 일반적 법률 질문 (예: "연장근로 수당 계산법")
    COMPLEX = "complex"         # 복합적 법률 해석 (예: "52시간제 예외 사업장 조건")
    EXPERT = "expert"           # 전문가 수준 분석 (예: "탄력근무제 주휴수당 산정")
```

**적응형 전략 선택**:

```python
def _determine_strategy(complexity: QueryComplexity) -> HyDEStrategy:
    if complexity == SIMPLE:
        return SINGLE
    elif complexity == MODERATE:
        return ENSEMBLE
    elif complexity == COMPLEX:
        return ADAPTIVE
    else:  # EXPERT
        return MULTI_PERSPECTIVE
```

---

### CRAG (Corrective Retrieval-Augmented Generation)

**파일**: `backend/app/ai/crag.py`

**일반 RAG의 한계**:
```
쿼리 --> 검색 --> 상위 K개 반환 --> LLM 응답
                      |
                      v
           품질 검증 없이 사용
           (관련 없는 문서도 포함될 수 있음)
```

**CRAG 개선**:
```
쿼리 --> 검색 --> 품질 평가 --> 교정 전략 적용 --> LLM 응답
                      |              |
                      v              v
               8단계 품질 등급    7가지 교정 전략
```

**품질 평가 등급** (8단계):
| 등급 | 설명 | 교정 필요 |
|------|------|----------|
| EXCELLENT | 완벽히 관련됨 | No |
| GOOD | 대부분 관련됨 | No |
| CORRECT | 정확하지만 부분적 | Optional |
| PARTIAL | 일부만 관련됨 | Yes |
| AMBIGUOUS | 모호함 | Yes |
| WEAK | 약하게 관련됨 | Yes |
| INCORRECT | 관련 없음 | Yes |
| HARMFUL | 오해 소지 있음 | Yes (Critical) |

**교정 전략** (7가지):
1. **NONE**: 품질 충분, 그대로 사용
2. **REFINE**: 검색 결과에서 핵심만 추출
3. **AUGMENT**: Graph DB로 관련 문서 확장
4. **REWRITE**: 쿼리 재작성 후 재검색
5. **DECOMPOSE**: 복잡한 쿼리를 하위 쿼리로 분해
6. **MULTI_HOP**: 다단계 추론 검색
7. **FALLBACK**: 웹 검색 또는 기본 응답

**그래프 확장 예시**:
```cypher
// Neo4j 쿼리: 관련 법률 조항 확장
MATCH (d:Document)-[:CITES]->(law:LawArticle)
WHERE d.content CONTAINS "연장근로"
MATCH (law)-[:RELATED_TO]->(related:LawArticle)
RETURN related.content
```

---

### RAPTOR (Recursive Abstractive Processing for Tree-Organized Retrieval)

**파일**: `backend/app/ai/raptor.py`

**목적**: 계약서 내용을 계층적 트리 구조로 요약하여 다중 레벨 검색 지원

**트리 구조**:
```
Level 2:    [전체 계약 요약]
                  |
Level 1:   [임금 관련]  [근로시간]  [휴가/휴일]
                |           |           |
Level 0:   [조항1] [조항2] [조항3] [조항4] [조항5] ...
           (원본 텍스트 청크)
```

**구축 과정**:
```
1. 텍스트 청킹 (Level 0 노드 생성)
       |
2. 유사한 청크 클러스터링
       |
3. 클러스터별 LLM 요약 (Level 1 생성)
       |
4. 반복하여 상위 레벨 생성
       |
5. 최종 루트 노드 (전체 요약)
```

**클러스터링 방법**:
```python
# 콘텐츠 다양성에 따른 적응형 클러스터링
if categories_diverse:
    method = SEMANTIC         # 의미 기반 클러스터링
elif node_count > 20:
    method = DBSCAN          # 밀도 기반 (노이즈 제거)
else:
    method = AGGLOMERATIVE   # 계층적 클러스터링
```

**검색 시 활용**:
```
쿼리: "이 계약서에서 가장 위험한 조항은?"
       |
       v
Level 2에서 시작 --> 전체 요약 확인
       |
       v
Level 1로 이동 --> 관련 카테고리 탐색
       |
       v
Level 0 --> 구체적인 위험 조항 반환
```

---

### Constitutional AI (헌법적 AI)

**파일**: `backend/app/ai/constitutional_ai.py`

**목적**: AI 분석이 근로자 보호 원칙을 준수하는지 검증하고 자동 수정

**6가지 노동법 헌법 원칙** (실제 코드):

```python
class ConstitutionalPrinciple(Enum):
    HUMAN_DIGNITY = "근로조건은 인간의 존엄성을 보장해야 한다"
    WORKER_PROTECTION = "해석이 모호할 때는 근로자에게 유리하게 해석한다"
    MINIMUM_STANDARD = "근로기준법은 최저 기준이며, 이에 미달하는 조건은 무효다"
    EQUALITY = "동일 가치 노동에 대해서는 동일 임금을 지급해야 한다"
    SAFETY = "근로자의 안전과 건강을 위협하는 조건은 허용되지 않는다"
    TRANSPARENCY = "근로조건은 명확하게 서면으로 명시되어야 한다"
```

| 원칙 | 설명 | 관련 법령 |
|------|------|----------|
| HUMAN_DIGNITY | 인간 존엄성 보장 | 헌법 제32조, 근로기준법 제1조 |
| WORKER_PROTECTION | 근로자 유리 해석 | 대법원 판례 |
| MINIMUM_STANDARD | 최저기준 강행규정 | 근로기준법 제15조 |
| EQUALITY | 동일노동 동일임금 | 근로기준법 제6조 |
| SAFETY | 안전보건 보장 | 산업안전보건법 |
| TRANSPARENCY | 서면 명시 의무 | 근로기준법 제17조 |

**2단계 프로세스**:
```
Stage 1: Critique (비평)
       |
       v
AI 분석 결과가 6가지 원칙을 위반하는지 검토
       |
       +-- 위반 발견 시 --> Stage 2: Revise (수정)
       |
       +-- 위반 없음 --> 그대로 반환
```

**예시**:
```
원본 분석: "이 위약금 조항은 계약 위반 시 손해배상을 규정하고 있습니다."

Critique: "제3조 위반 - 근로기준법 제20조에 따라 위약 예정 금지 원칙이
          적용됨을 언급하지 않음"

수정된 분석: "이 위약금 조항은 근로기준법 제20조(위약 예정의 금지)에
            위반될 소지가 있습니다. 사용자는 근로계약 불이행에 대한
            위약금이나 손해배상액을 예정하는 계약을 체결할 수 없습니다."
```

---

### LLM-as-a-Judge (LLM 평가자)

**파일**: `backend/app/ai/judge.py`

**목적**: AI 분석의 신뢰도를 5가지 차원에서 평가

**평가 차원** (실제 코드):

```python
class JudgmentCategory(Enum):
    ACCURACY = "정확성"       # 법적 사실의 정확성
    CONSISTENCY = "일관성"    # 분석 내 논리적 일관성
    COMPLETENESS = "완전성"   # 모든 위험 요소 포착 여부
    RELEVANCE = "관련성"      # 계약서 맥락과의 관련성
    LEGAL_BASIS = "법적 근거" # 법령/판례 인용의 적절성
```

| 차원 | 가중치 | 설명 |
|------|--------|------|
| ACCURACY (정확성) | 25% | 법적 사실의 정확성 |
| LEGAL_BASIS (법적 근거) | 25% | 법령/판례 인용의 적절성 |
| CONSISTENCY (일관성) | 20% | 분석 내 논리적 일관성 |
| COMPLETENESS (완전성) | 15% | 모든 위험 요소 포착 여부 |
| RELEVANCE (관련성) | 15% | 계약서 맥락과의 관련성 |

**신뢰도 배지**:
```
점수 >= 0.8: HIGH (높음) - 녹색 배지
점수 >= 0.6: MEDIUM (보통) - 노란색 배지
점수 < 0.6: LOW (낮음) - 빨간색 배지
```

**Fact-Check 페널티**:
```python
# 핵심 사실 오류 발견 시 점수 차감
if fact_check_finds_critical_error:
    overall_score *= 0.7  # 30% 페널티
```

---

### Reasoning Trace (추론 과정 추적)

**파일**: `backend/app/ai/reasoning_trace.py`

**목적**: AI가 어떻게 결론에 도달했는지 시각화하여 설명 가능한 AI 구현

**Knowledge Graph 구조**:
```
노드 유형 (12종):
- CONTRACT_CLAUSE: 계약서 조항
- LEGAL_REFERENCE: 법령 조항
- PRECEDENT: 판례
- RISK_PATTERN: 위험 패턴
- REASONING_STEP: 추론 단계
- CONCLUSION: 최종 결론
- EVIDENCE: 증거
...

엣지 유형 (11종):
- CITES: 인용
- LEADS_TO: 결론 도출
- SUPPORTS: 지지
- CONTRADICTS: 반박
- APPLIES: 적용
- DERIVES_FROM: 유래
...
```

**추론 경로 예시**:
```
[계약서 조항: "시급 9,000원"]
       |
       | CITES
       v
[법령: 근로기준법 제6조, 최저임금법 제6조]
       |
       | LEADS_TO
       v
[위험 패턴: 최저임금 미달]
       |
       | SUPPORTS
       v
[결론: 위반 판정 - 시급이 2025년 최저임금(10,030원) 미만]
```

**시각화 출력**:
- Mermaid 다이어그램 (마크다운 호환)
- Cytoscape JSON (프론트엔드 인터랙티브 시각화)
- Neo4j 저장 (영구 보관)

---

### Generative Redlining (수정안 자동 생성)

**파일**: `backend/app/ai/redlining.py`

**목적**: 위험 조항에 대한 법적으로 안전한 대안 텍스트 자동 생성

**변경 유형**:
- DELETE: 삭제 권장
- INSERT: 추가 권장
- MODIFY: 수정 권장
- KEEP: 유지

**출력 형식**:
```diff
- 제5조 (위약금) 근로자가 계약 기간 중 퇴사할 경우
- 교육비 전액을 배상한다.
+ 제5조 (교육비) 사용자가 근로자의 직무 수행을 위해
+ 교육을 실시한 경우, 교육 비용의 상환에 관한 사항은
+ 근로기준법 제20조를 준수하여 별도 합의한다.
```

---

## LangGraph 채팅 에이전트

### 아키텍처

**파일**: `backend/app/ai/langgraph_agent.py`

```
사용자 질문
     |
     v
[Query Analyzer 노드] --> 도구 사용 여부 판단
     |
     +-- 도구 필요 --> [Tools 노드] --> 검색 수행
     |                      |
     |                      v
     +-- 도구 불필요 -----> [Respond 노드] --> 응답 생성
                                |
                                v
                           SSE 스트리밍
```

### 도구 (Tools)

#### 1. Vector DB 검색 (Elasticsearch)
```python
@tool
async def search_vector_db(query: str, doc_type: str = None, limit: int = 5):
    """법령, 판례, 행정해석 검색"""
    # Multi-field 검색 (text 필드 2배 가중치)
    # doc_type: law, precedent, interpretation, manual
```

**사용 트리거 키워드**: "법", "조항", "위반", "기준", "판례", "해석"

#### 2. Graph DB 검색 (Neo4j)
```python
@tool
async def search_graph_db(clause_type: str, keywords: list[str]):
    """위험 패턴 및 관련 법령 그래프 탐색"""
    # RiskPattern 노드 검색
    # Category 기반 문서 확장
```

**사용 트리거 키워드**: "위험", "패턴", "문제"

#### 3. 웹 검색 (Tavily API)
```python
@tool
async def web_search(query: str, max_results: int = 3):
    """정부 기관 웹사이트 검색"""
    # 도메인 필터링: moel.go.kr, law.go.kr, minwon.go.kr
```

**사용 트리거 키워드**: "신고", "방법", "대응", "상담", "기관", "어디", "어떻게"

### 컨텍스트 인식 검색 쿼리 생성

**문제 상황**:
```
대화 맥락: 계약서에서 "최저임금 미달" 위반 발견
사용자: "신고 방법 알려줘"
단순 검색: "신고 방법" --> 일반적인 민원 신고 정보
```

**해결책**:
```python
async def generate_search_query(
    user_message: str,
    chat_history: list,      # 최근 4개 메시지
    analysis_summary: str    # 계약서 분석 결과 (위반 사항)
) -> str:
    """대화 맥락과 분석 결과를 반영한 검색 쿼리 생성"""

    prompt = f"""
    [계약서 분석 결과 - 가장 중요]
    {analysis_summary}

    [이전 대화]
    {chat_history}

    [현재 질문]
    {user_message}

    검색 쿼리 생성 (30자 이내):
    """

    # 결과: "최저임금 미달 신고 고용노동부"
```

### SSE 스트리밍

**이벤트 유형**:
| 이벤트 | 설명 | 데이터 |
|--------|------|--------|
| step | 처리 단계 | `{"step": "analyzing", "message": "질문 분석 중..."}` |
| tool | 도구 실행 | `{"tool": "search_vector_db", "status": "searching"}` |
| token | 응답 토큰 | `{"content": "근로기준법"}` |
| done | 완료 | `{"full_response": "..."}` |
| error | 오류 | `{"message": "오류 내용"}` |

---

## 프론트엔드 분석 페이지

### 2컬럼 레이아웃

**파일**: `frontend/src/app/analysis/[id]/page.tsx`

```
+----------------------------------+----------------------------------+
|           왼쪽 패널 (50%)         |          오른쪽 패널 (50%)        |
|                                  |                                  |
|   [원본 문서 뷰어]                |   [탭: 개요 | 위험조항 | 전체텍스트] |
|   - 줌 인/아웃 (50%~200%)        |   - 위험 조항 카드                |
|   - 위험 조항 하이라이팅          |   - 클릭 시 왼쪽에서 해당 부분    |
|   - 텍스트 선택 -> AI 질문       |     하이라이팅                    |
|                                  |                                  |
+----------------------------------+----------------------------------+
                                   |
                              [채팅 패널]
                              (오른쪽 오버레이)
```

### 위험 조항 하이라이팅

**3단계 매칭 전략**:
```
1. 정확 매칭 (Exact Match)
   - 원본 텍스트 그대로 검색

2. 정규화 매칭 (Normalized Match)
   - 공백 차이 무시하고 검색

3. 부분 매칭 (Partial Match)
   - 첫 50자로 시작점 찾기
```

**심각도별 스타일링**:
```css
HIGH: border-l-4 border-red-400 bg-red-50
MEDIUM: border-l-4 border-amber-400 bg-amber-50
LOW: border-l-4 border-blue-400 bg-blue-50
```

### AI 채팅 패널

**기능**:
- SSE 스트리밍으로 실시간 응답
- 마크다운 렌더링 (react-markdown)
- 도구 실행 상태 표시
- 대화 히스토리 유지
- 생성 중단 기능

**텍스트 선택 -> AI 질문**:
```
1. 계약서 텍스트 드래그 선택
2. 툴팁 표시: "AI에게 질문하기"
3. 클릭 시 채팅 패널 열림
4. 자동 질문 생성: "'{선택 텍스트}' - 이 조항은 무슨 의미이고, 위험한가요?"
```

---

## 데이터베이스 아키텍처

### PostgreSQL (관계형 데이터)

**주요 테이블**:
- `users`: 사용자 정보
- `contracts`: 계약서 메타데이터 + 분석 결과 (JSONB)
- `alembic_version`: 마이그레이션 버전

**Contract 테이블 스키마**:
```sql
CREATE TABLE contracts (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id),
    filename VARCHAR(255),
    file_url VARCHAR(500),
    extracted_text TEXT,
    analysis_result JSONB,  -- AI 분석 결과 저장
    status VARCHAR(50),     -- pending, analyzing, completed, failed
    created_at TIMESTAMP,
    updated_at TIMESTAMP
);
```

### Elasticsearch (Vector DB)

**인덱스**: `docscanner_chunks`

#### MUVERA 임베딩 (Multi-Vector Retrieval via Fixed Dimensional Encodings)

**출처**: Google Research (2024)

- 논문: [arXiv:2405.19504](https://arxiv.org/abs/2405.19504)
- 블로그: [Google Research Blog](https://research.google/blog/muvera-making-multi-vector-retrieval-as-fast-as-single-vector-search/)
- 발표: NeurIPS 2024

**기존 임베딩 vs MUVERA**:

| 구분 | 기존 단일 벡터 임베딩 | MUVERA (다중 벡터 → FDE) |
|------|---------------------|-------------------------|
| 방식 | 전체 텍스트 → 1개 벡터 | 문장별 임베딩 → FDE 압축 |
| 문제점 | 긴 문서에서 세부 의미 손실 | - |
| 장점 | 빠른 검색 (단순 내적) | 세부 의미 보존 + 빠른 검색 |
| 검색 | MIPS (내적) | MIPS (FDE 내적 ≈ Chamfer 유사도) |

**예시**:

```
[기존 방식]
"근로시간은 1일 8시간, 1주 40시간을 초과할 수 없다.
 연장근로는 당사자 합의로 1주 12시간 한도."
        ↓
   [단일 벡터] ← 두 문장의 의미가 섞임

[MUVERA 방식]
"근로시간은 1일 8시간, 1주 40시간을 초과할 수 없다."  → [벡터1]
"연장근로는 당사자 합의로 1주 12시간 한도."           → [벡터2]
        ↓
   [FDE 압축] ← 각 문장의 의미가 파티션별로 보존
        ↓
   [단일 벡터 (1024차원)]
```

**해결하려는 문제**:

ColBERT 같은 다중 벡터 모델은 토큰당 하나의 임베딩을 생성하여 검색 정확도가 높지만, Chamfer 유사도 계산 비용이 크고 기존 MIPS(Maximum Inner Product Search) 알고리즘을 직접 적용할 수 없음.

**MUVERA 해결책**:

다중 벡터 집합을 FDE(Fixed Dimensional Encoding)로 단일 벡터로 압축. FDE의 내적이 원래 Chamfer 유사도를 근사하도록 설계.

**FDE 알고리즘 상세**:

```
입력: 문장 임베딩 집합 {v1, v2, ..., vn} (각 1024차원)

1. SimHash 파티셔닝
   - 랜덤 초평면(hyperplane)으로 벡터 공간을 2^k개 파티션으로 분할
   - 각 벡터가 어느 파티션에 속하는지 Gray Code로 인덱싱
   - 본 프로젝트: k=3 → 8개 파티션

2. 파티션별 집계 (Aggregation)
   - 문서: 각 파티션에 속한 벡터들의 평균 (AVERAGE)
   - 쿼리: 각 파티션에 속한 벡터들의 합 (SUM)
   - 비어있는 파티션: 가장 가까운 벡터로 채움 (fill_empty_partitions)

3. 최종 FDE 생성
   - 모든 파티션의 집계 결과를 연결 (concatenate)
   - Count Sketch로 최종 차원 압축 (8192 → 1024)

출력: 단일 FDE 벡터 (1024차원)
```

**왜 문서는 AVERAGE, 쿼리는 SUM인가?**

Chamfer 유사도의 비대칭성을 반영하기 위함. 문서는 정규화(평균)하여 길이에 무관하게 만들고, 쿼리는 합산하여 매칭 강도를 높임.

```
[청크 (최대 1,000자)]
          |
          v
[문장 단위 분할 (SentenceSplitter)]
          |
          v
[문장별 임베딩 (KURE-v1, 다중 벡터)]
          |
          v
    [FDE 압축]
          |
          +-- 임베딩 공간을 초평면으로 분할
          +-- 각 영역의 벡터들을 평균화
          |
          v
    [단일 벡터 (1024차원)]
          |
          v
    [기존 MIPS 알고리즘으로 고속 검색]
```

**본 프로젝트 MUVERA 처리 과정**:

1. 기존 청크 로드 (Legal API 14,549개 + PDF 674개)
2. `SentenceSplitter`로 청크를 문장 단위로 재분할 (마침표/느낌표/물음표 기준, 최소 10자)
3. 문서 타입 태그 추가 (예: `[판례]`, `[법령해석례]`)
4. 각 문장을 KURE-v1로 개별 임베딩 (multi-vector)
5. FDE로 압축하여 단일 1024차원 벡터 생성
6. Elasticsearch에 인덱싱

**성능 향상** (PLAID 대비):

| 지표 | 개선 |
|------|------|
| 후보 검색량 | 5~20배 감소 |
| 지연시간 | 90% 감소 |
| 재현율(Recall) | 10% 향상 |

**본 프로젝트 적용**:

- 임베딩 모델: `nlpai-lab/KURE-v1` (한국어 법률 특화)
- FDE 설정: repetitions=1, simhash_projections=3, partitions=8
- 최종 차원: 1024

---

#### 구축된 데이터 현황

**총 15,223개 청크** (2025년 11월 25일 기준)

| 데이터 유형 | 문서 수 | 청크 수 | 문서당 평균 청크 |
|------------|--------|--------|----------------|
| 판례 (precedent) | 969 | 10,576 | 10.9 |
| 고용노동부 해설 (labor_ministry) | 1,827 | 3,384 | 1.9 |
| 법령해석례 (interpretation) | 135 | 589 | 4.4 |
| **Legal API 소계** | **2,931** | **14,549** | - |
| 업무 매뉴얼 (manual) | - | 296 | - |
| 표준취업규칙 (employment_rules) | - | 367 | - |
| 가이드 (guide) | - | 4 | - |
| 리플릿 (leaflet) | - | 7 | - |
| **PDF 문서 소계** | - | **674** | - |

**청킹 설정**:

- 최대 청크 길이: 1,000자
- 최소 청크 길이: 150자
- 청크당 평균 문장 수: 3.96

---

**문서 구조**:
```json
{
    "text": "근로기준법 제50조 (근로시간) 1주간의 근로시간은...",
    "source": "근로기준법 제50조",
    "doc_type": "precedent",  // precedent, interpretation, labor_ministry, manual
    "title": "근로시간",
    "keywords": ["근로시간", "1주", "40시간"],
    "embedding": [0.1, 0.2, ...]  // 1024차원 MUVERA FDE 벡터
}
```

**검색 방식**: BM25 + Vector 하이브리드 검색

### Neo4j (Graph DB)

**목적**: 법률 문서 간의 인용 관계, 카테고리 분류, 위험 패턴 연결을 그래프로 구조화하여 멀티홉 검색 지원

#### 그래프 구축 파이프라인

```
5_build_graph.py        → Document 노드 생성 (15,223개 청크)
        |
6_create_relationships.py → 노드 라벨 세분화 + Category/Source 관계 생성
        |
7_seed_ontology.py      → 온톨로지 구축 (ClauseType, RiskPattern)
        |
8_build_multihop_links.py → LLM 기반 법령 인용 관계 추출 (CITES)
```

#### 노드 유형

| 노드 라벨 | 설명 | 생성 방식 |
|----------|------|----------|
| `Document` | 기본 문서 노드 (15,223개) | 5_build_graph.py |
| `Precedent` | 판례 (10,576개) | 6_create_relationships.py |
| `Interpretation` | 법령해석례 + 고용노동부 해설 (3,973개) | 6_create_relationships.py |
| `Manual` | 실무 매뉴얼, 가이드, 리플릿 (307개) | 6_create_relationships.py |
| `Law` | 인용된 법령 조항 (LLM 추출) | 8_build_multihop_links.py |
| `Category` | 카테고리 허브 | 6_create_relationships.py |
| `ClauseType` | 조항 유형 (6종) | 7_seed_ontology.py |
| `RiskPattern` | 위험 패턴 (4종) | 7_seed_ontology.py |

#### 온톨로지 (지식 체계)

**조항 유형 (ClauseType)** - 6종:

| 조항 유형 | 필수 여부 | 관련 법령 |
|----------|----------|----------|
| 임금 | 필수 | 근로기준법 제17조 |
| 근로시간 | 필수 | 근로기준법 제50조 |
| 휴일_휴가 | 필수 | 근로기준법 제55조, 제60조 |
| 계약기간 | 필수 | 기간제법 |
| 해고_퇴직 | 선택 | 근로기준법 제23조 |
| 손해배상 | 선택 | 근로기준법 제20조 |

**위험 패턴 (RiskPattern)** - 4종:

| 위험 패턴 | 위험도 | 트리거 키워드 |
|----------|--------|--------------|
| 포괄임금제 | High | "포괄하여", "제수당 포함" |
| 과도한_위약금 | High | "배상하여야", "위약금" |
| 최저임금_미달 | High | "최저임금", "수습기간 90%" |
| 부당_해고_조항 | Medium | "즉시 해고", "갑의 판단" |

#### 관계 (Relationships)

```cypher
// 카테고리 분류
(:Document)-[:CATEGORIZED_AS]->(:Category)

// 출처 연결
(:Document)-[:SOURCE_IS]->(:Source)

// 위험 패턴 → 조항 유형
(:RiskPattern)-[:IS_A_TYPE_OF]->(:ClauseType)

// 위험 패턴 → 판례/해석 (근거 자료)
(:RiskPattern)-[:HAS_CASE]->(:Precedent)
(:RiskPattern)-[:HAS_INTERPRETATION]->(:Interpretation)

// LLM 추출 법령 인용 관계
(:Precedent)-[:CITES]->(:Law)
(:Interpretation)-[:CITES]->(:Law)
```

#### 멀티홉 검색 예시

```
[계약서 조항: "포괄임금제"]
        |
        v
[RiskPattern: 포괄임금제] --HAS_CASE--> [Precedent: 대법원 2019다12345]
        |                                        |
        v                                        v
[ClauseType: 임금]                    [Law: 근로기준법 제56조]
```

---

### 데이터 수집 현황

#### Legal API 데이터 (open.law.go.kr)

| 데이터 유형 | 수집 건수 | 청크 수 | 수집 일자 |
|------------|----------|--------|----------|
| 판례 (precedent) | 969건 | 10,576 | 2025-10-27 |
| 법령해석례 (interpretation) | 135건 | 589 | 2025-10-27 |
| 고용노동부 해설 (labor_ministry) | 1,827건 | 3,384 | 2025-10-27 |
| **소계** | **2,931건** | **14,549** | - |

**수집 키워드**: 근로계약, 임금, 최저임금, 연장근로, 해고, 퇴직금, 연차휴가 등

#### PDF 문서

| 문서명 | 유형 | 청크 수 |
|-------|------|--------|
| 개정 표준근로계약서 (2025년) | employment_rules | 367 |
| 채용절차의 공정화에 관한 법률 업무 매뉴얼 | manual | 296 |
| 2025년 적용 최저임금 안내 | guide | 4 |
| 채용절차의 공정화에 관한 법률 리플릿 | leaflet | 7 |
| **소계** | - | **674** |

### Redis (캐시 & 작업 큐)

**용도**:
- Celery 작업 브로커
- API 응답 캐싱
- 세션 저장

---

## 기술 스택 요약

### Backend

| 기술 | 용도 | 버전 |
|------|------|------|
| FastAPI | REST API 프레임워크 | 0.121.3 |
| Celery | 비동기 작업 처리 | 5.5.3 |
| SQLAlchemy | ORM | 2.0.44 |
| Alembic | DB 마이그레이션 | 1.17.2 |
| LangGraph | AI 에이전트 프레임워크 | 0.2.0+ |
| LangChain | LLM 통합 | 0.3.0+ |
| OpenAI | GPT-4o API | 1.95.0 |
| Google Generative AI | Gemini API | 0.8.3 |
| Tavily | 웹 검색 API | 0.5.0+ |
| sentence-transformers | 임베딩 생성 | 5.1.2 |
| pdfplumber | PDF 파싱 | 0.11.4 |
| python-docx | DOCX 파싱 | 1.1.2 |
| Neo4j | 그래프 DB 드라이버 | 5.14.1 |
| Elasticsearch | 벡터 검색 | 8.11.0 |

### Frontend

| 기술 | 용도 | 버전 |
|------|------|------|
| Next.js | React 프레임워크 | 15.5.4 |
| React | UI 라이브러리 | 19.1.0 |
| TypeScript | 타입 안전성 | 5.x |
| Tailwind CSS | 스타일링 | 3.4.18 |
| Radix UI | UI 컴포넌트 | 다수 |
| react-markdown | 마크다운 렌더링 | 9.0.1 |
| remark-gfm | GFM 지원 | 4.0.0 |
| react-pdf | PDF 뷰어 | 10.2.0 |
| Recharts | 차트 라이브러리 | 3.2.1 |

### Infrastructure

| 기술 | 용도 | 포트 |
|------|------|------|
| PostgreSQL | 관계형 DB | 5435 |
| Elasticsearch | Vector DB | 9200 |
| Neo4j | Graph DB | 7474, 7687 |
| Redis | 캐시 & 큐 | 6379 |
| Docker | 컨테이너화 | - |

---

## 부록: 기술 적용 전후 비교

### 1. HyDE 적용 효과

**Before (일반 키워드 검색)**:
```
쿼리: "야근 수당 안 줘도 되나요?"
결과:
  1. "야근 식대 지원 안내" (관련성 낮음)
  2. "수당 지급 절차" (부분 관련)
  3. "야근 신청서 양식" (관련성 낮음)
```

**After (HyDE 적용)**:
```
쿼리: "야근 수당 안 줘도 되나요?"
HyDE 가상 문서: "근로기준법 제56조에 따르면 연장근로에 대해..."
결과:
  1. "근로기준법 제56조 (연장/야간/휴일 근로)" (정확)
  2. "연장근로수당 산정 방법 (행정해석)" (정확)
  3. "야간근로 가산수당 판례" (정확)
```

### 2. CRAG 적용 효과

**Before (일반 RAG)**:
```
품질 평가 없이 상위 5개 문서 사용
  -> 관련 없는 문서도 LLM에 전달
  -> 잘못된 정보 기반 응답 가능성
```

**After (CRAG 적용)**:
```
검색 -> 품질 평가 (PARTIAL) -> 교정 (AUGMENT)
  -> Graph DB로 관련 문서 확장
  -> 품질 재평가 (GOOD)
  -> 고품질 문서만 LLM에 전달
```

### 3. Neuro-Symbolic 적용 효과

**Before (Pure LLM)**:
```
입력: "월급 200만원, 주 5일, 하루 9시간 근무"
LLM 계산: "시급은 약 10,000원 정도로 최저임금과 비슷합니다"
문제: 부정확한 계산, 연장근로 수당 누락
```

**After (Neuro-Symbolic)**:
```
입력: "월급 200만원, 주 5일, 하루 9시간 근무"
LLM 추출: base_salary=2,000,000, daily_hours=9, weekly_days=5
Python 계산:
  - 월 근로시간 = 9 * 22 = 198시간
  - 시급 = 2,000,000 / 198 = 10,101원 (최저임금 충족)
  - 연장근로 = (9-8) * 22 = 22시간
  - 연장수당 필요액 = 22 * 10,101 * 0.5 = 111,111원
결과: "최저임금은 충족하나, 월 111,111원의 연장근로수당 미지급"
```

### 4. Constitutional AI 적용 효과

**Before (일반 LLM 응답)**:
```
분석: "이 위약금 조항은 교육비 배상을 규정하고 있습니다.
       회사가 교육에 투자한 비용을 회수하려는 조항입니다."
문제: 근로자 불이익 조항임을 명시하지 않음
```

**After (Constitutional AI 적용)**:
```
Critique: "제3조(최저기준 보장) 위반 - 근로기준법 제20조
          위약 예정 금지 원칙 미언급"

수정된 분석: "이 위약금 조항은 근로기준법 제20조(위약 예정의 금지)를
            위반할 수 있습니다. 법적으로 사용자는 근로계약 불이행에
            대한 위약금이나 손해배상액을 미리 정할 수 없습니다.
            이 조항의 삭제 또는 수정을 권고합니다."
```

---

*최종 업데이트: 2025년 12월*
