# DocScanner AI - Q&A 대비 자료

발표 후 예상 질문과 상세 기술 설명을 정리한 문서입니다.

---

## 1. MUVERA / FDE (Fixed Dimensional Encoding)

### Q: MUVERA가 정확히 뭔가요? 직접 구현한 건가요?

**A**: MUVERA는 Google Research에서 2024년 NeurIPS에 발표한 "Multi-Vector Retrieval via Fixed Dimensional Encodings" 논문의 기법입니다. 저희가 이 논문을 참고하여 직접 구현했습니다.

**핵심 아이디어**:
- 기존 RAG는 문서를 하나의 벡터로 임베딩 (정보 손실)
- ColBERT는 토큰마다 벡터 생성 (검색 비용 높음)
- MUVERA는 "다중 벡터의 정보를 보존하면서 단일 벡터 검색 속도"를 달성

### Q: FDE가 구체적으로 어떻게 동작하나요?

**A**: 저희 구현 ([fde_generator.py](../ai/preprocessing/fde_generator.py))을 기준으로 설명드리면:

```
[입력] 문서의 문장들 각각 임베딩 (N개의 1024차원 벡터)
         ↓
[Step 1] SimHash 파티셔닝
         - 각 문장 벡터에 랜덤 투영 행렬 곱셈 (6개 투영)
         - 투영 결과의 부호(>0)를 Gray Code로 변환
         - 64개(2^6) 파티션 중 하나에 할당
         ↓
[Step 2] 파티션별 집계
         - Document: 파티션 내 벡터들의 평균 (AVERAGE)
         - Query: 파티션 내 벡터들의 합 (SUM)
         ↓
[Step 3] 연결 (Concatenate)
         - 64개 파티션 결과를 이어붙임
         ↓
[Step 4] Count Sketch 압축
         - 최종 1024차원으로 압축
         ↓
[출력] 단일 1024차원 FDE 벡터
```

**왜 Document는 AVERAGE, Query는 SUM인가요?**
- Document: 길이에 무관한 정규화된 표현 필요
- Query: 짧은 쿼리도 충분한 신호 강도 필요

### Q: SimHash가 뭔가요?

**A**: Locality-Sensitive Hashing(LSH)의 일종으로, **유사한 벡터가 같은 버킷(파티션)에 들어갈 확률이 높도록** 해싱하는 기법입니다.

```python
# 실제 구현 (fde_generator.py:76-80)
def _simhash_partition_index_gray(sketch_vector: np.ndarray) -> int:
    partition_index = 0
    for val in sketch_vector:
        partition_index = _append_to_gray_code(partition_index, val > 0)
    return partition_index
```

- 랜덤 투영 후 부호만 취함 (0 또는 1)
- 유사한 벡터는 비슷한 방향 → 같은 부호 → 같은 파티션
- 이를 통해 의미적으로 유사한 문장들이 같은 파티션에 모임

### Q: 성능 향상 수치의 근거가 뭔가요?

**A**: 대본에서 언급한 "5-20x 속도 향상, 90% 정확도"는 MUVERA 원 논문의 실험 결과입니다. 저희 시스템에서 직접 측정한 수치는 아닙니다. 향후 평가 계획에 포함되어 있습니다.

---

## 2. Knowledge Graph 구축

### Q: 그래프를 어떻게 구축했나요?

**A**: 5단계 파이프라인으로 구축했습니다:

**Step 1: 노드 생성** ([5_build_graph.py](../ai/preprocessing/5_build_graph.py))
```cypher
MERGE (d:Document {id: row.chunk_id})
SET d.content = row.content,
    d.source = row.source,
    d.category = row.category,
    d.type = row.doc_type
```
- 모든 청크를 Document 노드로 생성
- 약 4,000개 노드 (판례, 해석례, 법령해설, 매뉴얼)

**Step 2: 라벨링 및 관계 생성** ([6_create_relationships.py](../ai/preprocessing/6_create_relationships.py))
```cypher
-- 문서 타입별 라벨 부여
MATCH (d:Document) WHERE d.type = 'precedent' SET d:Precedent
MATCH (d:Document) WHERE d.type IN ['interpretation', 'labor_ministry'] SET d:Interpretation

-- 카테고리 허브 연결
MERGE (c:Category {name: d.category})
MERGE (d)-[:CATEGORIZED_AS]->(c)
```

**Step 3: 온톨로지 시드** ([7_seed_ontology.py](../ai/preprocessing/7_seed_ontology.py))
```cypher
-- 조항 유형 노드
MERGE (c:ClauseType {name: '임금'})
SET c.isRequired = true,
    c.explanation = '근로기준법 제17조에 따라...'

-- 위험 패턴 노드 및 연결
MERGE (r:RiskPattern {name: '포괄임금제'})
SET r.riskLevel = 'High',
    r.triggers = ['포괄하여', '포함하여 지급', ...]
MERGE (r)-[:IS_A_TYPE_OF]->(c:ClauseType {name: '임금'})
```

**Step 4: 멀티홉 링크** ([8_build_multihop_links.py](../ai/preprocessing/8_build_multihop_links.py))
```python
# LLM으로 판례/해석에서 인용된 법령 추출
extraction = citation_chain.invoke({"text": doc["content"][:4000]})
# 예: {law_name: '근로기준법', article: '제23조'}

# 인용 관계 생성
MERGE (d)-[:CITES]->(l:Law {name: '근로기준법 제23조'})
```

### Q: 그래프 스키마가 어떻게 되나요?

**A**:
```
노드 타입:
- Document (Precedent, Interpretation, Manual, Law)
- Category (근로기준법, 최저임금법, ...)
- Source (출처)
- ClauseType (임금, 근로시간, 휴일_휴가, ...)
- RiskPattern (포괄임금제, 과도한_위약금, ...)
- Law (근로기준법 제20조, ...)

관계 타입:
- (Document)-[:CATEGORIZED_AS]->(Category)
- (Document)-[:SOURCE_IS]->(Source)
- (Document)-[:CITES]->(Law)
- (RiskPattern)-[:IS_A_TYPE_OF]->(ClauseType)
- (RiskPattern)-[:HAS_CASE]->(Precedent)
- (RiskPattern)-[:HAS_INTERPRETATION]->(Interpretation)
```

### Q: 멀티홉 검색이 뭔가요?

**A**: 질문에 직접 답하는 문서가 없을 때, 그래프 관계를 따라가며 관련 정보를 찾는 방식입니다.

예시: "포괄임금제가 위법한가요?"
```
1홉: (Query) → 벡터검색 → (관련 판례)
2홉: (관련 판례) -[:CITES]-> (근로기준법 제56조)
3홉: (근로기준법 제56조) <-[:CITES]- (다른 유사 판례)
```

CRAG에서 구현: ([crag.py:941-984](../backend/app/ai/crag.py#L941-L984))
```python
query = """
MATCH path = (d:Document)-[:CITES*1..2]->(related)
WHERE d.id IN $anchor_ids
RETURN related.content, length(path) AS hops
ORDER BY hops ASC
"""
```

---

## 3. HyDE (Hypothetical Document Embeddings)

### Q: HyDE가 뭔가요?

**A**: "Precise Zero-Shot Dense Retrieval without Relevance Labels" (Gao et al.) 논문의 기법입니다.

**핵심 아이디어**: 사용자 질문을 그대로 임베딩하면 "질문 벡터"가 되지만, **이상적인 답변 문서를 먼저 생성**한 후 임베딩하면 "답변 벡터"가 되어 실제 답변 문서와 더 유사해집니다.

```
[기존 방식]
"포괄임금제 위법?" → 임베딩 → 질문 벡터 → 검색

[HyDE 방식]
"포괄임금제 위법?"
    → LLM이 가상 답변 생성: "포괄임금제는 근로기준법 제56조에 따라..."
    → 가상 답변 임베딩 → 답변 벡터 → 검색
```

### Q: 저희 HyDE 구현의 특징은?

**A**: ([hyde.py](../backend/app/ai/hyde.py))

1. **다관점 생성** (Multi-Perspective):
   - 법률 전문가 관점 프롬프트
   - 판례 분석가 관점 프롬프트
   - 행정해석 관점 프롬프트
   - 3개 문서 생성 후 가중 앙상블

2. **적응형 전략** (Adaptive Strategy):
   ```python
   def _analyze_query_complexity(self, query):
       # 길이, 법률 용어 수, 복합 질문 여부로 복잡도 판단
       if total_score < 0.2: return SIMPLE
       elif total_score < 0.5: return MODERATE
       else: return COMPLEX
   ```
   - SIMPLE: 단일 문서 생성
   - MODERATE: 앙상블 (temperature 다르게)
   - COMPLEX/EXPERT: 다관점 생성

3. **법률 용어 매핑**:
   ```python
   LEGAL_TERMINOLOGY_MAP = {
       "잘라": ["해고", "근로관계 종료", "계약 해지"],
       "월급": ["임금", "급여", "보수", "근로기준법 제43조"],
       "야근": ["연장근로", "시간외근로", "근로기준법 제53조"],
       ...
   }
   ```

---

## 4. CRAG (Corrective RAG)

### Q: CRAG가 뭔가요?

**A**: "Corrective Retrieval Augmented Generation" (Yan et al. 2024) 논문의 기법으로, 검색 결과의 품질을 평가하고 부족하면 자동으로 보정하는 방식입니다.

### Q: 저희 CRAG 구현의 동작 방식은?

**A**: ([crag.py](../backend/app/ai/crag.py))

```
[Step 1] 초기 검색
    벡터 검색으로 top-k 문서 가져옴
         ↓
[Step 2] 품질 평가 (LLM 기반 루브릭)
    - 법적 관련성: 0-5점
    - 사실 정확성: 0-5점
    - 완전성: 0-5점
    → quality: excellent/good/correct/partial/ambiguous/weak/incorrect
         ↓
[Step 3] 보정 전략 결정
    - NONE: 충분함 → 바로 사용
    - REFINE: 관련 부분만 추출
    - REWRITE: 쿼리 재작성 후 재검색
    - DECOMPOSE: 복합 질문 분해
    - AUGMENT: 그래프로 확장
    - FALLBACK: BM25 폴백 검색
    - MULTI_HOP: 그래프 다단계 탐색
         ↓
[Step 4] 보정 실행 (최대 3회 반복)
         ↓
[Step 5] 최종 컨텍스트 생성
```

### Q: 품질 평가 루브릭 예시?

**A**:
```
[법적 관련성 평가]
점수 5: 질문에 직접 답변 가능한 법령 조항/판례 포함
점수 4: 관련 법령 포함, 대부분 답변 가능
점수 3: 관련 법적 개념 언급, 간접적 도움
점수 2: 법적 맥락은 맞으나 직접 관련 없음
점수 1: 관련성 거의 없음
점수 0: 완전히 무관 또는 잘못된 정보
```

---

## 5. 벡터 검색 / Elasticsearch

### Q: Elasticsearch를 어떻게 사용하나요?

**A**: ([4_index.py](../ai/preprocessing/4_index.py))

**인덱스 설정**:
```python
index_settings = {
    "mappings": {
        "properties": {
            "text": {
                "type": "text",
                "analyzer": "nori"  # 한국어 형태소 분석기
            },
            "embedding": {
                "type": "dense_vector",
                "dims": 1024  # MUVERA FDE 차원
            },
            "source": {
                "type": "keyword"
            }
        }
    }
}
```

**검색 방식**:
1. **벡터 검색 (KNN)**:
   ```python
   search_body = {
       "knn": {
           "field": "embedding",
           "query_vector": query_fde.tolist(),
           "k": 5,
           "num_candidates": 50
       }
   }
   ```

2. **BM25 폴백** (CRAG에서 사용):
   ```python
   search_body = {
       "query": {
           "multi_match": {
               "query": query,
               "fields": ["text^2", "title"],
               "analyzer": "nori"
           }
       }
   }
   ```

### Q: nori 분석기가 뭔가요?

**A**: Elasticsearch의 한국어 형태소 분석 플러그인입니다.
- "근로계약서" → ["근로", "계약", "서"]
- "포괄임금제" → ["포괄", "임금", "제"]
- BM25 검색 시 형태소 단위로 매칭

---

## 6. Neuro-Symbolic AI

### Q: Neuro-Symbolic이 뭔가요?

**A**: Neural Network(LLM)와 Symbolic Reasoning(규칙 기반)을 결합한 접근입니다.

**저희 시스템에서**:
- **Neural**: LLM의 자연어 이해, 문서 생성
- **Symbolic**: 그래프 DB의 명시적 관계, 위험 패턴 규칙

예시:
```python
# Symbolic: 명시적 위험 패턴 규칙 (7_seed_ontology.py)
risk_patterns = [
    {
        "name": "포괄임금제",
        "triggers": ["포괄하여", "포함하여 지급"],
        "riskLevel": "High",
        "law_keywords": ["제56조", "연장근로"]
    }
]

# Neural: LLM이 맥락을 이해하고 적용
# 계약서 조항이 위험 패턴의 triggers와 매칭되면
# LLM이 구체적 위반 내용을 설명
```

---

## 7. Constitutional AI

### Q: Constitutional AI가 뭔가요?

**A**: Anthropic에서 제안한 자기 수정 AI 기법입니다.

**핵심**: AI가 스스로 출력을 평가하고 수정하는 피드백 루프
- Critique: "이 응답이 법적으로 정확한가?"
- Revision: "부정확하면 수정"

**저희 시스템에서** (CRAG의 일부):
```python
# Step 1: 초기 응답 생성
initial_response = llm.generate(query, context)

# Step 2: 품질 평가 (Critique)
evaluation = crag._evaluate_quality(query, docs)
if evaluation.quality < THRESHOLD:
    # Step 3: 보정 (Revision)
    corrected = crag._refine_documents(query, docs)
```

---

## 8. 데이터

### Q: 어떤 데이터를 사용했나요?

**A**:
| 데이터 타입 | 출처 | 수량 |
|------------|------|------|
| 판례 | 대법원 판례 공개 데이터 | ~1,000건 |
| 법령해석례 | 고용노동부 API | ~2,000건 |
| 법령해설 | 고용노동부 매뉴얼 | ~500페이지 |
| 표준계약서 | 고용노동부 | 10종 |

### Q: 데이터 수집은 어떻게?

**A**:
1. **판례**: 법령정보센터 Open API
2. **해석례**: 고용노동부 고객상담센터 API
3. **매뉴얼**: PDF 크롤링 후 OCR (Gemini Vision)

---

## 9. 시스템 아키텍처

### Q: 기술 스택이 뭔가요?

**A**:
- **Frontend**: Next.js 15, React 19, Tailwind CSS
- **Backend**: FastAPI, LangGraph, Celery
- **AI**: GPT-5-mini (분석), Gemini 2.5-flash (OCR), KURE-v1 (임베딩)
- **DB**: PostgreSQL (메타데이터), Elasticsearch (벡터), Neo4j (그래프)

### Q: LangGraph는 뭔가요?

**A**: LangChain에서 만든 에이전트 오케스트레이션 프레임워크입니다.
- 상태 기반 워크플로우 정의
- 조건부 분기, 반복 지원
- 저희 분석 파이프라인의 12단계를 그래프로 정의

---

## 10. 향후 계획

### Q: 하이브리드 스코어 퓨전이 뭔가요?

**A**:
```
Final Score = w1 * MUVERA_score + w2 * Reranker_score + w3 * Graph_authority

- MUVERA_score: 벡터 유사도
- Reranker_score: Cross-encoder 정밀 점수
- Graph_authority: PageRank + Citation Count
```

### Q: 평가는 어떻게 할 계획인가요?

**A**:
1. **위험 조항 탐지 정확도**: 실제 위험 계약서 샘플로 Precision/Recall 측정
2. **법률 근거 인용 정확성**: 전문가 검토
3. **사용자 만족도**: 파일럿 테스트

---

## 예상 꼬리 질문

### Q: ColBERT와의 차이점?

**A**: ColBERT는 토큰별 벡터를 모두 저장하고 MaxSim으로 검색합니다. MUVERA는 이를 FDE로 압축해서 단일 벡터 검색만으로 유사한 성능을 냅니다.

### Q: 왜 Neo4j를 선택했나요?

**A**:
- 법률 도메인은 관계가 중요 (법령 → 판례 → 해석)
- Cypher 쿼리로 멀티홉 탐색이 간단
- 온톨로지 표현에 적합

### Q: 실시간 분석이 가능한가요?

**A**: 현재 계약서 분석에 약 30초-1분 소요됩니다. SSE로 진행 상황을 스트리밍하여 사용자 경험을 개선했습니다.

### Q: Hallucination 방지는?

**A**:
1. CRAG의 품질 평가로 관련 없는 문서 필터링
2. 답변 시 반드시 검색된 문서 기반으로 생성
3. 법적 근거(법령 조항, 판례 번호) 명시적 인용 강제
