# AI Pipeline Architecture V2

DocScanner.ai V2 계약서 분석 파이프라인의 구현 상세 문서.

## 목차

1. [파이프라인 개요](#파이프라인-개요)
2. [전체 흐름도](#전체-흐름도)
3. [모듈별 상세 구현](#모듈별-상세-구현)
4. [모델 할당 구성](#모델-할당-구성)
5. [데이터 흐름](#데이터-흐름)

---

## 파이프라인 개요

### 핵심 구성 요소

V2 파이프라인은 `AdvancedAIPipeline` 클래스(`backend/app/ai/pipeline.py`)에서 orchestration되며, 다음 모듈들로 구성됩니다:

| 모듈 | 역할 | 파일 |
|------|------|------|
| ClauseAnalyzer | 조항 추출 및 위반 분석 | `clause_analyzer.py` |
| HyDERetriever | 가상 문서 기반 검색 향상 | `hyde.py` |
| CorrectiveRAG | 자기 수정 RAG | `crag.py` |
| RAPTORIndexer | 계층적 문서 인덱싱 | `raptor.py` |
| ConstitutionalAI | 노동법 원칙 검증 | `constitutional_ai.py` |
| LLMJudge | 신뢰도 평가 | `judge.py` |

### 설계 원칙

1. **Hybrid AI**: LLM 기반 추론 + Python 기반 정확한 수치 계산 (Neuro-Symbolic)
2. **Self-Correcting**: 검색 품질 평가 후 자동 보정 (CRAG)
3. **Constitutional**: 노동법 6대 원칙에 따른 윤리적 검증
4. **Explainable**: 모든 판단에 법적 근거와 신뢰도 점수 제공

---

## 전체 흐름도

```
계약서 텍스트 입력
         |
         v
+--------------------------------------------------+
|              ClauseAnalyzer                       |
|  +--------------------------------------------+  |
|  | 1. LLM 조항 추출 (JSON 스키마 강제)          |  |
|  |    - Fallback: 정규표현식 매칭              |  |
|  +--------------------------------------------+  |
|                      |                           |
|                      v                           |
|  +--------------------------------------------+  |
|  | 2. 개별 조항 위반 분석                       |  |
|  |    - Hybrid Search: Vector DB + Graph DB   |  |
|  |    - RRF(Reciprocal Rank Fusion) 병합      |  |
|  +--------------------------------------------+  |
|                      |                           |
|                      v                           |
|  +--------------------------------------------+  |
|  | 3. Holistic Analysis (전체 계약서 분석)      |  |
|  |    - NeuroSymbolicCalculator: 최저임금 계산 |  |
|  |    - 52시간 상한 검증                       |  |
|  +--------------------------------------------+  |
|                      |                           |
|                      v                           |
|  +--------------------------------------------+  |
|  | 4. 동일 조항 위반 병합 (LLM)                 |  |
|  +--------------------------------------------+  |
|                      |                           |
|                      v                           |
|  +--------------------------------------------+  |
|  | 5. 위반 위치 매핑 (Gemini)                   |  |
|  |    - UI 하이라이팅용 정확한 텍스트 위치      |  |
|  +--------------------------------------------+  |
+--------------------------------------------------+
         |
         v
+--------------------------------------------------+
|                    CRAG                          |
|  +--------------------------------------------+  |
|  | 1. 검색 결과 관련성 평가                     |  |
|  |    - RELEVANT / PARTIALLY / NOT_RELEVANT   |  |
|  +--------------------------------------------+  |
|                      |                           |
|           +----------+----------+                |
|           |          |          |                |
|       RELEVANT   PARTIAL    NOT_RELEVANT         |
|           |          |          |                |
|        그대로     필터링     쿼리 재작성          |
|        사용      + 보강      후 재검색            |
|           |          |          |                |
|           +----------+----------+                |
+--------------------------------------------------+
         |
         v
+--------------------------------------------------+
|                   RAPTOR                         |
|  +--------------------------------------------+  |
|  | 1. 클러스터링 (agglomerative/DBSCAN/        |  |
|  |    k-means/semantic)                        |  |
|  +--------------------------------------------+  |
|                      |                           |
|                      v                           |
|  +--------------------------------------------+  |
|  | 2. 법률 특화 요약 (카테고리별 프롬프트)       |  |
|  |    - 임금, 근로시간, 휴일휴가, 해고퇴직      |  |
|  +--------------------------------------------+  |
|                      |                           |
|                      v                           |
|  +--------------------------------------------+  |
|  | 3. 적응적 검색 (쿼리 유형별 레벨 선택)       |  |
+--------------------------------------------------+
         |
         v
+--------------------------------------------------+
|              Constitutional AI                    |
|  +--------------------------------------------+  |
|  | 1. Critique: 6가지 원칙 위반 검토            |  |
|  +--------------------------------------------+  |
|                      |                           |
|                      v                           |
|  +--------------------------------------------+  |
|  | 2. Revise: 위반 시 분석 결과 수정            |  |
+--------------------------------------------------+
         |
         v
+--------------------------------------------------+
|                 LLM-as-a-Judge                   |
|  +--------------------------------------------+  |
|  | 1. 5가지 카테고리 평가 (가중치 적용)          |  |
|  | 2. 팩트 체크: 치명적 오류 검사               |  |
|  | 3. 신뢰도 레벨: HIGH / MEDIUM / LOW         |  |
+--------------------------------------------------+
         |
         v
최종 분석 결과 (위반 목록 + 신뢰도 점수 + 수정 권고안)
```

---

## 모듈별 상세 구현

### 1. ClauseAnalyzer

**파일**: `backend/app/ai/clause_analyzer.py`

#### 1.1 조항 추출 (`_extract_clauses`)

LLM을 사용하여 계약서에서 조항을 구조화된 형태로 추출합니다.

**프롬프트 구조**:
```python
system_prompt = """당신은 한국 근로계약서 분석 전문가입니다.
주어진 근로계약서에서 개별 조항들을 추출하여 구조화된 형태로 반환하세요."""
```

**JSON 스키마 강제**:
```python
extraction_schema = {
    "type": "object",
    "properties": {
        "clauses": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "clause_number": {"type": "string"},
                    "title": {"type": "string"},
                    "content": {"type": "string"},
                    "category": {
                        "type": "string",
                        "enum": ["임금", "근로시간", "휴일휴가", "해고퇴직", "기타"]
                    }
                }
            }
        }
    }
}
```

**Fallback 메커니즘**:

LLM 실패 시 정규표현식 기반 추출:
```python
# 제N조, 제N항 등의 패턴 매칭
clause_pattern = r'제\s*(\d+)\s*조[^\n]*\n(.*?)(?=제\s*\d+\s*조|$)'
```

#### 1.2 개별 조항 위반 분석 (`_analyze_clause`)

각 조항에 대해 Hybrid Search로 법률 컨텍스트를 검색하고 위반 여부를 분석합니다.

**Hybrid Search 구현**:
1. Vector DB (Pinecone): 의미적 유사성 기반 검색
2. Graph DB (Neo4j): 법률 조항 간 관계 기반 검색
3. RRF(Reciprocal Rank Fusion)로 두 결과 병합

**분석 프롬프트**:
```python
analysis_prompt = f"""
## 분석 대상 조항
{clause.content}

## 관련 법률 조항
{legal_context}

## 분석 지침
1. 이 조항이 근로기준법을 위반하는지 판단하세요
2. 위반 시 구체적인 법률 조항과 근거를 제시하세요
3. 위반의 심각도를 평가하세요 (high/medium/low)
4. 수정 권고안을 제시하세요
"""
```

**출력 스키마**:
```python
violation_schema = {
    "violations": [{
        "violation_type": "string",
        "severity": "high|medium|low",
        "description": "string",
        "legal_basis": "string",
        "recommendation": "string"
    }]
}
```

#### 1.3 Holistic Analysis (전체 계약서 분석)

개별 조항 분석 후 전체 계약서 차원의 분석을 수행합니다.

**NeuroSymbolic Calculator**:

LLM이 값을 추출하고, Python이 정확한 수치 계산을 수행합니다.

```python
# LLM이 값 추출
extracted = llm_extract({
    "monthly_wage": "...",
    "weekly_hours": "...",
    "monthly_hours": "..."
})

# Python 계산 (정확한 수치 연산)
hourly_wage = monthly_wage / monthly_hours
minimum_wage_2024 = 9860
is_violation = hourly_wage < minimum_wage_2024
```

**52시간 상한 검증**:
```python
weekly_hours = extract_weekly_hours(contract_text)
if weekly_hours > 52:
    add_violation("주 52시간 초과")
```

#### 1.4 동일 조항 위반 병합 (`_merge_same_clause_violations`)

같은 조항에서 발견된 여러 위반을 LLM으로 병합:

```python
merge_prompt = f"""
다음은 동일한 조항에서 발견된 여러 위반 사항입니다:
{violations_list}

이들을 논리적으로 병합하여 하나의 통합된 위반 분석 결과로 정리하세요.
중복을 제거하고, 관련된 내용을 그룹화하세요.
"""
```

#### 1.5 위반 위치 매핑 (ViolationLocationMapper)

UI에서 하이라이팅을 위해 Gemini를 사용하여 위반 텍스트의 정확한 위치를 찾습니다.

```python
class ViolationLocationMapper:
    def find_violation_location(self, contract_text: str, violation: dict) -> dict:
        prompt = f"""
        계약서 텍스트에서 다음 위반과 관련된 정확한 텍스트를 찾으세요:
        위반 내용: {violation['description']}

        원본 텍스트에서 해당 부분을 정확히 추출하세요.
        """

        # 반환: {"start": 시작위치, "end": 끝위치, "text": "해당 텍스트"}
```

---

### 2. HyDE (Hypothetical Document Embedding)

**파일**: `backend/app/ai/hyde.py`

법률 검색 품질 향상을 위해 가상의 법률 문서를 먼저 생성합니다.

**동작 원리**:
```
기존 RAG:
  쿼리 → 쿼리 임베딩 → 유사 문서 검색

HyDE:
  쿼리 → LLM이 가상 법률 문서 생성 → 가상 문서 임베딩 → 유사 문서 검색
```

**구현**:
```python
class HyDERetriever:
    def generate_hypothetical_document(self, query: str) -> str:
        prompt = f"""
        다음 질문에 대해 답변하는 가상의 법률 문서를 작성하세요:
        질문: {query}

        이 문서는 근로기준법의 관련 조항을 인용하고,
        법적 해석을 포함해야 합니다.
        """

        hypothetical_doc = self.llm.generate(prompt)
        return hypothetical_doc

    def retrieve(self, query: str) -> List[Document]:
        # 1. 가상 문서 생성
        hypo_doc = self.generate_hypothetical_document(query)

        # 2. 가상 문서를 임베딩하여 검색
        embedding = self.embed(hypo_doc)

        # 3. Vector DB에서 유사 문서 검색
        results = self.vector_db.search(embedding)
        return results
```

**장점**: 짧은 쿼리 대신 풍부한 의미 정보가 담긴 가상 문서로 검색하여 정확도 향상

---

### 3. CRAG (Corrective RAG)

**파일**: `backend/app/ai/crag.py`

검색 결과의 관련성을 평가하고 필요시 재검색하는 자기 수정 RAG입니다.

**관련성 평가 기준**:
```python
class RetrievalQuality(Enum):
    RELEVANT = "relevant"           # 신뢰도 > 0.8
    PARTIALLY_RELEVANT = "partial"  # 0.3 < 신뢰도 < 0.8
    NOT_RELEVANT = "not_relevant"   # 신뢰도 < 0.3
```

**구현**:
```python
class CorrectiveRAG:
    def retrieve_with_correction(self, query: str, context: str) -> dict:
        # 1단계: 초기 검색
        initial_results = self.retriever.search(query)

        # 2단계: 관련성 평가
        relevance_prompt = f"""
        질문: {query}
        검색 결과: {initial_results}

        각 검색 결과의 관련성을 평가하세요:
        - RELEVANT: 질문에 직접적으로 관련됨
        - PARTIALLY_RELEVANT: 부분적으로 관련됨
        - NOT_RELEVANT: 관련 없음
        """

        evaluations = self.llm.evaluate(relevance_prompt)

        # 3단계: 결과에 따른 분기
        if all_relevant(evaluations):
            return {"action": "USE", "results": initial_results}
        elif all_not_relevant(evaluations):
            refined_query = self.rewrite_query(query)
            new_results = self.retriever.search(refined_query)
            return {"action": "REFINED", "results": new_results}
        else:
            filtered = filter_relevant(initial_results, evaluations)
            return {"action": "FILTERED", "results": filtered}
```

---

### 4. RAPTOR (Recursive Abstractive Processing for Tree-Organized Retrieval)

**파일**: `backend/app/ai/raptor.py`

법률 문서를 계층적 트리 구조로 인덱싱합니다.

**트리 구조**:
```
                    [루트 요약]
                   "이 계약서는 최저임금 미달,
                    과도한 연장근로 등 3개의
                    주요 법적 위험이 있습니다"
                         |
           +-------------+-------------+
           |                           |
    [임금 관련 요약]            [근로시간 요약]
    "시급 8,000원으로        "주 52시간 초과
     최저임금 미달"           연장근로 조항"
           |                           |
    +------+------+             +------+------+
    |             |             |             |
 [청크1]       [청크2]       [청크3]       [청크4]
```

**클러스터링 방식**:
- Agglomerative (기본)
- DBSCAN
- K-means
- Semantic

**법률 특화 요약 프롬프트**:
```python
LEGAL_FOCUSED_PROMPTS = {
    "임금": """
        이 문서들의 핵심 임금 관련 내용을 요약하세요:
        - 최저임금 기준
        - 수당 계산 방식
        - 임금 지급 시기 및 방법
    """,
    "근로시간": """
        근로시간 관련 핵심 내용:
        - 법정 근로시간
        - 연장근로 제한
        - 휴게시간 규정
    """,
    "휴일휴가": "...",
    "해고퇴직": "..."
}
```

**적응적 검색**:
```python
def search(self, query: str) -> List[Document]:
    # 쿼리 유형 분석
    query_type = self.analyze_query(query)

    # 적응적 레벨 선택
    if query_type == "specific":
        start_level = 0  # 리프부터 (세부 정보)
    else:
        start_level = self.tree.depth // 2  # 중간부터 (일반 질문)

    results = self.tree_search(query, start_level)
    return results
```

---

### 5. Constitutional AI (헌법적 AI)

**파일**: `backend/app/ai/constitutional_ai.py`

노동법 원칙에 기반한 윤리적 검토를 수행합니다.

**6가지 노동법 헌법 원칙**:

| 원칙 | 설명 | 검토 내용 |
|------|------|----------|
| HUMAN_DIGNITY | 인간 존엄성 보장 | 근로자를 단순 자원으로 취급하지 않는가? |
| WORKER_PROTECTION | 작성자 불이익 원칙 (In dubio pro operario) | 모호한 조항을 사용자에게 유리하게 해석하지 않았는가? |
| MINIMUM_STANDARD | 최저 기준 효력 | 법정 최저 기준 이하를 허용하지 않는가? |
| EQUALITY | 균등 대우 원칙 | 차별적 해석이 없는가? |
| SAFETY | 안전 우선 원칙 | 안전 관련 규정을 경시하지 않았는가? |
| TRANSPARENCY | 명시 의무 | 불명확한 조항을 지적했는가? |

**2단계 프로세스**:

```python
class ConstitutionalAIReviewer:
    def review(self, analysis_result: dict) -> dict:
        # 1단계: Critique (비판)
        critique_prompt = f"""
        다음 분석 결과가 노동법 헌법 원칙을 준수하는지 검토하세요:

        분석 결과: {analysis_result}

        각 원칙에 대해 위반 여부를 평가하세요.
        """

        critique = self.llm.generate(critique_prompt)

        # 2단계: Revise (수정)
        if critique.has_violations:
            revise_prompt = f"""
            다음 비판을 반영하여 분석 결과를 수정하세요:

            원본 분석: {analysis_result}
            비판 내용: {critique}

            노동법 헌법 원칙에 부합하도록 수정된 분석을 제시하세요.
            """

            revised = self.llm.generate(revise_prompt)
            return revised

        return analysis_result
```

---

### 6. LLM-as-a-Judge (신뢰도 평가)

**파일**: `backend/app/ai/judge.py`

분석 결과의 신뢰도를 독립적으로 평가합니다.

**5가지 평가 카테고리**:

| 카테고리 | 가중치 | 설명 |
|----------|--------|------|
| ACCURACY | 0.25 | 법률 조항 번호, 수치 등의 정확성 |
| LEGAL_BASIS | 0.25 | 법적 근거의 적절성과 명확성 |
| CONSISTENCY | 0.20 | 분석 내 논리적 일관성 |
| COMPLETENESS | 0.15 | 모든 위반 사항 식별 여부 |
| RELEVANCE | 0.15 | 분석의 관련성과 적시성 |

**평가 프로세스**:
```python
class LLMJudge:
    def evaluate(self, analysis: dict, original_contract: str) -> dict:
        # 1. 카테고리별 평가 (0-10 점수)
        scores = {}
        for category, config in EVALUATION_CATEGORIES.items():
            score = self.evaluate_category(analysis, category)
            scores[category] = score

        # 2. 가중 평균 계산
        total_score = sum(
            scores[cat] * config["weight"]
            for cat, config in EVALUATION_CATEGORIES.items()
        )

        # 3. 팩트 체크 (치명적 오류 검사)
        fact_check = self.check_critical_errors(analysis)
        # - 잘못된 법률 조항 번호
        # - 틀린 수치 (최저임금 등)
        # - 존재하지 않는 법률 인용

        if fact_check.has_critical_errors:
            total_score *= 0.5  # 심각한 패널티

        # 4. 신뢰도 레벨 결정
        if total_score >= 80:
            confidence = "HIGH"
        elif total_score >= 60:
            confidence = "MEDIUM"
        else:
            confidence = "LOW"

        return {
            "scores": scores,
            "total_score": total_score,
            "confidence": confidence,
            "fact_check": fact_check
        }
```

---

## 모델 할당 구성

**파일**: `backend/app/core/config.py`

각 모듈별로 최적화된 모델이 할당됩니다:

| 역할 | 모델 | 용도 |
|------|------|------|
| retrieval | text-embedding-3-large | OpenAI 임베딩 |
| reasoning | gpt-4o | 주요 추론 |
| hyde | gpt-4o-mini | 가상 문서 생성 |
| scan | gemini-1.5-flash | 빠른 스캔 |
| clause_analyzer | gpt-4o | 조항 분석 |
| crag | gpt-4o-mini | CRAG 평가 |
| raptor | gpt-4o-mini | RAPTOR 요약 |
| redliner | gpt-4o | 수정안 생성 |
| location | gemini-1.5-flash | 위치 매핑 |
| judge | gpt-4o | 신뢰도 평가 |
| constitutional | gpt-4o | 헌법적 검토 |

---

## 데이터 흐름

### 입력에서 출력까지

```
계약서 원문
    |
    v
+---------------------------------------------------------------+
| ClauseAnalyzer                                                 |
|  - LLM: 조항 추출 (JSON 스키마 강제)                            |
|  - Hybrid Search: Vector DB + Graph DB -> RRF 병합             |
|  - LLM: 각 조항 위반 분석                                       |
|  - NeuroSymbolic: 최저임금/근로시간 계산                        |
|  - LLM: 동일 조항 위반 병합                                     |
|  - Gemini: 위반 위치 매핑                                       |
+---------------------------------------------------------------+
    |
    v
+---------------------------------------------------------------+
| CRAG (Self-Correcting Retrieval)                               |
|  - 관련성 평가: RELEVANT / PARTIAL / NOT_RELEVANT              |
|  - 필요시 쿼리 재작성 및 재검색                                 |
+---------------------------------------------------------------+
    |
    v
+---------------------------------------------------------------+
| RAPTOR (Hierarchical Retrieval)                                |
|  - 클러스터링: 의미적으로 유사한 문서 그룹화                    |
|  - 법률 특화 요약: 카테고리별 프롬프트                          |
|  - 적응적 검색: 쿼리 유형에 따른 레벨 선택                      |
+---------------------------------------------------------------+
    |
    v
+---------------------------------------------------------------+
| Constitutional AI                                              |
|  - Critique: 6가지 노동법 원칙 위반 검토                        |
|  - Revise: 위반 시 분석 결과 수정                               |
+---------------------------------------------------------------+
    |
    v
+---------------------------------------------------------------+
| LLM-as-a-Judge                                                 |
|  - 5가지 카테고리 평가 (가중치 적용)                            |
|  - 팩트 체크: 치명적 오류 검사                                  |
|  - 신뢰도 레벨: HIGH / MEDIUM / LOW                            |
+---------------------------------------------------------------+
    |
    v
최종 분석 결과
  - 위반 목록 (위치 정보 포함)
  - 신뢰도 점수
  - 수정 권고안
  - 법적 근거
```

### 토큰 사용량 추적

각 모듈은 `contract_id`를 기준으로 토큰 사용량을 추적합니다:

```python
class TokenTracker:
    def log_usage(self, contract_id: str, module: str, tokens: int):
        self.usage[contract_id][module] += tokens

    def get_summary(self, contract_id: str) -> dict:
        return {
            "clause_analyzer": self.usage[contract_id]["clause_analyzer"],
            "crag": self.usage[contract_id]["crag"],
            "raptor": self.usage[contract_id]["raptor"],
            "constitutional": self.usage[contract_id]["constitutional"],
            "judge": self.usage[contract_id]["judge"],
            "total": sum(self.usage[contract_id].values())
        }
```

---

## 참고 문헌

1. Gao et al. (2022). "Precise Zero-Shot Dense Retrieval without Relevance Labels" - HyDE
2. Yan et al. (2024). "Corrective Retrieval Augmented Generation" - CRAG
3. Sarthi et al. (2024). "RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval"
4. Anthropic (2022). "Constitutional AI: Harmlessness from AI Feedback"
5. Zheng et al. (2023). "Judging LLM-as-a-Judge"

---

*Last Updated: 2024-12*
