# AI Pipeline Architecture

DocScanner AI의 고급 AI 분석 파이프라인 아키텍처 상세 문서.

## 목차

1. [파이프라인 개요](#파이프라인-개요)
2. [기술 스택 및 흐름도](#기술-스택-및-흐름도)
3. [각 모듈 상세 설명](#각-모듈-상세-설명)
4. [Graph Vector 활용](#graph-vector-활용)
5. [기술적 난이도 분석](#기술적-난이도-분석)

---

## 파이프라인 개요

DocScanner AI는 근로계약서를 분석하여 법적 위험을 탐지하는 시스템으로, 12개의 AI 모듈이 순차적으로 실행됩니다.

### 핵심 설계 원칙

1. **Hybrid AI**: 규칙 기반(Symbolic) + LLM 기반(Neural) 결합
2. **Self-Correcting**: 검색 품질을 평가하고 자동 보정
3. **Explainable**: 모든 판단에 대한 추론 과정 시각화
4. **Constitutional**: 노동법 원칙에 따른 검증 레이어

---

## 기술 스택 및 흐름도

```
[계약서 업로드]
       |
       v
+------------------+
|  1. PII Masking  |  개인정보 비식별화 (규칙 기반)
+------------------+
       |
       v
+------------------+
|   2. Chunking    |  텍스트 분할 (의미 단위)
+------------------+
       |
       v
+------------------+     +------------------+
|     3. HyDE      | --> |   Embedding      |  가상 문서 생성 후 벡터화
+------------------+     +------------------+
       |                        |
       v                        v
+------------------+     +------------------+
|     4. CRAG      | <-- |  Graph Vector DB |  검색 + 그래프 확장 + 품질 보정
+------------------+     +------------------+
       |
       v
+------------------+
|    5. RAPTOR     |  계층적 요약 트리 구축
+------------------+
       |
       v
+------------------+
|  6. Stress Test  |  수치 시뮬레이션 (Neuro-Symbolic)
+------------------+
       |
       v
+------------------+
|   7. Redlining   |  자동 수정 제안 생성
+------------------+
       |
       v
+------------------+
| 8. Constitutional|  노동법 원칙 검증
+------------------+
       |
       v
+------------------+
|    9. Judge      |  분석 결과 신뢰도 평가
+------------------+
       |
       v
+------------------+
| 10. Reasoning    |  추론 과정 그래프 생성
+------------------+
       |
       v
+------------------+
|    11. DSPy      |  피드백 기록 (자가 진화)
+------------------+
       |
       v
[최종 분석 결과]
```

---

## 각 모듈 상세 설명

### 1. PII Masking (개인정보 비식별화)

**파일**: `backend/app/ai/pii_masking.py`

#### 작동 원리

정규표현식 기반으로 개인정보를 탐지하고 가역적으로 마스킹합니다.

```python
# 탐지 패턴 예시
PATTERNS = {
    "주민등록번호": r"\d{6}-[1-4]\d{6}",
    "전화번호": r"01[0-9]-\d{3,4}-\d{4}",
    "이메일": r"[\w.-]+@[\w.-]+\.\w+",
    "계좌번호": r"\d{3}-\d{2,6}-\d{2,6}",
}
```

#### 입력/출력 예시

```
입력: "홍길동(010-1234-5678)의 계좌 123-456-789로 입금"
출력: "[이름_1]([전화번호_1])의 계좌 [계좌번호_1]로 입금"
```

#### 장점

- API 비용 절감 (LLM 미사용)
- 개인정보보호법 준수
- 복원 가능 (분석 후 원본 복구)

#### 왜 필요한가?

1. 근로계약서에는 민감한 개인정보가 포함됨
2. LLM에 원본 데이터 전송 시 보안 위험
3. 마스킹된 데이터로도 법적 분석은 동일하게 가능

---

### 2. HyDE (Hypothetical Document Embeddings)

**파일**: `backend/app/ai/hyde.py`

**논문**: "Precise Zero-Shot Dense Retrieval without Relevance Labels" (Gao et al., 2022)

#### 작동 원리

사용자 쿼리만으로는 관련 문서를 정확히 검색하기 어렵습니다. HyDE는 LLM을 사용하여 "이상적인 답변 문서"를 먼저 생성하고, 이를 임베딩하여 검색합니다.

```
[기존 방식]
쿼리: "최저임금 위반"
  -> 쿼리 임베딩
  -> 유사 문서 검색

[HyDE 방식]
쿼리: "최저임금 위반"
  -> LLM이 가상 문서 생성:
     "근로기준법 제6조에 따르면 최저임금은 시간당 9,860원이며,
      이에 미달하는 임금 지급은 법 위반에 해당한다..."
  -> 가상 문서 임베딩
  -> 유사 문서 검색 (더 정확한 결과)
```

#### 입력/출력 예시

```python
# 입력
query = "이 계약서에서 위약금 조항이 적법한가요?"

# HyDE 생성 결과
hypothetical_document = """
근로기준법 제20조에 따르면, 사용자는 근로계약 불이행에 대한
위약금 또는 손해배상액을 예정하는 계약을 체결하지 못한다.
따라서 '계약기간 미준수 시 위약금 500만원' 같은 조항은
무효이며, 실제 발생한 손해만 배상 청구할 수 있다.
"""
```

#### 장점

1. **검색 정확도 향상**: 쿼리와 문서 간의 의미적 갭(semantic gap) 해소
2. **Zero-shot**: 추가 학습 데이터 없이 바로 적용 가능
3. **도메인 특화**: 법률 도메인에 맞는 가상 문서 생성

#### 기술적 구현

```python
class HyDEGenerator:
    def generate(self, query: str, prompt_type: str) -> HyDEResult:
        # 1. 쿼리 복잡도 분석
        complexity = self._analyze_complexity(query)

        # 2. 전략 선택 (단순/다중/계층적)
        strategy = self._select_strategy(complexity)

        # 3. 가상 문서 생성
        if strategy == GenerationStrategy.MULTI_PERSPECTIVE:
            # 다양한 관점에서 여러 문서 생성
            docs = self._generate_multi_perspective(query)

        # 4. 주 문서 선택 (가장 관련성 높은 것)
        primary = self._select_primary(docs)

        return HyDEResult(
            hypothetical_documents=docs,
            primary_document=primary,
            strategy_used=strategy
        )
```

---

### 3. CRAG (Corrective Retrieval-Augmented Generation)

**파일**: `backend/app/ai/crag.py`

**논문**: "Corrective Retrieval Augmented Generation" (Yan et al., 2024)

#### 작동 원리

기존 RAG는 검색된 문서를 그대로 사용합니다. CRAG는 검색 품질을 평가하고, 품질이 낮으면 **그래프 확장** 또는 **쿼리 재작성**으로 자동 보정합니다.

```
[기존 RAG]
쿼리 -> 검색 -> 문서 -> LLM -> 답변

[CRAG]
쿼리 -> 검색 -> 품질 평가 -> [분기]
                              |
                   +----------+----------+
                   |          |          |
               CORRECT    AMBIGUOUS   INCORRECT
                   |          |          |
                그대로    그래프확장   쿼리재작성
                사용      + 보강       후 재검색
                   |          |          |
                   +----------+----------+
                              |
                           LLM -> 답변
```

#### 품질 평가 기준

```python
class RetrievalQuality(Enum):
    CORRECT = "correct"      # 신뢰도 > 0.8
    INCORRECT = "incorrect"  # 신뢰도 < 0.3
    AMBIGUOUS = "ambiguous"  # 그 외
```

#### 그래프 확장 (Graph Expansion)

품질이 `AMBIGUOUS`일 때, Knowledge Graph를 활용하여 관련 문서를 추가로 검색합니다.

```python
def _expand_with_graph(self, docs, max_hops=2):
    """
    그래프를 따라 관련 노드 탐색

    예시:
    [최저임금] --관련법령--> [근로기준법 제6조]
                              |
                         --참조판례--> [대법원 2020다12345]
    """
    expanded = []
    for doc in docs:
        # 1. 문서에서 엔티티 추출
        entities = self._extract_entities(doc)

        # 2. 각 엔티티의 연결 노드 탐색
        for entity in entities:
            neighbors = self.graph.get_neighbors(entity, max_hops)
            expanded.extend(neighbors)

    return expanded
```

#### 입력/출력 예시

```python
# 입력
query = "포괄임금제가 적법한 경우는?"
initial_docs = [...검색된 문서...]

# CRAG 처리 과정
1. 품질 평가: AMBIGUOUS (관련성 점수 0.5)
2. 그래프 확장:
   - "포괄임금제" -> "대법원 2019다123" (판례)
   - "포괄임금제" -> "고용노동부 해석례" (유권해석)
3. 쿼리 재작성: "포괄임금제 적법 요건 판례"
4. 재검색 후 품질: CORRECT (0.85)

# 출력
CRAGResult(
    quality=CORRECT,
    correction_iterations=2,
    all_docs=[...원본 + 확장 문서...],
    confidence_score=0.85
)
```

#### 장점

1. **자가 보정**: 검색 실패를 자동으로 감지하고 수정
2. **Knowledge Graph 활용**: 법률 도메인의 관계 정보 활용
3. **신뢰도 제공**: 검색 결과의 품질을 수치로 제공

---

### 4. RAPTOR (Recursive Abstractive Processing for Tree-Organized Retrieval)

**파일**: `backend/app/ai/raptor.py`

**논문**: "RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval" (Sarthi et al., 2024)

#### 작동 원리

긴 문서를 계층적 요약 트리로 구조화합니다. 상위 노드는 하위 노드들의 요약을 담고 있어, 질문에 따라 적절한 추상화 수준에서 정보를 검색할 수 있습니다.

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
 "제3조       "제4조        "제5조        "제6조
  임금..."     수당..."      근로..."      휴일..."
```

#### 구현 과정

```python
def build_tree(self, chunks: List[str]) -> RAPTORTree:
    # 1. 리프 노드 생성 (원본 청크)
    leaf_nodes = [Node(text=chunk) for chunk in chunks]

    # 2. 유사한 노드 클러스터링
    clusters = self._cluster_nodes(leaf_nodes)

    # 3. 각 클러스터 요약 생성 (부모 노드)
    parent_nodes = []
    for cluster in clusters:
        summary = self._summarize_cluster(cluster)
        parent = Node(text=summary, children=cluster)
        parent_nodes.append(parent)

    # 4. 재귀적으로 상위 레벨 구축
    if len(parent_nodes) > 1:
        return self.build_tree(parent_nodes)

    return RAPTORTree(root=parent_nodes[0])
```

#### 입력/출력 예시

```python
# 입력: 10페이지 계약서 (약 20개 청크)

# 출력: RAPTOR 트리
RAPTORTree(
    nodes={
        "root_1": Node(
            text="본 근로계약서는 정규직 채용 계약으로,
                  임금 조건에서 최저임금 미달 위험이 있으며...",
            level=0
        ),
        "node_2": Node(
            text="임금 관련 조항 요약: 기본급 180만원,
                  포괄임금제 적용...",
            level=1,
            children=["chunk_1", "chunk_2", "chunk_3"]
        ),
        ...
    },
    root_ids=["root_1"]
)
```

#### 장점

1. **다중 해상도 검색**: 세부 정보 vs 전체 맥락 모두 검색 가능
2. **효율적 요약**: 전체 문서를 읽지 않고도 핵심 파악
3. **계층적 구조**: 법률 문서의 조항 구조와 잘 매핑됨

---

### 5. Legal Stress Test (법적 수치 시뮬레이션)

**파일**: `backend/app/ai/legal_stress_test.py`

#### 작동 원리

**Neuro-Symbolic AI** 접근법을 사용합니다:
- **Neural**: LLM으로 계약서에서 수치 정보 추출
- **Symbolic**: 추출된 수치를 규칙 기반으로 계산/검증

```python
class LegalStressTest:
    def run(self, contract_text: str) -> StressTestResult:
        # 1. [Neural] 수치 정보 추출
        wage_info = self._extract_wage_info(contract_text)
        work_time = self._extract_work_time(contract_text)

        # 2. [Symbolic] 법적 기준 검증
        violations = []

        # 최저임금 검증
        if wage_info.hourly_wage < MINIMUM_WAGE_2024:
            violations.append(ViolationDetail(
                type="최저임금 미달",
                current_value=wage_info.hourly_wage,
                legal_standard=MINIMUM_WAGE_2024,
                shortage=MINIMUM_WAGE_2024 - wage_info.hourly_wage
            ))

        # 연장근로수당 검증
        if work_time.overtime_hours > 0:
            expected_overtime_pay = self._calc_overtime(wage_info, work_time)
            if wage_info.overtime_pay < expected_overtime_pay:
                violations.append(...)

        # 3. 체불 예상액 계산
        total_underpayment = sum(v.shortage for v in violations)

        return StressTestResult(
            violations=violations,
            total_underpayment=total_underpayment,
            annual_underpayment=total_underpayment * 12
        )
```

#### 검증 항목

| 항목 | 법적 기준 | 검증 내용 |
|------|----------|----------|
| 최저임금 | 근로기준법 제6조 | 시급 >= 9,860원 (2024) |
| 연장근로수당 | 근로기준법 제56조 | 통상임금의 1.5배 |
| 야간근로수당 | 근로기준법 제56조 | 통상임금의 1.5배 (22시~06시) |
| 주휴수당 | 근로기준법 제55조 | 주 15시간 이상 시 유급 휴일 |
| 연차휴가 | 근로기준법 제60조 | 1년 근속 시 15일 |

#### 입력/출력 예시

```python
# 입력
contract_text = """
제3조(임금)
월 기본급: 180만원 (포괄임금제 적용, 연장근로 포함)
근무시간: 주 52시간 (기본 40시간 + 연장 12시간)
"""

# 출력
StressTestResult(
    violations=[
        ViolationDetail(
            type="최저임금 미달",
            description="시급 8,653원으로 최저임금(9,860원) 미달",
            current_value=8653,
            legal_standard=9860,
            monthly_shortage=251,660,
            annual_shortage=3,019,920,
            legal_basis="근로기준법 제6조"
        ),
        ViolationDetail(
            type="연장근로수당 미지급",
            description="연장근로 12시간에 대한 가산수당 미포함",
            monthly_shortage=187,200,
            annual_shortage=2,246,400,
            legal_basis="근로기준법 제56조"
        )
    ],
    total_underpayment=438,860,
    annual_underpayment=5,266,320,
    risk_level=RiskLevel.HIGH
)
```

#### 장점

1. **정량적 분석**: "위험하다"가 아닌 "연 526만원 체불 예상" 같은 구체적 수치
2. **법적 근거 명시**: 모든 판단에 법조항 명시
3. **하이브리드 AI**: LLM의 이해력 + 규칙의 정확성 결합

---

### 6. Generative Redlining (자동 수정 제안)

**파일**: `backend/app/ai/redlining.py`

#### 작동 원리

위험 조항을 탐지하고, 법적으로 안전한 대안 문구를 생성합니다.

```python
class GenerativeRedlining:
    # 위험 패턴 정의
    RISK_PATTERNS = {
        "위약금": {
            "pattern": r"(위약금|손해배상).*\d+",
            "severity": "High",
            "suggestion": "근로계약 불이행에 대한 위약금 예정 금지",
            "legal_basis": "근로기준법 제20조"
        },
        "포괄임금": {
            "pattern": r"포괄\s*(임금|급여)",
            "severity": "High",
            "suggestion": "기본급과 각종 수당을 별도 명시",
            "legal_basis": "근로기준법 제56조"
        }
    }

    def redline(self, text: str) -> RedlineResult:
        changes = []

        # 1. 규칙 기반 탐지
        for name, pattern in self.RISK_PATTERNS.items():
            matches = re.finditer(pattern["pattern"], text)
            for match in matches:
                changes.append(RedlineChange(
                    original_text=match.group(),
                    revised_text=f"[수정 필요: {pattern['suggestion']}]",
                    reason=f"{name} 위험 패턴 탐지",
                    legal_basis=pattern["legal_basis"]
                ))

        # 2. LLM 기반 심층 분석 (규칙으로 못 찾은 경우)
        if not changes:
            changes = self._llm_analyze(text)

        return RedlineResult(changes=changes)
```

#### 입력/출력 예시

```python
# 입력
text = "계약기간 미준수 시 위약금 500만원을 배상한다"

# 출력
RedlineChange(
    change_type=ChangeType.MODIFY,
    original_text="위약금 500만원",
    revised_text="[수정 필요: 근로계약 불이행에 대한 위약금 예정 금지. 실손해 배상만 가능]",
    reason="위약금 위험 패턴 탐지",
    legal_basis="근로기준법 제20조",
    severity="High"
)
```

#### 장점

1. **Diff 형식 출력**: 변경 전/후를 명확히 비교
2. **법적 근거 제공**: 왜 수정해야 하는지 설명
3. **하이브리드 탐지**: 규칙(빠름) + LLM(정교함) 결합

---

### 7. Constitutional AI (노동법 원칙 검증)

**파일**: `backend/app/ai/constitutional_ai.py`

**논문**: "Constitutional AI: Harmlessness from AI Feedback" (Anthropic, 2022)

#### 작동 원리

Anthropic의 Constitutional AI를 노동법 도메인에 적용합니다. AI의 분석 결과가 노동법의 기본 원칙을 위반하지 않는지 검증합니다.

```python
class ConstitutionalPrinciple(Enum):
    HUMAN_DIGNITY = "근로조건은 인간의 존엄성을 보장해야 한다"
    WORKER_PROTECTION = "해석이 모호할 때는 근로자에게 유리하게 해석한다"
    MINIMUM_STANDARD = "근로기준법은 최저 기준이며, 이에 미달하는 조건은 무효다"
    EQUALITY = "동일 가치 노동에 대해서는 동일 임금을 지급해야 한다"
    SAFETY = "근로자의 안전과 건강을 위협하는 조건은 허용되지 않는다"
    TRANSPARENCY = "근로조건은 명확하게 서면으로 명시되어야 한다"
```

#### 검증 프로세스

```
[AI 분석 결과]
      |
      v
+---------------------+
| 원칙 1: 인간 존엄성  | --> 위반 여부 판단
+---------------------+
      |
      v
+---------------------+
| 원칙 2: 근로자 보호  | --> 위반 여부 판단
+---------------------+
      |
      v
      ...
      |
      v
[수정된 분석 결과]
```

#### 입력/출력 예시

```python
# 입력 (AI의 초기 분석)
analysis = "이 계약서의 포괄임금제 조항은 회사의 인건비 절감에 효과적입니다"

# Constitutional AI 검토 결과
ConstitutionalReview(
    original_response=analysis,
    critiques=[
        CritiqueResult(
            principle=WORKER_PROTECTION,
            violation_detected=True,
            critique="분석이 사용자(회사) 관점에 치우쳐 있음.
                     포괄임금제는 근로자에게 불리할 수 있으므로
                     유효 요건 충족 여부를 먼저 검토해야 함",
            severity="high",
            suggestion="포괄임금제의 법적 유효 요건과 근로자
                       권리 침해 가능성을 함께 분석해야 합니다"
        )
    ],
    revised_response="이 계약서의 포괄임금제 조항은 유효 요건
                     (업무 특성, 연장근로 예측 가능성 등)을
                     충족하는지 검토가 필요하며, 실제 연장근로
                     시간에 따른 추가 수당 청구권이 있을 수 있습니다",
    is_constitutional=False
)
```

#### 장점

1. **편향 방지**: AI가 한쪽(사용자/근로자)에 치우치지 않도록 보정
2. **원칙 기반**: 명시적인 원칙에 따른 검증
3. **자기 수정**: 문제 발견 시 자동으로 응답 수정

---

### 8. LLM-as-a-Judge (분석 신뢰도 평가)

**파일**: `backend/app/ai/judge.py`

**논문**: "Judging LLM-as-a-Judge" (Zheng et al., 2023)

#### 작동 원리

별도의 LLM이 분석 결과의 품질을 다각도로 평가합니다.

```python
class JudgmentCategory(Enum):
    ACCURACY = "정확성"       # 법률 해석의 정확성
    CONSISTENCY = "일관성"    # 근거와 결론의 일관성
    COMPLETENESS = "완전성"   # 분석의 완전성
    RELEVANCE = "관련성"      # 질문/계약서와의 관련성
    LEGAL_BASIS = "법적 근거" # 법적 근거의 타당성
```

#### 평가 프로세스

```python
def evaluate(self, analysis: str, context: str) -> JudgmentResult:
    scores = []

    for category in JudgmentCategory:
        prompt = f"""
        다음 분석의 '{category.value}'를 0.0~1.0 점수로 평가하세요.

        [분석 내용]
        {analysis}

        [원본 계약서]
        {context}

        평가 기준:
        - 1.0: 매우 우수
        - 0.7: 적절함
        - 0.4: 개선 필요
        - 0.0: 부적절
        """

        score = self._get_score(prompt)
        scores.append(JudgmentScore(category=category, score=score))

    return JudgmentResult(
        scores=scores,
        overall_score=sum(s.score for s in scores) / len(scores),
        is_reliable=overall_score >= 0.7
    )
```

#### 입력/출력 예시

```python
# 입력
analysis = "이 계약서는 근로기준법 제20조 위반 소지가 있습니다"
context = "계약서 전문..."

# 출력
JudgmentResult(
    scores=[
        JudgmentScore(category=ACCURACY, score=0.85),
        JudgmentScore(category=CONSISTENCY, score=0.90),
        JudgmentScore(category=COMPLETENESS, score=0.70),
        JudgmentScore(category=RELEVANCE, score=0.95),
        JudgmentScore(category=LEGAL_BASIS, score=0.80)
    ],
    overall_score=0.84,
    confidence_level="High",
    is_reliable=True,
    verdict="분석 결과가 신뢰할 수 있으며, 법적 근거가 명확함",
    recommendations=[
        "완전성 향상을 위해 관련 판례 추가 검토 권장"
    ]
)
```

#### 장점

1. **자가 평가**: AI 분석의 품질을 AI가 검증
2. **다차원 평가**: 단일 점수가 아닌 여러 기준으로 평가
3. **사용자 신뢰**: 분석 결과의 신뢰도를 명시적으로 제공

---

### 9. Reasoning Trace (추론 시각화)

**파일**: `backend/app/ai/reasoning_trace.py`

#### 작동 원리

AI의 추론 과정을 그래프 형태로 시각화하여 **설명 가능한 AI(XAI)**를 구현합니다.

```python
class TraceNodeType(Enum):
    INPUT = "input"           # 입력 (계약서)
    EVIDENCE = "evidence"     # 근거 (법조항, 판례)
    REASONING = "reasoning"   # 추론 단계
    CONCLUSION = "conclusion" # 결론
```

#### 그래프 구조

```
[입력: 계약서]
      |
      +---> [근거1: 근로기준법 제20조]
      |           |
      |           v
      |     [추론: 위약금 조항 존재]
      |           |
      +---> [근거2: 대법원 판례] ----+
                  |                  |
                  v                  v
            [추론: 무효 판단]  [신뢰도: 0.9]
                  |
                  v
            [결론: 위약금 조항 무효]
```

#### 입력/출력 예시

```python
# 입력
contract_excerpt = "계약 해지 시 위약금 300만원..."
analysis_result = {"conclusion": "위약금 조항 무효"}
retrieved_docs = [{"source": "근로기준법 제20조", "text": "..."}]

# 출력
ReasoningTrace(
    nodes=[
        TraceNode(id="n1", type=INPUT, label="계약서",
                  content="계약 해지 시 위약금 300만원..."),
        TraceNode(id="n2", type=EVIDENCE, label="법적 근거",
                  content="근로기준법 제20조: 위약금 예정 금지"),
        TraceNode(id="n3", type=REASONING, label="위험 판단",
                  content="계약서에 위약금 조항이 존재함"),
        TraceNode(id="n4", type=CONCLUSION, label="결론",
                  content="해당 위약금 조항은 무효")
    ],
    edges=[
        TraceEdge(source="n1", target="n3", label="분석"),
        TraceEdge(source="n2", target="n3", label="근거"),
        TraceEdge(source="n3", target="n4", label="도출")
    ],
    overall_confidence=0.92
)
```

#### 장점

1. **투명성**: AI 판단의 근거를 명확히 제시
2. **디버깅**: 잘못된 판단의 원인 파악 용이
3. **사용자 이해**: 비전문가도 AI의 논리를 따라갈 수 있음

---

### 10. DSPy (Declarative Self-improving Python)

**파일**: `backend/app/ai/dspy_optimizer.py`

**프레임워크**: Stanford NLP의 DSPy

#### 작동 원리

프롬프트를 수동으로 작성하지 않고, 데이터 기반으로 자동 최적화합니다.

```python
class DSPyOptimizer:
    def __init__(self):
        self.examples = []  # 학습 데이터

    def record_feedback(self, query, response, feedback_type):
        """사용자 피드백 기록"""
        self.examples.append({
            "query": query,
            "response": response,
            "feedback": feedback_type  # positive/negative/neutral
        })

    def optimize_prompt(self):
        """
        피드백 데이터를 기반으로 프롬프트 자동 최적화

        예: "positive" 피드백을 받은 응답 패턴 학습
        """
        positive_examples = [e for e in self.examples
                           if e["feedback"] == "positive"]

        # DSPy의 자동 프롬프트 최적화
        optimized = dspy.BootstrapFewShot(
            max_bootstrapped_demos=4,
            metric=self._quality_metric
        ).compile(
            self.base_program,
            trainset=positive_examples
        )

        return optimized
```

#### 자가 진화 과정

```
[초기 상태]
프롬프트: "계약서를 분석하세요"
성능: 70%

       |
       | 피드백 수집
       v

[1000건 분석 후]
- 긍정 피드백: 650건
- 부정 피드백: 150건
- 중립: 200건

       |
       | DSPy 최적화
       v

[최적화 후]
프롬프트: "근로계약서를 분석하세요.
         최저임금, 근로시간, 위약금 조항에
         특히 주의하세요..."
성능: 85%
```

#### 장점

1. **자동 개선**: 수동 프롬프트 엔지니어링 불필요
2. **데이터 기반**: 실제 사용 패턴 학습
3. **지속적 진화**: 시간이 지날수록 성능 향상

---

## Graph Vector 활용

### Knowledge Graph 구조

DocScanner AI는 법률 도메인 특화 Knowledge Graph를 구축하여 사용합니다.

```
[엔티티 유형]
- 법조항 (Article)
- 판례 (Precedent)
- 해석례 (Interpretation)
- 법률 용어 (Term)

[관계 유형]
- 참조 (references)
- 개정 (amends)
- 해석 (interprets)
- 관련 (related_to)
```

### 그래프 + 벡터 하이브리드 검색

```python
class GraphVectorRetriever:
    def retrieve(self, query: str, top_k: int = 10):
        # 1. 벡터 검색 (의미적 유사도)
        vector_results = self.vector_db.search(
            embedding=self.embed(query),
            top_k=top_k
        )

        # 2. 그래프 확장 (관계 기반)
        expanded = []
        for doc in vector_results:
            # 관련 노드 탐색
            neighbors = self.graph.get_neighbors(
                doc.id,
                relation_types=["references", "interprets"],
                max_hops=2
            )
            expanded.extend(neighbors)

        # 3. 재랭킹 (벡터 점수 + 그래프 중심성)
        all_docs = vector_results + expanded
        reranked = self._rerank(all_docs, query)

        return reranked[:top_k]
```

### 활용 예시

```
쿼리: "포괄임금제 위법"

1. 벡터 검색 결과:
   - 근로기준법 제56조 (0.85)
   - 포괄임금제 해설 문서 (0.82)

2. 그래프 확장:
   - 근로기준법 제56조 --참조--> 대법원 2019다12345
   - 근로기준법 제56조 --해석--> 고용노동부 유권해석 2020-1234
   - 대법원 2019다12345 --관련--> 대법원 2021다5678

3. 최종 결과 (벡터 + 그래프 통합):
   - 근로기준법 제56조
   - 대법원 2019다12345 (그래프로 발견)
   - 포괄임금제 해설 문서
   - 고용노동부 유권해석 (그래프로 발견)
```

---

## 기술적 난이도 분석

### 높은 난이도 (High Complexity)

| 기술 | 난이도 | 이유 |
|------|--------|------|
| CRAG | ★★★★★ | 품질 평가 + 그래프 확장 + 재검색의 복합 로직 |
| RAPTOR | ★★★★☆ | 계층적 클러스터링 + 재귀적 요약 |
| Constitutional AI | ★★★★☆ | 다중 원칙 검증 + 자기 수정 메커니즘 |

### 중간 난이도 (Medium Complexity)

| 기술 | 난이도 | 이유 |
|------|--------|------|
| HyDE | ★★★☆☆ | 가상 문서 생성은 단순하나 전략 선택이 중요 |
| Stress Test | ★★★☆☆ | Neuro-Symbolic 결합의 복잡성 |
| Reasoning Trace | ★★★☆☆ | 그래프 구조화 + 신뢰도 계산 |

### 낮은 난이도 (Low Complexity)

| 기술 | 난이도 | 이유 |
|------|--------|------|
| PII Masking | ★★☆☆☆ | 정규표현식 기반 규칙 |
| Redlining | ★★☆☆☆ | 패턴 매칭 + LLM 호출 |
| Judge | ★★☆☆☆ | 단순 평가 프롬프트 |

### 기술적 도전 과제

1. **비동기 처리**: Celery 환경에서 asyncio 충돌 해결
   ```python
   # 해결책: ThreadPoolExecutor로 별도 스레드에서 실행
   def retrieve_sync(self, query):
       with ThreadPoolExecutor() as executor:
           future = executor.submit(self._async_retrieve, query)
           return future.result()
   ```

2. **Reasoning Model 호환성**: gpt-5-mini 등은 temperature 미지원
   ```python
   def _is_reasoning_model(self) -> bool:
       return any(kw in self.model.lower()
                  for kw in ["o1", "o3", "gpt-5"])
   ```

3. **그래프 + 벡터 통합**: 두 검색 결과의 점수 정규화
   ```python
   def _normalize_scores(self, vector_score, graph_centrality):
       # Min-Max 정규화 후 가중 평균
       return 0.7 * vector_score + 0.3 * graph_centrality
   ```

---

## 참고 문헌

1. Gao et al. (2022). "Precise Zero-Shot Dense Retrieval without Relevance Labels" - HyDE
2. Yan et al. (2024). "Corrective Retrieval Augmented Generation" - CRAG
3. Sarthi et al. (2024). "RAPTOR: Recursive Abstractive Processing" - RAPTOR
4. Anthropic (2022). "Constitutional AI: Harmlessness from AI Feedback"
5. Zheng et al. (2023). "Judging LLM-as-a-Judge"
6. Stanford NLP - DSPy Framework

---

*Last Updated: 2024-12*
