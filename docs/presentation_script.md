# AI Legal Guardian - 발표 대본

COSE361 인공지능 / 10분 발표 / 이성민

---

## Slide 0: Title (30초)

안녕하세요. 고려대학교 컴퓨터학과 23학번 이성민입니다.

오늘 발표 주제는 "AI Legal Guardian: 사회적 약자를 위한 계약서 검토 에이전트 시스템"입니다.

이 프로젝트는 수업에서 배운 **Rational Agent** 개념을 실제 법률 도메인에 적용한 것입니다. 법률 지식이 부족한 근로자들이 불공정한 계약으로부터 스스로를 보호할 수 있도록 돕는 AI 에이전트를 구현했습니다.

---

## Slide 1: Problem Statement (1분)

먼저 문제 상황을 살펴보겠습니다. 기존 계약서 검토 방식에는 크게 세 가지 한계가 있습니다.

**첫째, 정보 비대칭입니다.**
근로자는 불공정 조항을 식별할 법률 전문 지식이 부족합니다. 법률 상담은 비용이 많이 들고, 중요한 문제가 계약 체결 후에야 발견되는 경우가 많습니다.

**둘째, 복잡한 법적 기준입니다.**
근로기준법에는 11개 이상의 정량적 검증 항목이 있고, 2025년 최저임금 10,030원처럼 매년 업데이트됩니다. 또한 조항 간 의존성으로 전체적 분석이 필요합니다.

**셋째, 기존 AI의 한계입니다.**
정규식 기반 시스템은 의미적 위반을 놓치고, 순수 LLM은 법률 인용을 hallucination합니다. 수업에서 배운 **Uncertainty** 개념처럼, 현실 세계는 불확실하고 LLM의 출력도 확률적입니다. 무엇보다 검증 가능한 추론 과정이 없습니다.

---

## Slide 2: Solution Overview (40초)

저희 DocScanner AI는 이러한 문제를 해결하기 위해 수업에서 배운 **MDP 프레임워크**를 적용했습니다.

에이전트의 **State**는 현재 분석 중인 계약서 조항, **Action**은 검색/분석/응답 생성, **Reward**는 법적 정확성과 윤리적 원칙 준수입니다. 에이전트의 목표는 **"법적 안전"이라는 Utility를 최대화**하는 것입니다.

사용자가 계약서를 업로드하면, 12단계의 AI 파이프라인을 통해 분석하고, 지식 그래프를 기반으로 법률 근거를 검증하여 최종 보고서를 생성합니다.

---

## Slide 3: MUVERA Embedding - Perception I (1분 10초)

이제 핵심 기술들을 설명드리겠습니다. 먼저 Perception, 즉 인지 단계입니다.

이 부분은 **Tutorial 2-1에서 다룬 Multi-modal Embedding (CLIP)** 개념의 확장입니다. CLIP이 이미지와 텍스트를 같은 벡터 공간에 매핑했듯이, 저희는 법률 문서와 사용자 질문을 같은 공간에 매핑합니다.

**MUVERA**는 Google Research에서 2024년 NeurIPS에 발표한 기법입니다.

기존 방식의 문제점을 설명드리면, 단일 벡터 임베딩은 긴 문서에서 세부 정보를 손실합니다. ColBERT 같은 다중 벡터 방식은 정확하지만 계산 비용이 높습니다.

**MUVERA의 해결책**은:
1. 청크를 문장 단위로 분리합니다
2. 각 문장을 KURE-v1이라는 한국어 법률 특화 모델로 임베딩합니다
3. **FDE(Fixed Dimensional Encoding)** 기법으로 압축합니다
4. 최종적으로 의미가 보존된 단일 1024차원 벡터를 출력합니다

**FDE가 핵심인데요**, 여러 문장 벡터를 하나로 합치는 방법입니다. 먼저 **SimHash**로 각 문장 벡터를 8개 파티션 중 하나에 할당합니다. SimHash는 유사한 벡터가 같은 파티션에 들어가도록 하는 Locality-Sensitive Hashing의 일종입니다. 그 다음 각 파티션 내에서 벡터들을 평균하고, 8개 파티션의 결과를 연결(concatenate)하면 고정 차원의 단일 벡터가 됩니다.

이렇게 하면 ColBERT처럼 **토큰별 세부 정보를 보존**하면서도, 단일 벡터처럼 **빠른 내적(dot product) 검색**이 가능합니다.

Tutorial에서 CLIP이 이미지-텍스트 쌍을 효율적으로 표현했듯이, MUVERA는 다중 문장을 효율적으로 표현합니다.

---

## Slide 4: Context-Aware Retrieval - Perception II (1분 10초)

두 번째 인지 기술은 맥락 인식 검색입니다. 이 부분은 수업에서 배운 **Search 이론, 특히 Informed Search**의 적용입니다.

**Knowledge Graph**는 법률 지식을 그래프로 구조화한 것입니다. 8가지 노드 유형(Document, Precedent, Law, RiskPattern 등)이 있고, "조항 -> 위험패턴 -> 판례 -> 법령" 같은 경로를 탐색합니다.

수업에서 배운 **A* Search**처럼, 단순히 모든 경로를 탐색하는 것이 아니라 **휴리스틱**을 사용합니다. 저희의 휴리스틱은 **HyDE(Hypothetical Document Embeddings)**입니다.

예를 들어 "연장근로 수당이 필요한가요?"라는 질문에, LLM이 먼저 "근로기준법 제56조에 따르면 연장근로에 대해 50% 가산 임금을 지급해야 합니다"라는 **가상 답변**을 생성합니다. 이 가상 답변이 목표(Goal)까지의 거리를 추정하는 휴리스틱 h(n) 역할을 합니다.

**CRAG(Corrective RAG)**는 검색된 문서의 품질을 8단계로 평가합니다. 수업에서 배운 **Probabilistic Reasoning**처럼, 각 문서의 관련성을 확률적으로 평가하고, 품질이 낮으면 교정 전략을 적용합니다.

---

## Slide 5: Neuro-Symbolic AI - Reasoning (1분 10초)

이제 Reasoning, 추론 단계입니다.

수업에서 **Bayesian Networks**와 **Machine Learning**의 한계를 배웠습니다. 확률 모델은 학습 데이터에 의존하고, 명시적인 규칙을 보장하지 못합니다. 특히 LLM은 계산을 자주 틀립니다.

반면 **Logic-based AI**는 정확하지만 자연어를 처리하지 못합니다.

저희의 **Neuro-Symbolic 접근법**은 두 패러다임을 결합합니다:
- **Neuro (LLM)**: Tutorial 1-1의 **KeyBERT**처럼 텍스트에서 핵심 엔티티를 추출합니다. 급여, 근무시간, 날짜 등을 구조화된 형태로 추출합니다.
- **Symbolic (Python)**: 추출된 값으로 시급, 연장근로 수당, 미지급액을 **논리적으로 정확히** 계산합니다.

예를 들어, LLM이 "월급 300만원, 하루 9시간, 주 5일"을 추출하면, Python이 "시급 = 300만원 / (9시간 x 22일) = 15,151원"을 계산합니다.

이를 통해 수업의 **Probabilistic 접근과 Logic 접근의 한계를 동시에 극복**했습니다.

---

## Slide 6: Constitutional AI - Action (1분)

마지막으로 Action 단계입니다. 이 부분이 수업 내용과 가장 밀접합니다.

수업에서 배운 **MDP**에서 에이전트는 **Reward를 최대화**하는 방향으로 행동합니다. 그런데 법률 도메인에서 Reward를 어떻게 정의할까요?

**Tutorial 2-2 (Agent Interaction with Reward Logic)**에서 다뤘듯이, 에이전트에게 상황에 맞는 보상을 주어 행동을 제어할 수 있습니다.

저희는 **6가지 헌법적 원칙**을 Reward Function으로 정의했습니다:
1. **인간 존엄성**: 근로조건은 인간의 존엄성을 존중해야 함
2. **근로자 보호**: 모호한 조항은 근로자에게 유리하게 해석
3. **최저 기준**: 법적 최저 기준 미달 조항은 무효
4. **평등**: 동일 노동 동일 임금
5. **안전**: 건강을 위협하는 근로조건 금지
6. **투명성**: 근로조건은 서면으로 명시

에이전트는 초안을 생성한 후, 이 원칙들에 비추어 **스스로 평가(Self-Critique)**하고 수정합니다. 이것은 수업에서 배운 **Adversarial Search의 Minimax**와 유사합니다. Critique가 응답을 공격하고, Agent가 방어(수정)하는 구조입니다.

결과적으로 **Rule-based Reward Logic**을 통해 윤리적으로 행동하는 **Rational Agent**를 만들었습니다.

---

## Slide 7: System Architecture (40초)

시스템 아키텍처입니다.

- **Frontend**: Next.js 15, React 19, Tailwind CSS
- **Backend**: FastAPI, LangGraph Agent, Celery Tasks
- **AI Pipeline**: GPT-5-mini/Gemini 2.5-flash, KURE-v1 임베딩, Tavily 검색
- **Data Layer**: PostgreSQL, Elasticsearch 벡터 DB, Neo4j 그래프 DB

전체가 하나의 **Rational Agent System**으로 동작하며, 각 컴포넌트가 Perception-Reasoning-Action 사이클을 구현합니다.

---

## Slide 8: Data & Implementation (40초)

구축한 지식 베이스 통계입니다.

총 **15,223개 청크**, **2,931개 문서**를 처리했습니다.

데이터 출처별로 보면:
- 판례: 969건 / 10,576 청크
- 고용노동부 해설: 1,827건 / 3,384 청크
- 법령 해석: 135건 / 589 청크
- 2025년 PDF 문서: 674 청크

이 데이터가 Knowledge Graph의 노드와 엣지가 되어 **Graph Search**의 기반이 됩니다.

---

## Slide 9: Live Demo (1분)

실제 시스템을 시연하겠습니다.

[데모 진행]
1. 계약서 PDF 업로드
2. AI가 실시간으로 분석하는 과정 (SSE 스트리밍)
3. 위험 조항 탐지 결과
4. 법률 근거와 판례가 포함된 조언 확인

---

## Slide 10: Discussion (50초)

향후 개선 방향입니다. 세 가지를 계획하고 있습니다.

**첫째, 하이브리드 스코어 퓨전입니다.**
기존에는 MUVERA 유사도만 사용했지만, Cross-encoder 리랭커의 정밀도와 그래프 권위 점수(PageRank + Citation Count)를 결합하려 합니다. 수업에서 배운 **Utility Theory**처럼 가중 합으로 최종 Utility를 계산합니다:
`Final = w1*MUVERA + w2*Reranker + w3*GraphAuth`

**둘째, 시스템 평가를 진행할 예정입니다.**
실제 근로계약서 샘플을 수집하여 위험 조항 탐지 정확도, 법률 근거 인용의 정확성, 사용자 만족도 등을 정량적으로 평가할 계획입니다.

**셋째, 채팅 에이전트 고도화입니다.**
현재는 계약서 분석 결과만 제공하지만, 노동청 신고 절차, 필요 서류, 관할 기관 안내 등 **신고 방식을 상세하게 안내**할 수 있도록 채팅 에이전트를 확장할 예정입니다. 분석에서 끝나지 않고 실제 행동까지 연결하는 것이 목표입니다.

---

## Slide 11: Conclusion (30초)

정리하겠습니다.

이 프로젝트는 수업에서 배운 AI 이론들을 실제 문제에 적용한 것입니다:

| 수업 내용 | 프로젝트 적용 |
|----------|-------------|
| MDP / Rational Agent | Constitutional AI의 Reward Logic |
| Informed Search (A*) | Knowledge Graph + HyDE 휴리스틱 |
| Probabilistic Reasoning | CRAG 품질 평가 |
| Tutorial 2-1 (Embedding) | MUVERA 다중 벡터 임베딩 |
| Tutorial 2-2 (Agent Reward) | 6가지 헌법적 원칙 기반 보상 |
| Tutorial 1-1 (KeyBERT) | Neuro-Symbolic 엔티티 추출 |

결과적으로 **법적 안전이라는 Utility를 최대화**하는 Rational Agent를 구현하여, 사회적 약자가 불공정 계약으로부터 스스로를 보호할 수 있도록 돕습니다.

감사합니다.

---

## 예상 질문 대비

**Q: MDP에서 State, Action, Reward가 구체적으로 뭔가요?**
A: State는 (현재 조항, 지금까지 분석 결과, 검색된 문서), Action은 (검색, 분석, 응답 생성, 수정), Reward는 6가지 헌법적 원칙 준수 여부입니다. Transition은 결정론적이 아니라 LLM 출력에 따라 확률적입니다.

**Q: A* Search와 HyDE의 연결이 정확히 뭔가요?**
A: A*에서 f(n) = g(n) + h(n)입니다. g(n)은 현재까지 비용, h(n)은 목표까지 추정 거리입니다. HyDE의 가상 답변이 h(n) 역할을 합니다. 가상 답변과 유사한 문서일수록 목표(정답)에 가깝다고 추정합니다.

**Q: Constitutional AI가 Minimax와 유사하다고 했는데?**
A: Minimax에서 상대방은 내 점수를 최소화하려 합니다. Constitutional AI에서 Critique는 응답의 약점을 찾아 공격하고, Agent는 이를 방어(수정)합니다. 이 과정이 수렴하면 robust한 응답이 됩니다.

**Q: Neuro-Symbolic에서 LLM이 엔티티 추출을 틀리면?**
A: Structured Output으로 JSON 스키마를 강제하고, 추출값의 범위 검증을 수행합니다. 비정상값 감지 시 재추출을 요청합니다. 이것도 일종의 Corrective loop입니다.

**Q: CRAG의 8단계 품질 평가가 Bayesian과 어떻게 연결되나요?**
A: 각 문서의 관련성을 P(relevant|document, query)로 추정합니다. 이 확률이 threshold 이하면 교정 전략을 적용합니다. 수업에서 배운 조건부 확률과 베이즈 정리의 응용입니다.
