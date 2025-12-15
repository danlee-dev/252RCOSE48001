제안서, 발표 자료, 포트폴리오의 **[3. 핵심 기술 및 차별성]** 챕터에 그대로 사용하실 수 있도록, 지금까지 논의된 모든 아키텍처, 최신 논문 기술, 기능적 차별점을 완벽하게 통합하여 정리해 드립니다.

심사위원(교수님)들이 중요하게 보는 **학술적 독창성(Novelty)**, **기술적 완성도(Completeness)**, **사회적 가치(Ethics)**를 모두 충족하는 **"Full Version 기술 명세서"**입니다.

---

# 📘 DocScanner.ai : 기술 명세서 (Final Version)
> **"Vision으로 읽고, Graph로 연결하며, Constitutional AI로 판단하는 자율 진화형 법률 에이전트"**

---

## 1. Core Architecture: Hybrid GraphRAG (지식의 구조화)
기존의 단순 검색(Vector Search) 한계를 넘어, 문맥과 법률적 인과관계를 동시에 추론하는 차세대 아키텍처입니다.

### 1.1. Dual-Knowledge Base (이중 지식 저장소)
* **Vector DB (Elasticsearch + MUVERA):** 문맥적 유사도 기반 검색.
    * **MUVERA 기술:** 긴 법률 문서를 문장 단위로 임베딩(KURE-v1) 후, SimHash로 압축하여 의미 손실을 최소화한 고속 검색 지원.
* **Graph DB (Neo4j):** 법률 지식 간의 논리적 연결 구조 저장.
    * **Ontology:** `Document` -[CITES]-> `Law`, `RiskPattern` -[HAS_CASE]-> `Precedent` 등 법적 인과관계 모델링.

### 1.2. Vector-to-Graph Traversal (추론 검색)
* **핵심 로직:** 벡터 검색으로 진입점(Anchor Node)을 찾고, 그래프 엣지를 타고 확장(Traversal)하여 숨겨진 정보를 추적.

* **작동 방식:** 사용자의 계약 조항과 유사한 판례를 찾은 뒤(Vector), 그 판례가 인용한 법령과 연결된 위험 패턴을 2-hop/3-hop으로 확장 검색(Graph)하여 "법적 근거"를 수집.

---

## 2. Advanced Retrieval: 검색 고도화 (최신 논문 적용)
사용자의 불명확한 질의와 긴 계약서의 문맥을 완벽히 이해하기 위한 학술적 접근입니다.

### 2.1. RAPTOR (Recursive Abstractive Processing for Tree-Organized Retrieval)
* **출처:** ICLR 2024 논문
* **배경:** 긴 계약서를 단순히 자르면(Chunking) 전체 맥락을 놓치는 문제 발생.
* **기술:** 문서를 **트리 구조(Tree Structure)**로 요약 및 인덱싱. (하위 조항 → 중간 요약 → 전체 요약)

* **활용:** "이 계약서가 전체적으로 근로자에게 불리한가?"와 같은 **거시적(Macro) 질문**에 대해 상위 요약 노드를 검색하여 답변.

### 2.2. HyDE (Hypothetical Document Embeddings)
* **배경:** "사장님이 맘대로 잘라요" 같은 구어체 질문과 "해고의 정당한 사유" 같은 법률 용어 사이의 **의미적 간극(Semantic Gap)** 존재.
* **기술:** 질문을 바로 검색하지 않고, LLM이 **"가상의 완벽한 법률 답변"**을 먼저 생성한 뒤, 이를 임베딩하여 검색.

* **효과:** 법률 비전문가의 저급한 질문으로도 전문적인 판례와 법령을 정확히 타격.

---

## 3. Dynamic Reasoning: 동적 추론 및 시뮬레이션
정적인 텍스트 분석을 넘어, 상황에 따라 전략을 수정하고 수치를 검증하는 엔진입니다.

### 3.1. Graph-Guided CRAG (Corrective RAG)
* **개념:** 검색된 지식의 품질을 평가하고, 불충분할 경우 그래프 DB를 통해 **지식을 스스로 보강(Self-Correction)**.
* **Dify Workflow:** 검색(Retrieval) → 평가(Evaluator) → (실패 시) 그래프 확장(Fallback) → 답변 생성.
* **효과:** 할루시네이션 최소화 및 검색 실패에 대한 자가 복구(Resilience) 구현.

### 3.2. Legal Stress Test (Neuro-Symbolic AI)
* **개념:** LLM의 추론 능력(Neuro)과 Python 코드의 계산 능력(Symbolic) 결합.
* **구현:** **Dify Code Interpreter** 노드 활용.
* **기능:** 계약서 내의 급여, 근무시간을 추출하여 최저임금 위반액, 주휴수당, 1년 근무 시 예상 체불 금액을 **코드로 정밀 시뮬레이션**.
* **효과:** "1년 뒤 약 120만 원의 체불이 예상됩니다"와 같은 구체적/수치적 위험 경고.

### 3.3. DSPy 기반 Dynamic Few-Shot (Self-Evolving)
* **출처:** Stanford NLP 논문 (DSPy)
* **개념:** 정적인 프롬프트 엔지니어링을 탈피, 데이터 기반으로 **프롬프트를 자동 최적화(Prompt Programming)**.
* **구현:** 사용자 피드백(좋아요/싫어요)이 누적될수록, 성공한 분석 사례를 프롬프트 예시(Few-Shot)로 동적으로 주입.
* **효과:** 사용자가 많아질수록 AI 변호사의 성능이 **자가 진화(Self-Evolving)**하는 시스템.

---

## 4. Input & Security: 입력 처리 및 보안 (현실적 문제 해결)
실제 서비스 운영을 고려한 기술적 디테일입니다.

### 4.1. Multimodal Parsing (Vision RAG)
* **기술:** VLM(Vision Language Model, 예: GPT-4o Vision)을 활용.
* **기능:** 계약서 이미지를 통째로 인식하여 OCR이 읽기 힘든 복잡한 **표(Table), 체크박스, 레이아웃**을 구조화된 마크다운 텍스트로 변환.
* **효과:** 임금 구성 항목 등 핵심 데이터의 파싱 정확도 획기적 개선.

### 4.2. PII Masking Pipeline (Privacy-Preserving)
* **기술:** LLM 전송 전(Pre-processing) 단계의 로컬 Python 스크립트.
* **기능:** 정규표현식 및 Presidio를 통해 주민번호, 이름, 주소, 전화번호를 자동 탐지하여 마스킹(`<MASKED>`) 처리.
* **효과:** 외부 LLM 사용에 따른 **개인정보 유출 원천 차단** (Enterprise급 보안).

---

## 5. Ethics & Reliability: 윤리 및 신뢰성 (사회적 가치)
AI가 법적/윤리적 선을 넘지 않도록 통제하는 최상위 안전장치입니다.

### 5.1. Constitutional AI (헌법적 AI)
* **배경:** Anthropic의 RLAIF(Reinforcement Learning from AI Feedback) 개념 적용.
* **핵심 기술:** 시스템 프롬프트 최상단에 **'근로기준법 헌법(Labor Law Constitution)'** 정의.
    * *원칙 1: 근로조건은 인간의 존엄성을 보장해야 한다.*
    * *원칙 2: 해석이 모호할 때는 '작성자 불이익(In dubio pro operario)' 원칙을 따른다.*
* **Critique Agent:** 답변 생성 전, 위 헌법에 위배되는지 **자기 비판(Critique) 및 수정(Revise)** 수행.
* **효과:** 법적 회색 지대에서도 사회적 합의에 부합하는 **윤리적 법률 조언** 제공.

### 5.2. LLM-as-a-Judge (신뢰도 평가)
* **기능:** 별도의 심판관(Judge) 에이전트가 분석 결과와 근거 자료의 일치성을 채점.
* **출력:** 사용자에게 답변과 함께 **신뢰도 점수(Confidence Score)**를 제공하여 맹신 방지.

---

## 6. UX/XAI: 설명 가능한 AI
사용자가 AI의 분석 결과를 직관적으로 이해하도록 돕습니다.

### 6.1. Generative Redlining (자동 수정 제안)
* **기능:** 독소 조항을 지적하는 데 그치지 않고, **법적으로 안전한 문장으로 재작성**하여 제공.
* **UI:** Git의 Diff View처럼 원본과 수정본을 대조(Red/Blue)하여 시각화.

### 6.2. Reasoning Trace (추론 과정 시각화)
* **기능:** AI가 결론에 도달한 경로를 지식 그래프로 시각화.
* **UI:** `[내 계약서 제5조] ─(유사)─ [대법원 판례] ─(인용)─ [근로기준법]`의 연결망을 사용자에게 제시.

---

### 📊 [Summary] 심사위원을 위한 기술 요약표 (Cheat Sheet)

| 구분 | 적용 기술 및 방법론 | 기대 효과 (차별점) |
| :--- | :--- | :--- |
| **Data Structure** | **RAPTOR** (ICLR 2024) | 트리 구조 요약을 통한 **거시적/미시적 문맥 동시 이해** |
| **Retrieval** | **HyDE** / **Hybrid GraphRAG** | 의미적 간극 해소 및 **법적 인과관계 추적** |
| **Reasoning** | **Graph-Guided CRAG** | 지식 그래프를 활용한 **검색 오류 자가 보정(Self-Correction)** |
| **Simulation** | **Neuro-Symbolic (Stress Test)** | Code Interpreter를 연동한 **수치 오류 없는 정밀 시뮬레이션** |
| **Optimization** | **DSPy / Dynamic Few-Shot** | 사용자 피드백을 학습하여 프롬프트가 **자가 진화(Self-Evolving)** |
| **Safety** | **Constitutional AI** | '근로기준법 헌법' 기반의 **윤리적 자기 검열(Critique)** |
| **Privacy** | **PII Masking** | 개인정보 비식별화를 통한 **Enterprise급 보안성 확보** |

이 기술 명세서는 **"최신 연구 트렌드"**와 **"현업 수준의 완성도"**를 모두 잡은 완벽한 구성입니다. 이대로 구현하고 발표하신다면 캡스톤 디자인 대회에서 압도적인 평가를 받으실 수 있습니다.
