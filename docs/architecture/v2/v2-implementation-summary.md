# DocScanner.ai V2 Implementation Summary

이 문서는 `feature/frontend-interactive-ui` 브랜치와 `develop` 브랜치 간의 주요 변경사항을 정리합니다.

---

## 1. 전체 UI 수정

### 1.1 디자인 시스템 통합

**파일**: `frontend/DESIGN_GUIDE.md`, `frontend/DESIGN_GUIDE_V2.md`

- Pretendard 폰트 기반 타이포그래피 표준화
- 통일된 색상 팔레트 도입:
  - Primary Green (Dark): `#3d5a47`
  - Page Background: `#f0f5f1`
  - Badge Colors: Success/Warning/Danger 톤 다운된 색상
- 모든 컴포넌트에 `tracking-tight` (-2.5% 자간) 적용
- Card Apple 스타일 적용 (`rounded-[16px]`, subtle shadows)

### 1.2 공유 레이아웃 시스템

**파일**: [frontend/src/app/(main)/layout.tsx](frontend/src/app/(main)/layout.tsx)

- 상단 Navigation Bar 공유
- Notification Panel (슬라이드 패널)
- Search Panel (글로벌 검색)
- Mobile Sidebar (반응형)
- Recent Revisions 동적 로딩 (API에서 최신 수정 버전 3개 가져오기)

### 1.3 로그인/회원가입 페이지 리디자인

**파일**: [frontend/src/app/login/page.tsx](frontend/src/app/login/page.tsx), [frontend/src/app/register/page.tsx](frontend/src/app/register/page.tsx)

- 인터랙티브 AI Orb 애니메이션 배경
- 현대적인 폼 디자인
- Glassmorphism 효과

### 1.4 분석 페이지 UI 개선

**파일**: [frontend/src/app/analysis/[id]/page.tsx](frontend/src/app/analysis/[id]/page.tsx)

- AI Avatar 컴포넌트 추가 (분석 상태 표시)
- ChatPanel 리디자인:
  - Liquid glass input effect (focus 시 scale-[1.02] 애니메이션)
  - Quick prompts 라운드 버튼 (`rounded-[14px]`)
  - ring-0으로 focus border 제거
- Risk clause 카드 스타일 개선:
  - 위험도별 아이콘 및 색상 분리
  - 확장 시 계층적 정보 표시 (위험 사유 > 수정 제안 > 법적 근거 > 참조 출처)
  - Markdown 볼드 강조 지원

### 1.5 계약서 수정 히스토리 페이지

**파일**: [frontend/src/app/(main)/contract-revisions/page.tsx](frontend/src/app/(main)/contract-revisions/page.tsx)

- 검색 드롭다운 (계약서 선택 시 날짜/버전 수 표시)
- 버전 히스토리 타임라인 UI
- 계약서별 구분을 위한 subtle left-border 색상 (CONTRACT_ACCENTS)
- 통계 카드: 총 계약서 / 수정된 계약서 (v2+ 존재하는 것만 카운트) / 총 버전

### 1.6 커스텀 차트 컴포넌트

**파일**: [frontend/src/components/charts/index.tsx](frontend/src/components/charts/index.tsx)

- DonutChart, BarChart, ProgressBar
- 디자인 가이드 색상 팔레트 적용

### 1.7 내용증명 작성 페이지 (신규)

**파일**: [frontend/src/app/(main)/certification/page.tsx](frontend/src/app/(main)/certification/page.tsx)

5단계 마법사 형태의 내용증명 작성 기능:

| 단계 | 이름 | 설명 |
|------|------|------|
| 1 | 피해 현황 | 피해 유형, 날짜, 금액, 상세 설명 입력 |
| 2 | 계약서 선택 | 분석된 계약서 중 관련 문서 선택 |
| 3 | 증거 수집 전략 | AI 기반 맞춤형 증거 수집 가이드 |
| 4 | 내용증명 작성 | 발신인/수신인 정보, 요구사항, 기한 설정 |
| 5 | 완료 | PDF 다운로드 |

**피해 유형 옵션**:
- 대금 미지급
- 하자/불량
- 이행 지연
- 계약 위반
- 사기/허위
- 기타

**UI 특징**:
- 질문별 카드 형태 인터페이스
- 분석 대기 중 법률 TIP 카드 순환 표시
- 증거 수집 전략 우선순위별 분류 (high/medium/low)
- 법적 효력 안내 정보 포함

---

## 2. 계약서 하이라이팅

### 2.1 텍스트 기반 하이라이팅

**파일**: [frontend/src/components/pdf-viewer/index.tsx](frontend/src/components/pdf-viewer/index.tsx)

- `HighlightClause` 타입 도입:
  ```typescript
  interface HighlightClause {
    id: string;
    text: string;         // 하이라이팅할 텍스트
    matchedText?: string; // 텍스트 기반 매칭용
    startIndex?: number;  // 위치 기반 매칭용
    endIndex?: number;
    severity: "high" | "medium" | "low";
  }
  ```

- 두 가지 하이라이팅 방식 지원:
  1. **텍스트 매칭**: `matchedText` 필드로 계약서 내 정확한 문자열 검색
  2. **인덱스 기반**: `startIndex`/`endIndex`로 위치 매핑

### 2.2 ViolationLocationMapper (Backend)

**파일**: [backend/app/ai/clause_analyzer.py:2328](backend/app/ai/clause_analyzer.py#L2328)

Gemini 2.5 Flash를 사용하여 위반 조항의 정확한 텍스트 위치 매핑:

```python
class ViolationLocationMapper:
    """
    분석된 위반 사항들의 정확한 텍스트 위치를 Gemini를 통해 찾고,
    suggestion을 참고하여 suggested_text를 생성하는 모듈.
    """
```

- `start_index`, `end_index`: 계약서 전문에서의 정확한 문자 위치
- `matched_text`: 하이라이팅할 실제 텍스트
- `suggested_text`: 법적으로 적합하게 수정된 대체 텍스트

### 2.3 분석 페이지 연동

Risk clause 클릭 시:
1. 해당 조항의 하이라이트로 PDF 뷰어 스크롤
2. `isActive` 상태로 선택된 조항 시각적 강조
3. 클릭 토글로 확장/축소

---

## 3. 계약서 수정 (Document Versioning)

### 3.1 문서 버전 관리 API

**파일**: [backend/app/api/v1/contracts.py:448-670](backend/app/api/v1/contracts.py#L448-L670)

Google Docs 스타일 버전 관리 시스템:

| Endpoint | Method | 설명 |
|----------|--------|------|
| `/{contract_id}/versions` | GET | 모든 버전 목록 조회 |
| `/{contract_id}/versions` | POST | 새 버전 생성 |
| `/{contract_id}/versions/{version_number}` | GET | 특정 버전 조회 |
| `/{contract_id}/versions/{version_number}/restore` | POST | 특정 버전으로 복원 |

### 3.2 DocumentVersion 모델

**파일**: [backend/app/models/contract.py](backend/app/models/contract.py)

```python
class DocumentVersion:
    id: int
    contract_id: int
    version_number: int
    content: str              # 수정된 계약서 전문
    changes: dict             # 변경 내역 JSON
    change_summary: str       # 변경 요약
    is_current: bool          # 현재 활성 버전 여부
    created_by: str           # 생성자 (user/ai/system)
    created_at: datetime
```

### 3.3 suggested_text 활용

분석 결과의 `suggested_text` 필드를 사용하여:
1. 사용자가 수정 제안 클릭 시 원본 텍스트 자동 대체
2. 새 버전 생성 (POST `/versions`)
3. 버전 히스토리에 기록

---

## 4. 계약서 수정 히스토리

### 4.1 프론트엔드 구현

**파일**: [frontend/src/app/(main)/contract-revisions/page.tsx](frontend/src/app/(main)/contract-revisions/page.tsx)

- 계약서별 버전 그룹화
- 타임라인 형태의 버전 히스토리 표시
- 버전 간 비교 기능 (diff 표시)
- 특정 버전으로 복원 기능

### 4.2 동적 데이터 로딩

```typescript
// 최근 수정 버전 로딩 (layout.tsx)
async function loadRecentRevisions() {
  const contracts = await contractsApi.list(0, 20);
  // 모든 계약서의 버전 정보 가져오기
  // 날짜순 정렬 후 상위 3개 반환
  // 제목 중복 시 날짜 suffix 추가
}
```

---

## 5. 전반적인 분석 로직 수정

### 5.1 Neuro-Symbolic 접근법 (핵심 변경)

**파일**: [backend/app/ai/clause_analyzer.py:139-438](backend/app/ai/clause_analyzer.py#L139-L438)

기존 LLM 단독 계산에서 Neuro-Symbolic 하이브리드로 전환:

```python
class NeuroSymbolicCalculator:
    """
    Neuro: LLM이 계약서에서 숫자/값 추출
    Symbolic: Python으로 정밀 체불액 계산
    """
    MINIMUM_WAGE_2025 = 10_030  # 시급
    OVERTIME_RATE = Decimal("1.5")  # 연장근로 50% 가산
```

**계산 흐름**:
1. LLM이 조항에서 `monthly_salary`, `daily_hours`, `break_minutes` 등 추출 (Neuro)
2. Python `Decimal` 타입으로 정밀 계산 (Symbolic)
3. 법정 임금 = 기본근로 + 주휴수당 + 연장근로수당
4. 체불액 = 법정 임금 - 실제 지급 임금

**이점**:
- LLM의 hallucination 방지 (계산 오류 제거)
- 재현 가능한 결과
- 법적 기준(2025년 최저임금 등) 정확 반영

### 5.2 동일 조항 위반 병합 (Violation Merging)

**파일**: [backend/app/ai/clause_analyzer.py:1916-2052](backend/app/ai/clause_analyzer.py#L1916-L2052)

같은 조항에서 여러 위반이 발견될 경우 하나로 통합:

```python
def _merge_same_clause_violations(self, violations):
    """
    동일 조항에서 발생한 여러 위반 사항을 하나로 병합
    1. clause_number로 그룹화
    2. 2개 이상이면 Gemini Flash로 병합
    3. 심각도는 가장 높은 것 사용
    """
```

**병합 내용**:
- `violation_type`: 통합된 위반 유형
- `description`: 모든 위반 사유를 포함한 종합 설명
- `legal_basis`: 모든 관련 법조문 나열
- `suggested_text`: 모든 문제를 해결하는 통합 수정안
- `crag_sources`: 모든 참조 출처 합침

### 5.3 하이브리드 법률 검색 (Hybrid Search)

**파일**: [backend/app/ai/clause_analyzer.py:1430-1500](backend/app/ai/clause_analyzer.py#L1430-L1500)

Vector DB + Graph DB 결합 검색:

```python
def _get_hybrid_legal_context(self, clause_type, clause_text):
    # 1. Vector DB (Elasticsearch): 의미적 유사도
    law_docs = self._search_by_category(query, "law")
    precedent_docs = self._search_by_category(query, "precedent")

    # 2. Graph DB (Neo4j): 구조적 관계
    graph_risk_patterns = self._search_graph_risk_patterns(clause_type)
    graph_docs = self._search_graph_documents_by_category(clause_type)

    # 3. 결과 병합 및 랭킹
```

### 5.4 Pipeline V2 아키텍처

**파일**: [backend/app/ai/pipeline.py](backend/app/ai/pipeline.py)

```
Pipeline Flow V2:
1. PII Masking (개인정보 비식별화)
2. LLM Clause Extraction (조항 분할 + 값 추출)
3. Clause-by-Clause CRAG Analysis (조항별 법률 검색 + LLM 위반 분석)
4. RAPTOR Indexing (계층적 요약)
5. Constitutional AI Review (메타 평가)
6. LLM-as-a-Judge (신뢰도 평가)
7. Reasoning Trace (추론 시각화)
```

### 5.5 develop vs feature 브랜치 상세 비교

아래 표는 `develop` 브랜치와 `feature/frontend-interactive-ui` 브랜치의 핵심 분석 로직 차이를 정리합니다.

#### 5.5.1 clause_analyzer.py 비교

| 항목 | develop 브랜치 | feature 브랜치 |
|------|---------------|----------------|
| **파일 크기** | ~1,400 lines | ~2,500 lines (+1,100 lines) |
| **NeuroSymbolicCalculator** | 없음 | 구현 완료 (lines 139-438) |
| **ViolationLocationMapper** | 없음 | 구현 완료 (lines 2328+) |
| **Violation Merging** | 없음 | `_merge_same_clause_violations` (lines 1916-2052) |
| **ClauseViolation 필드** | `original_text[:200]` 잘림 | `matched_text`, `suggested_text` 추가 |
| **텍스트 매칭** | 없음 | `_refine_text_matching`, `_fuzzy_find_text` |
| **하이브리드 검색** | Vector DB만 사용 | Vector DB + Graph DB (Neo4j) |

#### 5.5.2 config.py LLM 모델 비교

| 모델 역할 | develop 브랜치 | feature 브랜치 |
|-----------|---------------|----------------|
| **Retrieval** | gemini-2.0-flash-lite | gemini-2.5-flash-lite |
| **Reasoning** | gpt-4o | gpt-4.1 |
| **HyDE** | gpt-4o-mini | gpt-4o-mini |
| **Scan** | gemini-2.0-flash | gemini-2.5-flash-preview |
| **Clause Analyzer** | 없음 | gpt-4o (신규) |
| **CRAG** | 없음 | gpt-4o-mini (신규) |
| **RAPTOR** | 없음 | gemini-2.5-flash-preview (신규) |
| **Redliner** | 없음 | gpt-4o (신규) |
| **Location** | 없음 | gemini-2.5-flash-preview (신규) |
| **Judge** | 없음 | gpt-4o (신규) |
| **Constitutional** | 없음 | gpt-4o (신규) |
| **총 모델 수** | 4개 | 10개 |

#### 5.5.3 핵심 개선 사항

##### Neuro-Symbolic 연산

develop:

```python
# LLM에게 직접 계산 요청 -> 부정확한 결과
prompt = "이 계약서의 체불액을 계산해주세요"
result = llm.invoke(prompt)  # 종종 hallucination 발생
```

feature:

```python
# Neuro: LLM이 값 추출
extracted = llm.extract_values(clause)  # monthly_salary, hours, etc.

# Symbolic: Python이 정밀 계산
calculator = NeuroSymbolicCalculator()
result = calculator.compute_unpaid_wages(
    monthly_salary=Decimal(extracted["monthly_salary"]),
    daily_hours=Decimal(extracted["daily_hours"]),
    break_minutes=Decimal(extracted["break_minutes"])
)
# 2025년 최저임금 10,030원 기준 정확한 연산
```

##### 위반 사항 병합

develop:

```python
# 같은 조항에서 3개 위반 발견 시 -> 3개 별도 카드 표시
violations = [
    {"clause": "제5조", "issue": "최저임금 미달"},
    {"clause": "제5조", "issue": "연장근로수당 미지급"},
    {"clause": "제5조", "issue": "주휴수당 누락"}
]
# UI에 3개 카드가 따로 표시됨
```

feature:

```python
# 같은 조항 위반 -> 하나로 병합
merged = _merge_same_clause_violations(violations)
# 결과: 제5조에 대한 종합 분석 1개
# - violation_type: "임금 관련 다중 위반"
# - description: "최저임금 미달, 연장근로수당 미지급, 주휴수당 누락"
# - severity: 가장 높은 것 (high)
# - suggested_text: 모든 문제를 해결하는 통합 수정안
```

##### 텍스트 위치 매핑

develop:

```python
# 하이라이팅 불가능
violation = {
    "original_text": clause_text[:200]  # 잘린 텍스트
}
```

feature:

```python
# ViolationLocationMapper로 정확한 위치 찾기
mapper = ViolationLocationMapper()
result = mapper.map_locations(violation, full_contract_text)
# 결과:
# - start_index: 1523
# - end_index: 1789
# - matched_text: "정확히 매칭된 원문"
# - suggested_text: "법적으로 수정된 대체 텍스트"
```

---

## 6. LLM 모델 구성 수정

### 6.1 모델별 역할 분담

**파일**: [backend/app/core/config.py](backend/app/core/config.py), [.env.example](.env.example)

| 모델 | 환경변수 | 기본값 | 역할 |
|------|----------|--------|------|
| Retrieval | `LLM_RETRIEVAL_MODEL` | gemini-2.5-flash-lite | 빠른 검색/임베딩 |
| Clause Analyzer | `LLM_CLAUSE_ANALYZER_MODEL` | gpt-4o | 조항별 상세 분석 |
| Reasoning | `LLM_REASONING_MODEL` | gpt-4.1 | 종합 분석 |
| HyDE | `LLM_HYDE_MODEL` | gpt-4o-mini | 가설 문서 생성 |
| CRAG | `LLM_CRAG_MODEL` | gpt-4o-mini | Corrective RAG |
| RAPTOR | `LLM_RAPTOR_MODEL` | gemini-2.5-flash-preview | 계층적 요약 |
| Scan/OCR | `LLM_SCAN_MODEL` | gemini-2.5-flash-preview | 문서 스캔 |
| Location | `LLM_LOCATION_MODEL` | gemini-2.5-flash-preview | 위반 위치 매핑 |
| Judge | `LLM_JUDGE_MODEL` | gpt-4o | 신뢰도 평가 |
| Constitutional | `LLM_CONSTITUTIONAL_MODEL` | gpt-4o | 헌법적 검토 |

### 6.2 모델 선택 기준

- **OpenAI (gpt-4o, gpt-4.1)**: 복잡한 추론, 정확한 분석
- **Gemini (2.5-flash-lite, 2.5-flash-preview)**: 빠른 처리, 비용 효율적
  - Reasoning 모델(o1, o3)은 temperature 미지원으로 별도 처리

---

## 7. 기타 주요 변경사항

### 7.1 토큰 사용량 추적

**파일**: [backend/app/core/token_usage_tracker.py](backend/app/core/token_usage_tracker.py)

- 모든 LLM 호출에 대한 토큰 사용량 기록
- 모듈별/모델별 비용 분석
- `record_llm_usage()` 함수로 통합 관리

### 7.2 AI 아바타 컴포넌트

**파일**: [frontend/src/components/ai-avatar/index.tsx](frontend/src/components/ai-avatar/index.tsx)

- 분석 상태 시각화 (대기/분석중/완료)
- 인터랙티브 애니메이션
- 다양한 크기 지원 (AIAvatar, AIAvatarSmall)

### 7.3 Swirl Orb 애니메이션

**파일**: [frontend/src/components/swirl-orb/index.tsx](frontend/src/components/swirl-orb/index.tsx)

- 로그인/메인 페이지 배경 애니메이션
- WebGL 기반 고성능 렌더링

---

## 8. 커밋 히스토리

```
147f34c feat: add version management, document versioning, and UI improvements
f46e24d feat: add search dropdown and improve contract distinction in revisions page
60b741d feat: add gradient background to upload sidebar and dynamic recent revisions
0e01a18 feat: add liquid glass effect and interaction improvements to chat panel
32c12eb feat: refine analysis page design with unified color palette and interactive AI sphere
50e3b58 feat: add interactive AI avatar and chat panel redesign
5f5ab35 feat: add design guide and custom chart components for dashboard
e5d8970 feat: add shared navigation layout with notification and search panels
fb8e7e3 feat: redesign login/register pages with modern interactive UI
```
