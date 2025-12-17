# Chat Agent Architecture V2

이 문서는 DocScanner.ai의 LangGraph 기반 채팅 에이전트 아키텍처를 상세히 설명합니다.

---

## 1. 개요

채팅 에이전트는 계약서 분석 결과를 기반으로 사용자의 질문에 실시간으로 답변하는 대화형 AI 시스템입니다.

### 1.1 핵심 특징

| 특징 | 설명 |
|------|------|
| LangGraph 기반 | 상태 머신 기반 워크플로우 관리 |
| SSE 스트리밍 | Server-Sent Events로 실시간 토큰 스트리밍 |
| 멀티턴 대화 | 대화 히스토리 유지 및 컨텍스트 이해 |
| 동적 도구 선택 | LLM 기반 쿼리 분석으로 필요한 도구만 호출 |
| 분석 컨텍스트 주입 | 계약서 분석 결과를 에이전트 컨텍스트에 포함 |

### 1.2 사용 모델

| 용도 | 모델 | 역할 |
|------|------|------|
| Query Analysis | gemini-2.0-flash-lite | 쿼리 분석 및 도구 선택 |
| Response Generation | gemini-2.0-flash-lite | 최종 답변 생성 (스트리밍) |

---

## 2. 시스템 아키텍처

```
                                    +------------------+
                                    |    Frontend      |
                                    |  (Next.js SSE)   |
                                    +--------+---------+
                                             |
                                             | POST /api/v1/agent/{id}/stream
                                             v
+-----------------------------------------------------------------------------------+
|                              FastAPI Backend                                       |
|  +-----------------------------------------------------------------------------+  |
|  |                           agent_chat.py                                      |  |
|  |  +-----------------------------------------------------------------------+  |  |
|  |  |  1. Contract Retrieval                                                |  |  |
|  |  |     - DB에서 계약서 조회                                               |  |  |
|  |  |     - extracted_text 추출                                              |  |  |
|  |  |     - analysis_result에서 분석 요약 빌드                                |  |  |
|  |  +-----------------------------------------------------------------------+  |  |
|  |  |  2. build_analysis_summary()                                          |  |  |
|  |  |     - 위험도, 체불액, 위험 조항 요약                                    |  |  |
|  |  |     - HIGH/MEDIUM/LOW 분류                                             |  |  |
|  |  +-----------------------------------------------------------------------+  |  |
|  |  |  3. SSE Event Generator                                               |  |  |
|  |  |     - agent.chat_stream() 호출                                         |  |  |
|  |  |     - StreamEvent를 SSE 형식으로 변환                                   |  |  |
|  |  +-----------------------------------------------------------------------+  |  |
|  +-----------------------------------------------------------------------------+  |
+-----------------------------------------------------------------------------------+
                                             |
                                             v
+-----------------------------------------------------------------------------------+
|                           langgraph_agent.py                                       |
|  +-----------------------------------------------------------------------------+  |
|  |                        ContractChatAgent                                     |  |
|  |  +-----------------------------------------------------------------------+  |  |
|  |  |  chat_stream() - 메인 스트리밍 함수                                    |  |  |
|  |  |                                                                       |  |  |
|  |  |  Step 1: Query Analysis (LLM)                                         |  |  |
|  |  |    - analyze_query_with_llm()                                         |  |  |
|  |  |    - 도구 선택 및 쿼리 최적화                                          |  |  |
|  |  |                                                                       |  |  |
|  |  |  Step 2: Tool Execution                                               |  |  |
|  |  |    - search_vector_db (Elasticsearch)                                 |  |  |
|  |  |    - search_graph_db (Neo4j)                                          |  |  |
|  |  |    - web_search (Tavily)                                              |  |  |
|  |  |                                                                       |  |  |
|  |  |  Step 3: Response Generation (Streaming)                              |  |  |
|  |  |    - 컨텍스트 구성                                                     |  |  |
|  |  |    - LLM 스트리밍 호출                                                 |  |  |
|  |  +-----------------------------------------------------------------------+  |  |
|  +-----------------------------------------------------------------------------+  |
+-----------------------------------------------------------------------------------+
```

---

## 3. 처리 흐름

### 3.1 전체 플로우

```
[사용자 질문]
      |
      v
+---------------------+
| 1. API 요청 수신     |  POST /api/v1/agent/{contract_id}/stream
+---------------------+
      |
      v
+---------------------+
| 2. 계약서 조회       |  DB에서 contract_text, analysis_result 로드
+---------------------+
      |
      v
+---------------------+
| 3. 분석 요약 생성    |  build_analysis_summary()
+---------------------+     - 전체 위험도
      |                     - 체불 예상액 (월/연)
      |                     - 위험 조항 목록 (HIGH/MEDIUM/LOW)
      v
+---------------------+
| 4. 쿼리 분석 (LLM)   |  analyze_query_with_llm()
+---------------------+     - 사용자 질문 분석
      |                     - 필요한 도구 결정
      |                     - 각 도구별 최적화 쿼리 생성
      v
+---------------------+
| 5. 도구 실행         |  선택된 도구만 병렬/순차 실행
+---------------------+
      |
      +---> [Vector DB] 법령/판례 검색
      |
      +---> [Graph DB] 위험 패턴 검색
      |
      +---> [Web Search] 신고 방법/기관 정보
      |
      v
+---------------------+
| 6. 컨텍스트 구성     |  도구 결과 + 계약서 + 분석 요약
+---------------------+
      |
      v
+---------------------+
| 7. 응답 생성 (스트림) |  LLM 토큰 스트리밍
+---------------------+
      |
      v
[SSE로 프론트엔드 전송]
```

### 3.2 SSE 이벤트 타입

| Event Type | 설명 | Data 예시 |
|------------|------|-----------|
| `step` | 처리 단계 알림 | `{"step": "analyzing", "message": "질문 분석 중..."}` |
| `tool` | 도구 실행 상태 | `{"tool": "search_vector_db", "status": "searching"}` |
| `token` | 응답 토큰 (스트리밍) | `{"content": "근로기준법"}` |
| `done` | 응답 완료 | `{"full_response": "전체 응답 텍스트"}` |
| `error` | 에러 발생 | `{"message": "에러 메시지"}` |

---

## 4. 컴포넌트 상세

### 4.1 AgentState (상태 정의)

```python
class AgentState(TypedDict):
    messages: Sequence[BaseMessage]   # 대화 히스토리
    contract_text: str                # 계약서 원문
    contract_id: int                  # 계약서 ID
    current_step: str                 # 현재 처리 단계
    tool_results: list                # 도구 실행 결과
    final_response: str               # 최종 응답
```

### 4.2 ToolPlan (도구 선택 계획)

```python
@dataclass
class ToolPlan:
    use_vector_db: bool = False       # Elasticsearch 사용 여부
    use_graph_db: bool = False        # Neo4j 사용 여부
    use_web_search: bool = False      # Tavily 웹 검색 사용 여부
    vector_query: str = ""            # Vector DB용 최적화 쿼리
    graph_clause_type: str = ""       # Graph DB 조항 유형
    graph_keywords: list = []         # Graph DB 키워드
    web_query: str = ""               # 웹 검색용 쿼리
    reasoning: str = ""               # 도구 선택 이유
```

### 4.3 StreamEvent (SSE 이벤트)

```python
@dataclass
class StreamEvent:
    event_type: str   # "step", "tool", "token", "done", "error"
    data: dict

    def to_sse(self) -> str:
        return f"data: {json.dumps({'type': self.event_type, **self.data})}\n\n"
```

---

## 5. 도구 (Tools)

### 5.1 search_vector_db

**용도**: 법령, 판례, 해석례 검색

**백엔드**: Elasticsearch

```python
@tool
async def search_vector_db(
    query: str,
    doc_type: Optional[str] = None,  # law, precedent, interpretation, manual
    limit: int = 5
) -> str
```

**검색 필드**:
- `text` (가중치 2배)
- `title`
- `keywords`

**반환 필드**: `text`, `source`, `doc_type`, `title`, `score`

### 5.2 search_graph_db

**용도**: 위험 패턴 및 관련 법률 구조 검색

**백엔드**: Neo4j

```python
@tool
async def search_graph_db(
    clause_type: str,     # 근로시간, 임금, 휴게시간, 위약금 등
    keywords: list[str]
) -> str
```

**검색 노드**:
- `RiskPattern`: 위험 패턴 정의
- `Document`: 관련 법령 문서
- `Category`: 조항 카테고리

**Cypher 쿼리 예시**:
```cypher
MATCH (r:RiskPattern)
WHERE any(kw IN $keywords WHERE r.name CONTAINS kw)
OPTIONAL MATCH (r)-[:RELATES_TO]->(d:Document)
RETURN r.name, r.explanation, r.riskLevel, collect(d.content)
```

### 5.3 web_search

**용도**: 신고 방법, 기관 정보, 최신 정책 검색

**백엔드**: Tavily API (또는 Fallback 정적 데이터)

```python
@tool
async def web_search(
    query: str,
    max_results: int = 3
) -> str
```

**검색 도메인 제한**:
- `moel.go.kr` (고용노동부)
- `law.go.kr` (법제처)
- `minwon.go.kr` (민원마당)
- `nlcy.go.kr` (노동권익센터)

**Fallback 리소스** (API 키 없을 때):
- 고용노동부 민원마당
- 국민신문고
- 노동권익 상담센터 (1350)

### 5.4 analyze_contract_clause

**용도**: 계약서 특정 조항 분석

```python
@tool
def analyze_contract_clause(
    clause_text: str,
    question: str
) -> str
```

---

## 6. LLM 기반 쿼리 분석

### 6.1 analyze_query_with_llm()

사용자 질문을 분석하여 어떤 도구를 사용할지 결정합니다.

**입력**:
- `user_message`: 사용자 질문
- `chat_history`: 이전 대화 (최근 3개)
- `analysis_summary`: 계약서 분석 요약

**분석 프롬프트 구조**:
```
## 사용 가능한 도구
1. vector_db: 법령, 판례, 해석례 검색
2. graph_db: 위험 패턴 및 관련 법률 구조 검색
3. web_search: 신고 방법, 기관 정보, 최신 정책 검색

## 현재 계약서 분석 결과
{analysis_summary}

## 이전 대화
{history_text}

## 사용자 질문
{user_message}

## 출력 형식 (JSON)
{
  "reasoning": "도구 선택 이유",
  "tools": {
    "vector_db": {"use": true/false, "query": "..."},
    "graph_db": {"use": true/false, "clause_type": "...", "keywords": [...]},
    "web_search": {"use": true/false, "query": "..."}
  }
}
```

### 6.2 도구 선택 규칙

| 질문 유형 | 선택 도구 |
|----------|----------|
| 법적 근거, 조항, 위반, 판례 | `vector_db` |
| 위험, 패턴, 문제점 | `graph_db` |
| 신고, 대응, 상담, 기관, 방법 | `web_search` |
| 일반 인사, 단순 질문 | `vector_db` (기본값) |

### 6.3 Fallback 로직

LLM 분석 실패 시 키워드 기반 휴리스틱 사용:

```python
# 키워드 매칭 예시
if any(kw in query for kw in ["법", "조항", "위반"]):
    plan.use_vector_db = True

if any(kw in query for kw in ["신고", "방법", "상담"]):
    plan.use_web_search = True

# 아무것도 매칭되지 않으면 vector_db 기본 사용
if not (plan.use_vector_db or plan.use_graph_db or plan.use_web_search):
    plan.use_vector_db = True
```

---

## 7. 분석 컨텍스트 주입

### 7.1 build_analysis_summary()

계약서 분석 결과를 에이전트 컨텍스트용 텍스트로 변환합니다.

**입력**: `analysis_result` (JSON)

**출력 구조**:
```
[전체 위험도] HIGH

[체불 예상액] 월 234,000원 / 연 2,808,000원

[발견된 위험 조항 5건]

* HIGH 위험 (2건):
  - [제7조] 위약금 조항
    사유: 근로자에게 불리한 위약금 예정...
    법적근거: 근로기준법 제20조
    수정제안: 위약금 조항 삭제...

* MEDIUM 위험 (2건):
  - [제3조] 연장근로 조항
    ...

* LOW 위험 (1건)

[분석 요약]
본 계약서는 전반적으로 근로기준법 위반 소지가 있는...
```

### 7.2 시스템 프롬프트

```python
SYSTEM_PROMPT = """당신은 한국 근로계약서 분석 전문 AI 어시스턴트입니다.

## 역할
- 사용자가 업로드한 계약서를 분석하고 질문에 답변합니다
- 노동법 관련 정보를 정확하게 제공합니다
- 위반 사항에 대한 대응/예방/신고 방법을 안내합니다

## 핵심 답변 원칙
1. 질문에만 집중: 사용자가 묻지 않은 내용은 답변하지 마세요
2. 구체적으로 안내: 어디서 어떻게 하는지 단계별로 설명하세요
3. 마크다운 형식: 깔끔하게 정리하되 불필요한 서론/결론은 생략

## 이 계약서 분석 결과
{analysis_context}

## 계약서 원문 (발췌)
{contract_context}
"""
```

---

## 8. LangGraph 워크플로우

### 8.1 그래프 구조

```
                 +-------------+
                 |   analyze   |  (Entry Point)
                 +------+------+
                        |
                        v
                  [should_use_tools]
                   /            \
                  /              \
           "tools"              "respond"
              |                    |
              v                    |
        +---------+                |
        |  tools  |                |
        +----+----+                |
             |                     |
             +----------+----------+
                        |
                        v
                  +---------+
                  | respond |
                  +----+----+
                       |
                       v
                     [END]
```

### 8.2 노드 함수

| 노드 | 함수 | 역할 |
|------|------|------|
| `analyze` | `query_analyzer()` | 쿼리 분석 및 상태 업데이트 |
| `tools` | `call_tools()` | 도구 실행 |
| `respond` | `generate_response()` | 최종 응답 생성 |

### 8.3 조건부 엣지

```python
async def should_use_tools(state: AgentState) -> Literal["tools", "respond"]:
    """도구 사용 여부 결정"""
    last_message = state["messages"][-1].content

    tool_keywords = ["법", "조항", "위반", "신고", "대응", "예방", ...]

    if any(kw in last_message for kw in tool_keywords):
        return "tools"
    return "respond"
```

---

## 9. 스트리밍 응답 생성

### 9.1 chat_stream() 메인 루프

```python
async def chat_stream(self, message, contract_text, ...):
    # Step 1: 분석 시작 알림
    yield StreamEvent("step", {"step": "analyzing", "message": "질문 분석 중..."})

    # Step 2: LLM 기반 도구 선택
    tool_plan = await analyze_query_with_llm(message, chat_history, analysis_summary)

    # Step 3: 선택된 도구 실행
    if tool_plan.use_vector_db:
        yield StreamEvent("tool", {"tool": "search_vector_db", "status": "searching"})
        result = await search_vector_db.ainvoke(...)
        yield StreamEvent("tool", {"tool": "search_vector_db", "status": "complete"})

    # Step 4: 응답 생성 시작
    yield StreamEvent("step", {"step": "generating", "message": "답변 생성 중..."})

    # Step 5: LLM 스트리밍
    async for chunk in self.llm.astream(llm_messages):
        yield StreamEvent("token", {"content": chunk.content})

    # Step 6: 완료
    yield StreamEvent("done", {"full_response": full_response})
```

### 9.2 프론트엔드 SSE 처리

```typescript
// Next.js 프론트엔드 예시
const eventSource = new EventSource(`/api/v1/agent/${contractId}/stream`);

eventSource.onmessage = (event) => {
  const data = JSON.parse(event.data);

  switch (data.type) {
    case "step":
      setStatus(data.message);
      break;
    case "tool":
      setToolStatus(data.tool, data.status);
      break;
    case "token":
      appendToResponse(data.content);
      break;
    case "done":
      setComplete(data.full_response);
      break;
    case "error":
      setError(data.message);
      break;
  }
};
```

---

## 10. API 엔드포인트

### 10.1 스트리밍 채팅

```
POST /api/v1/agent/{contract_id}/stream
```

**Request Body**:
```json
{
  "message": "사용자 질문",
  "conversation_id": "optional-uuid",
  "include_contract": true
}
```

**Response**: SSE Stream

### 10.2 히스토리 포함 스트리밍

```
POST /api/v1/agent/{contract_id}/stream/history
```

**Request Body**:
```json
{
  "message": "후속 질문",
  "history": [
    {"role": "user", "content": "이전 질문"},
    {"role": "assistant", "content": "이전 답변"}
  ],
  "include_contract": true
}
```

### 10.3 동기 채팅 (Fallback)

```
POST /api/v1/agent/{contract_id}
```

SSE를 지원하지 않는 클라이언트용 폴백 엔드포인트.

**Response**:
```json
{
  "answer": "전체 응답 텍스트",
  "tools_used": ["법령/판례 검색", "웹 검색"],
  "contract_id": 123
}
```

### 10.4 헬스 체크

```
GET /api/v1/agent/health
```

**Response**:
```json
{
  "status": "healthy",
  "agent": "ready"
}
```

---

## 11. 시퀀스 다이어그램

```
Frontend          API             Agent           Tools           LLM
   |               |                |               |               |
   |--POST /stream-|                |               |               |
   |               |--get_contract--|               |               |
   |               |<--contract-----|               |               |
   |               |                |               |               |
   |               |--build_summary-|               |               |
   |               |                |               |               |
   |<--SSE:step----|--chat_stream---|               |               |
   |  "analyzing"  |                |               |               |
   |               |                |--analyze_query|               |
   |               |                |               |--LLM query--->|
   |               |                |               |<--tool plan---|
   |               |                |               |               |
   |<--SSE:tool----|                |--vector_db--->|               |
   |  "searching"  |                |<--results-----|               |
   |<--SSE:tool----|                |               |               |
   |  "complete"   |                |               |               |
   |               |                |               |               |
   |<--SSE:step----|                |               |               |
   |  "generating" |                |               |               |
   |               |                |               |--stream------>|
   |<--SSE:token---|                |               |<--chunk-------|
   |<--SSE:token---|                |               |<--chunk-------|
   |<--SSE:token---|                |               |<--chunk-------|
   |               |                |               |               |
   |<--SSE:done----|                |               |               |
   |               |                |               |               |
```

---

## 12. 에러 처리

### 12.1 예외 처리 흐름

```python
try:
    async for event in agent.chat_stream(...):
        yield event.to_sse()
except Exception as e:
    error_event = StreamEvent("error", {"message": str(e)})
    yield error_event.to_sse()
```

### 12.2 도구 실패 처리

각 도구는 에러 발생 시 JSON 형식으로 에러 반환:

```python
except Exception as e:
    return json.dumps({"error": str(e), "results": []})
```

---

*마지막 업데이트: 2025-12-17*
