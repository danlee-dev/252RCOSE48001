"""
LangGraph-based Contract Analysis Chat Agent

Features:
- Query decomposition (fast model)
- Tool-based retrieval (Vector DB, Graph DB, Web Search)
- Streaming response generation
- Real-time tool execution status
"""

import os
import json
import asyncio
from typing import (
    TypedDict,
    Annotated,
    Sequence,
    Literal,
    AsyncGenerator,
    Any,
    Optional,
)
from dataclasses import dataclass, field
from datetime import datetime

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    AIMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI

from app.core.config import settings


# ============ State Definition ============


class AgentState(TypedDict):
    """Agent state for LangGraph"""

    messages: Annotated[Sequence[BaseMessage], "Chat history"]
    contract_text: str
    contract_id: int
    current_step: str
    tool_results: list
    final_response: str


# ============ Tools Definition ============


@tool
async def search_vector_db(
    query: str, doc_type: Optional[str] = None, limit: int = 5
) -> str:
    """
    Search Vector DB (Elasticsearch) for legal documents.

    Args:
        query: Search query in Korean
        doc_type: Document type filter (law, precedent, interpretation, manual)
        limit: Maximum number of results

    Returns:
        JSON string of search results with source and text
    """
    try:
        from elasticsearch import Elasticsearch

        es_host = os.getenv("ELASTICSEARCH_URL", "http://localhost:9200")
        es = Elasticsearch([es_host])

        # Build query
        must_clauses = [
            {
                "multi_match": {
                    "query": query,
                    "fields": ["text^2", "title", "keywords"],
                    "type": "best_fields",
                }
            }
        ]

        filter_clauses = []
        if doc_type:
            filter_clauses.append({"term": {"doc_type": doc_type}})

        search_body = {
            "query": {
                "bool": {
                    "must": must_clauses,
                    "filter": filter_clauses if filter_clauses else None,
                }
            },
            "size": limit,
            "_source": ["text", "source", "doc_type", "title"],
        }

        # Remove None filter
        if not filter_clauses:
            del search_body["query"]["bool"]["filter"]

        response = es.search(index="docscanner_chunks", body=search_body)

        results = []
        for hit in response.get("hits", {}).get("hits", []):
            source = hit.get("_source", {})
            results.append(
                {
                    "source": source.get("source", ""),
                    "text": source.get("text", "")[:500],
                    "doc_type": source.get("doc_type", ""),
                    "score": hit.get("_score", 0),
                }
            )

        if not results:
            return json.dumps(
                {"message": "검색 결과가 없습니다.", "results": []}, ensure_ascii=False
            )

        return json.dumps({"results": results}, ensure_ascii=False)

    except Exception as e:
        return json.dumps({"error": str(e), "results": []}, ensure_ascii=False)


@tool
async def search_graph_db(clause_type: str, keywords: list[str]) -> str:
    """
    Search Graph DB (Neo4j) for risk patterns and related documents.

    Args:
        clause_type: Clause type (e.g., 근로시간, 임금, 휴게시간, 위약금)
        keywords: Keywords to search for

    Returns:
        JSON string of risk patterns and related legal documents
    """
    try:
        from neo4j import GraphDatabase

        uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        user = os.getenv("NEO4J_USER", "neo4j")
        password = os.getenv("NEO4J_PASSWORD", "")

        driver = GraphDatabase.driver(uri, auth=(user, password))
        results = []

        with driver.session() as session:
            # Search RiskPattern nodes
            query = """
            MATCH (r:RiskPattern)
            WHERE any(kw IN $keywords WHERE
                r.name CONTAINS kw OR
                any(t IN r.triggers WHERE t CONTAINS kw)
            )
            OPTIONAL MATCH (r)-[:RELATES_TO]->(d:Document)
            RETURN r.name AS pattern_name,
                   r.explanation AS explanation,
                   r.riskLevel AS risk_level,
                   collect(d.content)[0..2] AS related_docs
            LIMIT 3
            """

            result = session.run(query, keywords=keywords)
            for record in result:
                results.append(
                    {
                        "type": "risk_pattern",
                        "name": record["pattern_name"],
                        "explanation": record["explanation"],
                        "risk_level": record["risk_level"],
                        "related_docs": [d[:300] for d in record["related_docs"] if d],
                    }
                )

            # Search Category-related documents
            category_map = {
                "휴게시간": ["근로시간"],
                "근로시간": ["근로시간"],
                "임금": ["임금"],
                "위약금": ["기타"],
                "연차휴가": ["휴일휴가"],
            }
            categories = category_map.get(clause_type, ["기타"])

            doc_query = """
            MATCH (d:Document)-[:CATEGORIZED_AS]->(c:Category)
            WHERE c.name IN $categories
              AND any(kw IN $keywords WHERE d.content CONTAINS kw)
            RETURN d.content AS content, d.source AS source
            LIMIT 2
            """

            doc_result = session.run(
                doc_query, categories=categories, keywords=keywords
            )
            for record in doc_result:
                if record["content"]:
                    results.append(
                        {
                            "type": "document",
                            "source": record["source"],
                            "content": record["content"][:400],
                        }
                    )

        driver.close()

        if not results:
            return json.dumps(
                {"message": "관련 위험 패턴이 없습니다.", "results": []},
                ensure_ascii=False,
            )

        return json.dumps({"results": results}, ensure_ascii=False)

    except Exception as e:
        return json.dumps({"error": str(e), "results": []}, ensure_ascii=False)


@tool
async def web_search(query: str, max_results: int = 3) -> str:
    """
    Search the web for information about labor law, reporting methods, etc.

    Args:
        query: Search query in Korean
        max_results: Maximum number of results

    Returns:
        JSON string of search results
    """
    try:
        # Try Tavily first (if API key available)
        tavily_key = os.getenv("TAVILY_API_KEY")
        print(f">>> [web_search] TAVILY_API_KEY exists: {bool(tavily_key)}")

        if tavily_key:
            from tavily import TavilyClient

            client = TavilyClient(api_key=tavily_key)
            print(f">>> [web_search] Searching Tavily for: {query}")

            response = client.search(
                query=query,
                search_depth="basic",
                max_results=max_results,
                include_domains=[
                    "moel.go.kr",
                    "law.go.kr",
                    "minwon.go.kr",
                    "nlcy.go.kr",
                ],
            )
            print(
                f">>> [web_search] Tavily response received: {len(response.get('results', []))} results"
            )

            results = []
            for r in response.get("results", []):
                results.append(
                    {
                        "title": r.get("title", ""),
                        "url": r.get("url", ""),
                        "content": r.get("content", "")[:300],
                    }
                )

            return json.dumps(
                {"results": results, "source": "tavily"}, ensure_ascii=False
            )

        # Fallback: Return helpful static information
        print(">>> [web_search] No Tavily API key, using fallback resources")
        helpful_resources = [
            {
                "title": "고용노동부 민원마당",
                "url": "https://minwon.moel.go.kr",
                "content": "임금체불, 부당해고 등 노동관련 민원 신고 및 상담 가능",
            },
            {
                "title": "국민신문고",
                "url": "https://www.epeople.go.kr",
                "content": "정부 민원 통합 접수 시스템, 노동관련 민원 신고 가능",
            },
            {
                "title": "노동권익 상담센터",
                "url": "tel:1350",
                "content": "고용노동부 상담전화 1350, 근로기준법 위반 상담 및 신고 안내",
            },
        ]

        return json.dumps(
            {
                "results": helpful_resources,
                "source": "fallback",
                "note": "웹 검색 API가 설정되지 않아 기본 정보를 제공합니다.",
            },
            ensure_ascii=False,
        )

    except Exception as e:
        print(f">>> [web_search] ERROR: {e}")
        return json.dumps({"error": str(e), "results": []}, ensure_ascii=False)


@tool
def analyze_contract_clause(clause_text: str, question: str) -> str:
    """
    Analyze a specific clause from the contract based on the question.

    Args:
        clause_text: The clause text to analyze
        question: User's question about the clause

    Returns:
        Analysis result
    """
    # This tool provides structured analysis context
    return json.dumps(
        {
            "clause": clause_text[:500],
            "question": question,
            "analysis_prompt": "Based on the clause and question, provide legal analysis.",
        },
        ensure_ascii=False,
    )


# ============ LLM Setup ============


def get_fast_llm():
    """Get fast LLM for query decomposition and tool calls"""
    return ChatGoogleGenerativeAI(
        model="gemini-2.0-flash-lite",
        google_api_key=os.getenv("GEMINI_API_KEY"),
        temperature=0.1,
        streaming=True,
    )


def get_response_llm():
    """Get LLM for response generation"""
    return ChatGoogleGenerativeAI(
        model="gemini-2.0-flash-lite",
        google_api_key=os.getenv("GEMINI_API_KEY"),
        temperature=0.3,
        streaming=True,
    )


@dataclass
class ToolPlan:
    """Plan for which tools to use and with what queries"""

    use_vector_db: bool = False
    use_graph_db: bool = False
    use_web_search: bool = False
    vector_query: str = ""
    graph_clause_type: str = ""
    graph_keywords: list = field(default_factory=list)
    web_query: str = ""
    reasoning: str = ""


async def analyze_query_with_llm(
    user_message: str,
    chat_history: list = None,
    analysis_summary: str = "",
) -> ToolPlan:
    """
    Use LLM to analyze user query and decide which tools to use.

    Returns:
        ToolPlan with tool selection and optimized queries for each tool
    """
    llm = get_fast_llm()

    # Build context from chat history
    history_text = ""
    if chat_history:
        recent_messages = chat_history[-3:]
        for msg in recent_messages:
            role = "사용자" if msg.get("role") == "user" else "AI"
            history_text += f"{role}: {msg.get('content', '')[:150]}\n"

    prompt = f"""사용자 질문을 분석하여 필요한 도구를 결정하고, 각 도구에 맞는 최적화된 쿼리를 생성하세요.

## 사용 가능한 도구
1. **vector_db**: 법령, 판례, 해석례 검색 (근로기준법, 최저임금법 등 법적 근거 필요시)
2. **graph_db**: 위험 패턴 및 관련 법률 구조 검색 (특정 조항 유형의 위험성 분석시)
3. **web_search**: 신고 방법, 기관 정보, 최신 정책 검색 (실용적 정보, 연락처, 절차 안내시)

## 현재 계약서 분석 결과
{analysis_summary[:500] if analysis_summary else "분석 결과 없음"}

## 이전 대화
{history_text if history_text else "없음"}

## 사용자 질문
{user_message}

## 지시사항
JSON 형식으로 응답하세요. 반드시 아래 형식을 따르세요:

```json
{{
  "reasoning": "도구 선택 이유 (한 문장)",
  "tools": {{
    "vector_db": {{
      "use": true/false,
      "query": "법령/판례 검색용 최적화 쿼리"
    }},
    "graph_db": {{
      "use": true/false,
      "clause_type": "임금/근로시간/휴게시간/위약금/사회보험/연차휴가/기타 중 하나",
      "keywords": ["키워드1", "키워드2"]
    }},
    "web_search": {{
      "use": true/false,
      "query": "웹 검색용 최적화 쿼리 (신고방법, 기관정보 등)"
    }}
  }}
}}
```

주의사항:
- 최소 1개 도구는 반드시 선택
- 단순 인사나 일반 질문이면 vector_db만 선택
- 신고/대응/상담 관련이면 web_search 필수
- 특정 조항 위험성 질문이면 graph_db 포함
- 쿼리는 해당 도구에 최적화 (vector_db는 법률용어, web_search는 일상용어)

JSON:"""

    try:
        print(f">>> [analyze_query_with_llm] User message: {user_message[:50]}...")
        response = await llm.ainvoke(prompt)
        content = response.content.strip()

        # Extract JSON from response
        json_start = content.find("{")
        json_end = content.rfind("}") + 1
        if json_start != -1 and json_end > json_start:
            json_str = content[json_start:json_end]
            result = json.loads(json_str)

            tools = result.get("tools", {})
            plan = ToolPlan(
                use_vector_db=tools.get("vector_db", {}).get("use", False),
                use_graph_db=tools.get("graph_db", {}).get("use", False),
                use_web_search=tools.get("web_search", {}).get("use", False),
                vector_query=tools.get("vector_db", {}).get("query", user_message),
                graph_clause_type=tools.get("graph_db", {}).get("clause_type", "기타"),
                graph_keywords=tools.get("graph_db", {}).get("keywords", []),
                web_query=tools.get("web_search", {}).get("query", user_message),
                reasoning=result.get("reasoning", ""),
            )

            # Ensure at least one tool is selected
            if not (plan.use_vector_db or plan.use_graph_db or plan.use_web_search):
                plan.use_vector_db = True
                plan.vector_query = user_message

            print(f">>> [analyze_query_with_llm] Plan: vector={plan.use_vector_db}, graph={plan.use_graph_db}, web={plan.use_web_search}")
            print(f">>> [analyze_query_with_llm] Reasoning: {plan.reasoning}")
            return plan

    except Exception as e:
        print(f">>> [analyze_query_with_llm] Error: {e}, using fallback")

    # Fallback: simple keyword-based selection
    query_lower = user_message.lower()
    plan = ToolPlan()

    if any(kw in query_lower for kw in ["법", "조항", "위반", "기준", "판례", "해석"]):
        plan.use_vector_db = True
        plan.vector_query = user_message

    if any(kw in query_lower for kw in ["위험", "패턴", "문제"]):
        plan.use_graph_db = True
        clause_types = ["임금", "근로시간", "휴게시간", "위약금", "사회보험", "연차"]
        plan.graph_clause_type = next((ct for ct in clause_types if ct in query_lower), "기타")
        plan.graph_keywords = [w for w in query_lower.split() if len(w) > 1][:5]

    if any(kw in query_lower for kw in ["신고", "방법", "대응", "예방", "상담", "기관", "어디", "어떻게"]):
        plan.use_web_search = True
        plan.web_query = user_message

    # Default to vector search
    if not (plan.use_vector_db or plan.use_graph_db or plan.use_web_search):
        plan.use_vector_db = True
        plan.vector_query = user_message

    return plan


# ============ Agent Nodes ============

SYSTEM_PROMPT = """당신은 한국 근로계약서 분석 전문 AI 어시스턴트입니다.

## 역할
- 사용자가 업로드한 계약서를 분석하고 질문에 답변합니다
- 노동법 관련 정보를 정확하게 제공합니다
- 위반 사항에 대한 대응/예방/신고 방법을 안내합니다

## 핵심 답변 원칙
1. **질문에만 집중**: 사용자가 묻지 않은 내용은 답변하지 마세요. 사족을 달지 마세요.
2. **구체적으로 안내**: "신고할 수 있습니다"가 아니라, 어디서 어떻게 하는지 단계별로 설명하세요.
3. **마크다운 형식**: 깔끔하게 정리하되 불필요한 서론/결론은 생략하세요.

## 신고방법 안내 시 필수 포함 사항
신고방법을 물어보면, 반드시 아래 형식으로 **구체적인 단계**를 안내하세요:

### 온라인 신고 (고용노동부 민원마당)
1. 고용노동부 민원마당(https://minwon.moel.go.kr) 접속
2. 회원가입 또는 로그인 (공동인증서/간편인증 가능)
3. [민원신청] > [서식민원] 클릭
4. "임금체불 진정서" 또는 해당 민원 유형 선택
5. 신청서 작성 후 증빙자료(근로계약서, 급여명세서 등) 첨부
6. 제출 후 접수번호 확인

### 전화 상담
- 고용노동부 상담센터: **1350** (평일 09:00~18:00)
- 상담 후 진정서 접수 안내 가능

### 방문 신고
- 관할 고용노동청 방문 (근무지 주소 기준)
- 신분증, 근로계약서 사본, 급여명세서 등 지참

## 이 계약서 분석 결과 (중요)
{analysis_context}

## 계약서 원문 (발췌)
{contract_context}
"""


async def query_analyzer(state: AgentState) -> dict:
    """Analyze query and decide which tools to use"""
    messages = state["messages"]
    last_message = messages[-1].content if messages else ""

    llm = get_fast_llm()

    # Analyze query to determine tool usage
    analysis_prompt = f"""사용자 질문을 분석하고 필요한 도구를 결정하세요.

질문: {last_message}

다음 중 필요한 도구를 선택하세요:
1. search_vector_db - 법령, 판례, 해석례 검색 필요시
2. search_graph_db - 위험 패턴, 관련 법률 구조 검색 필요시
3. web_search - 신고 방법, 기관 정보, 최신 정보 필요시

JSON 형식으로 응답:
{{"tools": ["tool_name"], "reasoning": "이유"}}
"""

    return {"current_step": "analyzing"}


async def should_use_tools(state: AgentState) -> Literal["tools", "respond"]:
    """Decide whether to use tools or respond directly"""
    messages = state["messages"]
    last_message = messages[-1].content if messages else ""

    # Keywords that suggest tool usage
    tool_keywords = [
        "법",
        "조항",
        "위반",
        "신고",
        "대응",
        "예방",
        "방법",
        "판례",
        "해석",
        "기준",
        "기관",
        "상담",
        "근로기준법",
    ]

    needs_tools = any(kw in last_message for kw in tool_keywords)

    if needs_tools:
        return "tools"
    return "respond"


async def call_tools(state: AgentState) -> dict:
    """Call appropriate tools based on the query"""
    messages = state["messages"]
    contract_text = state.get("contract_text", "")
    last_message = messages[-1].content if messages else ""

    tool_results = []

    # Determine which tools to call based on query content
    query_lower = last_message.lower()

    # Vector DB search for legal content
    if any(kw in query_lower for kw in ["법", "조항", "위반", "기준", "판례"]):
        result = await search_vector_db.ainvoke(
            {"query": last_message, "doc_type": None, "limit": 3}
        )
        tool_results.append({"tool": "법령/판례 검색", "result": result})

    # Graph DB search for risk patterns
    if any(kw in query_lower for kw in ["위험", "패턴", "위반", "문제"]):
        # Extract clause type from query
        clause_types = ["임금", "근로시간", "휴게시간", "위약금", "사회보험", "연차"]
        detected_type = next((ct for ct in clause_types if ct in query_lower), "기타")

        keywords = [w for w in query_lower.split() if len(w) > 1][:5]

        result = await search_graph_db.ainvoke(
            {"clause_type": detected_type, "keywords": keywords}
        )
        tool_results.append({"tool": "위험패턴 검색", "result": result})

    # Web search for reporting/practical info
    if any(
        kw in query_lower
        for kw in ["신고", "방법", "대응", "예방", "상담", "기관", "어디"]
    ):
        result = await web_search.ainvoke({"query": last_message, "max_results": 3})
        tool_results.append({"tool": "웹 검색", "result": result})

    # If no specific tools matched, do a general vector search
    if not tool_results:
        result = await search_vector_db.ainvoke({"query": last_message, "limit": 3})
        tool_results.append({"tool": "관련 법령 검색", "result": result})

    return {"tool_results": tool_results, "current_step": "tools_complete"}


async def generate_response(state: AgentState) -> dict:
    """Generate final response using tool results"""
    messages = state["messages"]
    contract_text = state.get("contract_text", "")
    tool_results = state.get("tool_results", [])
    last_message = messages[-1].content if messages else ""

    # Format tool results for context
    context_parts = []

    if contract_text:
        context_parts.append(f"## 계약서 내용\n{contract_text[:2000]}")

    for tr in tool_results:
        tool_name = tr.get("tool", "도구")
        result = tr.get("result", "{}")
        try:
            parsed = json.loads(result) if isinstance(result, str) else result
            if "results" in parsed and parsed["results"]:
                context_parts.append(f"\n## {tool_name} 결과")
                for i, r in enumerate(parsed["results"][:3], 1):
                    if isinstance(r, dict):
                        source = r.get("source", r.get("title", ""))
                        text = r.get("text", r.get("content", r.get("explanation", "")))
                        context_parts.append(f"[{i}] {source}\n{text}")
        except:
            pass

    context = "\n\n".join(context_parts)

    # Build response prompt
    llm = get_response_llm()

    system_msg = SYSTEM_PROMPT.format(
        contract_context=contract_text[:1500] if contract_text else "계약서 정보 없음"
    )

    response_prompt = f"""## 수집된 정보
{context}

## 사용자 질문
{last_message}

## 지시사항
위 정보를 바탕으로 사용자 질문에 명확하고 도움이 되는 답변을 작성하세요.
- 마크다운 형식 사용
- 법적 근거가 있으면 명시
- 실용적인 조언 포함
- 필요시 관련 기관/연락처 안내
"""

    return {"current_step": "responding", "final_response": response_prompt}


# ============ Graph Builder ============


def build_agent_graph():
    """Build the LangGraph agent"""

    # Create graph
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("analyze", query_analyzer)
    workflow.add_node("tools", call_tools)
    workflow.add_node("respond", generate_response)

    # Set entry point
    workflow.set_entry_point("analyze")

    # Add edges
    workflow.add_conditional_edges(
        "analyze", should_use_tools, {"tools": "tools", "respond": "respond"}
    )
    workflow.add_edge("tools", "respond")
    workflow.add_edge("respond", END)

    return workflow.compile()


# ============ Streaming Agent ============


@dataclass
class StreamEvent:
    """Event for streaming to frontend"""

    event_type: str  # "step", "tool", "token", "done", "error"
    data: dict

    def to_sse(self) -> str:
        return f"data: {json.dumps({'type': self.event_type, **self.data}, ensure_ascii=False)}\n\n"


class ContractChatAgent:
    """Main agent class for contract chat"""

    def __init__(self):
        self.graph = build_agent_graph()
        self.llm = get_response_llm()

    async def chat_stream(
        self,
        message: str,
        contract_text: str = "",
        contract_id: int = 0,
        chat_history: list = None,
        analysis_summary: str = "",
    ) -> AsyncGenerator[StreamEvent, None]:
        """
        Stream chat response with real-time updates

        Yields:
            StreamEvent objects for SSE streaming
        """

        # Build initial state
        messages = []
        if chat_history:
            for msg in chat_history:
                if msg.get("role") == "user":
                    messages.append(HumanMessage(content=msg["content"]))
                else:
                    messages.append(AIMessage(content=msg["content"]))

        messages.append(HumanMessage(content=message))

        initial_state: AgentState = {
            "messages": messages,
            "contract_text": contract_text,
            "contract_id": contract_id,
            "current_step": "start",
            "tool_results": [],
            "final_response": "",
        }

        try:
            # Step 1: Analyzing query
            yield StreamEvent(
                "step", {"step": "analyzing", "message": "질문 분석 중..."}
            )
            await asyncio.sleep(0.1)

            # Step 2: LLM-based query analysis and tool selection
            tool_plan = await analyze_query_with_llm(
                user_message=message,
                chat_history=chat_history,
                analysis_summary=analysis_summary,
            )

            tool_results = []

            # Execute tools based on LLM's plan
            if tool_plan.use_vector_db:
                yield StreamEvent(
                    "tool",
                    {
                        "tool": "search_vector_db",
                        "status": "searching",
                        "message": f"법령/판례 검색 중: {tool_plan.vector_query[:25]}...",
                    },
                )
                result = await search_vector_db.ainvoke(
                    {"query": tool_plan.vector_query, "limit": 3}
                )
                tool_results.append({"tool": "법령/판례 검색", "result": result})
                yield StreamEvent(
                    "tool",
                    {
                        "tool": "search_vector_db",
                        "status": "complete",
                        "message": "법령/판례 검색 완료",
                    },
                )

            if tool_plan.use_graph_db:
                yield StreamEvent(
                    "tool",
                    {
                        "tool": "search_graph_db",
                        "status": "searching",
                        "message": f"위험 패턴 검색 중: {tool_plan.graph_clause_type}...",
                    },
                )
                result = await search_graph_db.ainvoke(
                    {
                        "clause_type": tool_plan.graph_clause_type,
                        "keywords": tool_plan.graph_keywords or [message[:20]],
                    }
                )
                tool_results.append({"tool": "위험패턴 검색", "result": result})
                yield StreamEvent(
                    "tool",
                    {
                        "tool": "search_graph_db",
                        "status": "complete",
                        "message": "위험 패턴 검색 완료",
                    },
                )

            if tool_plan.use_web_search:
                yield StreamEvent(
                    "tool",
                    {
                        "tool": "web_search",
                        "status": "searching",
                        "message": f"웹 검색 중: {tool_plan.web_query[:25]}...",
                    },
                )
                result = await web_search.ainvoke(
                    {"query": tool_plan.web_query, "max_results": 3}
                )
                tool_results.append({"tool": "웹 검색", "result": result})
                yield StreamEvent(
                    "tool",
                    {
                        "tool": "web_search",
                        "status": "complete",
                        "message": "웹 검색 완료",
                    },
                )

            # Step 3: Generate response with streaming
            yield StreamEvent(
                "step", {"step": "generating", "message": "답변 생성 중..."}
            )

            # Debug: log analysis summary usage
            if analysis_summary:
                print(f">>> [chat_stream] Using analysis summary ({len(analysis_summary)} chars)")

            # Build context from tool results
            context_parts = []

            if contract_text:
                context_parts.append(f"## 계약서 내용 (발췌)\n{contract_text[:1500]}")

            for tr in tool_results:
                tool_name = tr.get("tool", "")
                result = tr.get("result", "{}")
                try:
                    parsed = json.loads(result) if isinstance(result, str) else result
                    if "results" in parsed and parsed["results"]:
                        context_parts.append(f"\n## {tool_name} 결과")
                        for i, r in enumerate(parsed["results"][:3], 1):
                            if isinstance(r, dict):
                                source = r.get(
                                    "source", r.get("title", r.get("name", ""))
                                )
                                text = r.get(
                                    "text", r.get("content", r.get("explanation", ""))
                                )
                                if source or text:
                                    context_parts.append(
                                        f"[{i}] {source}\n{text[:300]}"
                                    )
                except:
                    pass

            context = "\n\n".join(context_parts)

            # Build messages for LLM
            system_content = SYSTEM_PROMPT.format(
                analysis_context=(
                    analysis_summary if analysis_summary else "분석 결과 없음"
                ),
                contract_context=(
                    contract_text[:1000] if contract_text else "계약서 정보 없음"
                ),
            )

            llm_messages = [
                SystemMessage(content=system_content),
                HumanMessage(
                    content=f"""## 수집된 참고 자료
{context}

## 사용자 질문
{message}

## 지시사항
1. **질문에만 답변하세요.** 묻지 않은 내용, 불필요한 서론/결론, 일반적인 조언은 생략하세요.
2. **신고방법 질문 시**: 시스템 프롬프트에 있는 구체적인 단계(사이트 접속 > 로그인 > 메뉴 클릭 > 서류 제출)를 그대로 안내하세요.
3. **마크다운 형식**으로 깔끔하게 정리하세요.

나쁜 예시 (사족이 많음):
"임금 체불은 심각한 문제입니다. 근로기준법에 따르면... (중략) ...어려운 상황이시겠지만 힘내세요!"

좋은 예시 (핵심만):
"### 임금체불 신고 방법
1. 고용노동부 민원마당(https://minwon.moel.go.kr) 접속
2. 로그인 후 [민원신청] > [서식민원] 클릭
3. '임금체불 진정서' 선택 후 작성
4. 근로계약서, 급여명세서 등 증빙자료 첨부
5. 제출

**전화 문의**: 1350 (고용노동부, 평일 09:00~18:00)"
"""
                ),
            ]

            # Stream response tokens
            full_response = ""
            async for chunk in self.llm.astream(llm_messages):
                if chunk.content:
                    full_response += chunk.content
                    yield StreamEvent("token", {"content": chunk.content})

            # Done
            yield StreamEvent("done", {"full_response": full_response})

        except Exception as e:
            yield StreamEvent("error", {"message": str(e)})


# Singleton instance
_agent_instance = None


def get_chat_agent() -> ContractChatAgent:
    """Get or create chat agent instance"""
    global _agent_instance
    if _agent_instance is None:
        _agent_instance = ContractChatAgent()
    return _agent_instance
