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
from typing import TypedDict, Annotated, Sequence, Literal, AsyncGenerator, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage, ToolMessage
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
    query: str,
    doc_type: Optional[str] = None,
    limit: int = 5
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
        must_clauses = [{
            "multi_match": {
                "query": query,
                "fields": ["text^2", "title", "keywords"],
                "type": "best_fields"
            }
        }]

        filter_clauses = []
        if doc_type:
            filter_clauses.append({"term": {"doc_type": doc_type}})

        search_body = {
            "query": {
                "bool": {
                    "must": must_clauses,
                    "filter": filter_clauses if filter_clauses else None
                }
            },
            "size": limit,
            "_source": ["text", "source", "doc_type", "title"]
        }

        # Remove None filter
        if not filter_clauses:
            del search_body["query"]["bool"]["filter"]

        response = es.search(index="docscanner_chunks", body=search_body)

        results = []
        for hit in response.get("hits", {}).get("hits", []):
            source = hit.get("_source", {})
            results.append({
                "source": source.get("source", ""),
                "text": source.get("text", "")[:500],
                "doc_type": source.get("doc_type", ""),
                "score": hit.get("_score", 0)
            })

        if not results:
            return json.dumps({"message": "검색 결과가 없습니다.", "results": []}, ensure_ascii=False)

        return json.dumps({"results": results}, ensure_ascii=False)

    except Exception as e:
        return json.dumps({"error": str(e), "results": []}, ensure_ascii=False)


@tool
async def search_graph_db(
    clause_type: str,
    keywords: list[str]
) -> str:
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
                toString(r.triggers) CONTAINS kw
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
                results.append({
                    "type": "risk_pattern",
                    "name": record["pattern_name"],
                    "explanation": record["explanation"],
                    "risk_level": record["risk_level"],
                    "related_docs": [d[:300] for d in record["related_docs"] if d]
                })

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

            doc_result = session.run(doc_query, categories=categories, keywords=keywords)
            for record in doc_result:
                if record["content"]:
                    results.append({
                        "type": "document",
                        "source": record["source"],
                        "content": record["content"][:400]
                    })

        driver.close()

        if not results:
            return json.dumps({"message": "관련 위험 패턴이 없습니다.", "results": []}, ensure_ascii=False)

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
                include_domains=["moel.go.kr", "law.go.kr", "minwon.go.kr", "nlcy.go.kr"]
            )
            print(f">>> [web_search] Tavily response received: {len(response.get('results', []))} results")

            results = []
            for r in response.get("results", []):
                results.append({
                    "title": r.get("title", ""),
                    "url": r.get("url", ""),
                    "content": r.get("content", "")[:300]
                })

            return json.dumps({"results": results, "source": "tavily"}, ensure_ascii=False)

        # Fallback: Return helpful static information
        print(">>> [web_search] No Tavily API key, using fallback resources")
        helpful_resources = [
            {
                "title": "고용노동부 민원마당",
                "url": "https://minwon.moel.go.kr",
                "content": "임금체불, 부당해고 등 노동관련 민원 신고 및 상담 가능"
            },
            {
                "title": "국민신문고",
                "url": "https://www.epeople.go.kr",
                "content": "정부 민원 통합 접수 시스템, 노동관련 민원 신고 가능"
            },
            {
                "title": "노동권익 상담센터",
                "url": "tel:1350",
                "content": "고용노동부 상담전화 1350, 근로기준법 위반 상담 및 신고 안내"
            }
        ]

        return json.dumps({
            "results": helpful_resources,
            "source": "fallback",
            "note": "웹 검색 API가 설정되지 않아 기본 정보를 제공합니다."
        }, ensure_ascii=False)

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
    return json.dumps({
        "clause": clause_text[:500],
        "question": question,
        "analysis_prompt": "Based on the clause and question, provide legal analysis."
    }, ensure_ascii=False)


# ============ LLM Setup ============

def get_fast_llm():
    """Get fast LLM for query decomposition and tool calls"""
    return ChatGoogleGenerativeAI(
        model="gemini-2.0-flash-lite",
        google_api_key=os.getenv("GEMINI_API_KEY"),
        temperature=0.1,
        streaming=True
    )


def get_response_llm():
    """Get LLM for response generation"""
    return ChatGoogleGenerativeAI(
        model="gemini-2.0-flash-lite",
        google_api_key=os.getenv("GEMINI_API_KEY"),
        temperature=0.3,
        streaming=True
    )


async def generate_search_query(
    user_message: str,
    chat_history: list = None,
    contract_context: str = "",
    analysis_summary: str = ""
) -> str:
    """
    Generate optimized search query based on context.
    Uses LLM to understand the conversation and create a specific search query.
    """
    llm = get_fast_llm()

    # Build context from chat history
    history_text = ""
    if chat_history:
        recent_messages = chat_history[-4:]  # Last 4 messages for context
        for msg in recent_messages:
            role = "사용자" if msg.get("role") == "user" else "AI"
            history_text += f"{role}: {msg.get('content', '')[:200]}\n"

    prompt = f"""대화 맥락을 파악하여 웹 검색에 최적화된 검색 쿼리를 생성하세요.

[이전 대화]
{history_text if history_text else "없음"}

[계약서 분석 결과 - 가장 중요]
{analysis_summary if analysis_summary else "분석 결과 없음"}

[현재 사용자 질문]
{user_message}

[지시사항]
- 계약서 분석 결과(발견된 위험 조항)를 최우선으로 반영
- 대화 맥락과 연결하여 구체적인 검색 쿼리 생성
- 노동법/근로기준법 관련 키워드 포함
- 정부 기관 관련 정보를 찾기 쉬운 쿼리로 작성
- 검색 쿼리만 출력 (설명 없이, 30자 이내)

검색 쿼리:"""

    try:
        print(f">>> [generate_search_query] Original: {user_message}")
        print(f">>> [generate_search_query] Analysis summary: {analysis_summary[:100] if analysis_summary else 'None'}")
        response = await llm.ainvoke(prompt)
        optimized_query = response.content.strip()
        # Clean up query (remove quotes, extra whitespace)
        optimized_query = optimized_query.replace('"', '').replace("'", "").strip()
        print(f">>> [generate_search_query] Optimized: {optimized_query}")
        return optimized_query if optimized_query else user_message
    except Exception as e:
        print(f">>> [generate_search_query] Error: {e}, using original query")
        return user_message


# ============ Agent Nodes ============

SYSTEM_PROMPT = """당신은 한국 근로계약서 분석 전문 AI 어시스턴트입니다.

## 역할
- 사용자가 업로드한 계약서를 분석하고 질문에 답변합니다
- 노동법 관련 정보를 정확하게 제공합니다
- 위반 사항에 대한 대응/예방/신고 방법을 안내합니다

## 답변 원칙
1. 정확한 법적 근거와 함께 설명
2. 실용적이고 구체적인 조언 제공
3. 필요시 관련 기관/신고 방법 안내
4. 마크다운 형식으로 깔끔하게 정리

## 참고 자료 활용
- 제공된 법령, 판례, 해석례를 우선 참조
- 웹 검색 결과는 신뢰할 수 있는 출처만 인용
- 출처를 명시하여 신뢰성 확보

## 현재 계약서 정보
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
        "법", "조항", "위반", "신고", "대응", "예방", "방법",
        "판례", "해석", "기준", "기관", "상담", "근로기준법"
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
        result = await search_vector_db.ainvoke({
            "query": last_message,
            "doc_type": None,
            "limit": 3
        })
        tool_results.append({
            "tool": "법령/판례 검색",
            "result": result
        })

    # Graph DB search for risk patterns
    if any(kw in query_lower for kw in ["위험", "패턴", "위반", "문제"]):
        # Extract clause type from query
        clause_types = ["임금", "근로시간", "휴게시간", "위약금", "사회보험", "연차"]
        detected_type = next((ct for ct in clause_types if ct in query_lower), "기타")

        keywords = [w for w in query_lower.split() if len(w) > 1][:5]

        result = await search_graph_db.ainvoke({
            "clause_type": detected_type,
            "keywords": keywords
        })
        tool_results.append({
            "tool": "위험패턴 검색",
            "result": result
        })

    # Web search for reporting/practical info
    if any(kw in query_lower for kw in ["신고", "방법", "대응", "예방", "상담", "기관", "어디"]):
        result = await web_search.ainvoke({
            "query": last_message,
            "max_results": 3
        })
        tool_results.append({
            "tool": "웹 검색",
            "result": result
        })

    # If no specific tools matched, do a general vector search
    if not tool_results:
        result = await search_vector_db.ainvoke({
            "query": last_message,
            "limit": 3
        })
        tool_results.append({
            "tool": "관련 법령 검색",
            "result": result
        })

    return {
        "tool_results": tool_results,
        "current_step": "tools_complete"
    }


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

    system_msg = SYSTEM_PROMPT.format(contract_context=contract_text[:1500] if contract_text else "계약서 정보 없음")

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

    return {
        "current_step": "responding",
        "final_response": response_prompt
    }


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
        "analyze",
        should_use_tools,
        {
            "tools": "tools",
            "respond": "respond"
        }
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
        analysis_summary: str = ""
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
            "final_response": ""
        }

        try:
            # Step 1: Analyzing query
            yield StreamEvent("step", {"step": "analyzing", "message": "질문 분석 중..."})
            await asyncio.sleep(0.1)

            # Step 2: Determine tools and execute
            query_lower = message.lower()
            tool_results = []

            # Check which tools to use
            use_vector = any(kw in query_lower for kw in ["법", "조항", "위반", "기준", "판례", "해석"])
            use_graph = any(kw in query_lower for kw in ["위험", "패턴", "문제"])
            use_web = any(kw in query_lower for kw in ["신고", "방법", "대응", "예방", "상담", "기관", "어디", "어떻게"])

            # Default to vector search if nothing specific
            if not (use_vector or use_graph or use_web):
                use_vector = True

            # Execute tools with streaming updates
            if use_vector:
                yield StreamEvent("tool", {"tool": "search_vector_db", "status": "searching", "message": "법령/판례 검색 중..."})
                result = await search_vector_db.ainvoke({"query": message, "limit": 3})
                tool_results.append({"tool": "법령/판례 검색", "result": result})
                yield StreamEvent("tool", {"tool": "search_vector_db", "status": "complete", "message": "법령/판례 검색 완료"})

            if use_graph:
                yield StreamEvent("tool", {"tool": "search_graph_db", "status": "searching", "message": "위험 패턴 검색 중..."})
                clause_types = ["임금", "근로시간", "휴게시간", "위약금", "사회보험"]
                detected_type = next((ct for ct in clause_types if ct in query_lower), "기타")
                keywords = [w for w in query_lower.split() if len(w) > 1][:5]
                result = await search_graph_db.ainvoke({"clause_type": detected_type, "keywords": keywords})
                tool_results.append({"tool": "위험패턴 검색", "result": result})
                yield StreamEvent("tool", {"tool": "search_graph_db", "status": "complete", "message": "위험 패턴 검색 완료"})

            if use_web:
                yield StreamEvent("tool", {"tool": "web_search", "status": "searching", "message": "검색 쿼리 생성 중..."})
                # Generate optimized search query based on context and analysis
                optimized_query = await generate_search_query(
                    user_message=message,
                    chat_history=chat_history,
                    analysis_summary=analysis_summary
                )
                yield StreamEvent("tool", {"tool": "web_search", "status": "searching", "message": f"웹 검색 중: {optimized_query[:30]}..."})
                result = await web_search.ainvoke({"query": optimized_query, "max_results": 3})
                tool_results.append({"tool": "웹 검색", "result": result})
                yield StreamEvent("tool", {"tool": "web_search", "status": "complete", "message": "웹 검색 완료"})

            # Step 3: Generate response with streaming
            yield StreamEvent("step", {"step": "generating", "message": "답변 생성 중..."})

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
                                source = r.get("source", r.get("title", r.get("name", "")))
                                text = r.get("text", r.get("content", r.get("explanation", "")))
                                if source or text:
                                    context_parts.append(f"[{i}] {source}\n{text[:300]}")
                except:
                    pass

            context = "\n\n".join(context_parts)

            # Build messages for LLM
            system_content = SYSTEM_PROMPT.format(
                contract_context=contract_text[:1000] if contract_text else "계약서 정보 없음"
            )

            llm_messages = [
                SystemMessage(content=system_content),
                HumanMessage(content=f"""## 수집된 참고 자료
{context}

## 사용자 질문
{message}

## 지시사항
위 자료를 참고하여 사용자 질문에 정확하고 도움이 되는 답변을 작성하세요.
마크다운 형식을 사용하고, 법적 근거와 실용적 조언을 포함하세요.""")
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
