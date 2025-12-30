import os
import json
import asyncio
from dotenv import load_dotenv
from neo4j import GraphDatabase
from tqdm import tqdm
from langchain_openai import ChatOpenAI 
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import List

# 1. 환경 설정
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, "../../"))
load_dotenv(os.path.join(root_dir, ".env"))

URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
AUTH = (os.getenv("NEO4J_USER", "neo4j"), os.getenv("NEO4J_PASSWORD", "password"))
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") # .env에 OPENAI_API_KEY 추가 필요

# 2. LLM 출력 구조 정의 (Pydantic) - LLM이 이 형식으로만 응답하도록 강제합니다.
class Citation(BaseModel):
    law_name: str = Field(description="법령 이름 (예: 근로기준법)")
    article: str = Field(description="조항 번호 (예: 제23조, 제56조)")
    
class CitationResult(BaseModel):
    citations: List[Citation]

# 3. LLM 설정 및 체인 구성
llm = ChatOpenAI(model="gpt-4o", temperature=0, api_key=OPENAI_API_KEY)
structured_llm = llm.with_structured_output(CitationResult)

prompt = ChatPromptTemplate.from_messages([
    ("system", "너는 법률 데이터 분석가다. 입력된 텍스트에서 '인용된 법령'과 '조항'을 정확히 추출하여 JSON 리스트로 반환해라. 추측하지 말고 텍스트에 명시된 것만 추출해야 한다. (예: '근로기준법 제23조 제1항'이 있다면 {law_name: '근로기준법', article: '제23조 1항'}로 반환)"),
    ("human", "{text}")
])
citation_chain = prompt | structured_llm

class MultiHopBuilder:
    def __init__(self):
        self.driver = GraphDatabase.driver(URI, auth=AUTH)

    def close(self):
        self.driver.close()

    async def build_citations(self):
        print("🔗 LLM 기반 멀티홉 인용 관계(CITES) 추출 시작...")
        
        # 1. 처리 대상 문서 가져오기 (판례, 해석)
        fetch_query = """
            MATCH (d:Document)
            WHERE (d:Precedent OR d:Interpretation) AND d.content IS NOT NULL
            RETURN d.id AS id, d.content AS content, d.type AS doc_type
        """
        with self.driver.session() as session:
            result = session.run(fetch_query)
            documents = [record for record in result]

        print(f"대상 문서: {len(documents)}개")

        # 2. LLM 추출 및 연결 생성
        with self.driver.session() as session:
            for doc in tqdm(documents):
                try:
                    # LLM 추출 (API 호출) - 텍스트 길이 제한 필요
                    extraction = citation_chain.invoke({"text": doc["content"][:4000]})
                    
                    if not extraction.citations:
                        continue

                    for citation in extraction.citations:
                        law_node_name = f"{citation.law_name} {citation.article}".strip()
                        if not law_node_name: continue

                        # 🔴 [멀티홉 최종 연결] (Precedent/Interpretation) -[:CITES]-> (Law)
                        query = """
                        MATCH (d:Document {id: $doc_id})
                        MERGE (l:Law {name: $law_name}) 
                        MERGE (d)-[:CITES]->(l)
                        """
                        session.run(query, doc_id=doc["id"], law_name=law_node_name)
                        
                except Exception as e:
                    print(f"⚠️ Error processing {doc['id']} ({doc['doc_type']}): {e}")

        print("✅ 멀티홉 인용 관계 구축 완료!")

if __name__ == "__main__":
    builder = MultiHopBuilder()
    try:
        # LLM 호출은 비동기이므로 asyncio.run으로 실행합니다.
        asyncio.run(builder.build_citations())
    finally:
        builder.close()