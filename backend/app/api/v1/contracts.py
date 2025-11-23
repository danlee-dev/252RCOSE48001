from typing import List, Optional, Any, Dict
from fastapi import APIRouter, Depends, UploadFile, File, status, HTTPException, Header
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy import desc, func
from app.core.database import get_db, AsyncSessionLocal
from app.schemas.contract import ContractResponse, ContractDetailResponse
from app.api import deps
from app.models.user import User
from app.models.contract import Contract 
from app.utils.file_storage import save_contract_file 
from app.core.celery_app import celery_app 
import requests
import asyncio
import os
import sys
from elasticsearch import Elasticsearch
from neo4j import Driver, basic_auth
from sentence_transformers import SentenceTransformer
import numpy as np
from app.api.deps import verify_internal_api_key, get_es_client, get_neo4j_driver
import re 
from app.tasks.analysis_tasks import analyze_contract_task 
# 🔴 [FDE IMPORT] FDE 관련 클래스/함수를 가져옵니다. (main.py가 경로를 설정한다고 가정)
from ai.preprocessing.fde_generator import ( 
    FixedDimensionalEncodingConfig,
    generate_query_fde,
    EncodingType,
    ProjectionType,
)


router = APIRouter()

# --- Helper Class for Splitting (3_embed_muvera.py의 로직 통합) ---
class APISentenceSplitter:
    @staticmethod
    def split_sentences(text: str, min_length: int = 10) -> List[str]:
        sentences = re.split(r'[.!?]\s+', text)
        sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) >= min_length]
        if not sentences:
            sentences = [text]
        return sentences
# -------------------------------------------------------------------


# 💡 모델 및 FDE 설정 (서버 시작 시 한 번만 로드)
try:
    GLOBAL_EMBEDDING_MODEL = SentenceTransformer("nlpai-lab/KURE-v1")
    GLOBAL_EMBEDDING_MODEL.max_seq_length = 512
    EMBEDDING_DIM = GLOBAL_EMBEDDING_MODEL.get_sentence_embedding_dimension() 
    
    # 🔴 [FDE 설정] 3_embed_muvera.py의 설정 (1024차원) 반영
    FDE_CONFIG = FixedDimensionalEncodingConfig(
        dimension=EMBEDDING_DIM,
        num_repetitions=1, 
        num_simhash_projections=3, 
        seed=42,
        encoding_type=EncodingType.AVERAGE, # 문서 생성에 AVERAGE 사용되었으므로 쿼리도 AVERAGE 설정 기반으로 해야 함
        projection_type=ProjectionType.DEFAULT_IDENTITY,
        final_projection_dimension=1024 
    )
    
except Exception as e:
    print(f"❌ Embedding Model Load Failed: {e}")
    GLOBAL_EMBEDDING_MODEL = None
    FDE_CONFIG = None
    
INDEX_NAME = "docscanner_chunks"


# -------------------------------------------------------------------------
# 🔴 [FastAPI 라우터] 메인 BE 로직 (업로드 및 조회 유지)
# -------------------------------------------------------------------------

@router.post("/", status_code=202, summary="계약서 업로드 및 AI 분석 시작")
async def upload_contract(
    file: UploadFile = File(..., description="업로드할 PDF 파일"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(deps.get_current_user)
):
    """
    **[보호됨]** PDF 계약서 파일을 업로드하고, Celery Task Queue에 AI 분석 작업을 등록합니다.
    
    이 엔드포인트는 파일 저장 후 즉시 응답(202 Accepted)하며, AI 분석은 백그라운드에서 비동기적으로 처리됩니다.
    
    - **요청 파라미터 (Input):**
        - `file`: 업로드할 **PDF 파일** (multipart/form-data로 전송). 현재 10MB 이하 권장.
    - **요청 헤더:**
        - `Authorization`: `Bearer <Access Token>` (로그인 필수)
    - **성공 응답 (202 Accepted):**
        - `message`: 작업 접수 확인
        - `contract_id`: 새로 생성된 계약서의 DB ID
        - `status`: PENDING (처리 대기 중)
    - **주요 오류 코드:**
        - `401 Unauthorized`: 유효하지 않은 토큰
        - `400 Bad Request`: 파일 형식이 PDF가 아님
        - `500 Internal Server Error`: 파일 시스템 저장 오류, Celery 등록 오류 등
    """
    # 1. 파일 저장 로직 실행
    try:
        file_url = await save_contract_file(current_user.id, file)
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"파일 저장 중 시스템 오류 발생: {e}")
        
    # 2. DB에 계약서 정보 저장 (status: PENDING)
    new_contract = Contract(
        user_id=current_user.id,
        title=file.filename,
        file_url=file_url,
        status="PENDING"
    )
    db.add(new_contract)
    await db.commit()
    await db.refresh(new_contract)
    
    # 3. Celery Task에 작업 등록
    analyze_contract_task.delay(new_contract.id) 
    
    return {
        "message": "Accepted", 
        "contract_id": new_contract.id, 
        "status": new_contract.status
    }

@router.get("/", response_model=List[ContractResponse], summary="내 계약서 목록 조회")
async def read_contracts(
    skip: int = 0, 
    limit: int = 10, 
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(deps.get_current_user)
):
    """
    **[보호됨]** 현재 로그인한 사용자가 업로드한 모든 계약서의 목록을 조회합니다. 
    결과는 페이지네이션을 지원하며, **업로드 최신순**으로 반환됩니다.
    
    - **요청 파라미터 (Query):**
        - `skip`: 건너뛸 항목 수 (페이지네이션 오프셋, 기본값 0).
        - `limit`: 한 번에 가져올 최대 항목 수 (페이지 크기, 기본값 10).
    - **응답 (Output):**
        - `200 OK`: 계약서 ID, 제목, 상태, 위험도 레벨 등 핵심 정보 목록.
    - **주요 오류 코드:**
        - `401 Unauthorized`: 유효하지 않은 토큰.
    """
    # 최신 계약서가 목록 맨 앞에 오도록 created_at을 기준으로 내림차순(DESC) 정렬합니다.
    stmt = (
        select(Contract)
        .where(Contract.user_id == current_user.id)
        .order_by(desc(Contract.created_at))
        .offset(skip)
        .limit(limit)
    )
    
    # DB에서 데이터 실행
    result = await db.execute(stmt)
    contracts = result.scalars().all() 
    
    return contracts

# -------------------------------------------------------------------------
# 🔴 [툴 API] Dify가 호출할 커스텀 툴 API 로직 구현
# -------------------------------------------------------------------------
    
# Muvera 검색 툴 (ES RRF 기반)
@router.get("/v1/search-muvera", 
            summary="Dify Tool: Muvera 멀티 벡터 검색", 
            include_in_schema=False, # 👈 사용자에게 노출 금지
            status_code=status.HTTP_200_OK)
async def search_muvera(
    query_text: str, 
    es: Elasticsearch = Depends(get_es_client), 
    internal_api_key: str = Depends(verify_internal_api_key)
):
    """
    **[내부 전용 API]** (Dify가 호출) 사용자의 조항을 FDE 벡터로 변환하여 ES에서 유사 조항을 검색합니다.
    
    - **사용 주체:** Dify AI Agent
    - **입력:** `query_text` (LLM이 추출한 계약 조항 텍스트)
    - **출력 스키마:** `{"context": [{"source": "...", "text": "..."}]}`
    - **인증:** `X-Internal-API-Key` 헤더 필요.
    """
    if GLOBAL_EMBEDDING_MODEL is None or FDE_CONFIG is None:
        raise HTTPException(status_code=503, detail="Embedding model or FDE config not loaded.")
        
    # 1. 쿼리 텍스트를 FDE 벡터로 변환 (MUVERA 로직 적용)
    try:
        # 문장 분할 
        sentences = APISentenceSplitter.split_sentences(query_text)
        
        # 각 문장 임베딩
        sentence_embeddings = GLOBAL_EMBEDDING_MODEL.encode(
            sentences, 
            convert_to_numpy=True, 
            normalize_embeddings=True
        )
        
        # 🔴 [핵심] FDE로 압축 (generate_query_fde 사용)
        query_vector_fde = generate_query_fde(sentence_embeddings, FDE_CONFIG)
        query_vector = query_vector_fde.tolist() 
        
    except Exception as e:
        print(f"Query FDE generation failed: {e}")
        raise HTTPException(status_code=500, detail="쿼리 벡터 생성 실패")
    
    # 2. Elasticsearch KNN 검색 쿼리
    search_query = {
        "field": "embedding",
        "k": 5, 
        "num_candidates": 50,
        "query_vector": query_vector, 
        "filter": {"bool": {"must_not": [{"exists": {"field": "type"}}]}},
    }
    
    try:
        # 3. ES 검색 실행
        response = es.search(
            index=INDEX_NAME,
            knn=search_query,
            _source=["text", "source", "doc_type"], 
            size=5
        )
        
        # 4. 결과 파싱 (Dify Context 형식에 맞춤)
        context = []
        for hit in response['hits']['hits']:
            source = hit['_source']
            context.append({
                "source": f"{source.get('source', 'N/A')}/{source.get('doc_type', 'N/A')}",
                "text": source['text']
            })
            
        return {"context": context}
        
    except Exception as e:
        print(f"ES search failed: {e}")
        raise HTTPException(status_code=500, detail="Elasticsearch 검색 실패")


# GraphDB 위험 규칙 검색 툴
@router.get("/v1/search-risk-pattern", 
            summary="Dify Tool: GraphDB 위험 규칙 검색", 
            include_in_schema=False, # 👈 사용자에게 노출 금지
            status_code=status.HTTP_200_OK)
async def search_risk_pattern(
    query_text: str, 
    driver: Driver = Depends(get_neo4j_driver),
    internal_api_key: str = Depends(verify_internal_api_key)
):
    """ 
    **[내부 전용 API]** 사용자의 조항과 관련된 법률 지식 그래프(Neo4j)에서
    위험 패턴, 규칙, 법령 관계 등을 검색하는 커스텀 툴입니다.
    
    - **사용 주체:** Dify AI Agent
    - **입력:** `query_text` (LLM이 추출한 계약 조항 텍스트)
    - **출력 스키마:** `{"context": [{"rule_name": "...", "text": "..."}]}`
    - **인증:** `X-Internal-API-Key` 헤더 필요.
    """
    
    # 1. 7_seed_ontology.py에 정의된 Regex 기반 검색 로직 사용
    cypher_query = """
    // 쿼리 텍스트에 포함된 단어를 바탕으로 RiskPattern 노드를 검색
    MATCH (r:RiskPattern)
    WHERE ANY(trigger IN r.triggers WHERE toLower($queryText) CONTAINS toLower(trigger))
    OPTIONAL MATCH (r)-[:IS_A_TYPE_OF]->(c:ClauseType)
    RETURN r.name AS name, r.explanation AS explanation, r.riskLevel AS level, c.name AS clauseType
    """
    
    try:
        # Neo4j 세션 열기 (basic_auth를 사용하여 연결 인증)
        with driver.session(auth=basic_auth(os.getenv("NEO4J_USER"), os.getenv("NEO4J_PASSWORD"))) as session:
            result = session.run(cypher_query, queryText=query_text).data()
            
            # 2. 결과 파싱 (Dify Context 형식에 맞춤)
            context = []
            for record in result:
                context.append({
                    "rule_name": record['name'],
                    "text": f"위험 패턴 '{record['name']}' ({record['clauseType']} 조항, 위험도: {record['level']}): {record['explanation']}"
                })
            
            # 3. 결과가 없으면 임시 메시지 반환
            if not context:
                return {"context": [{"text": "검색된 위험 규칙이 없습니다."}]}
            
            return {"context": context}
            
    except Exception as e:
        print(f"Neo4j search failed: {e}")
        raise HTTPException(status_code=500, detail="GraphDB 검색 실패")