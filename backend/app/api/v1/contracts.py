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
from app.utils.file_storage import save_contract_file, delete_contract_file 
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

@router.delete("/{contract_id}", status_code=204, summary="계약서 삭제")
async def delete_contract(
    contract_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(deps.get_current_user)
):
    """
    **[보호됨]** 특정 계약서를 삭제합니다. 
    DB의 계약서 정보와 업로드된 실제 PDF 파일이 모두 삭제됩니다.
    """
    # 1. 계약서 조회 (내 계약서인지 확인)
    stmt = select(Contract).where(Contract.id == contract_id, Contract.user_id == current_user.id)
    result = await db.execute(stmt)
    contract = result.scalar_one_or_none()

    if not contract:
        raise HTTPException(status_code=404, detail="계약서를 찾을 수 없습니다.")

    # 2. 파일 삭제 (DB 삭제 전 수행)
    try:
        delete_contract_file(contract.file_url)
    except Exception as e:
        print(f"File deletion warning: {e}")
        # 파일 삭제 실패해도 DB 삭제는 진행

    # 3. DB 삭제
    await db.delete(contract)
    await db.commit()
    
    return

# -------------------------------------------------------------------------
# 🔴 [툴 API] Dify가 호출할 커스텀 툴 API 로직 구현
# -------------------------------------------------------------------------
    
# Muvera 검색 툴
@router.get("/v1/search-muvera", 
            summary="[Tool] Muvera 멀티 벡터 검색 (유사 조항)", 
            include_in_schema=True, # 🔴 [수정] 테스트를 위해 노출
            status_code=status.HTTP_200_OK,
            responses={
                200: {
                    "description": "검색 성공",
                    "content": {
                        "application/json": {
                            "example": {
                                "context": [
                                    {
                                        "source": "근로기준법/law",
                                        "text": "제17조(근로조건의 명시) 사용자는 근로계약을 체결할 때에 근로자에게 다음 각 호의 사항을 명시하여야 한다."
                                    },
                                    {
                                        "source": "대법원 판례 2020다XXXX/precedent",
                                        "text": "근로계약서에 명시된 근로조건은..."
                                    }
                                ]
                            }
                        }
                    }
                }
            })
async def search_muvera(
    query_text: str = Query(
        ..., 
        description="분석할 계약 조항 텍스트 (예: '제3조 임금은 매월 25일에 지급한다.')",
        min_length=2
    ), 
    es: Elasticsearch = Depends(get_es_client), 
    internal_api_key: str = Depends(verify_internal_api_key)
):
    """
    **[Dify 전용 Tool]** 사용자의 계약 조항을 분석하여 **Elasticsearch**의 Multi-Vector Index에서
    가장 유사한 표준/법률 조항 청크(Chunk)를 검색합니다.
    
    - **역할:** RAG(Retrieval-Augmented Generation)를 위한 법률적 근거(Context) 제공
    - **입력:** 분석 대상 계약 조항 (자연어 문장)
    - **출력:** 유사도가 높은 상위 5개 법률/판례 조항 리스트
    
    **테스트 방법:**
    1. 상단 `Authorize` 버튼 클릭 -> `Client credentials location` 무시.
    2. 이 API의 자물쇠 아이콘 클릭 -> `X-Internal-API-Key` 입력란에 `.env`의 `INTERNAL_API_KEY` 값 입력.
    3. `query_text`에 "최저임금 미달" 등 검색어 입력 후 실행.
    """
    if GLOBAL_EMBEDDING_MODEL is None or FDE_CONFIG is None:
        raise HTTPException(status_code=503, detail="Embedding model or FDE config not loaded.")
        
    # 1. 쿼리 텍스트를 FDE 벡터로 변환 (MUVERA 로직 적용)
    try:
        sentences = APISentenceSplitter.split_sentences(query_text)
        sentence_embeddings = GLOBAL_EMBEDDING_MODEL.encode(
            sentences, 
            convert_to_numpy=True, 
            normalize_embeddings=True
        )
        
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
        response = es.search(
            index=INDEX_NAME,
            knn=search_query,
            _source=["text", "source", "doc_type"], 
            size=5
        )
        
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
            summary="[Tool] GraphDB 위험 규칙 검색 (Regex)", 
            include_in_schema=True, # 🔴 [수정] 테스트를 위해 노출
            status_code=status.HTTP_200_OK,
            responses={
                200: {
                    "description": "검색 성공",
                    "content": {
                        "application/json": {
                            "example": {
                                "context": [
                                    {
                                        "rule_name": "포괄임금제",
                                        "text": "위험 패턴 '포괄임금제' (임금 조항, 위험도: High): 연장근로수당을 포함하여 지급하는..."
                                    }
                                ]
                            }
                        }
                    }
                }
            })
async def search_risk_pattern(
    query_text: str = Query(
        ..., 
        description="분석할 계약 조항 텍스트 (예: '모든 수당을 포함하여 포괄 지급한다.')",
        min_length=2
    ), 
    driver: Driver = Depends(get_neo4j_driver),
    internal_api_key: str = Depends(verify_internal_api_key)
):
    """ 
    **[Dify 전용 Tool]** 사용자의 조항 텍스트에서 키워드(Regex)를 추출하여
    **Neo4j** 지식 그래프에 정의된 위험 패턴(RiskPattern)을 검색합니다.
    
    - **역할:** 규칙 기반(Rule-based)의 명확한 위험 요소 탐지
    - **입력:** 분석 대상 계약 조항
    - **출력:** 매칭된 위험 패턴의 이름, 설명, 위험도(High/Medium)
    
    **테스트 방법:**
    1. `X-Internal-API-Key` 헤더에 `.env`의 `INTERNAL_API_KEY` 값 입력.
    2. `query_text`에 "포괄하여 지급", "위약금" 등 위험 키워드가 포함된 문장 입력.
    """
    
    # 1. 7_seed_ontology.py에 정의된 Regex 기반 검색 로직 사용
    cypher_query = """
    MATCH (r:RiskPattern)
    WHERE ANY(trigger IN r.triggers WHERE toLower($queryText) CONTAINS toLower(trigger))
    OPTIONAL MATCH (r)-[:IS_A_TYPE_OF]->(c:ClauseType)
    RETURN r.name AS name, r.explanation AS explanation, r.riskLevel AS level, c.name AS clauseType
    """
    
    try:
        with driver.session(auth=basic_auth(os.getenv("NEO4J_USER"), os.getenv("NEO4J_PASSWORD"))) as session:
            result = session.run(cypher_query, queryText=query_text).data()
            
            context = []
            for record in result:
                context.append({
                    "rule_name": record['name'],
                    "text": f"위험 패턴 '{record['name']}' ({record['clauseType']} 조항, 위험도: {record['level']}): {record['explanation']}"
                })
            
            if not context:
                # 검색 결과가 없을 때 빈 리스트 대신 안내 메시지 반환 (Dify가 이해하기 좋음)
                return {"context": [{"text": "검색된 위험 규칙이 없습니다."}]}
            
            return {"context": context}
            
    except Exception as e:
        print(f"Neo4j search failed: {e}")
        raise HTTPException(status_code=500, detail="GraphDB 검색 실패")