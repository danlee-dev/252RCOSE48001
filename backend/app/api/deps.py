from typing import Generator, Optional
from fastapi import Depends, HTTPException, status, Header
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import jwt, JWTError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from app.core.config import settings
from app.core.database import get_db
from app.models.user import User
import os
from neo4j import GraphDatabase, Driver
from elasticsearch import Elasticsearch
import redis
from redis import Redis
import socket # 👈 소켓 타임아웃 설정에 필요

security = HTTPBearer()

# -------------------------------------------------------------------------
# 🔴 [DB 및 클라이언트 초기화] 서버 시작 시 한 번만 실행
# -------------------------------------------------------------------------

# Neo4j 드라이버 초기화
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
try:
    if NEO4J_URI:
        # 🔴 driver 초기화 시 타임아웃 설정 (5초)
        neo4j_driver = GraphDatabase.driver(
            NEO4J_URI, 
            auth=(NEO4J_USER, NEO4J_PASSWORD),
            connection_timeout=5.0  # 연결 타임아웃 5초 설정
        )
        neo4j_driver.verify_connectivity()
    else:
        neo4j_driver = None
except Exception as e:
    print(f"❌ Neo4j Driver initialization failed: {e}")
    # 오류 발생 시 드라이버를 None으로 설정하고, 서버 시작은 허용
    neo4j_driver = None

# Elasticsearch 클라이언트 초기화
ES_URL = os.getenv("ES_URL")
ES_API_KEY = os.getenv("ES_API_KEY")
try:
    if ES_URL:
        # Cloud (with API key) or Local
        if ES_API_KEY:
            es_client = Elasticsearch(
                ES_URL,
                api_key=ES_API_KEY,
                request_timeout=5.0
            )
        else:
            es_client = Elasticsearch(
                ES_URL,
                request_timeout=5.0
            )
        # 클라이언트가 실제로 연결 가능한지 핑 테스트
        if not es_client.ping():
             raise ConnectionError("ES ping failed after initialization.")
    else:
        es_client = None
except Exception as e:
    print(f"❌ Elasticsearch Client initialization failed: {e}")
    es_client = None

# Redis 클라이언트 초기화 (Railway REDIS_URL 지원)
REDIS_URL = os.getenv("REDIS_URL")
try:
    if REDIS_URL:
        # Railway/Cloud 환경: URL에서 직접 연결
        redis_client = redis.from_url(
            REDIS_URL,
            decode_responses=True,
            socket_timeout=3
        )
    else:
        # 로컬 환경: 개별 변수 사용
        REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
        REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
        redis_client = redis.Redis(
            host=REDIS_HOST,
            port=REDIS_PORT,
            decode_responses=True,
            socket_timeout=3
        )
    # 연결 테스트
    redis_client.ping()
except Exception as e:
    print(f"Warning: Redis connection failed: {e}")
    redis_client = None


# -------------------------------------------------------------------------
# 🔴 [의존성 주입 함수]
# -------------------------------------------------------------------------

def get_neo4j_driver() -> Driver:
    if neo4j_driver is None:
        raise HTTPException(status_code=503, detail="Graph Database connection is unavailable")
    return neo4j_driver

def get_es_client() -> Elasticsearch:
    if es_client is None:
        raise HTTPException(status_code=503, detail="Search Engine connection is unavailable")
    return es_client
    
def get_redis_client() -> Redis:
    if redis_client is None:
        raise HTTPException(status_code=503, detail="Redis Broker connection is unavailable")
    return redis_client

# -------------------------------------------------------------------------
# 🔴 내부 서비스 간 인증키 검증 로직 (Dify ↔ FastAPI Tool)
# -------------------------------------------------------------------------
async def verify_internal_api_key(
    x_internal_api_key: Optional[str] = Header(None, alias="X-Internal-API-Key")
) -> str:
    """
    Dify 등 내부 서비스의 툴 API 호출을 검증합니다.
    """
    INTERNAL_KEY = os.getenv("INTERNAL_API_KEY") 
    
    if not INTERNAL_KEY:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Server configuration error: INTERNAL_API_KEY is not set"
        )
    
    if x_internal_api_key != INTERNAL_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Internal API Key is missing or invalid"
        )
    return x_internal_api_key

# -------------------------------------------------------------------------
# (유저 인증 로직)
# -------------------------------------------------------------------------

async def get_current_user(
    db: AsyncSession = Depends(get_db),
    token_creds: HTTPAuthorizationCredentials = Depends(security)
) -> User:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="자격 증명을 검증할 수 없습니다.",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    token = token_creds.credentials 

    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        email: str = payload.get("sub")
        type: str = payload.get("type")
        
        if email is None or type != "access":
            raise credentials_exception
            
    except JWTError:
        raise credentials_exception
        
    result = await db.execute(select(User).where(User.email == email))
    user = result.scalar_one_or_none()
    
    if user is None:
        raise credentials_exception
    return user