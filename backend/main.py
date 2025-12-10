import sys
from pathlib import Path

# main.py (backend/) -> 1단계 위로 이동하여 프로젝트 루트를 찾음
project_root = Path(__file__).parent.parent.resolve()
# 이미 경로에 있다면 추가하지 않도록 확인
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import os
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# 프로젝트 루트의 .env 파일 로드
load_dotenv(dotenv_path=project_root / ".env")

from app.core.database import engine, Base
from app.api.v1 import auth, contracts, users, analysis, search, chat, agent_chat, scan, checklist
from fastapi.staticfiles import StaticFiles
import uvicorn

app = FastAPI(
    title="DocScanner AI API",
    description="""
    ## DocScanner AI 백엔드 API

    법률 문서 자동 분석 서비스를 위한 RESTful API입니다.
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS 설정 (환경변수로 제어, 쉼표로 구분)
# 예: CORS_ORIGINS=http://localhost:3000,https://myapp.com
cors_origins_str = os.getenv("CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000")
cors_origins = [origin.strip() for origin in cors_origins_str.split(",") if origin.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 🔴 [보완] DB 테이블 생성 이벤트 (Alembic 사용 시에는 주석 처리 권장)
# @app.on_event("startup")
# async def init_tables():
#     # Note: Alembic 사용 시 이 코드를 실행하면 안 됩니다.
#     # 다만, 개발 편의상 필요할 때만 주석을 해제하여 사용합니다.
#     # async with engine.begin() as conn:
#     #     await conn.run_sync(Base.metadata.create_all)
#     # print("✅ DB 테이블 생성 완료!")
    
# 정적 파일 경로 등록
app.mount("/storage", StaticFiles(directory="storage"), name="storage")

# 라우터 등록
app.include_router(auth.router, prefix="/api/v1/auth", tags=["Auth"])
app.include_router(users.router, prefix="/api/v1/users", tags=["Users"])
app.include_router(contracts.router, prefix="/api/v1/contracts", tags=["Contracts"])
app.include_router(analysis.router, prefix="/api/v1/analysis", tags=["Advanced AI Analysis"])
app.include_router(search.router, prefix="/api/v1/search", tags=["Legal Search (Dify Tool)"])
app.include_router(chat.router, prefix="/api/v1/chat", tags=["Chat with Agent"])
app.include_router(agent_chat.router, prefix="/api/v1", tags=["LangGraph Agent"])
app.include_router(scan.router, prefix="/api/v1", tags=["Quick Scan"])
app.include_router(checklist.router, prefix="/api/v1/checklist", tags=["Checklist"])

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)