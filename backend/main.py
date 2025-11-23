import sys
from pathlib import Path

# main.py (backend/) -> 1단계 위로 이동하여 프로젝트 루트를 찾음
project_root = Path(__file__).parent.parent.resolve()
# 이미 경로에 있다면 추가하지 않도록 확인
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from fastapi import FastAPI
from app.core.database import engine, Base
from app.api.v1 import auth, contracts, users
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

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)