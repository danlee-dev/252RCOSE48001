import os
import sys
from dotenv import load_dotenv
from sqlalchemy.engine import URL

# 1. 최상위 루트 폴더의 .env 파일 로드
current_dir = os.path.dirname(os.path.abspath(__file__))
# backend/app/core -> backend/app -> backend -> 루트 (3단계 위)
root_dir = os.path.abspath(os.path.join(current_dir, "../../../"))
load_dotenv(os.path.join(root_dir, ".env"))

class Settings:
    PROJECT_NAME: str = os.getenv("PROJECT_NAME", "DocScanner AI")
    API_V1_STR: str = os.getenv("API_V1_STR", "/api/v1")
    
    # 2. PostgreSQL 설정 가져오기
    POSTGRES_USER = os.getenv("POSTGRES_USER")
    POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD")
    POSTGRES_DB = os.getenv("POSTGRES_DB")
    POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
    
    # .env 로드가 실패하거나 값이 없을 때를 대비해 기본값을 '5435'로 설정 (안전장치)
    POSTGRES_PORT = os.getenv("POSTGRES_PORT", "5435")

    # 윈도우 환경에서 localhost 이슈 방지 (IPv4 127.0.0.1 강제)
    if sys.platform.startswith("win") and POSTGRES_HOST == "localhost":
        POSTGRES_HOST = "127.0.0.1"

    # DB URL 안전하게 생성 (SQLAlchemy 공식 권장 방식)
    _db_url_obj = URL.create(
        drivername="postgresql+psycopg",
        username=POSTGRES_USER,
        password=POSTGRES_PASSWORD,
        host=POSTGRES_HOST,
        port=int(POSTGRES_PORT),
        database=POSTGRES_DB,
    )
    
    # 문자열로 변환
    DATABASE_URL: str = _db_url_obj.render_as_string(hide_password=False)

    # 4. JWT 보안 설정
    SECRET_KEY: str = os.getenv("SECRET_KEY", "super_secret_key_fallback") # 키가 없을 경우 대비
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24      # 1일
    REFRESH_TOKEN_EXPIRE_DAYS: int = 14             # 14일

settings = Settings()