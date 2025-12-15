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
    # Railway 등 PaaS에서 제공하는 DATABASE_URL 우선 사용
    _raw_database_url = os.getenv("DATABASE_URL")

    if _raw_database_url:
        # Railway/Heroku 형식: postgres:// -> postgresql+psycopg://
        DATABASE_URL: str = _raw_database_url.replace(
            "postgres://", "postgresql+psycopg://"
        ).replace(
            "postgresql://", "postgresql+psycopg://"
        )
    else:
        # 로컬 개발용: 개별 환경변수에서 조합
        POSTGRES_USER = os.getenv("POSTGRES_USER")
        POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD")
        POSTGRES_DB = os.getenv("POSTGRES_DB")
        POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
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
        DATABASE_URL: str = _db_url_obj.render_as_string(hide_password=False)

    # 4. JWT 보안 설정
    SECRET_KEY: str = os.getenv("SECRET_KEY", "super_secret_key_fallback") # 키가 없을 경우 대비
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24      # 1일
    REFRESH_TOKEN_EXPIRE_DAYS: int = 14             # 14일

    # 5. LLM API 설정
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")

    # LLM 모델 선택 (하이브리드 접근법)
    LLM_RETRIEVAL_MODEL: str = os.getenv("LLM_RETRIEVAL_MODEL", "gemini-2.5-flash-lite")
    LLM_REASONING_MODEL: str = os.getenv("LLM_REASONING_MODEL", "gpt-5-mini")
    LLM_HYDE_MODEL: str = os.getenv("LLM_HYDE_MODEL", "gpt-4o-mini")
    LLM_SCAN_MODEL: str = os.getenv("LLM_SCAN_MODEL", "gemini-2.5-flash")

    # 6. 내부 서비스 인증
    INTERNAL_API_KEY: str = os.getenv("INTERNAL_API_KEY", "")

settings = Settings()