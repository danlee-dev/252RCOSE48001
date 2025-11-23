from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, declarative_base
from app.core.config import settings

# 1. 엔진 생성 (settings에 있는 올바른 URL 사용)
# echo=True: SQL 실행 로그 출력 (개발용)
engine = create_async_engine(settings.DATABASE_URL, echo=True)

# 2. 세션 생성기
AsyncSessionLocal = sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False
)

# 3. 모델 조상 클래스
Base = declarative_base()

# 4. 의존성 주입 함수
async def get_db():
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()