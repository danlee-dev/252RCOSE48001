import asyncio
import os
import sys
from dotenv import load_dotenv
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy import text
from urllib.parse import quote_plus

# 1. 윈도우 환경설정 패치
if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# 2. 환경변수 로드
current_dir = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.join(current_dir, "../.env")
load_dotenv(env_path)

# 3. 정보 가져오기
user = os.getenv("POSTGRES_USER")
password = os.getenv("POSTGRES_PASSWORD")
db_name = os.getenv("POSTGRES_DB")

# localhost 대신 127.0.0.1 사용 (IPv4 강제)
host = "127.0.0.1" 
port = os.getenv("POSTGRES_PORT", "5435")

print(f"\n🔍 [환경변수 확인]")
print(f"User: {user}")
print(f"DB: {db_name}")
print(f"Host: {host}")

# 4. 비밀번호 인코딩
encoded_pwd = quote_plus(password) if password else ""

# 5. 접속 URL
db_url = f"postgresql+psycopg://{user}:{encoded_pwd}@{host}:{port}/{db_name}"

async def test_connection():
    try:
        print("⏳ 접속 시도 중...")
        engine = create_async_engine(db_url, echo=False)
        
        async with engine.connect() as conn:
            result = await conn.execute(text("SELECT 1"))
            print("\n✅ [성공] 데이터베이스 연결 성공! (SELECT 1 결과: ", result.scalar(), ")")
            
    except Exception as e:
        print(f"\n❌ [실패] 연결 오류 발생:\n{e}")
    finally:
        await engine.dispose()

if __name__ == "__main__":
    asyncio.run(test_connection())