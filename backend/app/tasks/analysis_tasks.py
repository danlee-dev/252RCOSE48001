import os
import sys
import asyncio
from sqlalchemy.future import select
from app.core.database import AsyncSessionLocal
from app.models.contract import Contract
from app.core.celery_app import celery_app
from app.models.user import User 
import requests
from sqlalchemy.ext.asyncio import AsyncSession
from fastapi import HTTPException
from app.core.config import settings

# -------------------------------------------------------------------------
# 🔴 [Celery Task] Dify 호출 및 DB 업데이트 로직 (Worker에 의해 실행)
# -------------------------------------------------------------------------

@celery_app.task(name="analyze_contract")
def analyze_contract_task(contract_id: int):
    """
    Celery Task: Dify API를 호출하여 계약서를 분석하고 결과를 DB에 저장합니다.
    """
    # 🔴 [CRITICAL FIX] Task 실행 시점에 경로 재설정
    # Worker 프로세스가 Task 실행 시 app 모듈을 찾도록 보장합니다.
    from pathlib import Path
    backend_dir = Path(__file__).parent.parent.parent.resolve()
    if str(backend_dir) not in sys.path:
        sys.path.insert(0, str(backend_dir))

    async def run_analysis():
        async with AsyncSessionLocal() as db:
            stmt = select(Contract).where(Contract.id == contract_id)
            result = await db.execute(stmt)
            contract = result.scalar_one_or_none()

            if not contract:
                print(f"Error: Contract {contract_id} not found.")
                return

            print(f"[{contract_id}] Dify analysis STARTING for {contract.title}...")
            
            # DB 상태를 PROCESSING으로 즉시 업데이트
            contract.status = "PROCESSING"
            await db.commit()
            
            # 2. Dify API 동기식 호출
            try:
                DIFY_API_URL = os.getenv("DIFY_API_URL")
                DIFY_API_KEY = os.getenv("DIFY_API_KEY")
                
                response = requests.post(
                    DIFY_API_URL, 
                    headers={"Authorization": f"Bearer {DIFY_API_KEY}", "Content-Type": "application/json"}, 
                    json={
                        "inputs": {"file_url": contract.file_url}, 
                        "query": "이 계약서의 위험 조항을 분석하고 등급을 High/Medium/Low로 분류해줘.",
                        "user": str(contract.user_id)
                    },
                    timeout=300
                )
                response.raise_for_status() 
                dify_result = response.json()
                
                # 3. DB 상태 업데이트 및 결과 저장
                contract.status = "COMPLETED"
                contract.analysis_result = dify_result.get("answer", dify_result) 
                contract.risk_level = "Medium" 
                
                await db.commit()
                # TODO: WebSocket 푸시 알림 로직 추가
                print(f"[{contract_id}] Analysis COMPLETED. Status updated.")
                
            except requests.exceptions.RequestException as e:
                contract.status = "FAILED"
                print(f"[{contract_id}] Dify API Call FAILED: {e}")
                await db.commit()
            except Exception as e:
                contract.status = "FAILED"
                print(f"[{contract_id}] General Error in Worker: {e}")
                await db.commit()

    # 🔴 [CRITICAL FIX] Windows 환경 호환성 코드
    if sys.platform.startswith("win"):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        
    asyncio.run(run_analysis())