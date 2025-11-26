import os
import sys
import asyncio
import json
from sqlalchemy.future import select
from app.core.database import AsyncSessionLocal
from app.models.contract import Contract
from app.core.celery_app import celery_app
import requests

# -------------------------------------------------------------------------
# 🔴 [CRITICAL FIX] Worker 프로세스 내에서 app 모듈 경로를 찾도록 설정
# -------------------------------------------------------------------------
from pathlib import Path
backend_dir = Path(__file__).parent.parent.parent.resolve()
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir)) 
# -------------------------------------------------------------------------

# 🔴 [추가] 전처리기 클래스 import
from app.ai.preprocessor import ContractPreprocessor

@celery_app.task(name="analyze_contract")
def analyze_contract_task(contract_id: int):
    """
    Celery Task: 
    1. PDF 전처리 (텍스트 추출 및 청킹)
    2. Dify API 호출 (추출된 텍스트 전송)
    3. 결과 파싱 및 DB 저장 (JSONB)
    """
    
    async def run_analysis():
        async with AsyncSessionLocal() as db:
            stmt = select(Contract).where(Contract.id == contract_id)
            result = await db.execute(stmt)
            contract = result.scalar_one_or_none()

            if not contract:
                print(f"Error: Contract {contract_id} not found.")
                return

            print(f"[{contract_id}] Processing START for {contract.title}...")
            
            contract.status = "PROCESSING"
            await db.commit()
            
            try:
                # -------------------------------------------------------
                # 1. [전처리 단계] PDF -> 텍스트 추출
                # -------------------------------------------------------
                # DB에 저장된 file_url(/storage/...)을 로컬 절대 경로로 변환
                relative_path = contract.file_url.lstrip("/")
                pdf_path = backend_dir / relative_path
                
                processor = ContractPreprocessor()
                
                # (1) 텍스트 추출 (pdfplumber 사용)
                full_text = processor.extract_text(str(pdf_path))
                if not full_text:
                    raise Exception("PDF 텍스트 추출 실패 (빈 내용)")
                
                # (2) 청킹 (로그용 또는 추후 검색용)
                chunks = processor.chunk_text(full_text)
                print(f"[{contract_id}] Extracted text length: {len(full_text)}, Chunks: {len(chunks)}")

                # -------------------------------------------------------
                # 2. [Dify 호출 단계] 추출된 텍스트 전송
                # -------------------------------------------------------
                DIFY_API_URL = os.getenv("DIFY_API_URL")
                DIFY_API_KEY = os.getenv("DIFY_API_KEY")
                
                payload = {
                    "inputs": {
                        # 🔴 [핵심] 전처리된 텍스트를 Dify 변수에 주입
                        "contract_text": full_text, 
                        "file_url": contract.file_url # 참고용 원본 URL
                    },
                    "query": "이 계약서의 위험 조항을 분석해줘.", 
                    "user": str(contract.user_id),
                    "response_mode": "blocking"
                }
                
                headers = {"Authorization": f"Bearer {DIFY_API_KEY}", "Content-Type": "application/json"}
                
                print(f"[{contract_id}] Calling Dify API...")
                response = requests.post(DIFY_API_URL, headers=headers, json=payload, timeout=300)
                response.raise_for_status()
                
                dify_response = response.json()
                
                # -------------------------------------------------------
                # 3. [저장 단계] 결과 저장 (JSONB)
                # -------------------------------------------------------
                contract.status = "COMPLETED"
                contract.analysis_result = dify_response # Dify 전체 응답 저장
                
                # 임시 위험도 설정 (나중에 Dify 응답 파싱 로직 추가 필요)
                # 예: contract.risk_level = dify_response.get('data', {}).get('outputs', {}).get('risk_level', 'Unknown')
                contract.risk_level = "Check" 
                
                await db.commit()
                print(f"[{contract_id}] Analysis COMPLETED successfully.")
                
            except Exception as e:
                contract.status = "FAILED"
                print(f"[{contract_id}] Error: {e}")
                await db.commit()

    # Windows 환경 호환성 코드
    if sys.platform.startswith("win"):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        
    asyncio.run(run_analysis())