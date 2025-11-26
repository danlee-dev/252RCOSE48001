import os
from pathlib import Path
from fastapi import UploadFile, HTTPException
from app.core.config import settings # settings를 통해 BASE_DIR을 사용할 수 있도록 import

# 파일 저장 경로 설정 (FastAPI 서버가 실행되는 루트 폴더에 생성됨)
STORAGE_DIR = Path("storage/contracts")

# 파일 저장을 위한 초기화 (폴더 생성)
if not STORAGE_DIR.exists():
    STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    
async def save_contract_file(user_id: int, file: UploadFile) -> str:
    """
    업로드된 PDF 파일을 로컬 저장소에 저장하고 파일 URL을 반환합니다.
    """
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="PDF 파일만 업로드할 수 있습니다.")
        
    # 사용자별 폴더 생성
    user_storage_path = STORAGE_DIR / str(user_id)
    if not user_storage_path.exists():
        user_storage_path.mkdir(parents=True, exist_ok=True)
    
    # 파일 저장 (보안을 위해 실제 서비스에서는 UUID를 사용하는 것이 권장됨)
    filename = file.filename
    file_path = user_storage_path / filename
    
    # 파일 저장 (비동기 처리)
    contents = await file.read()
    with open(file_path, "wb") as f:
        f.write(contents)
        
    # 클라이언트가 접근할 수 있는 URL 반환 (FastAPI StaticFiles 경로와 일치해야 함)
    return f"/storage/contracts/{user_id}/{filename}"

# 파일 삭제 함수
def delete_contract_file(file_url: str):
    """
    저장된 파일을 삭제합니다.
    file_url: "/storage/contracts/1/contract.pdf" 형태
    """
    if not file_url:
        return
    # URL 앞의 /storage/ 등 제거 (필요시) -> 로컬 경로로 변환
    # file_url 예시: /storage/contracts/1/test.pdf
    # 실제 경로: backend/storage/contracts/1/test.pdf
    
    # 1. 상대 경로 파싱
    relative_path = file_url.lstrip("/")
    
    # 2. backend 루트 기준 경로 찾기
    # utils/file_storage.py -> utils -> app -> backend
    backend_root = Path(__file__).parent.parent.parent
    file_path = backend_root / relative_path
    
    # 3. 파일 삭제
    if file_path.exists():
        try:
            os.remove(file_path)
            print(f"🗑️ Deleted file: {file_path}")
        except OSError as e:
            print(f"Error deleting file: {e}")
    else:
        print(f"⚠️ File not found for deletion: {file_path}")