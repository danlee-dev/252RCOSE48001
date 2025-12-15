from fastapi import APIRouter, HTTPException
from typing import Dict, Any
import json
import os

router = APIRouter()

@router.get("/", summary="2025 고용계약 체크리스트 조회")
def get_checklist() -> Dict[str, Any]:
    """
    백엔드 데이터 파일(checklist_2025.json)을 로드하여 반환합니다.
    데이터 파일 위치: backend/data/checklist_2025.json
    """

    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, "..", "..", "..", "data", "checklist_2025.json")
    
    # 절대 경로로 변환 (경로 오류 방지)
    data_path = os.path.abspath(data_path)

    try:
        with open(data_path, "r", encoding="utf-8") as f:
            checklist_data = json.load(f)
            return checklist_data
            
    except FileNotFoundError:
        print(f"❌ Checklist file not found at: {data_path}")
        raise HTTPException(status_code=500, detail="서버 내부 오류: 체크리스트 데이터 파일을 찾을 수 없습니다.")
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="서버 내부 오류: 체크리스트 데이터 파일 형식이 올바르지 않습니다.")