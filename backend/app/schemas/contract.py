from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional, Any, Dict

class ContractResponse(BaseModel):
    id: int = Field(..., example=1)
    title: str = Field(..., example="2025년 표준근로계약서")
    status: str = Field(..., example="COMPLETED")
    risk_level: Optional[str] = Field(None, example="High")
    created_at: datetime

    class Config:
        from_attributes = True

class ContractDetailResponse(ContractResponse):
    file_url: str
    extracted_text: Optional[str] = Field(
        None,
        description="PDF에서 추출된 원본 텍스트",
        example="표준근로계약서\n\n제1조 (계약기간)..."
    )
    analysis_result: Optional[Dict[str, Any]] = Field(
        None,
        description="AI 분석 결과 JSON",
        example={
            "summary": "이 계약서는...",
            "riskClauses": [{"text": "포괄임금...", "level": "High"}]
        }
    )