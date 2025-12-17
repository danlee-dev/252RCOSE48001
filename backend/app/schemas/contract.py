from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional, Any, Dict, List


# 버전 관리 스키마
class DocumentVersionCreate(BaseModel):
    """문서 버전 생성 요청"""
    content: str = Field(..., description="수정된 문서 전체 텍스트")
    changes: Optional[Dict[str, Any]] = Field(None, description="변경 사항 상세")
    change_summary: Optional[str] = Field(None, description="변경 요약")
    created_by: str = Field(default="user", description="생성자 (user 또는 ai)")


class DocumentVersionResponse(BaseModel):
    """문서 버전 응답"""
    id: int
    contract_id: int
    version_number: int
    content: str
    changes: Optional[Dict[str, Any]] = None
    change_summary: Optional[str] = None
    is_current: bool
    created_at: datetime
    created_by: Optional[str] = None

    class Config:
        from_attributes = True


class DocumentVersionListResponse(BaseModel):
    """문서 버전 목록 응답"""
    versions: List[DocumentVersionResponse]
    current_version: int


class ContractResponse(BaseModel):
    id: int = Field(..., example=1)
    title: str = Field(..., example="2025년 표준근로계약서")
    status: str = Field(..., example="COMPLETED")
    risk_level: Optional[str] = Field(None, example="High")
    created_at: datetime

    class Config:
        from_attributes = True

class ContractStats(BaseModel):
    """계약서 통계"""
    total: int = Field(..., description="전체 계약서 수")
    completed: int = Field(..., description="완료된 계약서 수")
    processing: int = Field(..., description="처리 중인 계약서 수")
    failed: int = Field(..., description="실패한 계약서 수")


class ContractListResponse(BaseModel):
    """계약서 목록 페이지네이션 응답"""
    items: List[ContractResponse]
    total: int = Field(..., description="전체 계약서 수")
    skip: int = Field(..., description="건너뛴 항목 수")
    limit: int = Field(..., description="페이지당 항목 수")
    stats: ContractStats = Field(..., description="전체 통계")


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