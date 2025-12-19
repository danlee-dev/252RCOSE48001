from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime

# --- 1. 기본 정보 모델 (Base Models) ---

class LegalNoticeSender(BaseModel):
    """발신인(근로자) 정보"""
    name: str = Field(..., description="발신인 이름")
    address: Optional[str] = Field(None, description="발신인 주소")
    phone: Optional[str] = Field(None, description="발신인 연락처")

class LegalNoticeRecipient(BaseModel):
    """수신인(사업주/회사) 정보"""
    company_name: str = Field(..., description="수신인(회사) 이름")
    representative: Optional[str] = Field(None, description="대표자 이름")
    address: Optional[str] = Field(None, description="수신인 주소")

# --- 2. 증거 수집 가이드 관련 스키마 (GET /evidence-guide) ---

class EvidenceItem(BaseModel):
    """개별 증거 항목"""
    item_name: str = Field(..., description="증거 명칭 (예: 급여명세서)")
    description: str = Field(..., description="증거에 대한 설명 및 확보 방법")
    is_required: bool = Field(False, description="필수 증거 여부")

class ViolationEvidenceGuide(BaseModel):
    """위반 유형별 가이드"""
    violation_type: str = Field(..., description="위반 유형 (예: 임금체불)")
    severity: str = Field(..., description="위반 심각도 (HIGH/MEDIUM/LOW)")
    evidence_list: List[EvidenceItem] = Field(..., description="필요한 증거 목록")
    guide_text: str = Field(..., description="상세 가이드 텍스트 (Markdown)")

class EvidenceGuideResponse(BaseModel):
    """증거 수집 가이드 응답"""
    guides: List[ViolationEvidenceGuide]

# --- 3. 내용증명 세션 시작 (POST /start) ---

class LegalNoticeStartRequest(BaseModel):
    """세션 시작 요청"""
    contract_id: Optional[int] = Field(None, description="분석된 계약서 ID (선택)")
    sender_info: Optional[LegalNoticeSender] = None
    recipient_info: Optional[LegalNoticeRecipient] = None

class LegalNoticeSessionResponse(BaseModel):
    """세션 시작 응답"""
    session_id: str = Field(..., description="생성된 세션 UUID")
    status: str = Field(..., description="세션 상태 (collecting, generating, completed)")
    missing_info: List[str] = Field(default=[], description="아직 수집되지 않은 필수 정보 목록")
    ai_message: str = Field(..., description="AI의 첫 인사말 또는 질문")

# --- 3.1 세션 목록 조회 (GET /sessions) ---

class LegalNoticeSessionListItem(BaseModel):
    """세션 목록 아이템"""
    id: str
    title: Optional[str] = None
    status: str
    contract_id: Optional[int] = None
    contract_title: Optional[str] = None
    created_at: datetime
    updated_at: datetime

class LegalNoticeSessionListResponse(BaseModel):
    """세션 목록 응답"""
    items: List[LegalNoticeSessionListItem]
    total: int
    stats: Dict[str, int] = Field(default_factory=dict)

# --- 3.2 세션 상세 조회 (GET /sessions/{id}) ---

class LegalNoticeSessionDetail(BaseModel):
    """세션 상세 응답"""
    id: str
    title: Optional[str] = None
    status: str
    contract_id: Optional[int] = None
    contract_title: Optional[str] = None
    messages: List[Dict[str, Any]]
    collected_info: Dict[str, Any]
    final_content: Optional[str] = None
    evidence_guide: Optional[str] = None
    damage_summary: Optional[str] = None
    created_at: datetime
    updated_at: datetime

# --- 4. 대화형 정보 수집 (POST /chat) ---

class LegalNoticeChatRequest(BaseModel):
    """채팅 요청"""
    session_id: str = Field(..., description="진행 중인 세션 ID")
    user_message: str = Field(..., description="사용자 답변 내용")

class LegalNoticeChatResponse(BaseModel):
    """채팅 응답"""
    session_id: str
    ai_message: str = Field(..., description="AI의 다음 질문 또는 안내")
    is_complete: bool = Field(..., description="정보 수집 완료 여부")
    collected_info: Dict[str, Any] = Field(..., description="현재까지 수집된 정보 요약")

# --- 5. 최종 내용증명 생성 (POST /generate) ---

class LegalNoticeGenerateRequest(BaseModel):
    """생성 요청"""
    session_id: str = Field(..., description="정보 수집이 완료된 세션 ID")

class LegalNoticeResultResponse(BaseModel):
    """최종 결과 응답"""
    title: str = Field(..., description="내용증명 제목")
    content: str = Field(..., description="생성된 내용증명 본문 (Markdown 형식)")
    generated_at: datetime = Field(..., description="생성 일시")