from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import relationship
from app.core.database import Base
from app.core.timezone import now_utc

class LegalNoticeSession(Base):
    """내용증명 작성 진행 상태를 관리하는 세션 테이블"""
    __tablename__ = "legal_notice_sessions"

    id = Column(String, primary_key=True)  # 세션 식별을 위한 UUID
    contract_id = Column(Integer, ForeignKey("contracts.id", ondelete="CASCADE"), nullable=False)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)

    status = Column(String, default="collecting")  # 진행 상태: collecting(수집중), generating(생성중), completed(완료)
    
    # AI와의 대화 내역 저장 (List[Dict] 형태)
    # 예: [{"role": "ai", "content": "..."}, {"role": "user", "content": "..."}]
    messages = Column(JSONB, default=list)
    
    # 현재까지 수집된 필수 정보 저장 (Dict 형태)
    # 예: {"sender_name": "홍길동", "missing_fields": ["start_date"]}
    collected_info = Column(JSONB, default=dict)
    
    # 최종 생성된 내용증명 본문 (Markdown/Text)
    final_content = Column(Text, nullable=True)
    
    created_at = Column(DateTime(timezone=True), default=now_utc)
    updated_at = Column(DateTime(timezone=True), default=now_utc, onupdate=now_utc)

    # Contract, User 테이블과의 관계 설정
    contract = relationship("Contract", backref="legal_notice_sessions")
    user = relationship("User", backref="legal_notice_sessions")