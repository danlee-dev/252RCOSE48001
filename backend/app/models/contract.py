from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Text, Boolean
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import relationship
from app.core.database import Base
from app.core.timezone import now_utc

class Contract(Base):
    __tablename__ = "contracts"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)

    title = Column(String, default="분석중...")
    file_url = Column(String, nullable=False)
    status = Column(String, default="PENDING")
    extracted_text = Column(Text, nullable=True)  # PDF에서 추출된 원본 텍스트
    analysis_result = Column(JSONB, nullable=True)
    risk_level = Column(String, nullable=True)
    created_at = Column(DateTime(timezone=True), default=now_utc)

    user = relationship("User", backref="contracts")
    
    # cascade="all, delete-orphan" 추가 (계약서 삭제 시 버전들도 같이 삭제됨)
    versions = relationship(
        "DocumentVersion", 
        back_populates="contract", 
        order_by="DocumentVersion.version_number",
        cascade="all, delete-orphan" 
    )

class DocumentVersion(Base):
    """문서 버전 관리 테이블 - Google Docs 스타일 버전 히스토리"""
    __tablename__ = "document_versions"

    id = Column(Integer, primary_key=True, index=True)
    # ondelete="CASCADE"는 DB 레벨 설정, cascade="..."는 ORM 레벨 설정
    contract_id = Column(Integer, ForeignKey("contracts.id", ondelete="CASCADE"), nullable=False)

    version_number = Column(Integer, nullable=False, default=1)
    content = Column(Text, nullable=False)  # 해당 버전의 문서 텍스트
    changes = Column(JSONB, nullable=True)  # 변경 사항 상세 (수정된 조항들)
    change_summary = Column(String, nullable=True)  # 변경 요약 설명

    is_current = Column(Boolean, default=False)  # 현재 활성 버전인지
    created_at = Column(DateTime(timezone=True), default=now_utc)
    created_by = Column(String, nullable=True)  # "user" 또는 "ai"

    contract = relationship("Contract", back_populates="versions")