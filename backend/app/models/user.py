from sqlalchemy import Column, Integer, String, DateTime
from app.core.database import Base
from app.core.timezone import now_utc


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    password = Column(String, nullable=False)
    username = Column(String, nullable=False)
    hashed_refresh_token = Column(String, nullable=True)  # 리프레시 토큰 해싱 저장
    timezone = Column(String, default="Asia/Seoul")  # 사용자별 타임존 설정
    created_at = Column(DateTime(timezone=True), default=now_utc)