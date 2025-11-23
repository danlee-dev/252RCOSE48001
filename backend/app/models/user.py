from sqlalchemy import Column, Integer, String, DateTime
from sqlalchemy.sql import func
from app.core.database import Base

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    password = Column(String, nullable=False)
    username = Column(String, nullable=False)
    hashed_refresh_token = Column(String, nullable=True) # 리프레시 토큰 해싱 저장
    created_at = Column(DateTime(timezone=True), server_default=func.now())