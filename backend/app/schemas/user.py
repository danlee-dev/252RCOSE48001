from pydantic import BaseModel, EmailStr, Field
from datetime import datetime
from typing import Optional

# --- 공통 ---
class UserBase(BaseModel):
    email: EmailStr = Field(..., example="test@docscanner.ai", description="사용자 이메일")
    username: str = Field(..., example="홍길동", description="사용자 이름")

# --- 회원가입 요청 ---
class UserCreate(UserBase):
    password: str = Field(..., example="1234", description="비밀번호")

    class Config:
        json_schema_extra = {
            "example": {
                "email": "test@docscanner.ai",
                "username": "김코딩",
                "password": "1234"
            }
        }

# --- 로그인용 스키마 ---
class UserLogin(BaseModel):
    email: EmailStr = Field(..., example="test@docscanner.ai")
    password: str = Field(..., example="1234")

# --- 토큰 갱신 요청 ---
class TokenRefresh(BaseModel):
    refresh_token: str = Field(..., description="발급받은 리프레시 토큰")

# --- 비밀번호 변경 요청 스키마 ---
class UserPasswordUpdate(BaseModel):
    current_password: str = Field(..., example="1234", description="현재 비밀번호")
    new_password: str = Field(..., example="5678", description="새로운 비밀번호")

# --- 응답 (Response) ---
class UserResponse(UserBase):
    id: int
    created_at: datetime

    class Config:
        from_attributes = True

class Token(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"