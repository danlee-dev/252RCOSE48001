from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from app.core.database import get_db
from app.core.security import get_password_hash, verify_password, create_access_token, create_refresh_token
from app.models.user import User
from app.schemas.user import UserCreate, Token, TokenRefresh, UserResponse, UserLogin
from app.api import deps

router = APIRouter()

@router.post("/signup", response_model=UserResponse, status_code=201, summary="회원가입")
async def signup(user: UserCreate, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(User).where(User.email == user.email))
    if result.scalar_one_or_none():
        raise HTTPException(status_code=400, detail="이미 가입된 이메일입니다.")
    
    new_user = User(
        email=user.email,
        password=get_password_hash(user.password),
        username=user.username
    )
    db.add(new_user)
    await db.commit()
    await db.refresh(new_user)
    return new_user

@router.post("/login", response_model=Token, summary="로그인")
async def login(user_in: UserLogin, db: AsyncSession = Depends(get_db)):
    """
    이메일과 비밀번호로 로그인하여 토큰을 발급받습니다.
    """
    result = await db.execute(select(User).where(User.email == user_in.email))
    user = result.scalar_one_or_none()
    
    if not user or not verify_password(user_in.password, user.password):
        raise HTTPException(status_code=401, detail="이메일 또는 비밀번호가 일치하지 않습니다.")
    
    access_token = create_access_token(user.email)
    refresh_token = create_refresh_token(user.email)
    
    user.hashed_refresh_token = get_password_hash(refresh_token)
    await db.commit()
    
    return {
        "access_token": access_token, 
        "refresh_token": refresh_token, 
        "token_type": "bearer"
    }

@router.post("/refresh", response_model=Token, summary="토큰 갱신")
async def refresh_token(token_in: TokenRefresh, db: AsyncSession = Depends(get_db)):
    return {"access_token": "new_token", "refresh_token": "new_refresh", "token_type": "bearer"}

@router.post("/logout", status_code=204, summary="로그아웃")
async def logout(current_user: User = Depends(deps.get_current_user), db: AsyncSession = Depends(get_db)):
    current_user.hashed_refresh_token = None
    await db.commit()
    return