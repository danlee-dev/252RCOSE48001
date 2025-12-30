from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from app.api import deps
from app.core.database import get_db
from app.core.security import verify_password, get_password_hash
from app.models.user import User
from app.schemas.user import UserResponse, UserPasswordUpdate

router = APIRouter()

# 1. 내 정보 조회
@router.get("/me", response_model=UserResponse, summary="내 정보 조회")
async def read_user_me(current_user: User = Depends(deps.get_current_user)):
    """
    현재 로그인한 사용자의 정보를 조회합니다.
    (토큰 필요)
    """
    return current_user

# 2. 비밀번호 변경
@router.patch("/me/password", status_code=200, summary="비밀번호 변경")
async def update_password(
    body: UserPasswordUpdate,
    current_user: User = Depends(deps.get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    현재 비밀번호를 확인하고 새로운 비밀번호로 변경합니다.
    """
    # 1. 현재 비밀번호가 맞는지 확인
    if not verify_password(body.current_password, current_user.password):
        raise HTTPException(status_code=400, detail="현재 비밀번호가 일치하지 않습니다.")
    
    # 2. 새 비밀번호가 현재 비밀번호와 같은지 확인 (선택 사항)
    if body.current_password == body.new_password:
        raise HTTPException(status_code=400, detail="새 비밀번호는 현재 비밀번호와 다르게 설정해야 합니다.")

    # 3. 비밀번호 변경 및 저장
    current_user.password = get_password_hash(body.new_password)
    db.add(current_user)
    await db.commit()
    
    return {"message": "비밀번호가 성공적으로 변경되었습니다."}

# 3. 회원 탈퇴
@router.delete("/me", status_code=204, summary="회원 탈퇴")
async def delete_user_me(
    current_user: User = Depends(deps.get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    현재 로그인한 사용자의 계정을 영구 삭제합니다.
    """
    await db.delete(current_user)
    await db.commit()
    return