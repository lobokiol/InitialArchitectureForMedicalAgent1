from datetime import datetime
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel


class UserCreate(BaseModel):
    user_id: str
    name: Optional[str] = None


class UserInfo(BaseModel):
    user_id: str
    name: Optional[str] = None
    created_at: str


router = APIRouter(prefix="/users", tags=["users"])


@router.post("", response_model=UserInfo, deprecated=True)
async def create_user(body: UserCreate):
    raise HTTPException(
        status_code=410,
        detail={"detail": "use POST /auth/register", "code": "DEPRECATED"},
    )


@router.get("/{user_id}", response_model=UserInfo, deprecated=True)
async def get_user(user_id: str):
    raise HTTPException(
        status_code=410,
        detail={"detail": "use GET /auth/me", "code": "DEPRECATED"},
    )
