"""Deprecated user routes (not mounted). Prefer /auth/register and /auth/me."""

from typing import Optional

from fastapi import APIRouter, status
from pydantic import BaseModel

from app.gateway.errors import api_error


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
    raise api_error("DEPRECATED", "use POST /auth/register", status.HTTP_410_GONE)


@router.get("/{user_id}", response_model=UserInfo, deprecated=True)
async def get_user(user_id: str):
    raise api_error("DEPRECATED", "use GET /auth/me", status.HTTP_410_GONE)
