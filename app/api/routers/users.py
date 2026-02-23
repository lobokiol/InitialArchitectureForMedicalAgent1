from datetime import datetime
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.infra.redis_client import redis_client


class UserCreate(BaseModel):
    user_id: str
    name: Optional[str] = None


class UserInfo(BaseModel):
    user_id: str
    name: Optional[str] = None
    created_at: str


router = APIRouter(prefix="/users", tags=["users"])


@router.post("", response_model=UserInfo)
async def create_user(body: UserCreate):
    """
    创建或更新用户基本信息（仅依赖 Redis 存一份元数据）。
    """
    key = f"user:{body.user_id}:meta"
    now = datetime.utcnow().isoformat()
    # 如果已存在就保留原 created_at
    created_at = redis_client.hget(key, "created_at") or now
    redis_client.hset(
        key,
        mapping={
            "user_id": body.user_id,
            "name": body.name or "",
            "created_at": created_at,
        },
    )
    return UserInfo(user_id=body.user_id, name=body.name or "", created_at=created_at)


@router.get("/{user_id}", response_model=UserInfo)
async def get_user(user_id: str):
    key = f"user:{user_id}:meta"
    meta = redis_client.hgetall(key)
    if not meta:
        raise HTTPException(status_code=404, detail="user not found")
    return UserInfo(
        user_id=user_id,
        name=meta.get("name") or "",
        created_at=meta.get("created_at") or "",
    )
