from datetime import datetime
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.infra.redis_client import redis_client
from app.infra.redis_compat import hset_mapping

_memory_users: dict[str, dict[str, str]] = {}


class UserCreate(BaseModel):
    user_id: str
    name: Optional[str] = None


class UserInfo(BaseModel):
    user_id: str
    name: Optional[str] = None
    created_at: str


router = APIRouter(prefix="/users", tags=["users"])


def _upsert_user(user_id: str, name: str | None) -> UserInfo:
    now = datetime.utcnow().isoformat()
    if redis_client is not None:
        key = f"user:{user_id}:meta"
        created_at = redis_client.hget(key, "created_at") or now
        hset_mapping(
            redis_client,
            key,
            {
                "user_id": user_id,
                "name": name or "",
                "created_at": created_at,
            },
        )
        return UserInfo(user_id=user_id, name=name or "", created_at=created_at)

    rec = _memory_users.get(user_id)
    if not rec:
        rec = {"user_id": user_id, "name": name or "", "created_at": now}
        _memory_users[user_id] = rec
    elif name:
        rec["name"] = name
    return UserInfo(user_id=rec["user_id"], name=rec.get("name") or "", created_at=rec["created_at"])


@router.post("", response_model=UserInfo)
async def create_user(body: UserCreate):
    """创建或更新用户基本信息（Redis 或进程内回退）。"""
    return _upsert_user(body.user_id, body.name)


@router.get("/{user_id}", response_model=UserInfo)
async def get_user(user_id: str):
    if redis_client is not None:
        key = f"user:{user_id}:meta"
        meta = redis_client.hgetall(key)
        if not meta:
            raise HTTPException(status_code=404, detail="user not found")
        return UserInfo(
            user_id=user_id,
            name=meta.get("name") or "",
            created_at=meta.get("created_at") or "",
        )

    rec = _memory_users.get(user_id)
    if not rec:
        raise HTTPException(status_code=404, detail="user not found")
    return UserInfo(
        user_id=user_id,
        name=rec.get("name") or "",
        created_at=rec.get("created_at") or "",
    )
