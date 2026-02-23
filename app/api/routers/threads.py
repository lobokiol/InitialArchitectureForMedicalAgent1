from typing import Optional, List

from fastapi import APIRouter, Query
from pydantic import BaseModel

from app.services.chat_service import get_session_manager


class ThreadInfo(BaseModel):
    thread_id: str
    title: str
    created_at: str
    last_active_at: str
    is_deleted: bool


class CreateThreadRequest(BaseModel):
    user_id: str
    title: Optional[str] = None


class CreateThreadResponse(BaseModel):
    thread_id: str
    title: str


class DeleteThreadResponse(BaseModel):
    deleted: bool
    new_current_thread_id: Optional[str] = None


class SwitchThreadRequest(BaseModel):
    user_id: str
    thread_id: str


class SwitchThreadResponse(BaseModel):
    thread_id: str
    title: str


router = APIRouter(prefix="/threads", tags=["threads"])

session_manager = get_session_manager()


@router.get("", response_model=List[ThreadInfo])
async def list_threads(user_id: str = Query(..., description="用户 ID")):
    return session_manager.list_threads(user_id)


@router.post("", response_model=CreateThreadResponse)
async def create_thread(body: CreateThreadRequest):
    thread_id = session_manager.create_thread(body.user_id, title=body.title)
    meta = session_manager.client.hgetall(f"thread:{thread_id}:meta")
    title = meta.get("title", thread_id) if meta else thread_id
    return CreateThreadResponse(thread_id=thread_id, title=title)


@router.delete("/{thread_id}", response_model=DeleteThreadResponse)
async def delete_thread(thread_id: str, user_id: str = Query(..., description="用户 ID")):
    meta_key = f"thread:{thread_id}:meta"
    if not session_manager.client.exists(meta_key):
        return DeleteThreadResponse(deleted=False, new_current_thread_id=None)

    new_current = session_manager.delete_thread(user_id, thread_id)
    return DeleteThreadResponse(
        deleted=True,
        new_current_thread_id=new_current,
    )


@router.get("/current", response_model=ThreadInfo)
async def get_current_thread(user_id: str = Query(..., description="用户 ID")):
    """
    获取当前用户的当前会话，如果没有则创建一个默认会话。
    """
    cur = session_manager.get_current_thread(user_id)
    if not cur:
        cur = session_manager.create_thread(user_id, title="默认对话")
    meta = session_manager.client.hgetall(f"thread:{cur}:meta")
    if not meta:
        raise ValueError("current thread meta missing")
    return ThreadInfo(
        thread_id=cur,
        title=meta.get("title", cur),
        created_at=meta.get("created_at", ""),
        last_active_at=meta.get("last_active_at", ""),
        is_deleted=meta.get("is_deleted") == "1",
    )


@router.post("/switch", response_model=SwitchThreadResponse)
async def switch_thread(body: SwitchThreadRequest):
    """
    设置当前会话为指定 thread_id。
    """
    meta_key = f"thread:{body.thread_id}:meta"
    if not session_manager.client.exists(meta_key):
        raise ValueError("thread not found")

    session_manager.set_current_thread(body.user_id, body.thread_id)
    meta = session_manager.client.hgetall(meta_key)
    return SwitchThreadResponse(
        thread_id=body.thread_id,
        title=meta.get("title", body.thread_id),
    )
