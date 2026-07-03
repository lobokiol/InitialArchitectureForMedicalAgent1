from typing import Optional, List

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel

from app.gateway.deps import CurrentUser, get_current_user
from app.services.chat_service import get_session_manager


class ThreadInfo(BaseModel):
    thread_id: str
    title: str
    created_at: str
    last_active_at: str
    is_deleted: bool


class CreateThreadRequest(BaseModel):
    user_id: Optional[str] = None
    title: Optional[str] = None


class CreateThreadResponse(BaseModel):
    thread_id: str
    title: str


class DeleteThreadResponse(BaseModel):
    deleted: bool
    new_current_thread_id: Optional[str] = None


class SwitchThreadRequest(BaseModel):
    user_id: Optional[str] = None
    thread_id: str


class SwitchThreadResponse(BaseModel):
    thread_id: str
    title: str


router = APIRouter(prefix="/threads", tags=["threads"])

session_manager = get_session_manager()


def _raise_thread_error(exc: ValueError) -> None:
    code = str(exc)
    if code == "THREAD_NOT_OWNED":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={"detail": "thread not owned", "code": "THREAD_NOT_OWNED"},
        ) from None
    if code == "THREAD_NOT_FOUND":
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"detail": "thread not found", "code": "THREAD_NOT_FOUND"},
        ) from None
    raise exc


@router.get("", response_model=List[ThreadInfo])
async def list_threads(user: CurrentUser = Depends(get_current_user)):
    return session_manager.list_threads(user.phone)


@router.post("", response_model=CreateThreadResponse)
async def create_thread(
    body: CreateThreadRequest,
    user: CurrentUser = Depends(get_current_user),
):
    thread_id = session_manager.create_thread(user.phone, title=body.title)
    meta = session_manager.client.hgetall(f"thread:{thread_id}:meta")
    title = meta.get("title", thread_id) if meta else thread_id
    return CreateThreadResponse(thread_id=thread_id, title=title)


@router.delete("/{thread_id}", response_model=DeleteThreadResponse)
async def delete_thread(
    thread_id: str,
    user: CurrentUser = Depends(get_current_user),
):
    try:
        session_manager.assert_thread_owner(thread_id, user.phone)
    except ValueError as exc:
        _raise_thread_error(exc)

    new_current = session_manager.delete_thread(user.phone, thread_id)
    return DeleteThreadResponse(
        deleted=True,
        new_current_thread_id=new_current,
    )


@router.get("/current", response_model=ThreadInfo)
async def get_current_thread(user: CurrentUser = Depends(get_current_user)):
    """
    获取当前用户的当前会话，如果没有则创建一个默认会话。
    """
    cur = session_manager.get_current_thread(user.phone)
    if not cur:
        cur = session_manager.create_thread(user.phone, title="默认对话")
    else:
        try:
            session_manager.assert_thread_owner(cur, user.phone)
        except ValueError as exc:
            _raise_thread_error(exc)

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
async def switch_thread(
    body: SwitchThreadRequest,
    user: CurrentUser = Depends(get_current_user),
):
    """
    设置当前会话为指定 thread_id。
    """
    try:
        session_manager.assert_thread_owner(body.thread_id, user.phone)
    except ValueError as exc:
        _raise_thread_error(exc)

    session_manager.set_current_thread(user.phone, body.thread_id)
    meta = session_manager.client.hgetall(f"thread:{body.thread_id}:meta")
    return SwitchThreadResponse(
        thread_id=body.thread_id,
        title=meta.get("title", body.thread_id),
    )
