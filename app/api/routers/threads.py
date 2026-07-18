from typing import List, Optional

from fastapi import APIRouter, Depends, Request, status
from pydantic import BaseModel

from app.core import config
from app.gateway.deps import CurrentUser, get_current_user
from app.gateway.errors import api_error
from app.gateway.rate_limit import limiter
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
        raise api_error("THREAD_NOT_OWNED", "thread not owned", status.HTTP_403_FORBIDDEN) from None
    if code == "THREAD_NOT_FOUND":
        raise api_error("THREAD_NOT_FOUND", "thread not found", status.HTTP_404_NOT_FOUND) from None
    raise exc


@router.get("", response_model=List[ThreadInfo])
@limiter.limit(config.RATE_LIMIT_READ)
async def list_threads(request: Request, user: CurrentUser = Depends(get_current_user)):
    return session_manager.list_threads(user.phone)


@router.post("", response_model=CreateThreadResponse)
@limiter.limit(config.RATE_LIMIT_READ)
async def create_thread(
    request: Request,
    body: CreateThreadRequest,
    user: CurrentUser = Depends(get_current_user),
):
    thread_id = session_manager.create_thread(user.phone, title=body.title)
    info = session_manager.get_thread_info(thread_id)
    title = info["title"] if info else thread_id
    return CreateThreadResponse(thread_id=thread_id, title=title)


@router.delete("/{thread_id}", response_model=DeleteThreadResponse)
@limiter.limit(config.RATE_LIMIT_READ)
async def delete_thread(
    request: Request,
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
@limiter.limit(config.RATE_LIMIT_READ)
async def get_current_thread(request: Request, user: CurrentUser = Depends(get_current_user)):
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

    info = session_manager.get_thread_info(cur)
    if not info:
        raise api_error(
            "THREAD_META_MISSING",
            "current thread meta missing",
            status.HTTP_500_INTERNAL_SERVER_ERROR,
        )
    return ThreadInfo(**info)


@router.post("/switch", response_model=SwitchThreadResponse)
@limiter.limit(config.RATE_LIMIT_READ)
async def switch_thread(
    request: Request,
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
    info = session_manager.get_thread_info(body.thread_id)
    title = info["title"] if info else body.thread_id
    return SwitchThreadResponse(thread_id=body.thread_id, title=title)
