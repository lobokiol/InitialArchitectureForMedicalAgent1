from __future__ import annotations

import jwt as pyjwt
from fastapi import APIRouter, Depends, HTTPException, Request, status
from passlib.hash import bcrypt
from pydantic import BaseModel, Field

from app.core import config
from app.gateway.deps import CurrentUser, get_current_user
from app.gateway.errors import auth_error
from app.gateway.jwt import decode_token, issue_token_pair
from app.gateway.phone import normalize_phone
from app.gateway.rate_limit import limiter
from app.infra import wechat_client
from app.infra.token_store import get_token_store
from app.infra.user_store import get_user_store

router = APIRouter(prefix="/auth", tags=["auth"])


class RegisterRequest(BaseModel):
    phone: str
    password: str = Field(min_length=8)
    display_name: str | None = None


class LoginRequest(BaseModel):
    phone: str
    password: str


class RefreshRequest(BaseModel):
    refresh_token: str


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    expires_in: int
    token_type: str = "bearer"


class MeResponse(BaseModel):
    phone: str
    display_name: str | None
    created_at: str


class WechatLoginRequest(BaseModel):
    code: str


class WechatBindPhoneRequest(BaseModel):
    phone_code: str
    openid: str


class WechatLoginNeedPhoneResponse(BaseModel):
    need_phone: bool = True
    openid: str


def _issue_and_persist(phone: str) -> TokenResponse:
    pair = issue_token_pair(phone)
    claims = decode_token(pair.refresh_token, "refresh")
    if claims.jti is None:
        raise auth_error("AUTH_INVALID", "invalid token")
    ttl = config.JWT_REFRESH_EXPIRE_DAYS * 24 * 3600
    get_token_store().save_refresh(claims.jti, phone, ttl)
    return TokenResponse(
        access_token=pair.access_token,
        refresh_token=pair.refresh_token,
        expires_in=pair.expires_in,
    )


def _normalize_or_400(raw: str) -> str:
    try:
        return normalize_phone(raw)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"detail": "invalid phone", "code": "PHONE_INVALID"},
        )


@router.post("/register", response_model=TokenResponse)
@limiter.limit(config.RATE_LIMIT_AUTH_IP)
async def register(request: Request, body: RegisterRequest) -> TokenResponse:
    phone = _normalize_or_400(body.phone)
    pw_hash = bcrypt.hash(body.password)
    try:
        get_user_store().create_user(phone, pw_hash, body.display_name or "")
    except ValueError as exc:
        if str(exc) == "PHONE_EXISTS":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={"detail": "phone already registered", "code": "PHONE_EXISTS"},
            ) from None
        raise
    return _issue_and_persist(phone)


@router.post("/login", response_model=TokenResponse)
@limiter.limit(config.RATE_LIMIT_AUTH_IP)
async def login(request: Request, body: LoginRequest) -> TokenResponse:
    phone = _normalize_or_400(body.phone)
    if not get_user_store().verify_password(phone, body.password):
        raise auth_error("AUTH_INVALID", "invalid credentials")
    return _issue_and_persist(phone)


@router.post("/refresh", response_model=TokenResponse)
@limiter.limit("20/minute")
async def refresh(request: Request, body: RefreshRequest) -> TokenResponse:
    try:
        claims = decode_token(body.refresh_token, "refresh")
    except pyjwt.ExpiredSignatureError:
        raise auth_error("AUTH_EXPIRED", "refresh token expired")
    except pyjwt.InvalidTokenError:
        raise auth_error("AUTH_INVALID", "invalid refresh token")

    if claims.jti is not None and not get_token_store().is_refresh_valid(claims.jti):
        raise auth_error("AUTH_INVALID", "refresh token revoked")

    if claims.jti is not None:
        get_token_store().revoke_refresh(claims.jti)

    return _issue_and_persist(claims.sub)


@router.post("/logout")
async def logout(user: CurrentUser = Depends(get_current_user)) -> dict[str, bool]:
    get_token_store().revoke_all_for_phone(user.phone)
    return {"ok": True}


@router.post("/wechat/login")
@limiter.limit(config.RATE_LIMIT_AUTH_IP)
async def wechat_login(
    request: Request, body: WechatLoginRequest
) -> TokenResponse | WechatLoginNeedPhoneResponse:
    session = await wechat_client.code2session(body.code)
    openid = session["openid"]
    phone = get_user_store().get_phone_by_openid(openid)
    if phone:
        return _issue_and_persist(phone)
    return WechatLoginNeedPhoneResponse(openid=openid)


@router.post("/wechat/bind-phone", response_model=TokenResponse)
@limiter.limit(config.RATE_LIMIT_AUTH_IP)
async def wechat_bind_phone(
    request: Request, body: WechatBindPhoneRequest
) -> TokenResponse:
    phone = await wechat_client.get_phone_number(body.phone_code)
    get_user_store().upsert_wechat_user(phone, body.openid)
    return _issue_and_persist(phone)


@router.get("/me", response_model=MeResponse)
@limiter.limit(config.RATE_LIMIT_READ)
async def me(request: Request, user: CurrentUser = Depends(get_current_user)) -> MeResponse:
    record = get_user_store().get_user(user.phone)
    if record is None:
        raise auth_error("AUTH_INVALID", "user not found")
    return MeResponse(
        phone=record.phone,
        display_name=record.display_name,
        created_at=record.created_at,
    )
