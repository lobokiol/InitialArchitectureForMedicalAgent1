from __future__ import annotations

import httpx
from fastapi import HTTPException, status

from app.core import config
from app.gateway.phone import normalize_phone

_WECHAT_BASE = "https://api.weixin.qq.com"


def _require_credentials() -> None:
    if not config.WECHAT_APP_ID or not config.WECHAT_APP_SECRET:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"detail": "WeChat API not configured", "code": "WECHAT_API_ERROR"},
        )


async def code2session(code: str) -> dict:
    """Exchange wx.login code for openid and session_key."""
    _require_credentials()
    params = {
        "appid": config.WECHAT_APP_ID,
        "secret": config.WECHAT_APP_SECRET,
        "js_code": code,
        "grant_type": "authorization_code",
    }
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.get(f"{_WECHAT_BASE}/sns/jscode2session", params=params)
        resp.raise_for_status()
        data = resp.json()

    if data.get("errcode", 0) != 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"detail": data.get("errmsg", "invalid code"), "code": "WECHAT_CODE_INVALID"},
        )

    result: dict = {
        "openid": data["openid"],
        "session_key": data["session_key"],
    }
    if unionid := data.get("unionid"):
        result["unionid"] = unionid
    return result


async def _get_access_token() -> str:
    params = {
        "grant_type": "client_credential",
        "appid": config.WECHAT_APP_ID,
        "secret": config.WECHAT_APP_SECRET,
    }
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.get(f"{_WECHAT_BASE}/cgi-bin/token", params=params)
        resp.raise_for_status()
        data = resp.json()

    if data.get("errcode", 0) != 0:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"detail": data.get("errmsg", "token error"), "code": "WECHAT_API_ERROR"},
        )
    return data["access_token"]


async def get_phone_number(phone_code: str) -> str:
    """Resolve wx.getPhoneNumber code to E.164 phone string."""
    _require_credentials()
    access_token = await _get_access_token()
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.post(
            f"{_WECHAT_BASE}/wxa/business/getuserphonenumber",
            params={"access_token": access_token},
            json={"code": phone_code},
        )
        resp.raise_for_status()
        data = resp.json()

    if data.get("errcode", 0) != 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"detail": data.get("errmsg", "invalid phone code"), "code": "WECHAT_CODE_INVALID"},
        )

    phone_info = data.get("phone_info") or {}
    raw = phone_info.get("purePhoneNumber") or phone_info.get("phoneNumber") or ""
    if not raw:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"detail": "missing phone in WeChat response", "code": "WECHAT_API_ERROR"},
        )
    return normalize_phone(raw)
