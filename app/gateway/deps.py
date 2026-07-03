from __future__ import annotations

from dataclasses import dataclass

import jwt as pyjwt
from fastapi import Depends, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.gateway.errors import auth_error
from app.gateway.jwt import decode_token

security = HTTPBearer(auto_error=False)


@dataclass(frozen=True)
class CurrentUser:
    phone: str
    openid: str | None = None


async def get_current_user(
    request: Request,
    creds: HTTPAuthorizationCredentials | None = Depends(security),
) -> CurrentUser:
    if creds is None or not creds.credentials:
        raise auth_error("AUTH_MISSING", "authorization required")

    try:
        claims = decode_token(creds.credentials, "access")
    except pyjwt.ExpiredSignatureError:
        raise auth_error("AUTH_EXPIRED", "token expired")
    except pyjwt.InvalidTokenError:
        raise auth_error("AUTH_INVALID", "invalid token")

    user = CurrentUser(phone=claims.sub)
    request.state.user = user
    return user
