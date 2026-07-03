from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import jwt

from app.core import config

_ALGO = "HS256"


@dataclass(frozen=True)
class TokenClaims:
    sub: str
    type: str
    jti: str | None
    exp: int


@dataclass(frozen=True)
class TokenPair:
    access_token: str
    refresh_token: str
    expires_in: int


def _now() -> datetime:
    return datetime.now(timezone.utc)


def issue_token_pair(phone: str) -> TokenPair:
    access_min = config.JWT_ACCESS_EXPIRE_MINUTES
    refresh_days = config.JWT_REFRESH_EXPIRE_DAYS
    jti = uuid.uuid4().hex
    now = _now()
    access_exp = now + timedelta(minutes=access_min)
    refresh_exp = now + timedelta(days=refresh_days)
    access = jwt.encode(
        {"sub": phone, "type": "access", "iat": int(now.timestamp()), "exp": int(access_exp.timestamp())},
        config.JWT_SECRET,
        algorithm=_ALGO,
    )
    refresh = jwt.encode(
        {
            "sub": phone,
            "type": "refresh",
            "jti": jti,
            "iat": int(now.timestamp()),
            "exp": int(refresh_exp.timestamp()),
        },
        config.JWT_SECRET,
        algorithm=_ALGO,
    )
    return TokenPair(access_token=access, refresh_token=refresh, expires_in=access_min * 60)


def decode_token(token: str, expected_type: str) -> TokenClaims:
    payload = jwt.decode(token, config.JWT_SECRET, algorithms=[_ALGO])
    if payload.get("type") != expected_type:
        raise jwt.InvalidTokenError("wrong token type")
    return TokenClaims(
        sub=payload["sub"],
        type=payload["type"],
        jti=payload.get("jti"),
        exp=int(payload["exp"]),
    )
