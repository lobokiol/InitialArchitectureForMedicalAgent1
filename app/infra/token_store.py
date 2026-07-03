"""Redis-backed refresh token jti storage and revocation."""
from __future__ import annotations

from typing import Any

from app.core import config
from app.infra import redis_client as redis_mod

_REFRESH_PREFIX = "refresh:"
_PHONE_PREFIX = "refresh:phone:"


class TokenStore:
    def __init__(self, redis: Any | None) -> None:
        self._redis = redis

    def save_refresh(self, jti: str, phone: str, ttl_seconds: int) -> None:
        if self._redis is None:
            return

        self._redis.setex(f"{_REFRESH_PREFIX}{jti}", ttl_seconds, phone)
        phone_key = f"{_PHONE_PREFIX}{phone}"
        self._redis.lpush(phone_key, jti)

        while self._redis.llen(phone_key) > config.MAX_REFRESH_PER_PHONE:
            old_jti = self._redis.rpop(phone_key)
            if old_jti:
                self._redis.delete(f"{_REFRESH_PREFIX}{old_jti}")

    def is_refresh_valid(self, jti: str) -> bool:
        if self._redis is None:
            return True
        return self._redis.get(f"{_REFRESH_PREFIX}{jti}") is not None

    def revoke_refresh(self, jti: str) -> None:
        if self._redis is None:
            return

        key = f"{_REFRESH_PREFIX}{jti}"
        phone = self._redis.get(key)
        self._redis.delete(key)
        if phone:
            self._redis.lrem(f"{_PHONE_PREFIX}{phone}", 0, jti)

    def revoke_all_for_phone(self, phone: str) -> None:
        if self._redis is None:
            return

        phone_key = f"{_PHONE_PREFIX}{phone}"
        jtis = self._redis.lrange(phone_key, 0, -1)
        if jtis:
            self._redis.delete(*[f"{_REFRESH_PREFIX}{j}" for j in jtis])
        self._redis.delete(phone_key)


_token_store: TokenStore | None = None


def get_token_store() -> TokenStore:
    global _token_store
    if _token_store is None:
        _token_store = TokenStore(redis_mod.redis_client)
    return _token_store
