from slowapi import Limiter
from slowapi.util import get_remote_address
from starlette.requests import Request

from app.core import config
from app.infra.redis_client import redis_client


def rate_limit_key(request: Request) -> str:
    path = request.url.path
    if path == "/auth/me":
        user = getattr(request.state, "user", None)
        if user is not None:
            return user.phone
        return get_remote_address(request) or "unknown"
    if path.startswith("/auth/"):
        return get_remote_address(request) or "unknown"
    user = getattr(request.state, "user", None)
    if user is not None:
        return user.phone
    return get_remote_address(request) or "unknown"


_storage = config.REDIS_URI if redis_client is not None else "memory://"
limiter = Limiter(key_func=rate_limit_key, storage_uri=_storage, default_limits=[])
