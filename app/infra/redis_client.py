import redis
from contextlib import ExitStack
from typing import Any

from app.core import config
from app.core.logging import logger

_redis_available = False
redis_client: Any = None
checkpointer: Any = None
_stack = ExitStack()


def _init_redis() -> None:
    global _redis_available, redis_client, checkpointer

    if config.USE_MEMORY_CHECKPOINTER:
        from langgraph.checkpoint.memory import MemorySaver

        checkpointer = MemorySaver()
        logger.warning("USE_MEMORY_CHECKPOINTER=true：使用内存会话，重启后历史丢失")
        return

    try:
        from langgraph.checkpoint.redis import RedisSaver

        client = redis.Redis.from_url(config.REDIS_URI, decode_responses=True, socket_connect_timeout=2)
        client.ping()
        redis_client = client
        checkpointer = _stack.enter_context(RedisSaver.from_conn_string(config.REDIS_URI))
        checkpointer.setup()
        _redis_available = True
        logger.info("Redis checkpointer 已连接: %s", config.REDIS_URI)
    except Exception:
        from langgraph.checkpoint.memory import MemorySaver

        logger.warning(
            "Redis 不可用 (%s)，回退到 MemorySaver；线程元数据仍需要 Redis 或 USE_MEMORY_CHECKPOINTER=true",
            config.REDIS_URI,
        )
        checkpointer = MemorySaver()


def check_redis() -> dict[str, Any]:
    """Redis readiness for /ready (optional when USE_MEMORY_CHECKPOINTER=true)."""
    if config.USE_MEMORY_CHECKPOINTER:
        return {"ok": True, "mode": "memory", "required": False}

    if redis_client is None:
        return {
            "ok": False,
            "mode": "memory_fallback",
            "required": True,
            "error": f"Redis unreachable at {config.REDIS_URI}",
        }

    try:
        redis_client.ping()
        return {"ok": True, "mode": "redis", "required": True}
    except Exception as exc:
        return {"ok": False, "mode": "redis", "required": True, "error": str(exc)}


_init_redis()
