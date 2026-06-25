import sqlite3
from contextlib import ExitStack
from pathlib import Path
from typing import Any

import redis

from app.core import config
from app.core.logging import logger

_redis_available = False
_checkpointer_mode = "memory"
redis_client: Any = None
checkpointer: Any = None
_stack = ExitStack()


def _ping_redis() -> redis.Redis:
    client = redis.Redis.from_url(
        config.REDIS_URI,
        decode_responses=True,
        socket_connect_timeout=2,
    )
    client.ping()
    return client


def _sqlite_checkpointer():
    from langgraph.checkpoint.sqlite import SqliteSaver

    path = Path(config.LANGGRAPH_CHECKPOINT_DB)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), check_same_thread=False)
    return SqliteSaver(conn)


def _init_redis() -> None:
    global _redis_available, redis_client, checkpointer, _checkpointer_mode

    if config.USE_MEMORY_CHECKPOINTER:
        from langgraph.checkpoint.memory import MemorySaver

        checkpointer = MemorySaver()
        _checkpointer_mode = "memory"
        logger.warning("USE_MEMORY_CHECKPOINTER=true：使用内存会话，重启后历史丢失")
        return

    try:
        redis_client = _ping_redis()
        _redis_available = True
        logger.info("Redis 已连接（会话元数据）: %s", config.REDIS_URI)
    except Exception:
        logger.exception("Redis 不可达 (%s)", config.REDIS_URI)
        from langgraph.checkpoint.memory import MemorySaver

        checkpointer = MemorySaver()
        _checkpointer_mode = "memory"
        return

    try:
        from langgraph.checkpoint.redis import RedisSaver

        checkpointer = _stack.enter_context(RedisSaver.from_conn_string(config.REDIS_URI))
        checkpointer.setup()
        _checkpointer_mode = "redis"
        logger.info("LangGraph checkpoint 使用 RedisSaver: %s", config.REDIS_URI)
        return
    except Exception as exc:
        logger.warning(
            "Redis 无 RediSearch/RedisJSON（Windows 旧版 Redis 常见），"
            "checkpoint 回退 SQLite: %s (%s)",
            config.LANGGRAPH_CHECKPOINT_DB,
            exc,
        )

    try:
        checkpointer = _sqlite_checkpointer()
        _checkpointer_mode = "sqlite"
        logger.info("LangGraph checkpoint 使用 SqliteSaver: %s", config.LANGGRAPH_CHECKPOINT_DB)
    except Exception:
        logger.exception("SqliteSaver 初始化失败，回退 MemorySaver")
        from langgraph.checkpoint.memory import MemorySaver

        checkpointer = MemorySaver()
        _checkpointer_mode = "memory"


def check_redis() -> dict[str, Any]:
    """Redis / checkpoint readiness for /ready."""
    if config.USE_MEMORY_CHECKPOINTER:
        return {
            "ok": True,
            "mode": "memory",
            "checkpointer": "memory",
            "required": False,
        }

    if redis_client is None:
        return {
            "ok": False,
            "mode": "memory_fallback",
            "checkpointer": _checkpointer_mode,
            "required": True,
            "error": f"Redis unreachable at {config.REDIS_URI}",
        }

    try:
        redis_client.ping()
        return {
            "ok": True,
            "mode": "redis",
            "checkpointer": _checkpointer_mode,
            "required": True,
        }
    except Exception as exc:
        return {
            "ok": False,
            "mode": "redis",
            "checkpointer": _checkpointer_mode,
            "required": True,
            "error": str(exc),
        }


_init_redis()
