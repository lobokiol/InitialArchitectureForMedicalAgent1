import redis
from contextlib import ExitStack
from langgraph.checkpoint.redis import RedisSaver

from app.core import config

redis_client = redis.Redis.from_url(config.REDIS_URI, decode_responses=True)

# LangGraph checkpoint
_stack = ExitStack()
checkpointer = _stack.enter_context(RedisSaver.from_conn_string(config.REDIS_URI))
checkpointer.setup()
