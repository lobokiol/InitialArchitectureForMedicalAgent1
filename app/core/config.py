import getpass
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# Always load project-root .env (works when cwd is tests/, scripts/, etc.)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
_PROJECT_ROOT = PROJECT_ROOT  # backward-compatible alias
load_dotenv(PROJECT_ROOT / ".env", override=True)

_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})


def _require_env(name: str) -> str:
    """
    Fetch required environment variables.
    Falls back to an interactive prompt when running in a TTY, otherwise raises.
    """
    value = os.getenv(name)
    if value:
        return value
    if os.isatty(0):
        return getpass.getpass(f"Enter your {name}: ")
    raise RuntimeError(f"Environment variable {name} is required.")


def _env_bool(name: str, default: str = "false") -> bool:
    return os.getenv(name, default).strip().lower() in _TRUE_VALUES


def _env_int(name: str, default: str) -> int:
    return int(os.getenv(name, default))


def _env_float(name: str, default: str) -> float:
    return float(os.getenv(name, default))


# Core external service settings
DASHSCOPE_API_KEY: str = _require_env("DASHSCOPE_API_KEY")
ES_URL: str = os.getenv("ES_URL", "http://localhost:9200")
REDIS_URI: str = os.getenv("REDIS_URI", "redis://localhost:6379")
USE_MEMORY_CHECKPOINTER: bool = _env_bool("USE_MEMORY_CHECKPOINTER", "false")
# Windows 原生 Redis 3.x 无 RediSearch 时，LangGraph checkpoint 回退到此 SQLite
LANGGRAPH_CHECKPOINT_DB: str = os.getenv("LANGGRAPH_CHECKPOINT_DB", "data/langgraph_checkpoints.db")

# Index / collection defaults
RAG_KB_INDEX: str = os.getenv("RAG_KB_INDEX", "rag_knowledge")
RAG_KB_HYBRID_PIPELINE: str = os.getenv("RAG_KB_HYBRID_PIPELINE", "rag-knowledge-hybrid-pipeline")
RAG_DEPT_RULES_INDEX: str = os.getenv("RAG_DEPT_RULES_INDEX", "rag_department_rules")
# Thresholds for min-max normalized hybrid scores (~0–1); see rag-hybrid-pipeline-alignment spec
RAG_CLARIFY_MIN_SCORE: float = _env_float("RAG_CLARIFY_MIN_SCORE", "0.55")
RAG_CLARIFY_MIN_MARGIN: float = _env_float("RAG_CLARIFY_MIN_MARGIN", "0.10")
DISEASE_KB_INDEX: str = os.getenv("DISEASE_KB_INDEX", "disease_kb")

# Short-term history control
MAX_HISTORY_MSGS: int = _env_int("MAX_HISTORY_MSGS", "12")
TRIM_TRIGGER_MSGS: int = _env_int("TRIM_TRIGGER_MSGS", "24")

# Model defaults
CHAT_MODEL_NAME = os.getenv("CHAT_MODEL_NAME", "qwen3.6-flash")
CHAT_FALLBACK_MODEL_NAME = os.getenv("CHAT_FALLBACK_MODEL_NAME", "deepseek-v4-flash")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "text-embedding-v2")
CHAT_BASE_URL = os.getenv("CHAT_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
EMBEDDING_BASE_URL = os.getenv("EMBEDDING_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")

# Misc
LLM_TIMEOUT = _env_int("LLM_TIMEOUT", "60")
LLM_MAX_RETRIES = _env_int("LLM_MAX_RETRIES", "2")
LLM_TEMPERATURE = _env_float("LLM_TEMPERATURE", "0.0")

# Local knowledge-base scripts and JSONL data (formerly demo/)
SOURCE_DATA_DIR: str = os.getenv("SOURCE_DATA_DIR", "sourceData")
DISEASE_KB_PATH: str = os.getenv(
    "DISEASE_KB_PATH",
    str(PROJECT_ROOT / SOURCE_DATA_DIR / "data" / "disease_kb.jsonl"),
)

# Triage session persistence (SQLite, for eval)
TRIAGE_SESSION_DB_PATH: str = os.getenv("TRIAGE_SESSION_DB_PATH", "data/triage_sessions.db")
TRIAGE_SESSION_ENABLED: bool = _env_bool("TRIAGE_SESSION_ENABLED", "true")

MCP_ENABLED: bool = _env_bool("MCP_ENABLED", "true")
# Use the current interpreter so Linux/Windows/venv all resolve correctly.
MCP_SERVER_COMMAND: str = os.getenv(
    "MCP_SERVER_COMMAND",
    f"{sys.executable} hospital_mcp/server.py",
)
MCP_TIMEOUT_SECONDS: float = _env_float("MCP_TIMEOUT_SECONDS", "5.0")
CHAT_TIMEOUT_SECONDS: float = _env_float("CHAT_TIMEOUT_SECONDS", "60")
MCP_FOLLOWUP_ENABLED: bool = _env_bool("MCP_FOLLOWUP_ENABLED", "true")

JWT_SECRET: str = os.getenv("JWT_SECRET", "dev-only-change-me-in-local-env!!")
JWT_ACCESS_EXPIRE_MINUTES: int = _env_int("JWT_ACCESS_EXPIRE_MINUTES", "120")
JWT_REFRESH_EXPIRE_DAYS: int = _env_int("JWT_REFRESH_EXPIRE_DAYS", "7")
WECHAT_APP_ID: str = os.getenv("WECHAT_APP_ID", "")
WECHAT_APP_SECRET: str = os.getenv("WECHAT_APP_SECRET", "")
CORS_ORIGINS: list[str] = [
    o.strip() for o in os.getenv(
        "CORS_ORIGINS",
        "http://localhost:5173,https://servicewechat.com",
    ).split(",")
    if o.strip()
]
RATE_LIMIT_CHAT: str = os.getenv("RATE_LIMIT_CHAT", "")
RATE_LIMIT_READ: str = os.getenv("RATE_LIMIT_READ", "60/minute")
RATE_LIMIT_AUTH_IP: str = os.getenv("RATE_LIMIT_AUTH_IP", "10/minute")
MAX_REFRESH_PER_PHONE: int = _env_int("MAX_REFRESH_PER_PHONE", "3")
