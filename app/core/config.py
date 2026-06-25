import os
import getpass
from dotenv import load_dotenv

# Load environment variables early
load_dotenv(override=True)


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


# Core external service settings
DASHSCOPE_API_KEY: str = _require_env("DASHSCOPE_API_KEY")
ES_URL: str = os.getenv("ES_URL", "http://localhost:9200")
REDIS_URI: str = os.getenv("REDIS_URI", "redis://localhost:6379")
USE_MEMORY_CHECKPOINTER: bool = os.getenv("USE_MEMORY_CHECKPOINTER", "false").lower() in (
    "1",
    "true",
    "yes",
)
# Windows 原生 Redis 3.x 无 RediSearch 时，LangGraph checkpoint 回退到此 SQLite
LANGGRAPH_CHECKPOINT_DB: str = os.getenv("LANGGRAPH_CHECKPOINT_DB", "data/langgraph_checkpoints.db")

# Index / collection defaults
RAG_KB_INDEX: str = os.getenv("RAG_KB_INDEX", "rag_knowledge")
RAG_DEPT_RULES_INDEX: str = os.getenv("RAG_DEPT_RULES_INDEX", "rag_department_rules")
RAG_CLARIFY_MIN_SCORE: float = float(os.getenv("RAG_CLARIFY_MIN_SCORE", "1.2"))
RAG_CLARIFY_MIN_MARGIN: float = float(os.getenv("RAG_CLARIFY_MIN_MARGIN", "0.15"))
DISEASE_KB_INDEX: str = os.getenv("DISEASE_KB_INDEX", "disease_kb")
ES_INDEX_NAME: str = os.getenv("ES_INDEX_NAME", "hospital_procedures")
MAX_REWRITE: int = int(os.getenv("MAX_REWRITE", "2"))

# Short-term history control
MAX_HISTORY_MSGS: int = int(os.getenv("MAX_HISTORY_MSGS", "12"))
TRIM_TRIGGER_MSGS: int = int(os.getenv("TRIM_TRIGGER_MSGS", "24"))

# Model defaults
CHAT_MODEL_NAME = os.getenv("CHAT_MODEL_NAME", "qwen3-max")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "text-embedding-v2")
CHAT_BASE_URL = os.getenv("CHAT_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
EMBEDDING_BASE_URL = os.getenv("EMBEDDING_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")

# Misc
LLM_TIMEOUT = int(os.getenv("LLM_TIMEOUT", "60"))
LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "2"))
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.0"))

# Local knowledge-base scripts and JSONL data (formerly demo/)
SOURCE_DATA_DIR: str = os.getenv("SOURCE_DATA_DIR", "sourceData")

# Triage session persistence (SQLite, for eval)
TRIAGE_SESSION_DB_PATH: str = os.getenv("TRIAGE_SESSION_DB_PATH", "data/triage_sessions.db")
TRIAGE_SESSION_ENABLED: bool = os.getenv("TRIAGE_SESSION_ENABLED", "true").lower() in (
    "1",
    "true",
    "yes",
)
