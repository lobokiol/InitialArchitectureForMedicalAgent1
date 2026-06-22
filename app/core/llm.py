import httpx
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from app.core import config
from app.core.logging import logger


_chat_llm: ChatOpenAI | None = None
_embedding_model: OpenAIEmbeddings | None = None
_http_client: httpx.Client | None = None
_http_async_client: httpx.AsyncClient | None = None


def _dashscope_http_client() -> httpx.Client:
    global _http_client
    if _http_client is None:
        _http_client = httpx.Client(trust_env=False, timeout=config.LLM_TIMEOUT)
    return _http_client


def _dashscope_http_async_client() -> httpx.AsyncClient:
    global _http_async_client
    if _http_async_client is None:
        _http_async_client = httpx.AsyncClient(trust_env=False, timeout=config.LLM_TIMEOUT)
    return _http_async_client


def get_chat_llm() -> ChatOpenAI:
    global _chat_llm
    if _chat_llm is None:
        logger.info("Initializing ChatOpenAI client")
        _chat_llm = ChatOpenAI(
            base_url=config.CHAT_BASE_URL,
            api_key=config.DASHSCOPE_API_KEY,
            model=config.CHAT_MODEL_NAME,
            temperature=config.LLM_TEMPERATURE,
            timeout=config.LLM_TIMEOUT,
            max_retries=config.LLM_MAX_RETRIES,
            http_client=_dashscope_http_client(),
            http_async_client=_dashscope_http_async_client(),
        )
    return _chat_llm


def get_embedding_model() -> OpenAIEmbeddings:
    global _embedding_model
    if _embedding_model is None:
        logger.info("Initializing OpenAIEmbeddings client")
        _embedding_model = OpenAIEmbeddings(
            base_url=config.EMBEDDING_BASE_URL,
            api_key=config.DASHSCOPE_API_KEY,
            model=config.EMBEDDING_MODEL_NAME,
            deployment=config.EMBEDDING_MODEL_NAME,
            check_embedding_ctx_length=False,
            http_client=_dashscope_http_client(),
            http_async_client=_dashscope_http_async_client(),
        )
    return _embedding_model
