from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from app.core import config
from app.core.logging import logger


_chat_llm: ChatOpenAI | None = None
_embedding_model: OpenAIEmbeddings | None = None


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
        )
    return _embedding_model
