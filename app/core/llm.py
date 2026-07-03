from __future__ import annotations

from typing import Any

import httpx
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from app.core import config
from app.core.logging import logger


_chat_llms: dict[str, ChatOpenAI] = {}
_chat_llm_with_fallback: FallbackChatModel | None = None
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


def _create_chat_llm(model: str) -> ChatOpenAI:
    if model not in _chat_llms:
        logger.info("Initializing ChatOpenAI client model=%s", model)
        _chat_llms[model] = ChatOpenAI(
            base_url=config.CHAT_BASE_URL,
            api_key=config.DASHSCOPE_API_KEY,
            model=model,
            temperature=config.LLM_TEMPERATURE,
            timeout=config.LLM_TIMEOUT,
            max_retries=config.LLM_MAX_RETRIES,
            http_client=_dashscope_http_client(),
            http_async_client=_dashscope_http_async_client(),
        )
    return _chat_llms[model]


class FallbackChatModel:
    """Primary chat model with optional fallback on invoke failure."""

    def __init__(
        self,
        *,
        primary_name: str,
        fallback_name: str | None,
        primary: ChatOpenAI,
        fallback: ChatOpenAI | None,
    ) -> None:
        self._primary_name = primary_name
        self._fallback_name = fallback_name
        self._primary = primary
        self._fallback = fallback

    def _run_with_fallback(self, method: str, *args: Any, **kwargs: Any) -> Any:
        try:
            return getattr(self._primary, method)(*args, **kwargs)
        except Exception as exc:
            if self._fallback is None:
                raise
            logger.warning(
                "Primary LLM %s failed (%s), falling back to %s",
                self._primary_name,
                exc,
                self._fallback_name,
            )
            return getattr(self._fallback, method)(*args, **kwargs)

    def invoke(self, *args: Any, **kwargs: Any) -> Any:
        return self._run_with_fallback("invoke", *args, **kwargs)

    async def ainvoke(self, *args: Any, **kwargs: Any) -> Any:
        try:
            return await self._primary.ainvoke(*args, **kwargs)
        except Exception as exc:
            if self._fallback is None:
                raise
            logger.warning(
                "Primary LLM %s failed (%s), falling back to %s",
                self._primary_name,
                exc,
                self._fallback_name,
            )
            return await self._fallback.ainvoke(*args, **kwargs)

    def with_structured_output(self, schema: Any, **kwargs: Any) -> FallbackChatModel:
        primary = self._primary.with_structured_output(schema, **kwargs)
        fallback = (
            self._fallback.with_structured_output(schema, **kwargs)
            if self._fallback is not None
            else None
        )
        return FallbackChatModel(
            primary_name=self._primary_name,
            fallback_name=self._fallback_name,
            primary=primary,  # type: ignore[arg-type]
            fallback=fallback,  # type: ignore[arg-type]
        )


def get_chat_llm() -> FallbackChatModel:
    global _chat_llm_with_fallback
    if _chat_llm_with_fallback is None:
        fallback_name = (config.CHAT_FALLBACK_MODEL_NAME or "").strip() or None
        fallback = _create_chat_llm(fallback_name) if fallback_name else None
        _chat_llm_with_fallback = FallbackChatModel(
            primary_name=config.CHAT_MODEL_NAME,
            fallback_name=fallback_name,
            primary=_create_chat_llm(config.CHAT_MODEL_NAME),
            fallback=fallback,
        )
    return _chat_llm_with_fallback


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
