import asyncio
import time

from app.core import config
from app.services import chat_service


def _slow_chat_once(user_id, thread_id, message):
    time.sleep(0.5)
    return {"user_id": user_id, "thread_id": thread_id, "reply": "never"}


def _fast_chat_once(user_id, thread_id, message):
    return {
        "user_id": user_id,
        "thread_id": thread_id or "t-fixed",
        "reply": "ok",
        "used_docs": {"medical": [], "process": []},
        "node_trace": ["decision"],
    }


def test_chat_once_async_times_out(monkeypatch):
    monkeypatch.setattr(config, "CHAT_TIMEOUT_SECONDS", 0.1)
    monkeypatch.setattr(chat_service, "chat_once", _slow_chat_once)

    result = asyncio.run(chat_service.chat_once_async("u1", "t1", "头痛"))

    assert result["timed_out"] is True
    assert result["reply"] == chat_service.CHAT_TIMEOUT_MESSAGE
    assert result["user_id"] == "u1"
    assert result["thread_id"] == "t1"
    assert result["used_docs"] == {"medical": [], "process": []}
    assert result["node_trace"] == []


def test_chat_once_async_success_sets_timed_out_false(monkeypatch):
    monkeypatch.setattr(config, "CHAT_TIMEOUT_SECONDS", 5.0)
    monkeypatch.setattr(chat_service, "chat_once", _fast_chat_once)

    result = asyncio.run(chat_service.chat_once_async("u1", "t1", "头痛"))

    assert result["timed_out"] is False
    assert result["reply"] == "ok"
    assert result["node_trace"] == ["decision"]


def test_chat_once_async_timeout_ensures_thread_id(monkeypatch):
    monkeypatch.setattr(config, "CHAT_TIMEOUT_SECONDS", 0.1)
    monkeypatch.setattr(chat_service, "chat_once", _slow_chat_once)

    result = asyncio.run(chat_service.chat_once_async("u1", None, "头痛"))

    assert result["timed_out"] is True
    assert isinstance(result["thread_id"], str)
    assert len(result["thread_id"]) > 0
