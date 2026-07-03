import asyncio
import time

import pytest
from fastapi.testclient import TestClient

from app.core import config
from app.main import create_app
from app.infra import user_store as user_store_mod
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


@pytest.fixture
def client(tmp_path, monkeypatch):
    db = str(tmp_path / "auth.db")
    store = user_store_mod.UserStore(db)
    store.init_schema()
    monkeypatch.setattr(user_store_mod, "get_user_store", lambda: store)
    monkeypatch.setattr("app.api.routers.auth.get_user_store", lambda: store)
    return TestClient(create_app())


@pytest.fixture
def auth_headers(client):
    r = client.post("/auth/register", json={
        "phone": "13800138020",
        "password": "testpass123",
    })
    assert r.status_code == 200, r.text
    token = r.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


def test_chat_endpoint_returns_timeout_fallback(client, auth_headers, monkeypatch):
    monkeypatch.setattr(config, "CHAT_TIMEOUT_SECONDS", 0.1)

    def slow_chat_once(user_id, thread_id, message):
        time.sleep(0.5)
        return {"user_id": user_id, "thread_id": thread_id, "reply": "never"}

    monkeypatch.setattr(chat_service, "chat_once", slow_chat_once)

    r = client.post("/chat", json={"message": "头痛"}, headers=auth_headers)

    assert r.status_code == 200, r.text
    body = r.json()
    assert body["timed_out"] is True
    assert body["reply"] == chat_service.CHAT_TIMEOUT_MESSAGE
    assert body["user_id"] == "+8613800138020"
    assert isinstance(body["thread_id"], str) and body["thread_id"]


def test_chat_endpoint_success_timed_out_false(client, auth_headers, monkeypatch):
    def fast_chat_once(user_id, thread_id, message):
        return {
            "user_id": user_id,
            "thread_id": thread_id or "t1",
            "reply": "ok",
            "used_docs": {"medical": [], "process": []},
        }

    monkeypatch.setattr(chat_service, "chat_once", fast_chat_once)

    r = client.post("/chat", json={"message": "头痛"}, headers=auth_headers)

    assert r.status_code == 200, r.text
    assert r.json()["timed_out"] is False
    assert r.json()["reply"] == "ok"
