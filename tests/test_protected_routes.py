import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from app.infra import user_store as user_store_mod


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
        "phone": "13800138010",
        "password": "testpass123",
    })
    assert r.status_code == 200, r.text
    token = r.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


def test_chat_without_token_returns_401(client):
    r = client.post("/chat", json={"user_id": "x", "message": "hi"})
    assert r.status_code == 401


def test_chat_ignores_body_user_id(client, auth_headers, monkeypatch):
    def fake_chat_once(user_id, thread_id, message):
        return {
            "user_id": user_id,
            "thread_id": thread_id or "t1",
            "reply": "ok",
            "used_docs": {"medical": [], "process": []},
        }

    monkeypatch.setattr("app.services.chat_service.chat_once", fake_chat_once)
    r = client.post("/chat", json={
        "user_id": "evil-other",
        "message": "头痛",
    }, headers=auth_headers)
    assert r.status_code != 403
    assert r.status_code == 200
    assert r.json()["user_id"] == "+8613800138010"
