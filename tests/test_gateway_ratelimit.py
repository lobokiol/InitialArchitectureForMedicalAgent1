import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from app.infra import user_store as user_store_mod
from app.gateway.rate_limit import rate_limit_key


@pytest.fixture
def client(tmp_path, monkeypatch):
    db = str(tmp_path / "auth.db")
    store = user_store_mod.UserStore(db)
    store.init_schema()
    monkeypatch.setattr(user_store_mod, "get_user_store", lambda: store)
    monkeypatch.setattr("app.api.routers.auth.get_user_store", lambda: store)
    return TestClient(create_app())


def test_rate_limit_key_auth_uses_ip():
    class FakeRequest:
        url = type("URL", (), {"path": "/auth/login"})()
        state = type("State", (), {})()
        client = type("Client", (), {"host": "203.0.113.1"})()

    assert rate_limit_key(FakeRequest()) == "203.0.113.1"


def test_rate_limit_key_chat_uses_phone():
    class FakeRequest:
        url = type("URL", (), {"path": "/chat"})()
        state = type("State", (), {"user": type("User", (), {"phone": "+8613800138000"})()})()

    assert rate_limit_key(FakeRequest()) == "+8613800138000"


def test_auth_register_rate_limit_by_ip(client):
    for i in range(10):
        r = client.post("/auth/register", json={
            "phone": f"138001380{i:02d}",
            "password": "longpass1",
        })
        assert r.status_code == 200, r.text

    r = client.post("/auth/register", json={
        "phone": "13800138099",
        "password": "longpass1",
    })
    assert r.status_code == 429
