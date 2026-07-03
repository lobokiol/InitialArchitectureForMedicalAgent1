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


def test_register_login_me(client):
    r = client.post("/auth/register", json={
        "phone": "13800138000",
        "password": "testpass123",
        "display_name": "Demo",
    })
    assert r.status_code == 200, r.text
    tokens = r.json()
    assert "access_token" in tokens

    me = client.get("/auth/me", headers={"Authorization": f"Bearer {tokens['access_token']}"})
    assert me.status_code == 200
    assert me.json()["phone"] == "+8613800138000"


def test_login_wrong_password(client):
    client.post("/auth/register", json={"phone": "13800138001", "password": "okpass123"})
    r = client.post("/auth/login", json={"phone": "13800138001", "password": "wrong"})
    assert r.status_code == 401
    assert r.json()["detail"]["code"] == "AUTH_INVALID"


def test_refresh(client):
    reg = client.post("/auth/register", json={"phone": "13800138002", "password": "okpass123"}).json()
    r = client.post("/auth/refresh", json={"refresh_token": reg["refresh_token"]})
    assert r.status_code == 200
    assert "access_token" in r.json()
