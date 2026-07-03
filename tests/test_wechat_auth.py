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


def test_wechat_login_need_phone(client, monkeypatch):
    async def fake_code2session(code):
        return {"openid": "oid-1", "session_key": "sk"}

    monkeypatch.setattr("app.infra.wechat_client.code2session", fake_code2session)
    r = client.post("/auth/wechat/login", json={"code": "abc"})
    assert r.status_code == 200
    assert r.json()["need_phone"] is True
    assert r.json()["openid"] == "oid-1"


def test_wechat_bind_phone_issues_token(client, monkeypatch):
    async def fake_phone(code):
        return "+8613800138000"

    monkeypatch.setattr("app.infra.wechat_client.get_phone_number", fake_phone)
    r = client.post("/auth/wechat/bind-phone", json={
        "openid": "oid-1",
        "phone_code": "pc",
    })
    assert r.status_code == 200
    assert "access_token" in r.json()
