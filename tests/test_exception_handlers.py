import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.gateway.exception_handlers import register_exception_handlers
from app.main import create_app
from app.infra import user_store as user_store_mod


@pytest.fixture
def app_client(tmp_path, monkeypatch):
    db = str(tmp_path / "auth.db")
    store = user_store_mod.UserStore(db)
    store.init_schema()
    monkeypatch.setattr(user_store_mod, "get_user_store", lambda: store)
    monkeypatch.setattr("app.api.routers.auth.get_user_store", lambda: store)
    return TestClient(create_app())


def test_validation_error_returns_structured_422(app_client):
    r = app_client.post("/auth/register", json={"password": "longpass1"})
    assert r.status_code == 422
    body = r.json()["detail"]
    assert body["code"] == "VALIDATION_ERROR"
    assert body["detail"] == "请求参数不合法"
    assert len(body["errors"]) >= 1


def test_unhandled_exception_returns_internal_error():
    app = FastAPI()
    register_exception_handlers(app)

    @app.get("/boom")
    async def boom():
        raise RuntimeError("boom")

    r = TestClient(app, raise_server_exceptions=False).get("/boom")
    assert r.status_code == 500
    body = r.json()["detail"]
    assert body["code"] == "INTERNAL_ERROR"
    assert body["detail"] == "服务器内部错误"
    assert "boom" not in r.text


def test_rate_limit_returns_structured_429(app_client):
    for i in range(10):
        r = app_client.post("/auth/register", json={
            "phone": f"138001380{i:02d}",
            "password": "longpass1",
        })
        assert r.status_code == 200, r.text

    r = app_client.post("/auth/register", json={
        "phone": "13800138099",
        "password": "longpass1",
    })
    assert r.status_code == 429
    assert r.json()["detail"]["code"] == "RATE_LIMITED"
