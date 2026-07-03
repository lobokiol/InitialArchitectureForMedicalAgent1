from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from app.gateway.deps import CurrentUser, get_current_user
from app.gateway.jwt import issue_token_pair


def _make_client() -> TestClient:
    app = FastAPI()

    @app.get("/me")
    async def me(user: CurrentUser = Depends(get_current_user)):
        return {"phone": user.phone}

    return TestClient(app)


def test_missing_token_returns_auth_missing():
    r = _make_client().get("/me")
    assert r.status_code == 401
    assert r.json()["detail"]["code"] == "AUTH_MISSING"


def test_valid_access_token():
    pair = issue_token_pair("+8613800138000")
    r = _make_client().get("/me", headers={"Authorization": f"Bearer {pair.access_token}"})
    assert r.status_code == 200
    assert r.json()["phone"] == "+8613800138000"


def test_refresh_token_rejected():
    pair = issue_token_pair("+8613800138000")
    r = _make_client().get("/me", headers={"Authorization": f"Bearer {pair.refresh_token}"})
    assert r.status_code == 401
    assert r.json()["detail"]["code"] == "AUTH_INVALID"


def test_expired_token_returns_auth_expired(monkeypatch):
    from app.core import config

    monkeypatch.setattr(config, "JWT_ACCESS_EXPIRE_MINUTES", -1)
    pair = issue_token_pair("+8613800138000")
    r = _make_client().get("/me", headers={"Authorization": f"Bearer {pair.access_token}"})
    assert r.status_code == 401
    assert r.json()["detail"]["code"] == "AUTH_EXPIRED"
