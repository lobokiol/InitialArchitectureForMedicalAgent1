"""Lock health/ready routes after extraction from main."""

from fastapi.testclient import TestClient

from app.main import app
from app.services import chat_service


def test_healthz_ok():
    client = TestClient(app)
    resp = client.get("/healthz")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"


def test_ready_includes_langgraph_probe(monkeypatch):
    client = TestClient(app)
    monkeypatch.setattr(chat_service, "is_graph_ready", lambda: True)
    monkeypatch.setattr("app.infra.es_client.check_opensearch", lambda: {"ok": True})
    monkeypatch.setattr("app.infra.redis_client.check_redis", lambda: {"ok": True})
    monkeypatch.setattr(
        "app.infra.triage_session_store.check_triage_session_db", lambda: {"ok": True}
    )
    resp = client.get("/ready")
    assert resp.status_code == 200
    body = resp.json()
    assert body["langgraph"]["ok"] is True
    assert body["status"] == "ok"


def test_is_graph_ready_public_api():
    assert chat_service.is_graph_ready() is True
