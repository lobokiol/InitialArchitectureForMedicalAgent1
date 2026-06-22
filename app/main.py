from fastapi import FastAPI

from app.api.routers import chat, threads
from app.api.routers import users
from app.core.logging import logger  # ensure logging configured


def create_app() -> FastAPI:
    app = FastAPI(title="Medical RAG Assistant")
    app.include_router(chat.router)
    app.include_router(threads.router)
    app.include_router(users.router)
    return app


app = create_app()


@app.get("/healthz")
async def healthz():
    logger.info("health check ping")
    return {"status": "ok"}


@app.get("/ready")
async def ready():
    """LangGraph + OpenSearch + Redis + TriageDb readiness for local/dev."""
    from app.infra.es_client import check_opensearch
    from app.infra.redis_client import check_redis
    from app.infra.triage_session_store import check_triage_session_db
    from app.services import chat_service

    result: dict = {
        "status": "ok",
        "langgraph": {"ok": True},
        "opensearch": {},
        "redis": {},
        "triage_db": {},
    }

    try:
        result["opensearch"] = check_opensearch()
    except Exception as exc:
        result["status"] = "degraded"
        result["opensearch"] = {"ok": False, "error": str(exc)}

    try:
        result["redis"] = check_redis()
        if not result["redis"].get("ok"):
            result["status"] = "degraded"
    except Exception as exc:
        result["status"] = "degraded"
        result["redis"] = {"ok": False, "error": str(exc)}

    try:
        result["triage_db"] = check_triage_session_db()
        if not result["triage_db"].get("ok"):
            result["status"] = "degraded"
    except Exception as exc:
        result["status"] = "degraded"
        result["triage_db"] = {"ok": False, "error": str(exc)}

    if chat_service._app is None:
        result["status"] = "degraded"
        result["langgraph"] = {"ok": False}
    return result
