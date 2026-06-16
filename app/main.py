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
    """LangGraph + OpenSearch readiness for local/dev."""
    from app.infra.es_client import check_opensearch
    from app.services import chat_service

    result: dict = {"status": "ok", "langgraph": {"ok": True}, "opensearch": {}}
    try:
        result["opensearch"] = check_opensearch()
    except Exception as exc:
        result["status"] = "degraded"
        result["opensearch"] = {"ok": False, "error": str(exc)}
    if chat_service._app is None:
        result["status"] = "degraded"
        result["langgraph"] = {"ok": False}
    return result
