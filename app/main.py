from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

from app.core import config  # noqa: F401 — load project-root .env before LangChain/LangGraph
from app.api.routers import auth, chat, threads
from app.core.logging import logger  # ensure logging configured
from app.gateway.rate_limit import limiter


def create_app() -> FastAPI:
    app = FastAPI(title="Medical RAG Assistant")
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
    app.add_middleware(SlowAPIMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(auth.router)
    app.include_router(chat.router)
    app.include_router(threads.router)
    return app


app = create_app()


@app.get("/healthz")
async def healthz():
    logger.info("health check ping")
    return {"status": "ok", "log_sink": "foreground-api"}


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
