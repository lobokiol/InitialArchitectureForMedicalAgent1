from fastapi import APIRouter

from app.core.logging import logger
from app.services import chat_service

router = APIRouter(tags=["health"])


@router.get("/healthz")
async def healthz():
    logger.info("health check ping")
    return {"status": "ok", "log_sink": "foreground-api"}


@router.get("/ready")
async def ready():
    """LangGraph + OpenSearch + Redis + TriageDb readiness for local/dev."""
    from app.infra.es_client import check_opensearch
    from app.infra.redis_client import check_redis
    from app.infra.triage_session_store import check_triage_session_db

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

    if not chat_service.is_graph_ready():
        result["status"] = "degraded"
        result["langgraph"] = {"ok": False}
    return result
