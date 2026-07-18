from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from slowapi.middleware import SlowAPIMiddleware

from app.core import config  # noqa: F401 — load project-root .env before LangChain/LangGraph
from app.api.routers import auth, chat, health, threads
from app.core.logging import logger  # ensure logging configured
from app.gateway.exception_handlers import register_exception_handlers
from app.gateway.rate_limit import limiter


def create_app() -> FastAPI:
    app = FastAPI(title="Medical RAG Assistant")
    app.state.limiter = limiter
    register_exception_handlers(app)
    app.add_middleware(SlowAPIMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(health.router)
    app.include_router(auth.router)
    app.include_router(chat.router)
    app.include_router(threads.router)
    return app


app = create_app()
logger.info("FastAPI app created")
