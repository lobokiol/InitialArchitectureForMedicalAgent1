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
