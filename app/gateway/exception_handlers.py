from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from slowapi.errors import RateLimitExceeded
from starlette.responses import Response

from app.core.logging import logger


async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    payload = {
        "detail": "请求参数不合法",
        "code": "VALIDATION_ERROR",
        "errors": exc.errors(),
    }
    return JSONResponse(status_code=422, content={"detail": payload})


async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("unhandled error: %s %s", request.method, request.url.path)
    payload = {"detail": "服务器内部错误", "code": "INTERNAL_ERROR"}
    return JSONResponse(status_code=500, content={"detail": payload})


async def rate_limit_exception_handler(
    request: Request, exc: RateLimitExceeded
) -> Response:
    detail_payload: dict = {
        "detail": "Rate limit exceeded",
        "code": "RATE_LIMITED",
    }
    response = JSONResponse(status_code=429, content={"detail": detail_payload})
    view_rate_limit = getattr(request.state, "view_rate_limit", None)
    limiter = getattr(request.app.state, "limiter", None)
    if view_rate_limit is not None and limiter is not None:
        response = limiter._inject_headers(response, view_rate_limit)
        retry_after = response.headers.get("Retry-After")
        if retry_after is not None:
            try:
                detail_payload = {**detail_payload, "retry_after": int(retry_after)}
                response = JSONResponse(status_code=429, content={"detail": detail_payload})
                response = limiter._inject_headers(response, view_rate_limit)
            except ValueError:
                pass
    return response


def register_exception_handlers(app: FastAPI) -> None:
    app.add_exception_handler(RateLimitExceeded, rate_limit_exception_handler)
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    app.add_exception_handler(Exception, unhandled_exception_handler)
