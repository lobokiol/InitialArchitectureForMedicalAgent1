# Global Exception Handler Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Register global FastAPI exception handlers so 422, 429, and 500 responses use the same nested `{"detail": {"detail", "code", ...}}` contract as existing business errors.

**Architecture:** Add `app/gateway/exception_handlers.py` with `register_exception_handlers(app)` registering handlers for `RequestValidationError`, `RateLimitExceeded`, and `Exception`. Extend `errors.py` with `api_error()`. Wire from `create_app()` replacing slowapi's default 429 handler.

**Tech Stack:** FastAPI, Starlette, slowapi, pytest, TestClient

## Global Constraints

- Nested JSON only: `body["detail"]["code"]` — do not flatten to top-level `{"detail", "code"}`
- 422: `VALIDATION_ERROR`, summary `"请求参数不合法"`, always include Pydantic `errors` array
- 500: fixed `"服务器内部错误"` / `INTERNAL_ERROR`; log traceback server-side; never expose `str(exc)` to client
- 429: `RATE_LIMITED`; preserve slowapi `Retry-After` header via `_inject_headers`
- Do not add `HTTPException` global handler; do not modify router business code

**Spec:** `docs/superpowers/specs/2026-07-04-global-exception-handler-design.md`

---

### Task 1: `api_error()` helper

**Files:**
- Modify: `app/gateway/errors.py`
- Test: `tests/test_gateway_deps.py` (regression)

**Interfaces:**
- Produces: `api_error(code: str, detail: str, status_code: int, **extra: object) -> HTTPException`
- Produces: `auth_error(...)` unchanged signature, delegates to `api_error`

- [ ] **Step 1: Update errors.py**

```python
from fastapi import HTTPException, status


def api_error(
    code: str,
    detail: str,
    status_code: int,
    **extra: object,
) -> HTTPException:
    return HTTPException(
        status_code=status_code,
        detail={"detail": detail, "code": code, **extra},
    )


def auth_error(code: str, detail: str, status_code: int = status.HTTP_401_UNAUTHORIZED) -> HTTPException:
    return api_error(code, detail, status_code)
```

- [ ] **Step 2: Regression**

Run: `pytest tests/test_gateway_deps.py tests/test_auth_api.py -q`
Expected: PASS

---

### Task 2: Exception handlers module

**Files:**
- Create: `app/gateway/exception_handlers.py`
- Test: `tests/test_exception_handlers.py`

**Interfaces:**
- Produces: `register_exception_handlers(app: FastAPI) -> None`
- Produces: `validation_exception_handler`, `unhandled_exception_handler`, `rate_limit_exception_handler`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_exception_handlers.py
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
```

- [ ] **Step 2: Run tests to verify fail**

Run: `pytest tests/test_exception_handlers.py -v`
Expected: FAIL (module not found or wrong response shape)

- [ ] **Step 3: Implement exception_handlers.py**

```python
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
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_exception_handlers.py -v`
Expected: FAIL until Task 3 wires `main.py`

---

### Task 3: Wire into `create_app()`

**Files:**
- Modify: `app/main.py`

- [ ] **Step 1: Replace slowapi default handler**

Remove:
```python
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
...
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
```

Add:
```python
from app.gateway.exception_handlers import register_exception_handlers
...
register_exception_handlers(app)
```

Place `register_exception_handlers(app)` after `app.state.limiter = limiter`.

- [ ] **Step 2: Full regression**

Run: `pytest tests/test_exception_handlers.py tests/test_gateway_deps.py tests/test_auth_api.py tests/test_gateway_ratelimit.py -q`
Expected: PASS

---

### Task 4: Enhance ratelimit test assertion

**Files:**
- Modify: `tests/test_gateway_ratelimit.py`

- [ ] **Step 1: Add body assertion to existing test**

After `assert r.status_code == 429` add:
```python
assert r.json()["detail"]["code"] == "RATE_LIMITED"
```

- [ ] **Step 2: Final regression**

Run: `pytest tests/test_exception_handlers.py tests/test_gateway_deps.py tests/test_auth_api.py tests/test_gateway_ratelimit.py -q`
Expected: PASS
