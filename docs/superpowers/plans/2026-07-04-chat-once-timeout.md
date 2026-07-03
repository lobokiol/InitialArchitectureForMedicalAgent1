# chat_once 整体超时 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为 HTTP `/chat` 增加可配置整体超时（默认 60s）；超时返回固定兜底文案与 `timed_out: true`；sync `chat_once` 行为不变。

**Architecture:** 在 `chat_service` 新增 `chat_once_async`，用 `asyncio.wait_for` 包装 `run_in_executor(chat_once, ...)`；超时走 `_timeout_fallback()` 返回与成功路径同形 dict。路由层改调 async 入口并在 `ChatResponse` 暴露 `timed_out`。

**Tech Stack:** Python 3.10+, FastAPI, asyncio, pytest, pytest-asyncio (`asyncio_mode = auto`), LangGraph（不改动图本身）

## Global Constraints

- 超时边界：`chat_service.chat_once_async`（**不**在 sync `chat_once` 内加超时）
- `CHAT_TIMEOUT_SECONDS` 可配置，**默认 `60.0`**（`float(os.getenv("CHAT_TIMEOUT_SECONDS", "60"))`）
- 超时响应：HTTP **200**；`reply` = 固定兜底文案；`timed_out: true`
- 兜底文案（verbatim）：`处理时间较长，暂无法完成导诊。请稍后重试，或到医院分诊台咨询。`
- 孤儿线程：接受；`logger.warning` 记录 `user_id` / `thread_id` / 超时秒数
- 超时兜底：**不调** `triage_recorder`
- sync `chat_once`：**不修改**
- 测试始终 mock `chat_once`，不调真实 LLM

**Spec:** `docs/superpowers/specs/2026-07-04-chat-once-timeout-design.md`

---

## File Map

| File | Responsibility |
|------|----------------|
| `app/core/config.py` | `CHAT_TIMEOUT_SECONDS` 环境变量 |
| `app/services/chat_service.py` | `CHAT_TIMEOUT_MESSAGE`、`_timeout_fallback()`、`chat_once_async()` |
| `app/api/routers/chat.py` | 改调 `chat_once_async`；`ChatResponse.timed_out` |
| `tests/test_chat_timeout.py` | 超时/正常/thread_id 单测 + `/chat` 集成测 |

---

### Task 1: 配置 `CHAT_TIMEOUT_SECONDS`

**Files:**
- Modify: `app/core/config.py`（在 `MCP_TIMEOUT_SECONDS` 行附近追加）

**Interfaces:**
- Produces: `config.CHAT_TIMEOUT_SECONDS: float`（默认 `60.0`）

- [ ] **Step 1: 在 config.py 追加常量**

在 `app/core/config.py` 中 `MCP_TIMEOUT_SECONDS` 定义之后追加：

```python
CHAT_TIMEOUT_SECONDS: float = float(os.getenv("CHAT_TIMEOUT_SECONDS", "60"))
```

- [ ] **Step 2: 验证 import**

Run: `python -c "from app.core import config; assert config.CHAT_TIMEOUT_SECONDS == 60.0"`

Expected: 无输出、exit code 0

- [ ] **Step 3: Commit**

```bash
git add app/core/config.py
git commit -m "feat: add CHAT_TIMEOUT_SECONDS config"
```

---

### Task 2: `chat_once_async` 与超时兜底（TDD）

**Files:**
- Create: `tests/test_chat_timeout.py`
- Modify: `app/services/chat_service.py`

**Interfaces:**
- Consumes: `config.CHAT_TIMEOUT_SECONDS: float`
- Produces: `CHAT_TIMEOUT_MESSAGE: str`
- Produces: `_timeout_fallback(user_id: str, thread_id: str) -> Dict[str, Any]`
- Produces: `async def chat_once_async(user_id: str, thread_id: Optional[str], message: str) -> Dict[str, Any]`

- [ ] **Step 1: 写失败测试（async 层）**

创建 `tests/test_chat_timeout.py`：

```python
import time

import pytest

from app.core import config
from app.services import chat_service


def _slow_chat_once(user_id, thread_id, message):
    time.sleep(999)
    return {"user_id": user_id, "thread_id": thread_id, "reply": "never"}


def _fast_chat_once(user_id, thread_id, message):
    return {
        "user_id": user_id,
        "thread_id": thread_id or "t-fixed",
        "reply": "ok",
        "used_docs": {"medical": [], "process": []},
        "node_trace": ["decision"],
    }


@pytest.mark.asyncio
async def test_chat_once_async_times_out(monkeypatch):
    monkeypatch.setattr(config, "CHAT_TIMEOUT_SECONDS", 0.1)
    monkeypatch.setattr(chat_service, "chat_once", _slow_chat_once)

    result = await chat_service.chat_once_async("u1", "t1", "头痛")

    assert result["timed_out"] is True
    assert result["reply"] == chat_service.CHAT_TIMEOUT_MESSAGE
    assert result["user_id"] == "u1"
    assert result["thread_id"] == "t1"
    assert result["used_docs"] == {"medical": [], "process": []}
    assert result["node_trace"] == []


@pytest.mark.asyncio
async def test_chat_once_async_success_sets_timed_out_false(monkeypatch):
    monkeypatch.setattr(config, "CHAT_TIMEOUT_SECONDS", 5.0)
    monkeypatch.setattr(chat_service, "chat_once", _fast_chat_once)

    result = await chat_service.chat_once_async("u1", "t1", "头痛")

    assert result["timed_out"] is False
    assert result["reply"] == "ok"
    assert result["node_trace"] == ["decision"]


@pytest.mark.asyncio
async def test_chat_once_async_timeout_ensures_thread_id(monkeypatch):
    monkeypatch.setattr(config, "CHAT_TIMEOUT_SECONDS", 0.1)
    monkeypatch.setattr(chat_service, "chat_once", _slow_chat_once)

    result = await chat_service.chat_once_async("u1", None, "头痛")

    assert result["timed_out"] is True
    assert isinstance(result["thread_id"], str)
    assert len(result["thread_id"]) > 0
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest tests/test_chat_timeout.py -v`

Expected: FAIL — `CHAT_TIMEOUT_MESSAGE` 或 `chat_once_async` 未定义

- [ ] **Step 3: 实现 chat_service 超时层**

在 `app/services/chat_service.py` 顶部 import 区追加 `import asyncio` 与 `from app.core import config`。

在 `_app = build_app(checkpointer)` 之前（模块级，`_extract_reply` 之前亦可）追加常量：

```python
CHAT_TIMEOUT_MESSAGE = (
    "处理时间较长，暂无法完成导诊。请稍后重试，或到医院分诊台咨询。"
)
```

在 `chat_once` 函数**之前**追加：

```python
def _timeout_fallback(user_id: str, thread_id: str) -> Dict[str, Any]:
    return {
        "user_id": user_id,
        "thread_id": thread_id,
        "reply": CHAT_TIMEOUT_MESSAGE,
        "timed_out": True,
        "intent_result": None,
        "used_docs": {"medical": [], "process": []},
        "awaiting_dept_choice": False,
        "dept_choices": [],
        "awaiting_clarify": False,
        "clarify_phase": None,
        "clarify_choices": [],
        "multi_select": False,
        "dept_confidence": None,
        "dept_confidence_passed": None,
        "dept_confidence_reason": None,
        "locked_department": None,
        "recommended_department": None,
        "oncall_appointments": [],
        "oncall_fetch_error": None,
        "node_trace": [],
        "app_state": None,
    }


async def chat_once_async(
    user_id: str,
    thread_id: Optional[str],
    message: str,
) -> Dict[str, Any]:
    thread_id = _ensure_thread(user_id, thread_id)
    loop = asyncio.get_running_loop()
    try:
        result = await asyncio.wait_for(
            loop.run_in_executor(None, chat_once, user_id, thread_id, message),
            timeout=config.CHAT_TIMEOUT_SECONDS,
        )
        result["timed_out"] = False
        return result
    except asyncio.TimeoutError:
        logger.warning(
            "chat_once timed out (user_id=%s, thread_id=%s, timeout=%ss)",
            user_id,
            thread_id,
            config.CHAT_TIMEOUT_SECONDS,
        )
        return _timeout_fallback(user_id, thread_id)
```

**注意：** `chat_once` 定义在 `chat_once_async` 之后无妨（Python 在 `run_in_executor` 调用时才解析 `chat_once` 引用）；若 linter 报前向引用，将 `chat_once_async` 放在 `chat_once` **之后**。

- [ ] **Step 4: 运行测试确认通过**

Run: `pytest tests/test_chat_timeout.py -v`

Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add app/services/chat_service.py tests/test_chat_timeout.py
git commit -m "feat: add chat_once_async with configurable timeout fallback"
```

---

### Task 3: 路由接入与 HTTP 集成测试

**Files:**
- Modify: `app/api/routers/chat.py`
- Modify: `tests/test_chat_timeout.py`（追加 HTTP 测试）
- Modify: `tests/test_protected_routes.py`（更新 mock 说明，见 Step 4）

**Interfaces:**
- Consumes: `chat_service.chat_once_async(user_id, thread_id, message) -> Dict[str, Any]`
- Produces: `ChatResponse.timed_out: bool = False`
- Produces: `POST /chat` 经 `chat_once_async` 返回含 `timed_out` 的 JSON

- [ ] **Step 1: 写失败 HTTP 测试**

在 `tests/test_chat_timeout.py` 追加：

```python
import time

import pytest
from fastapi.testclient import TestClient

from app.core import config
from app.main import create_app
from app.infra import user_store as user_store_mod
from app.services import chat_service


@pytest.fixture
def client(tmp_path, monkeypatch):
    db = str(tmp_path / "auth.db")
    store = user_store_mod.UserStore(db)
    store.init_schema()
    monkeypatch.setattr(user_store_mod, "get_user_store", lambda: store)
    monkeypatch.setattr("app.api.routers.auth.get_user_store", lambda: store)
    return TestClient(create_app())


@pytest.fixture
def auth_headers(client):
    r = client.post("/auth/register", json={
        "phone": "13800138020",
        "password": "testpass123",
    })
    assert r.status_code == 200, r.text
    token = r.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


def test_chat_endpoint_returns_timeout_fallback(client, auth_headers, monkeypatch):
    monkeypatch.setattr(config, "CHAT_TIMEOUT_SECONDS", 0.1)

    def slow_chat_once(user_id, thread_id, message):
        time.sleep(999)
        return {"user_id": user_id, "thread_id": thread_id, "reply": "never"}

    monkeypatch.setattr(chat_service, "chat_once", slow_chat_once)

    r = client.post("/chat", json={"message": "头痛"}, headers=auth_headers)

    assert r.status_code == 200, r.text
    body = r.json()
    assert body["timed_out"] is True
    assert body["reply"] == chat_service.CHAT_TIMEOUT_MESSAGE
    assert body["user_id"] == "+8613800138020"
    assert isinstance(body["thread_id"], str) and body["thread_id"]


def test_chat_endpoint_success_timed_out_false(client, auth_headers, monkeypatch):
    def fast_chat_once(user_id, thread_id, message):
        return {
            "user_id": user_id,
            "thread_id": thread_id or "t1",
            "reply": "ok",
            "used_docs": {"medical": [], "process": []},
        }

    monkeypatch.setattr(chat_service, "chat_once", fast_chat_once)

    r = client.post("/chat", json={"message": "头痛"}, headers=auth_headers)

    assert r.status_code == 200, r.text
    assert r.json()["timed_out"] is False
    assert r.json()["reply"] == "ok"
```

- [ ] **Step 2: 运行 HTTP 测试确认失败**

Run: `pytest tests/test_chat_timeout.py::test_chat_endpoint_returns_timeout_fallback -v`

Expected: FAIL — 响应无 `timed_out` 字段或仍走旧 `run_in_executor` 路径

- [ ] **Step 3: 修改 chat.py**

`app/api/routers/chat.py` 变更：

1. 删除 `import asyncio`（若不再使用）
2. `ChatResponse` 增加字段：`timed_out: bool = False`
3. `chat_endpoint` 内将 `run_in_executor` 块替换为：

```python
    result = await chat_service.chat_once_async(
        user_id,
        body.thread_id,
        body.message,
    )
    return result
```

完整 `chat_endpoint` 应类似：

```python
@router.post("", response_model=ChatResponse)
@limiter.limit(config.RATE_LIMIT_CHAT)
async def chat_endpoint(
    request: Request,
    body: ChatRequest,
    user: CurrentUser = Depends(get_current_user),
):
    user_id = user.phone
    logger.info('POST /chat user_id=%s thread_id=%s message=%r', user_id, body.thread_id, body.message)
    result = await chat_service.chat_once_async(
        user_id,
        body.thread_id,
        body.message,
    )
    return result
```

- [ ] **Step 4: 更新既有 protected route 测试的 mock 返回值**

`tests/test_protected_routes.py` 中 `fake_chat_once` 无需改签名（仍 mock `chat_once`），但成功路径现在会由 `chat_once_async` 注入 `timed_out=False`。若 `fake_chat_once` 返回 dict 缺少 `used_docs` 等字段，Pydantic 可能仍接受（`used_docs` 在 `ChatResponse` 为必填）。

确认 `test_chat_ignores_body_user_id` 的 `fake_chat_once` 返回包含 `used_docs`（已有）。运行：

Run: `pytest tests/test_protected_routes.py tests/test_chat_timeout.py -v`

Expected: 全部 PASS

- [ ] **Step 5: 全量相关测试**

Run: `pytest tests/test_chat_timeout.py tests/test_protected_routes.py -v`

Expected: 全部 PASS

- [ ] **Step 6: Commit**

```bash
git add app/api/routers/chat.py tests/test_chat_timeout.py
git commit -m "feat: wire /chat to chat_once_async and expose timed_out"
```

---

## Self-Review（计划 vs Spec）

| Spec 要求 | 对应 Task |
|-----------|-----------|
| `CHAT_TIMEOUT_SECONDS` 默认 60 | Task 1 |
| `CHAT_TIMEOUT_MESSAGE` 固定文案 | Task 2 |
| `chat_once_async` + `_timeout_fallback` | Task 2 |
| `timed_out` 字段 + HTTP 200 | Task 3 |
| 成功路径 `timed_out=false` | Task 2 Step 3 + Task 3 |
| 不调 `triage_recorder`（超时路径无 invoke 完成） | Task 2 `_timeout_fallback` 不触发 recorder |
| sync `chat_once` 不变 | 无 Task 修改 `chat_once` |
| 三则测试（超时/正常/thread_id） | Task 2 + Task 3 |
| `logger.warning` 超时日志 | Task 2 `chat_once_async` |
| 非目标（无可中断图、不改 CLI） | 无对应 Task ✓ |

无 TBD / 占位符；类型与字段名与 spec 一致。

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-07-04-chat-once-timeout.md`.

**两种执行方式：**

1. **Subagent-Driven（推荐）** — 每个 Task 派生子 agent，Task 间人工/自动 review，迭代快
2. **Inline Execution** — 本会话按 `executing-plans` 逐步执行，批量 checkpoint _review

选哪种？
