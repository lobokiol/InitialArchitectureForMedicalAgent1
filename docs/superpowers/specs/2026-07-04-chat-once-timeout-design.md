# chat_once 整体超时与兜底回复设计

**日期**: 2026-07-04  
**状态**: 待实现  
**范围**: 为 HTTP `/chat` 入口增加可配置整体超时；超时返回固定兜底文案与 `timed_out` 标记；sync `chat_once` 保持不变

---

## 1. 背景与目标

### 1.1 现状

- `app/services/chat_service.py` 中 `chat_once` 为同步函数，通过 `_app.stream()` 跑完整 LangGraph，**无整体超时**。
- `app/api/routers/chat.py` 用 `run_in_executor` 在线程池调用 `chat_once`，避免阻塞事件循环。
- 子组件已有独立超时：`LLM_TIMEOUT` 默认 60s、`MCP_TIMEOUT_SECONDS` 默认 5s；CLI `BACKEND_TIMEOUT` 默认 120s。
- 图内 reject 节点（`rag_miss_reject`、`low_confidence_reject`、`answer_generate` 导诊台兜底等）已有固定中文兜底文案模式。

### 1.2 目标

| 场景 | 行为 |
|------|------|
| 导诊在配置时限内完成 | 与现网一致，`timed_out=false` |
| 导诊超过整体超时 | HTTP **200**，`reply` 为固定兜底文案，`timed_out=true` |
| 直连 sync `chat_once`（CLI/单测） | **不受**本次整体超时约束（行为不变） |

### 1.3 设计决策汇总

| 项 | 决策 |
|----|------|
| 超时边界 | `chat_service` 新增 `chat_once_async`（方案 B） |
| 超时时间 | `CHAT_TIMEOUT_SECONDS` 可配置，**默认 60.0**（与 `LLM_TIMEOUT` 对齐） |
| 客户端体验 | HTTP 200 + 兜底 `reply` + `timed_out: true`（方案 B） |
| 孤儿线程 | 接受后台 LangGraph 继续跑完；`logger.warning` 记录（方案 C） |
| sync `chat_once` | 不修改 |
| 超时兜底 | 不调 `triage_recorder`（无有效 state） |

---

## 2. 架构与数据流

```
POST /chat
  → chat_endpoint
  → chat_once_async(user_id, thread_id, message)
       ├─ thread_id = _ensure_thread(user_id, thread_id)   # 超时路径也需 thread_id
       ├─ asyncio.wait_for(
       │     run_in_executor(None, chat_once, user_id, thread_id, message),
       │     timeout=CHAT_TIMEOUT_SECONDS,
       │   )
       ├─ 成功 → result["timed_out"] = False → 返回
       └─ asyncio.TimeoutError
            → logger.warning(...)
            → _timeout_fallback(user_id, thread_id)
```

**说明**：`asyncio.wait_for` 取消的是 awaitable，**不能**终止 `run_in_executor` 内正在执行的同步 LangGraph。超时后孤儿线程可能继续写入 checkpoint；本轮不干预，下轮用户可继续对话。若后续出现状态污染，单独立项做可中断图。

---

## 3. 配置与常量

### 3.1 配置

`app/core/config.py` 新增：

```python
CHAT_TIMEOUT_SECONDS: float = float(os.getenv("CHAT_TIMEOUT_SECONDS", "60"))
```

### 3.2 兜底文案

`app/services/chat_service.py` 模块级常量：

```python
CHAT_TIMEOUT_MESSAGE = (
    "处理时间较长，暂无法完成导诊。请稍后重试，或到医院分诊台咨询。"
)
```

---

## 4. 组件变更

| 文件 | 变更 |
|------|------|
| `app/core/config.py` | `CHAT_TIMEOUT_SECONDS` |
| `app/services/chat_service.py` | `CHAT_TIMEOUT_MESSAGE`、`_timeout_fallback()`、`async def chat_once_async()` |
| `app/api/routers/chat.py` | `chat_endpoint` 改调 `chat_once_async`；`ChatResponse` 增加 `timed_out: bool = False` |

### 4.1 `chat_once_async`

```python
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

### 4.2 `_timeout_fallback`

返回与 `chat_once` 成功路径**同形** dict，字段取安全默认值：

```python
{
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
```

### 4.3 `chat.py` 路由

移除路由内手写 `run_in_executor`，改为：

```python
result = await chat_service.chat_once_async(user_id, body.thread_id, body.message)
return result
```

---

## 5. 错误处理

| 场景 | 行为 |
|------|------|
| 整体超时 | HTTP 200；`reply=CHAT_TIMEOUT_MESSAGE`；`timed_out=true`；`warning` 日志 |
| 正常完成 | `timed_out=false`；其余字段与现网一致 |
| `chat_once` 抛异常（如 checkpoint 失败） | **不捕获**；FastAPI 返回 500（本次不改） |
| 超时后孤儿线程跑完 | 不干预；可能写入 checkpoint |
| 超时兜底 | 不调 `triage_recorder` |

---

## 6. 测试

新建 `tests/test_chat_timeout.py`：

1. **超时兜底**：`monkeypatch` `chat_once` 为 `time.sleep(999)`；`CHAT_TIMEOUT_SECONDS=0.1`；经 `/chat` 或 `chat_once_async` 断言 `reply == CHAT_TIMEOUT_MESSAGE`、`timed_out is True`、HTTP 200。
2. **正常路径**：`monkeypatch chat_once` 返回固定 dict；断言 `timed_out is False`。
3. **thread_id 保留**：无 `thread_id` 请求超时后，响应含合法 `thread_id`。

测试始终 mock `chat_once`，不调真实 LLM。

---

## 7. 非目标（YAGNI）

- 不实现 LangGraph 可中断 / 线程取消
- 不改 CLI `BACKEND_TIMEOUT` 或 sync `chat_once` 行为
- 不新增 `/config` 暴露超时配置
- 超时场景不落库 `triage_recorder`
