# FastAPI 全局 Exception Handler：统一结构化错误

**日期**: 2026-07-04  
**状态**: 已实现  
**范围**: 在现有 FastAPI 应用内补齐全局异常处理，使 422 校验错误、429 限流、500 未捕获异常与现有业务错误使用同一嵌套 JSON 契约；不改动现有 router 抛错方式，不拍平响应形状。

**前置**: `app/gateway/errors.py`（`auth_error`）、`app/main.py`（slowapi `RateLimitExceeded` handler）、[API 网关设计 §8](./2026-07-04-api-gateway-jwt-wechat-ratelimit-design.md#8-错误处理)

---

## 1. 背景与目标

### 1.1 现状

- 业务错误已在各 router / `deps` 中通过 `HTTPException(detail={"detail": "...", "code": "..."})` 或 `auth_error()` 抛出
- 测试与客户端约定访问路径为 `response["detail"]["code"]`（FastAPI 将 `HTTPException.detail` 包在顶层 `detail` 键下，形成**嵌套**结构）
- `app/main.py` 仅注册 slowapi 默认 `_rate_limit_exceeded_handler`，429 响应为 `{"error": "..."}`，与网关错误契约不一致
- 无 `RequestValidationError`（422）全局处理——FastAPI 默认返回 `detail: [{loc, msg, type}]` 数组，无 `code`
- 无 `Exception` 兜底——未捕获异常返回通用 500，无 `code`

### 1.2 目标

| 能力 | 说明 |
|------|------|
| 422 统一格式 | 参数校验失败返回 `VALIDATION_ERROR` + 人话摘要 + Pydantic `errors` 数组 |
| 500 统一格式 | 未捕获异常返回固定文案 `INTERNAL_ERROR`，服务端记录完整 traceback |
| 429 对齐契约 | 替换 slowapi 默认 handler，返回 `RATE_LIMITED`，保留 `Retry-After` header |
| 向后兼容 | 保持嵌套 JSON；现有 `test_gateway_deps`、`test_auth_api` 等断言无需修改 |
| 最小侵入 | 不新增 `HTTPException` 全局 handler；不修改各 router 现有 `raise` 调用 |

### 1.3 明确不做

- **不拍平**响应为顶层 `{"detail": "...", "code": "..."}`（避免 breaking change）
- **不引入**自定义 `AppError` 异常类或大规模替换 `HTTPException`
- **不按环境**切换 422/500 的客户端可见内容（422 始终带 `errors`；500 始终固定文案）

---

## 2. 响应契约

### 2.1 嵌套结构（全站统一）

所有经全局 handler 处理的错误，HTTP body 形如：

```json
{
  "detail": {
    "detail": "<人话描述>",
    "code": "<机器可读码>",
    "<可选扩展字段>": "..."
  }
}
```

与现有业务错误一致；客户端继续用 `body["detail"]["code"]` 分支。

### 2.2 错误码表（本 spec 新增/规范部分）

| HTTP | 场景 | code | 扩展字段 |
|------|------|------|----------|
| 422 | 请求参数/Body 校验失败 | `VALIDATION_ERROR` | `errors`: Pydantic 原始数组 |
| 429 | slowapi 限流超限 | `RATE_LIMITED` | `retry_after`（秒，有则带） |
| 500 | 未捕获 `Exception` | `INTERNAL_ERROR` | 无 |

业务错误（401/403/400/410 等）仍由既有 router 定义，本 spec 不枚举变更。

### 2.3 响应示例

**422**

```json
{
  "detail": {
    "detail": "请求参数不合法",
    "code": "VALIDATION_ERROR",
    "errors": [
      {
        "type": "missing",
        "loc": ["body", "phone"],
        "msg": "Field required",
        "input": {}
      }
    ]
  }
}
```

**500**

```json
{
  "detail": {
    "detail": "服务器内部错误",
    "code": "INTERNAL_ERROR"
  }
}
```

**429**

```json
{
  "detail": {
    "detail": "Rate limit exceeded",
    "code": "RATE_LIMITED",
    "retry_after": 42
  }
}
```

`Retry-After` HTTP header 继续由 slowapi `_inject_headers` 注入，与 body 中 `retry_after` 并存。

---

## 3. 架构

```text
create_app()
  └─ register_exception_handlers(app)     # app/gateway/exception_handlers.py
       ├─ RateLimitExceeded           → rate_limit_exception_handler
       ├─ RequestValidationError      → validation_exception_handler
       └─ Exception                   → unhandled_exception_handler

业务路由（不变）
  └─ raise HTTPException(detail={"detail": "...", "code": "..."})
  └─ raise auth_error(...)            # 薄封装，行为不变
```

### 3.1 新增/修改文件

| 路径 | 变更 |
|------|------|
| `app/gateway/exception_handlers.py` | **新建**：三个 handler + `register_exception_handlers(app)` |
| `app/gateway/errors.py` | **修改**：新增 `api_error()`；`auth_error()` 改为调用 `api_error()` |
| `app/main.py` | **修改**：用 `register_exception_handlers(app)` 替代单独 `add_exception_handler(RateLimitExceeded, ...)` |
| `tests/test_exception_handlers.py` | **新建**：422 / 500 / 429 body 契约测试 |

---

## 4. 组件设计

### 4.1 `api_error()` — `app/gateway/errors.py`

```python
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

`auth_error` 对外签名与行为不变。

### 4.2 `validation_exception_handler`

- 捕获：`fastapi.exceptions.RequestValidationError`
- 状态码：`422`
- 组装：

```python
payload = {
    "detail": "请求参数不合法",
    "code": "VALIDATION_ERROR",
    "errors": exc.errors(),
}
return JSONResponse(status_code=422, content={"detail": payload})
```

- `errors` 使用 Pydantic/FastAPI 原始列表，不做裁剪或翻译

### 4.3 `unhandled_exception_handler`

- 捕获：`Exception`（注册为最低优先级兜底；不捕获 `HTTPException`，由 Starlette 默认路径处理）
- 日志：`logger.exception("unhandled error: %s %s", request.method, request.url.path)`
- 状态码：`500`
- 客户端 body：固定 `"服务器内部错误"` / `INTERNAL_ERROR`，**不包含** `str(exc)`

### 4.4 `rate_limit_exception_handler`

- 捕获：`slowapi.errors.RateLimitExceeded`
- 替换 slowapi 内置 `_rate_limit_exceeded_handler`
- 实现要点：
  1. 构建 `JSONResponse`，`content={"detail": {"detail": "Rate limit exceeded", "code": "RATE_LIMITED", ...}}`
  2. 若 `request.state.view_rate_limit` 存在，调用 `request.app.state.limiter._inject_headers(response, request.state.view_rate_limit)` 保留 header 行为
  3. 从注入后的 `Retry-After` header 或 `view_rate_limit` 解析 `retry_after` 写入 body（解析失败则省略该字段）

### 4.5 `register_exception_handlers(app)`

```python
def register_exception_handlers(app: FastAPI) -> None:
    app.add_exception_handler(RateLimitExceeded, rate_limit_exception_handler)
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    app.add_exception_handler(Exception, unhandled_exception_handler)
```

在 `create_app()` 中于 router 注册之前或之后均可；建议在 middleware 注册附近、router `include` 之前调用。

---

## 5. 与现有代码的关系

| 模块 | 关系 |
|------|------|
| `app/api/routers/*` | 不修改 |
| `app/gateway/deps.py` | 不修改（继续 `auth_error`） |
| `app/infra/wechat_client.py` | 不修改 |
| `tests/test_gateway_deps.py` | 回归通过，断言不变 |
| `tests/test_auth_api.py` | 回归通过 |
| `tests/test_gateway_ratelimit.py` | 增强：429 时断言 `detail.code == RATE_LIMITED` |

---

## 6. 测试策略

### 6.1 `tests/test_exception_handlers.py`

| 用例 | 方法 | 断言 |
|------|------|------|
| 校验失败 | 对需 body 的公开端点（如 `POST /auth/register`）发送缺字段 JSON | `422`；`detail.code == VALIDATION_ERROR`；`detail.errors` 非空 |
| 未捕获异常 | 在测试用最小 FastAPI app 上注册会 `raise RuntimeError` 的路由，并调用 `register_exception_handlers` | `500`；`INTERNAL_ERROR`；body 不含 `"boom"` 等测试字符串 |
| 限流 | 复用 `test_gateway_ratelimit` 的 register 循环 fixture | 第 11 次 `429`；`detail.code == RATE_LIMITED` |

### 6.2 回归

```bash
pytest tests/test_exception_handlers.py tests/test_gateway_deps.py tests/test_auth_api.py tests/test_gateway_ratelimit.py -q
```

---

## 7. 实现顺序

1. `errors.py` — 添加 `api_error`，重构 `auth_error`
2. `exception_handlers.py` — 三个 handler + `register_exception_handlers`
3. `main.py` — 接入注册函数
4. `tests/test_exception_handlers.py` — 新测试 + 增强 ratelimit 断言
5. 全量相关测试回归

---

## 8. 验收标准

- [x] 422/500/429 均符合 §2 嵌套契约
- [x] 500 响应不含异常原文；日志含 traceback
- [x] 429 保留 `Retry-After` header
- [x] 现有 auth/gateway 测试无修改即通过
- [x] 无 router 业务代码改动
