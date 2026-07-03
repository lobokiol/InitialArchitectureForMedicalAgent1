# API 网关层：JWT 鉴权 + 微信小程序 + 限流设计

**日期**: 2026-07-04  
**状态**: 待实现  
**范围**: 在现有 FastAPI 应用内新增 `app/gateway/` 网关层，实现 JWT 鉴权（Access 2h + Refresh 7d）、微信小程序手机号登录、Web 手机号密码登录、slowapi 按用户限流；改造 `chat`/`threads` 路由与 `front_Web` 客户端。

**前置**: 无（当前 API 无鉴权，客户端直接传 `user_id`）

---

## 1. 背景与目标

### 1.1 现状

- HTTP 入口为单体 FastAPI（`app/main.py`），无独立网关进程
- 身份靠客户端传入 `user_id`（body/query），服务端不校验
- 无微信集成、无 JWT、无限流中间件
- Redis 用于 session/checkpoint；SQLite 用于 triage 记录

### 1.2 目标

为医院导诊场景（微信小程序 + Web 演示前端）提供生产级 API 入口能力：

| 能力 | 说明 |
|------|------|
| JWT 鉴权 | Access + Refresh，静默续期 |
| 微信登录 | `wx.login` + 手机号授权，openid 作微信侧凭证 |
| Web 登录 | 手机号 + 密码，与小程序共用 JWT 签发 |
| 限流 | slowapi，按用户限流；`/chat` 严格，`/threads` 读操作宽松 |
| 归属校验 | `user_id`/`thread_id` 与 token 绑定，防越权 |

### 1.3 设计决策汇总

| 项 | 决策 |
|----|------|
| 客户端 | 微信小程序 + 现有 `front_Web` 双端并行 |
| 身份主键 | 手机号（E.164），JWT `sub` = 手机号 = 业务 `user_id` |
| 微信 openid | 仅存 `wechat_bindings`，不作为业务 `user_id` |
| Web 账号 | 正式账号体系：手机号注册/登录，与小程序共用 JWT |
| 网关形态 | **方案 3 混合式**：`Depends` 鉴权 + slowapi 按路由限流（应用内，无独立网关进程） |
| 用户主数据 | SQLite（`users` + `wechat_bindings`） |
| 会话/吊销 | Redis（refresh jti、可选 access 黑名单） |
| 限流粒度 | 按 JWT `sub`（已登录）；`/auth/*` 按 IP |
| Token 生命周期 | Access 2h，Refresh 7d，支持 `/auth/refresh` 静默续期 |

---

## 2. 整体架构

```text
Clients                    app/gateway/                    Business
─────────                  ─────────────                   ────────
微信小程序 ──┐
            ├──HTTPS──►  jwt.py         签发/解析 JWT
Web 前端  ──┘           deps.py        get_current_user
                        rate_limit.py  slowapi + key_func
                        whitelist.py   公开路径
                              │
                        api/routers/
                        ├── auth.py    /auth/*（公开，IP 限流）
                        ├── chat.py    受保护 + 10/min
                        ├── threads.py 受保护 + 60/min
                        └── users.py   废弃，合并到 /auth/me
                              │
                        infra/
                        ├── user_store.py   SQLite 用户表
                        └── token_store.py  Redis refresh/blacklist
```

### 2.1 新增模块

| 路径 | 职责 |
|------|------|
| `app/gateway/jwt.py` | 签发/解析 Access、Refresh；载荷 `sub`=手机号 |
| `app/gateway/deps.py` | `get_current_user` → `CurrentUser(phone, openid?)` |
| `app/gateway/rate_limit.py` | slowapi `Limiter` 实例与 `key_func` |
| `app/gateway/whitelist.py` | 免鉴权路径列表 |
| `app/api/routers/auth.py` | 登录/注册/刷新/登出/me |
| `app/infra/user_store.py` | SQLite CRUD：`users`、`wechat_bindings` |
| `app/infra/token_store.py` | Redis refresh jti 存储与吊销 |

### 2.2 `app/main.py` 改造

- 注册 slowapi `Limiter` 与 `SlowAPIMiddleware`
- 挂载 `auth` router
- 添加 `CORSMiddleware`（`localhost:5173` + `https://servicewechat.com`）

### 2.3 新增依赖

```text
PyJWT>=2.8.0
passlib[bcrypt]>=1.7.4
slowapi>=0.1.9
httpx>=0.27.0
```

---

## 3. 认证流程

### 3.1 微信小程序

```text
1. 小程序 wx.login() → code
2. POST /auth/wechat/login {code}
   → 后端调微信 code2session → openid
   → 若 openid 已绑手机号 → 直接返回 token
   → 否则返回 {need_phone: true}
3. 用户授权手机号 → wx.getPhoneNumber → phone_code
4. POST /auth/wechat/bind-phone {phone_code, openid}
   → 后端调微信 getPhoneNumber API → purePhoneNumber
   → upsert users + wechat_bindings
   → 签发 access + refresh
5. 后续请求 Header: Authorization: Bearer <access>
6. Access 过期 → POST /auth/refresh {refresh_token}
```

微信手机号解密在服务端完成（[getPhoneNumber](https://developers.weixin.qq.com/miniprogram/dev/OpenApiDoc/user-info/phone-number/getPhoneNumber.html)），需配置 `WECHAT_APP_ID` / `WECHAT_APP_SECRET`。

### 3.2 Web 前端

```text
1. POST /auth/register {phone, password, display_name?}
   → bcrypt 哈希 → SQLite users → 签发 token
2. POST /auth/login {phone, password}
   → 校验密码 → 签发 token
3. Axios 拦截器：401 → /auth/refresh → 失败跳登录页
4. 移除 localStorage 裸 user_id（triage_demo_user_id）
```

### 3.3 JWT 载荷

**Access Token**

```json
{
  "sub": "+8613800138000",
  "type": "access",
  "iat": 1710000000,
  "exp": 1710007200
}
```

**Refresh Token**

```json
{
  "sub": "+8613800138000",
  "type": "refresh",
  "jti": "uuid-v4",
  "iat": 1710000000,
  "exp": 1710604800
}
```

Refresh 的 `jti` 存入 Redis：`refresh:{jti}` → `{phone, issued_at}`，TTL 7 天。

### 3.4 Auth 端点

| 方法 | 路径 | 鉴权 | 限流 |
|------|------|------|------|
| POST | `/auth/wechat/login` | 无 | IP 10/min |
| POST | `/auth/wechat/bind-phone` | 无 | IP 10/min |
| POST | `/auth/register` | 无 | IP 10/min |
| POST | `/auth/login` | 无 | IP 10/min |
| POST | `/auth/refresh` | 无（凭 refresh body） | IP 20/min |
| POST | `/auth/logout` | Bearer | 60/min |
| GET | `/auth/me` | Bearer | 60/min |

---

## 4. 数据模型

### 4.1 SQLite（`app/infra/user_store.py`）

在现有 triage SQLite 同库中新增 `users`、`wechat_bindings` 表（不另建 `users.db`）。

```sql
CREATE TABLE users (
    phone         TEXT PRIMARY KEY,
    password_hash TEXT,
    display_name  TEXT,
    created_at    TEXT NOT NULL,
    updated_at    TEXT NOT NULL
);

CREATE TABLE wechat_bindings (
    openid    TEXT PRIMARY KEY,
    phone     TEXT NOT NULL REFERENCES users(phone),
    unionid   TEXT,
    bound_at  TEXT NOT NULL
);
```

- 微信用户：`password_hash` 为 NULL
- Web 用户：`password_hash` 为 bcrypt
- **手机号即业务 `user_id`**，与现有 chat/threads 语义一致

### 4.2 Redis（`app/infra/token_store.py`）

| Key | 用途 | TTL |
|-----|------|-----|
| `refresh:{jti}` | Refresh 有效性 | 7d |
| `refresh:phone:{phone}` | 该手机号活跃 refresh 列表（最多 3 个 jti） | — |
| `blacklist:access:{jti}` | 登出后立即失效 access（可选） | ≤2h |

Redis 不可用时降级：refresh 仅验 JWT 签名（日志告警），与项目现有 Redis fallback 策略一致。

### 4.3 与现有 Redis user meta 的关系

保留 `user:{phone}:meta` 作 display_name 缓存；登录/注册时同步写入。

---

## 5. 路由改造与限流

### 5.1 公开 vs 受保护

| 路径 | 鉴权 | 限流 key | 限额 |
|------|------|----------|------|
| `GET /healthz`, `GET /ready` | 无 | — | 无 |
| `POST /auth/*`（除 logout/me） | 无 | IP | 10/min |
| `POST /auth/refresh` | 无 | IP | 20/min |
| `POST /chat` | Bearer 必须 | `sub`（手机号） | **10/min** |
| `GET\|POST\|DELETE /threads/*` | Bearer 必须 | `sub` | **60/min** |
| `GET /auth/me`, `POST /auth/logout` | Bearer | `sub` | 60/min |

旧 `POST|GET /users` **废弃**，功能合并至 `/auth/register` 与 `/auth/me`。

### 5.2 业务路由改造

**chat.py**

- 添加 `Depends(get_current_user)`
- `user_id` 从 token 取，**忽略** `body.user_id`
- `@limiter.limit("10/minute")`

**threads.py**

- 移除 query/body 必填 `user_id`；从 token 注入
- 若 body 仍含 `user_id`，校验 `== token.sub`，否则 403
- 新增 `assert_thread_owner(thread_id, phone)` 于 `SessionManager` 或 service 层
- 读操作 `@limiter.limit("60/minute")`

### 5.3 slowapi 配置

```python
def rate_limit_key(request: Request) -> str:
    if request.url.path.startswith("/auth/"):
        return request.client.host or "unknown"
    user = getattr(request.state, "user", None)
    if user:
        return user.phone
    return request.client.host or "unknown"
```

- 有 Redis：`storage_uri=REDIS_URL`
- 无 Redis：`storage_uri=memory://`
- 超限返回 429：`{"detail": "Rate limit exceeded", "code": "RATE_LIMITED", "retry_after": N}`

---

## 6. 前端改造概要

### 6.1 front_Web

| 文件 | 改动 |
|------|------|
| 新增 `LoginPage` / `RegisterPage` | 手机号 + 密码表单 |
| `src/lib/api.ts` | Bearer header；401 拦截 → refresh → 重试 |
| `src/hooks/useUser.ts` | 从 `/auth/me` 取用户，移除 localStorage 裸 `user_id` |
| 路由 | 未登录跳转登录页 |

### 6.2 微信小程序（新目录或独立仓库）

| 步骤 | 说明 |
|------|------|
| `wx.login` | 获取 code 调 `/auth/wechat/login` |
| `wx.getPhoneNumber` | 获取 phone_code 调 `/auth/wechat/bind-phone` |
| 请求封装 | 存 token 至 `wx.setStorageSync`；带 Authorization |
| 续期 | 401 → `/auth/refresh` |

---

## 7. 环境变量

`.env.example` 补充：

```env
JWT_SECRET=change-me-in-production
JWT_ACCESS_EXPIRE_MINUTES=120
JWT_REFRESH_EXPIRE_DAYS=7
WECHAT_APP_ID=
WECHAT_APP_SECRET=
CORS_ORIGINS=http://localhost:5173,https://servicewechat.com
RATE_LIMIT_CHAT=10/minute
RATE_LIMIT_READ=60/minute
RATE_LIMIT_AUTH_IP=10/minute
MAX_REFRESH_PER_PHONE=3
```

---

## 8. 错误处理

| HTTP | 场景 | code 示例 |
|------|------|-----------|
| 401 | Token 缺失/过期/无效 | `AUTH_MISSING`, `AUTH_EXPIRED`, `AUTH_INVALID` |
| 403 | user_id 与 token 不匹配；thread 非本人 | `FORBIDDEN`, `THREAD_NOT_OWNED` |
| 429 | 限流 | `RATE_LIMITED` |
| 400 | 微信 code 无效、手机号已注册、密码弱 | `WECHAT_CODE_INVALID`, `PHONE_EXISTS` |
| 503 | 微信 API 不可用 | `WECHAT_API_ERROR` |

统一响应：`{"detail": "...", "code": "..."}`

---

## 9. 测试策略

| 层级 | 内容 |
|------|------|
| 单元 | `jwt.py` 签名校验；`user_store` CRUD；`rate_limit_key` 分支 |
| 集成 | login → chat 带 token；refresh 续期；跨用户 thread 返回 403 |
| Mock | 微信 API 用 `httpx` mock/respx，CI 不依赖外网 |
| 回归 | `scripts/verify_*.py` 增加 `--phone` + `--password` 或 `--token` |

---

## 10. 迁移与兼容

- **Breaking change**：旧「裸 `user_id`」调用方必须改为先登录拿 token
- `cli.py`：增加登录步骤或 `--token` 参数
- `README.md`：新增鉴权章节、环境变量、小程序对接说明
- eval 脚本：文档说明如何获取测试 token

---

## 11. 实现顺序建议

1. `user_store` + `token_store` + `jwt.py`
2. `auth` router（Web register/login/refresh 先行）
3. `deps.py` + 改造 `chat`/`threads`
4. `rate_limit.py` + slowapi 挂载
5. 微信登录端点
6. `front_Web` 登录页与拦截器
7. 测试与 README 更新

---

## 12. 不在本期范围

- 短信验证码（注册可先做密码，短信二期）
- 独立 Nginx/Kong 网关进程
- 微信小程序完整 UI（仅定义 API 契约）
- OAuth 第三方（除微信外）
