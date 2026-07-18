# 智能导诊助手 · Web

医院导诊 Agent 的 React + Vite 前端，对接后端 REST API（JWT 鉴权）。

## 演示

<img src="../sourceData/web页面展示.gif" width="50%" />

## 前置

后端 API 运行于 `http://127.0.0.1:8000`：

- **Windows：** `.\start-dev.ps1` 或 `.\scripts\start-api.ps1`
- **Docker：** `docker compose up -d`

## 启动

```bash
cd front_Web
cp .env.example .env       # Windows: copy .env.example .env
npm install
npm run dev
```

开发请求走 `/api` → `127.0.0.1:8000`。浏览器打开 Vite 地址（默认 `http://localhost:5173`）。

首次使用手机号 + 密码注册/登录；Axios 自动附加 Bearer，401 时尝试 `/auth/refresh`。

## 构建与测试

```bash
npm run build && npm run preview
npm test
```

## 功能

- 多轮导诊（症状澄清、科室鉴别单选/多选）
- JWT 登录与多会话（`/threads`）
- 斜杠命令：`/help` `/new` `/threads` `/switch` `/delete` `/user`（打开设置）`/exit`
- `/ready` 就绪状态、意图 / 置信度 / RAG 溯源展示
