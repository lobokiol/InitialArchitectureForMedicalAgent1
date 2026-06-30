## 前端展示

<img src="../sourceData/web页面展示.gif" width="50%" />

---

# 智能导诊助手

全屏零滚动的医院导诊 Agent 面试演示页，对接后端 REST API。

## 前置条件

后端 API 运行于 `http://127.0.0.1:8000`：

- **路径 A（本机）**：`.\scripts\start-api.ps1` 或 `.\start-dev.ps1`
- **路径 B（Docker）**：`docker compose up -d` 或 `.\scripts\docker-quickstart.ps1`

## 启动

```bash
cd front_Web
copy .env.example .env
npm install
npm run dev
```

开发环境请求走 `/api` 代理到 `127.0.0.1:8000`，避免浏览器 CORS 预检失败。

浏览器打开 Vite 提示的地址（默认 `http://localhost:5173`）。

## 构建

```bash
npm run build
npm run preview
```

## 测试

```bash
npm run test
```

## 功能

- 多轮导诊对话（症状澄清、科室鉴别单选/多选）
- 用户与会话管理（localStorage 持久化）
- 斜杠命令：`/help` `/new` `/threads` `/switch` `/delete` `/user`
- `/ready` 就绪状态、意图/置信度/RAG 溯源展示
