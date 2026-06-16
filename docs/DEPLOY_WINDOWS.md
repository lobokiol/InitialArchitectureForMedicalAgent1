# Windows 本地部署指南（InitialArchitectureForMedicalAgent1）

简化版导诊项目：**FastAPI + LangGraph + Redis + Elasticsearch + Milvus + DashScope（云端 LLM）**。

## 前置条件

| 软件 | 版本建议 | 说明 |
|------|---------|------|
| **Docker Desktop** | 最新 | 跑 Redis / ES / Milvus，[下载](https://www.docker.com/products/docker-desktop/) |
| **Python** | 3.10 ~ 3.12（推荐 3.11） | 你当前若是 3.13，部分包可能不兼容，建议装 3.11 |
| **DashScope API Key** | — | [阿里云百炼](https://dashscope.aliyun.com/) 开通，用于对话和向量 |

硬件建议：**内存 16GB+**（Milvus + ES 较吃内存）。

---

## 方式一：一键脚本（推荐）

```powershell
cd d:\InitialArchitectureForMedicalAgent1

# 1. 复制环境变量并填入 API Key
copy .env.example .env
notepad .env   # 修改 DASHSCOPE_API_KEY=sk-xxx

# 2. 一键启动依赖 + 装包 + 导数据
powershell -ExecutionPolicy Bypass -File scripts\setup-windows.ps1

# 3. 启动 API
powershell -ExecutionPolicy Bypass -File scripts\start-api.ps1
```

另开终端运行 CLI：

```powershell
cd d:\InitialArchitectureForMedicalAgent1
.\venv\Scripts\activate
python cli.py
```

浏览器访问：**http://localhost:8000/docs**

---

## 方式二：手动分步

### 1. 启动 Docker 依赖

```powershell
cd d:\InitialArchitectureForMedicalAgent1\demo
docker compose -f docker-compose.local.yml up -d
```

检查：

- Redis: `localhost:6379`
- ES: http://localhost:9200
- Milvus: `localhost:19530`

### 2. Python 环境

```powershell
cd d:\InitialArchitectureForMedicalAgent1
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
# 编辑 .env 填入 DASHSCOPE_API_KEY
```

### 3. 导入数据

```powershell
cd demo
python es.py          # 导入流程指南 → Elasticsearch
python milvus.py      # 导入症状 QA → Milvus（调用 DashScope embedding）
cd ..
```

### 4. 启动服务

```powershell
$env:PYTHONPATH="."
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

```powershell
python cli.py
```

---

## 常见问题

### Docker 命令找不到

安装 **Docker Desktop for Windows**，安装后 **重启 PowerShell**，确认：

```powershell
docker --version
docker compose version
```

### `DASHSCOPE_API_KEY` 报错

`.env` 必须存在且 key 有效；或在 PowerShell 临时设置：

```powershell
$env:DASHSCOPE_API_KEY="sk-xxx"
```

### Milvus / ES 连接失败

等 Docker 容器完全启动（首次 1～2 分钟）：

```powershell
docker compose -f demo/docker-compose.local.yml ps
docker logs medical-agent-es
docker logs medical-agent-milvus
```

### Python 3.13 装包失败

建议改用 Python 3.11 新建 venv：

```powershell
py -3.11 -m venv venv
```

---

## 与完整版主仓的区别

| 项目 | 本仓 Agent1 | 主仓 InitialArchitectureForMedicalAgent |
|------|------------|----------------------------------------|
| Neo4j KG | 无 | 有 |
| MCP Server | 无 | 有 |
| 槽位填槽 | 无（简化 RAG） | 有 |

完整版部署见主仓 `README.md` 与 `demo/mcp_docker-compose.yaml`。

---

## 停止服务

```powershell
docker compose -f demo/docker-compose.local.yml down
```

数据卷会保留；要清空：`docker compose -f demo/docker-compose.local.yml down -v`
