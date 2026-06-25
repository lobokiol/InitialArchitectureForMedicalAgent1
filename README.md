# 生产级医院导诊 Agentic 助手

基于 FastAPI + LangGraph + Redis + OpenSearch + DashScope 的医院导诊助手，提供 Rich CLI 多轮对话前端。

后端通过 LangGraph 状态机编排多轮对话、症状问诊、流程检索与意图识别，前端则以 CLI 形式演示多会话聊天体验（类似 ChatGPT 的会话列表）。

---

## 项目结构

```text
app/
  main.py                      # FastAPI 入口（/healthz、/ready）
  api/routers/                 # chat、threads、users
  core/                        # config、logging、llm（DashScope 兼容）
  domain/                      # AppState、routing、槽位/澄清/消歧模型
  graph/
    builder.py                 # LangGraph 主图编译
    nodes/                     # decision、slot_*、rag、clarify、dept_*、answer…
  infra/
    opensearch_rag.py          # OpenSearch 症状混合检索
    opensearch_disease_kb.py   # 疾病库检索
    opensearch_dept_rules.py   # 科室规则检索 + 本地 JSONL fallback
    disease_kb_store.py        # disease_kb.jsonl 加载
    redis_client.py            # Redis Checkpointer / MemorySaver 回退
    triage_session_store.py    # SQLite 导诊周期持久化
  ner/                         # 实体抽取、三分类路由
  triage/                      # 槽位填充、科室打分、规则打分、置信度
  sessions/manager.py          # 多会话元数据（Redis）
  services/
    chat_service.py            # API ↔ LangGraph 编排入口
    triage_recorder.py         # 完整导诊周期写入 SQLite

cli.py                         # Rich CLI 前端
sourceData/                    # 知识库 JSONL + OpenSearch 入库脚本
  data/                        # rag_knowledge、disease_kb、rag_department_rules…
  opensearch_rag_kb.py
  opensearch_disease_kb.py
  opensearch_dept_rules.py
scripts/                       # dev-services、repair_triage_fragments、评估脚本
tests/                         # 单元测试、run_eval 黄金用例
data/triage_sessions.db        # 导诊会话记录（运行时生成）
```

---

## 核心架构概览

一次 `/chat` 请求的链路：**CLI → FastAPI → `chat_service` → LangGraph（读/写 Checkpoint）→ OpenSearch / LLM → 回复；同时 `triage_recorder` 异步写入 SQLite**。

### 系统分层

```mermaid
flowchart TB
    subgraph L1["① 客户端"]
        CLI["cli.py<br/>Rich · 斜杠命令 · 多轮选项"]
    end

    subgraph L2["② API 层 app/api"]
        CHAT["POST /chat"]
        THREADS["/threads · /users"]
        HEALTH["GET /healthz · /ready"]
    end

    subgraph L3["③ 应用服务"]
        CS["chat_service<br/>pre_state · stream · 组装响应"]
        TR["triage_recorder<br/>完整导诊周期"]
        SM["SessionManager<br/>会话列表 / 当前 thread"]
    end

    subgraph L4["④ 编排与领域"]
        LG["LangGraph<br/>14 节点 · AppState"]
        NER["app/ner<br/>实体 · 三分类路由"]
        TRI["app/triage<br/>槽位 · 打分 · 置信度"]
        RT["app/domain/routing<br/>条件边"]
    end

    subgraph L5["⑤ 基础设施 app/infra · core/llm"]
        OS["OpenSearch 客户端"]
        RD["Redis Checkpointer<br/>或 MemorySaver"]
        SQ["SQLite triage_sessions"]
        LLM["DashScope Chat / Embedding"]
    end

    subgraph L6["⑥ 数据"]
        IDX[("索引<br/>rag_knowledge · disease_kb · rag_department_rules")]
        JSONL[("sourceData/data<br/>JSONL 源文件")]
    end

    CLI --> CHAT & THREADS
    CHAT --> CS
    THREADS --> SM
    CS --> LG
    CS --> TR
    LG --> NER & TRI & RT
    LG --> OS & LLM
    LG <-->|Checkpoint| RD
    SM --> RD
    TR --> SQ
    OS --> IDX
    JSONL -.->|opensearch 入库脚本| IDX
    HEALTH --> OS & RD & SQ & LG
```

| 层级 | 目录 / 模块 | 职责 |
|------|-------------|------|
| 客户端 | `cli.py` | 调用 REST API；渲染 Markdown；处理 `awaiting_clarify` / `awaiting_dept_choice` 多轮选项 |
| API | `app/api/routers` | 请求校验与响应序列化；`/ready` 聚合 OpenSearch、Redis、SQLite、LangGraph 状态 |
| 应用服务 | `chat_service` | 唯一对话入口：读 Checkpoint 判追问、stream 主图、提取回复 |
| 应用服务 | `triage_recorder` | 非阻塞记录导诊周期（`turns_json`、outcome、state 快照） |
| 应用服务 | `SessionManager` | `user_id` ↔ 多 `thread_id` 元数据（标题、活跃时间） |
| 编排 | `app/graph` | 编译 StateGraph；节点见 `builder.py` |
| 领域 | `ner` / `triage` / `domain` | 与图节点解耦的业务规则：NER、槽位、科室打分、路由谓词 |
| 基础设施 | `infra` + `core/llm` | 外部 I/O：检索、持久化、模型调用 |
| 数据 | OpenSearch + JSONL | 运行时查索引；开发态改 JSONL 后重新入库 |

---

## 配置与环境变量

核心环境变量集中在 `app/core/config.py` 中，项目会通过 `python-dotenv` 自动加载 `.env` 文件。

必填：

- `DASHSCOPE_API_KEY`：DashScope 兼容 OpenAI API 的密钥。

---

## 一键启动（Windows 本地开发）

`start-dev.cmd` / `start-dev.ps1` 转发到 `scripts/dev-services.ps1`，默认依次拉起：

1. **Redis**（Docker，`sourceData/redis/docker-compose.yaml`，可在配置中禁用）
2. **Triage SQLite** 初始化（`data/triage_sessions.db`）
3. **OpenSearch**（本地 zip，`esTools/...`）
4. **FastAPI**（后台，`logs/api.log`）
5. 可选验证：Redis、`/ready`、`rag_knowledge` 文档数等

### 前置准备（首次）

| 项 | 说明 |
|----|------|
| **Python 3.11** | 推荐 [uv](https://docs.astral.sh/uv/) + 项目根目录下 `.venv` |
| **OpenSearch 2.19** | 解压到 `esTools\opensearch-2.19.1-windows-x64\opensearch-2.19.1`（改 `scripts/dev-services.config.ps1` → `OpenSearch.Home`） |
| **DashScope API Key** | `copy .env.example .env`，填入 `DASHSCOPE_API_KEY` |
| **Docker Desktop** | 一键脚本默认启 Redis；无 Docker 时在 `.env` 设 `USE_MEMORY_CHECKPOINTER=true`，并在配置中设 `Redis.Enabled = $false` |

在项目根目录执行（首次）：

```powershell
# 1. Python 环境
uv venv --python 3.11
uv pip install -r requirements.txt

# 2. 环境变量
copy .env.example .env
# 编辑 .env，至少填入 DASHSCOPE_API_KEY

# 3. OpenSearch 入库（首次 / 知识库变更后；需 OpenSearch 已启动）
$env:PYTHONPATH = "."
.\.venv\Scripts\python.exe sourceData\opensearch_rag_kb.py --no-embed
.\.venv\Scripts\python.exe sourceData\opensearch_disease_kb.py --no-embed
.\.venv\Scripts\python.exe sourceData\opensearch_dept_rules.py
# 含向量混合检索时去掉 --no-embed（需消耗 Embedding 额度）
```

### 常用命令

在项目根目录执行：

```powershell
# 启动全部服务 + 健康检查（默认）
.\start-dev.cmd
# 或
.\start-dev.ps1

# 只看状态（OpenSearch / Redis / API / TriageDb）
.\start-dev.ps1 -Action status

# 只跑验证
.\start-dev.ps1 -Action verify

# 启动但不验证（更快）
.\start-dev.ps1 -Action start -SkipVerify

# 停止 API、OpenSearch、Dashboards、Redis
.\start-dev.ps1 -Action stop
```

若 PowerShell 禁止执行脚本，用：`powershell -ExecutionPolicy Bypass -File .\start-dev.ps1`

启动成功后：

| 入口 | 地址 |
|------|------|
| API 文档 | http://127.0.0.1:8000/docs |
| 健康检查 | http://127.0.0.1:8000/healthz |
| 就绪检查 | http://127.0.0.1:8000/ready |
| OpenSearch | http://127.0.0.1:9200 |
| CLI 对话 | 新开终端：`.\.venv\Scripts\python.exe cli.py` |
