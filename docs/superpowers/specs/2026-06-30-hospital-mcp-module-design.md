# 医院 MCP 完整模块设计

**日期**: 2026-06-30  
**状态**: 待实现  
**范围**: 将 `mcp_server` 升级为独立医院 MCP 服务 `hospital_mcp/`，将 `app/mcp` 升级为通用 Client；导诊 Agent 继续作为调用方。覆盖值班医生预约、科室介绍、导诊台到目标科室路线。数据均为 JSON Mock。

**前置**: `docs/superpowers/specs/2026-06-30-his-oncall-mcp-design.md`（v1 单工具已实现）

---

## 1. 背景与目标

### 1.1 现状

- `mcp_server/server.py`：单工具 `get_oncall_appointments`，stdio Mock
- `app/mcp/client.py`：固定 `call_tool("get_oncall_appointments")`
- `fetch_oncall` 节点：科室推荐成功后自动拉预约，前端 `AppointmentCards` 展示
- 无科室介绍、路线能力；无 `list_tools`；无 LLM 动态选工具

### 1.2 目标

构建完整医院 MCP 模块（Mock 阶段）：

| 能力 | 工具 | 触发 |
|------|------|------|
| 值班医生预约 | `get_oncall_appointments` | 科室推荐成功后**自动**（通道 1，保持现状） |
| 科室介绍 | `get_department_intro` | 用户追问时 **LLM 语义选工具**（通道 2） |
| 来院路线 | `get_department_route` | 用户追问时 **LLM 语义选工具**（通道 2） |

导诊 Agent（LangGraph + FastAPI + front_Web）仍为唯一业务调用方。

### 1.3 设计决策汇总

| 项 | 决策 |
|----|------|
| 架构 | 方案 A：**双通道**——预约固定节点 + 介绍/路线 LLM follow-up |
| 传输 | stdio（本期）；`MCP_TRANSPORT` 配置预留 SSE |
| 数据 | 全部 JSON Mock，存 `hospital_mcp/mock/*.json` |
| 预约展示 | 独立 `AppointmentCards`（不变） |
| 介绍/路线展示 | 写入助手 `reply` Markdown，**不做**独立前端卡片 |
| 追问科室上下文 | 优先 `last_recommended_department`；无则澄清，不调 MCP |
| follow-up 暴露给 LLM 的工具 | 仅 `get_department_intro`、`get_department_route`（不含 oncall，避免重复拉预约） |
| 号源追问 | 引导查看上一轮预约卡片或重新导诊，本期不在 follow-up 暴露 oncall 工具 |

---

## 2. 整体架构

```text
┌─────────────────────────────────────────────────────────────┐
│ 导诊 Agent                                                   │
│  ├─ fetch_hospital_info（原 fetch_oncall）  通道1：固定预约   │
│  └─ mcp_followup_agent（新）              通道2：LLM 选工具  │
└──────────────────────────┬──────────────────────────────────┘
                           │ app/mcp/McpHospitalClient
                           ▼
┌─────────────────────────────────────────────────────────────┐
│ hospital_mcp/（独立医院 MCP 服务，Mock 阶段同仓库）           │
│  get_oncall_appointments | get_department_intro |            │
│  get_department_route                                       │
└─────────────────────────────────────────────────────────────┘
```

### 2.1 LangGraph 路由

```text
trim_history → route_after_trim:
  ├─ symptom_clarify / dept_rules / dept_disambiguation（不变）
  ├─ mcp_followup（新）
  └─ decision（新 intake）

成功导诊路径（不变）:
  ... → fetch_hospital_info → answer_generate → END

追问路径（新）:
  mcp_followup_agent → END
```

### 2.2 `mcp_followup` 进入条件

1. `last_recommended_department` 非空  
2. 非 clarify / dept 反问 follow-up  
3. LLM 轻量判定：用户消息为科室信息追问（介绍、路线、在哪、怎么走等）

若无 `last_recommended_department`：返回固定澄清话术，不调 MCP。

---

## 3. MCP 工具契约

### 3.1 目录结构

```text
hospital_mcp/
├── server.py
├── adapters/
│   └── mock_store.py
└── mock/
    ├── departments.json   # 科室介绍 + floor/phone
    └── routes.json        # 导诊台 → 科室路线
```

`get_oncall_appointments` 医生列表可继续代码生成（`mock_store.doctors_for_department`），与 v1 一致。

### 3.2 `get_oncall_appointments(department: str)`

返回 JSON 数组，元素 `OnCallDoctor`：

```json
{"name": "消化内科·张医生", "time": "14:00-18:00", "slots": 3}
```

固定 3 条；`slots === 0` 表示已满。

### 3.3 `get_department_intro(department: str)`

```json
{
  "department": "消化内科",
  "summary": "诊治食管、胃肠、肝胆胰脾等疾病",
  "scope": ["胃炎", "胃溃疡", "肠炎", "肝病"],
  "visit_tips": "建议空腹前往，携带既往检查报告",
  "floor": "门诊楼 3 层",
  "phone": "0571-00001111"
}
```

未知科室：

```json
{"error": "department_not_found", "department": "未知科"}
```

### 3.4 `get_department_route(department: str, from_location: str = "导诊台")`

```json
{
  "department": "消化内科",
  "from": "导诊台",
  "to": "门诊楼 3 层 消化内科",
  "estimated_minutes": 5,
  "steps": [
    "从导诊台向东步行至门诊楼一楼大厅",
    "乘电梯至 3 层",
    "出电梯左转，沿指示牌到达消化内科分诊台"
  ],
  "landmarks": ["门诊楼", "3 层电梯厅", "消化内科分诊台"]
}
```

Mock 数据至少覆盖 `disease_kb` 中常见科室：消化内科、骨科、普外科、急诊等。

---

## 4. 通用 MCP Client（`app/mcp/`）

### 4.1 `McpHospitalClient`

```python
class McpHospitalClient:
    async def list_tools(self) -> list[Tool]: ...
    async def call_tool(self, name: str, arguments: dict) -> Any: ...
```

- stdio 连接：`MCP_SERVER_COMMAND`（默认 `python hospital_mcp/server.py`）
- `list_tools` 进程内缓存 TTL 5 分钟
- 超时：`MCP_TIMEOUT_SECONDS = 5.0`

### 4.2 类型化封装

```python
fetch_oncall_appointments(dept) -> list[OnCallDoctor]
fetch_department_intro(dept) -> DepartmentIntro
fetch_department_route(dept, from_location="导诊台") -> DepartmentRoute
```

### 4.3 共享模型（`app/domain/models.py`）

```python
class DepartmentIntro(BaseModel):
    department: str
    summary: str
    scope: list[str]
    visit_tips: str
    floor: str
    phone: str

class DepartmentRoute(BaseModel):
    department: str
    from_location: str  # JSON 字段 "from"
    to: str
    estimated_minutes: int
    steps: list[str]
    landmarks: list[str]
```

---

## 5. 导诊集成

### 5.1 通道 1：`fetch_hospital_info`（重命名 `fetch_oncall`）

- 触发条件、急诊跳过、跨轮 `session_reset` 清空逻辑**与 v1 一致**
- 成功时新增：`last_recommended_department = dept`
- `triage_state_reset_patch` 新增清空：`last_recommended_department = None`

### 5.2 通道 2：`mcp_followup_agent`

**流程**:

```text
1. dept = 用户句中科室名（若 NER/词典抽到）else last_recommended_department
2. tools = client.list_tools() 过滤为 intro + route 两项
3. LLM(user_message, tools) → tool_call
4. result = client.call_tool(name, {department: dept, ...})
5. LLM 根据 result JSON 生成 Markdown → AIMessage 写入 messages
```

**LLM 未选工具**：返回「请问您想了解科室介绍，还是来院路线？」

**工具返回 `department_not_found`**：LLM 生成「暂未找到该科室信息，请核对科室名称。」

### 5.3 API / 前端

| 字段 | 行为 |
|------|------|
| `oncall_appointments` | 不变，预约卡片 |
| `recommended_department` | 不变 |
| `reply` | follow-up 轮次含介绍/路线 Markdown |
| 新前端组件 | 无 |

`node_trace` 应包含 `mcp_followup_agent` 便于调试。

### 5.4 配置

```python
MCP_ENABLED: bool = True
MCP_SERVER_COMMAND: str = "python hospital_mcp/server.py"
MCP_TIMEOUT_SECONDS: float = 5.0
MCP_FOLLOWUP_ENABLED: bool = True  # 可关闭通道 2
```

---

## 6. 错误处理

| 场景 | 行为 |
|------|------|
| MCP Server 不可用 | 通道1：现有降级；通道2：Markdown 提示暂不可用 |
| 无 `last_recommended_department` | 固定澄清，不调 MCP |
| 新 intake（新症状/疾病） | `decision` reset，走正常导诊 |
| `MCP_FOLLOWUP_ENABLED=false` | 追问走 `decision`，可能 reject 或重新导诊 |

原则：预约为增强能力不阻断导诊；介绍/路线追问失败不阻断会话。

---

## 7. 测试计划

| 层级 | 内容 |
|------|------|
| mock_store | 三工具按科室返回、未知科室 error |
| McpHospitalClient | list_tools、call_tool 集成 |
| fetch_hospital_info | 写入 `last_recommended_department` |
| route_after_trim | 有/无 last_dept、mcp_followup 判定 |
| mcp_followup_agent | mock LLM 选 intro / route，reply 含 Markdown |
| 手动 | 胃炎导诊成功 → 卡片；追问「怎么走」→ 路线 Markdown；无导诊追问 → 澄清 |

---

## 8. 迁移说明（相对 v1）

| v1 | v2 |
|----|-----|
| `mcp_server/` | 重命名/迁移为 `hospital_mcp/` |
| `fetch_oncall` 节点 | 重命名为 `fetch_hospital_info`（行为不变） |
| 单工具 Client | `McpHospitalClient` + 类型化方法 |
| `MCP_SERVER_COMMAND` 默认值 | 更新为 `python hospital_mcp/server.py` |

保留 `mcp_server/` 软链接或过渡期 re-export 可选，实现阶段二选一。

---

## 9. 后续扩展（本期不做）

- SSE 远程部署与鉴权
- 真实 HIS HTTP 适配
- follow-up 暴露 `get_oncall_appointments` 给 LLM
- `dept_spans` NER 扩展（用户指定科室追问）
- MCP Resources（科室列表只读资源）
