# HIS 值班医生预约 MCP 集成设计

**日期**: 2026-06-30  
**状态**: 待实现  
**范围**: 科室推荐成功后，通过 MCP 调用医院 HIS 获取值班医生预约信息，返回前端独立卡片展示

---

## 1. 背景与目标

### 1.1 现状

- 导诊后端：FastAPI + LangGraph（14 节点），科室锁定后经 `dept_confidence` → `answer_generate` 输出 Markdown 回复。
- 前端：`front_Web/` 通过 `POST /chat` 获取 `ChatResponse`，展示科室推荐与对话气泡。
- `AppState` 已有 `need_tool_call` / `tool_call_result` 占位字段，尚未使用。
- 项目内尚无 MCP 实现；`app/tools/` 为空。

### 1.2 目标

构建 MCP 体系：

- **医院侧（服务端）**：MCP Server，暴露调用 HIS 的 `get_oncall_appointments` 工具。
- **本地侧（客户端）**：导诊后端作为 MCP Client，科室推荐成功后自动调用。
- **前端**：在科室推荐回复下方独立展示 3 张值班医生预约卡片，每张含无响应「预约」按钮。

### 1.3 设计决策汇总

| 项 | 决策 |
|----|------|
| 实现方案 | LangGraph 新节点 + MCP stdio Mock（方案 A） |
| 触发时机 | 症状链（`locked_department` + `dept_confidence_passed=true`）+ 疾病链（`disease_dept` 直接推荐）；**不含急诊** |
| 科室入参 | 症状链 → `locked_department`；疾病链 → `departments[0]` 的 `dept` 字段 |
| 数据格式 | 结构化 JSON：`{name, time, slots}` × 3 |
| 医院对接 | 先 Mock MCP Server 硬编码数据；后续 Server 内部换真实 HIS，工具签名不变 |
| 前端展示 | 独立卡片（不进 `reply` Markdown）；每张卡片含无响应「预约」按钮 |
| 失败策略 | 预约为增强能力，不阻断导诊主流程 |

---

## 2. 整体架构

```text
front_Web → POST /chat → LangGraph → fetch_oncall_appointments → MCP Client
                                                              ↓ stdio
                                                    Mock MCP Server（医院侧）
                                                              ↓（后续）
                                                         真实 HIS 系统
```

### 2.1 职责划分

| 组件 | 位置 | 职责 |
|------|------|------|
| `mcp_server/server.py` | 医院侧（仓库内 mock） | 暴露工具 `get_oncall_appointments(department)` |
| `app/mcp/client.py` | 本地后端 | MCP Client，调用上述工具 |
| `fetch_oncall_appointments` 节点 | LangGraph | 判断触发条件 → 调 Client → 写入 state |
| `AppointmentCards.tsx` | 前端 | 展示 3 张医生卡片 + 无响应「预约」按钮 |

### 2.2 图结构变更

```text
# 改前
dept_confidence(passed) → answer_generate → END
disease_dept            → answer_generate → END

# 改后
dept_confidence(passed) → fetch_oncall → answer_generate → END
disease_dept            → fetch_oncall → answer_generate → END
```

其他路径（`reject`、`low_confidence_reject`、`end_ask` 等）不变。

---

## 3. MCP 工具契约

### 3.1 Server 侧（`mcp_server/`）

| 字段 | 值 |
|------|-----|
| 工具名 | `get_oncall_appointments` |
| 入参 | `department: string` — 科室名称，如 `"骨科"` |
| 返回 | `list[OnCallDoctor]`，固定 3 条 |
| 传输 | stdio（开发）；预留 SSE 配置供生产切换 |

**Mock 返回示例**：

```json
[
  {"name": "张医生", "time": "14:00-18:00", "slots": 3},
  {"name": "李医生", "time": "08:00-12:00", "slots": 5},
  {"name": "王医生", "time": "全天", "slots": 0}
]
```

Mock 可按科室名微调文案（如在 `name` 前加科室前缀），结构不变。后续真实 HIS 只改 Server 内部实现。

### 3.2 Client 侧（`app/mcp/client.py`）

```python
async def fetch_oncall_appointments(department: str) -> list[OnCallDoctor]:
    """通过 MCP 调用医院 get_oncall_appointments 工具。"""
```

- 连接：`config.MCP_SERVER_COMMAND`（默认 `python mcp_server/server.py`）
- 超时：5 秒
- 解析为 Pydantic `OnCallDoctor` 列表

### 3.3 共享模型

```python
class OnCallDoctor(BaseModel):
    name: str
    time: str
    slots: int
```

---

## 4. 后端实现

### 4.1 `fetch_oncall_appointments` 节点

```python
def should_fetch(state: AppState) -> bool:
    if state.locked_department == "急诊":
        return False
    if state.locked_department and state.dept_confidence_passed is True:
        return True
    if state.disease_dept_result and state.disease_dept_result.departments:
        return True
    return False

def resolve_department(state: AppState) -> str | None:
    if state.locked_department:
        return state.locked_department
    ddr = state.disease_dept_result
    if ddr and ddr.departments:
        first = ddr.departments[0]
        if isinstance(first, dict):
            return first.get("dept") or first.get("department")
        return str(first) if first else None
    return None
```

节点执行流程：

1. `should_fetch(state)` 为 false → 返回 `{}` 跳过
2. `resolve_department(state)` 解析科室名
3. 调用 `fetch_oncall_appointments(dept)`
4. 写入 state：

```python
return {
    "oncall_appointments": doctors,
    "tool_call_result": {"department": dept, "doctors": doctors},
}
```

失败时：

```python
return {
    "oncall_appointments": [],
    "oncall_fetch_error": "暂无法获取预约信息",
}
```

`answer_generate` **不改动**——预约信息不进 `reply` Markdown。

### 4.2 API 响应扩展

`ChatResponse` 新增：

```python
oncall_appointments: List[OnCallDoctor] = []
oncall_fetch_error: Optional[str] = None
```

`AppState` 同步新增 `oncall_appointments: list[OnCallDoctor]` 与 `oncall_fetch_error: str | None`。

`chat_service.chat_once` 从最终 state 读出并透传。

### 4.3 配置项（`app/core/config.py`）

```python
MCP_ENABLED: bool = True
MCP_SERVER_COMMAND: str = "python mcp_server/server.py"
MCP_TIMEOUT_SECONDS: float = 5.0
```

`MCP_ENABLED=false` 时节点跳过，不影响导诊。

---

## 5. 前端实现

### 5.1 类型扩展（`front_Web/src/types/index.ts`）

```typescript
export interface OnCallDoctor {
  name: string;
  time: string;
  slots: number;
}

// ChatResponse 新增
oncall_appointments?: OnCallDoctor[];
oncall_fetch_error?: string;
```

### 5.2 组件 `AppointmentCards.tsx`

挂载在助手回复气泡**下方**（不在 `MessageBubble` 内）：

```text
┌─────────────────────────────────────┐
│ 根据您描述的症状，建议就诊科室：骨科  │  ← MessageBubble
└─────────────────────────────────────┘

  值班医生预约
┌──────────┐ ┌──────────┐ ┌──────────┐
│ 张医生    │ │ 李医生    │ │ 王医生    │
│14:00-18:00│ │08:00-12:00│ │  全天     │
│ 余号 3    │ │ 余号 5    │ │ 已满      │
│ [预约]    │ │ [预约]    │ │ [预约]    │  ← disabled，无 onClick 响应
└──────────┘ └──────────┘ └──────────┘
```

规则：

- 样式与现有绿色品牌色卡片一致；桌面横向排列，移动端纵向堆叠
- 「预约」按钮：`disabled`，视觉上可点但无响应
- `slots === 0` 时显示「已满」，按钮置灰
- 渲染条件：`chatSnapshot.oncall_appointments?.length > 0`
- `oncall_fetch_error` 有值时不展示卡片（可选：小字提示，首版静默）

---

## 6. 错误处理

| 场景 | 行为 |
|------|------|
| MCP Server 未启动 / 超时 | 科室推荐正常返回；`oncall_fetch_error="暂无法获取预约信息"`；不展示卡片 |
| 返回空列表 | 同失败处理 |
| `MCP_ENABLED=false` | 静默跳过，无错误、无卡片 |
| 急诊科室 | 节点跳过，不调用 MCP |

原则：**预约信息是增强，不能阻断导诊主流程。**

---

## 7. 目录结构（新增）

```text
mcp_server/
├── server.py          # Mock MCP Server（stdio）
└── mock_data.py       # 3 医生硬编码数据

app/mcp/
├── __init__.py
└── client.py          # MCP Client 封装

app/graph/nodes/
└── fetch_oncall.py    # 新图节点

front_Web/src/components/
└── AppointmentCards.tsx
```

---

## 8. 测试计划

| 层级 | 内容 |
|------|------|
| 单元 | `should_fetch` 触发条件；`resolve_department` 症状/疾病链解析 |
| 集成 | Mock MCP Server 启动 → 完整 `/chat` 返回 `oncall_appointments` |
| 前端 | 卡片渲染、按钮 disabled、`slots=0` 置灰 |
| 手动 | 症状链走完置信度 → 见 3 张卡片；疾病链直接推荐 → 见卡片；急诊 → 无卡片 |

---

## 9. 后续扩展（不在本期范围）

- MCP 传输从 stdio 切换为 SSE，连接医院远程 Server
- Mock Server 替换为真实 HIS HTTP 封装
- 「预约」按钮对接挂号系统
- 多科室并行查询（疾病链返回多个科室时）
