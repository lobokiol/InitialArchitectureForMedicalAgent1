# 首诊用户输入字数限制设计

**日期**: 2026-06-30  
**状态**: 待实现  
**范围**: 导诊首诊用户消息静默截断至 300 字；澄清/科室追问不截断；后端 + 前端双层约束

---

## 1. 背景与目标

### 1.1 现状

- `chat_service.chat_once` 将用户 `message` 原样写入 `HumanMessage`，无字数上限。
- 历史截断仅按消息**条数**（`trim_history`：`TRIM_TRIGGER_MSGS=24`，保留 `MAX_HISTORY_MSGS=12`），与单条字数无关。
- 前端 `ChatInput` 无 `maxLength`。
- 评测数据（`medical_small_100_eval_table_v2.csv`，n=100）：中位约 34 字，P99 约 121 字，最长 182 字；300 字可覆盖全部样本并留余量。

### 1.2 目标

| 场景 | 行为 |
|------|------|
| **首诊**（新 intake） | 超过 300 字时**静默截断**为前 300 字，不提示用户 |
| **追问**（澄清/科室待选） | **不截断**（常见「男」「19-35岁」等短回复） |

### 1.3 设计决策汇总

| 项 | 决策 |
|----|------|
| 超长处理 | 静默截断（非拒收、非提示） |
| 上限数值 | **300** 字，硬编码于 `app/core/config.py`（**不使用**环境变量） |
| 斜杠命令 | **不豁免**；与其它首诊输入同样适用 300 字规则（`/help` 等极短，无影响） |
| 实现方式 | 共享工具模块 `app/triage/message_limit.py` + `chat_service` 接入 |
| 前端 | `ChatInput` 在首诊 phase 设 `maxLength=300`；追问 phase 不限制 |
| 配置下发 | 不暴露 `/config` API；前端 `constants.ts` 与后端常量对齐为 300 |

---

## 2. 判定规则

### 2.1 首诊 vs 追问

复用现有 `is_awaiting_triage_followup(pre_state)`（`app/domain/routing.py`）：

- `pre_state` 为空 → **首诊**
- `clarify_state.status == "asking"` 且 `last_choices` 非空 → **追问**
- `dept_state.status == "asking"` 且 `last_choices` 非空 → **追问**
- 其余 → **首诊**（含上一轮导诊已结束、用户发起新描述）

与 `chat_service` 中已有的 `was_dept_followup = is_awaiting_triage_followup(pre_state)` 判定一致；截断使用同一 `is_followup` 标志。

### 2.2 计数方式

`len(text)`：Python / JavaScript 均按 Unicode 码点计，中文一字算一字。

### 2.3 截断位置

保留**前** 300 字：`text[:300]`。导诊主诉通常在句首，与评测数据分布一致。

---

## 3. 后端设计

### 3.1 配置

`app/core/config.py` 新增：

```python
MAX_INTAKE_MESSAGE_CHARS: int = 300
```

不通过 `os.getenv` 覆盖。

### 3.2 工具函数

新建 `app/triage/message_limit.py`：

```python
from app.core import config

def normalize_user_message(message: str, *, is_followup: bool) -> str:
    text = (message or "").strip()
    if is_followup:
        return text
    limit = config.MAX_INTAKE_MESSAGE_CHARS
    if len(text) <= limit:
        return text
    return text[:limit]
```

### 3.3 接入点

`app/services/chat_service.py` 的 `chat_once`：

```text
pre_state = _read_checkpoint_state(thread_id, user_id)
is_followup = bool(pre_state and is_awaiting_triage_followup(pre_state))
message = normalize_user_message(message, is_followup=is_followup)
was_dept_followup = is_followup   # 与现有变量语义一致，可复用 is_followup

inputs = {"messages": [HumanMessage(content=message)]}
```

- `triage_recorder.record_turn` 的 `user_message` 使用截断后的 `message`（与图内状态一致）。
- 超长首诊时打 debug 日志：`original_len`、`truncated_len`（不对用户展示）。

### 3.4 CLI / 直接 API

CLI 经 `POST /chat` 调用后端，无需单独改动。

---

## 4. 前端设计

### 4.1 常量

`front_Web/src/lib/constants.ts`：

```ts
/** 与 app/core/config.py MAX_INTAKE_MESSAGE_CHARS 保持一致 */
export const MAX_INTAKE_MESSAGE_CHARS = 300;
```

### 4.2 ChatInput

- 新增可选 prop：`maxLength?: number`
- 有值时传给 `<input maxLength={maxLength} />`

### 4.3 App.tsx

根据 `chat.phase` 决定是否限制：

| `chat.phase` | `maxLength` |
|--------------|-------------|
| `idle` | `300` |
| `awaiting_clarify` | 无（`undefined`） |
| `awaiting_dept` | 无 |
| `loading` | 无关（输入已 `disabled`） |

`useChat` 在导诊 settled 后将 phase 置回 `idle`，下一轮首诊重新启用 300 限制。

选项点击（`pickChoice` → `postAndUpdate`）发生在 `awaiting_*` phase，前端不限制；后端 `is_followup=True`，不截断。

---

## 5. 数据流

```text
用户输入
  → 前端：phase=idle 时 maxLength=300
  → POST /chat
  → chat_service：is_awaiting_triage_followup(pre_state)?
       否 → normalize_user_message（≤300）
       是 → 原样
  → LangGraph（HumanMessage）
```

---

## 6. 测试

新建 `tests/test_message_limit.py`：

| 用例 | `is_followup` | 输入 | 期望 |
|------|---------------|------|------|
| 首诊正常 | `False` | 50 字 | 原样 |
| 首诊超长 | `False` | 350 字 | 前 300 字 |
| 追问 | `True` | 任意长度 | 原样 |
| 空/空白 | `False` | `"   "` | `""` |

可选：手动验证前端 `idle` 下无法输入超过 300 字。

---

## 7. 明确不做

- 不向用户提示「已截断」
- 不做超长硬拒收
- 不使用环境变量配置上限
- 斜杠命令不做豁免
- 不修改 `trim_history` 条数逻辑
- 不新增 `/config` 接口

---

## 8. 涉及文件

| 文件 | 变更 |
|------|------|
| `app/core/config.py` | `MAX_INTAKE_MESSAGE_CHARS = 300` |
| `app/triage/message_limit.py` | 新建 `normalize_user_message` |
| `app/services/chat_service.py` | invoke 前 normalize |
| `front_Web/src/lib/constants.ts` | 新建常量 |
| `front_Web/src/components/ChatInput.tsx` | `maxLength` prop |
| `front_Web/src/App.tsx` | 按 phase 传入 `maxLength` |
| `tests/test_message_limit.py` | 单元测试 |
