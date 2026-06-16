# 科室消歧选择题交互 设计

**日期：** 2026-06-12  
**状态：** 已确认，待实现  
**前置：** `docs/superpowers/specs/2026-06-09-triage-slot-gate-design.md`、`docs/superpowers/specs/2026-06-11-triage-cli-fixes-design.md`  
**触发：** 反问环节 LLM 自由生成问句/解析回答，用户体验差且可能引入急症词、疾病名或 chunk 外信息

---

## 1. 背景与目标

当前科室消歧在 margin 不足时，由 `generate_dept_question()` 调用 LLM 生成开放式反问，用户自由输入后由 `parse_dept_answer()` 再次调用 LLM 解析。这导致：

- 问句可能包含 `emergency_flag` 中的急症描述（如「畸形」），与当前推荐科室无关
- LLM 可能编造 chunk 外的症状或疾病名
- 用户需再次自由描述，而非在有限选项中做选择

**目标：** 将反问改为**结构化单选题**；选项**仅**来自当前 RAG chunk 的 `department_recommendation[].condition` 中可区分的**症状**；用户只能点选；若无可用选项则直接锁定 `priority=1` 科室。

### 已确认决策

| 决策 | 选择 |
|------|------|
| 交互形态 | **A** — 结构化 API `dept_choices` + CLI `Prompt.ask(..., choices=...)` |
| 「都没有」 | **A** — 始终作为最后一项 |
| 选项来源 | 规则抽取（方案 1）；**反问环节不使用 LLM 生成/解析** |
| 语料改动 | 本期不新增 `rag_knowledge` 字段 |

---

## 2. 用户故事与示例

**输入：** 「脚后跟疼怎么办？」（召回 RK0013 足跟痛）

**期望反问：**

- 文案：`为更准确推荐科室，请选择您是否有以下情况：`
- 选项示例：`活动后加重`（骨科）、`多关节疼痛`（风湿免疫科）、`都没有`
- **不出现：** `畸形`、`不能负重`（来自 `emergency_flag`）、`足底筋膜炎`（疾病名）

**用户选择：** `多关节疼痛` → 风湿免疫科加分 → 锁定或进入下一轮（换 top-2 轴）

**用户选择：** `都没有` → 否定加权（骨科 -2 等）→ round+1，换鉴别轴

**无可选症状：** 不反问，直接输出 `priority=1` 科室（如骨科）

---

## 3. 方案对比（已选方案 1）

| 方案 | 说明 | 结论 |
|------|------|------|
| **1. 纯规则抽选项** | 从 condition + accompanying 白名单抽取，无 LLM | **已选** |
| 2. 语料预标注 | `disambiguation_options` 字段 | 维护成本高，本期不做 |
| 3. LLM 从白名单挑选 | 仍有越界风险 | 不符合「绝不让 LLM 自己生成」 |

---

## 4. API 与 CLI 交互

### 4.1 反问响应（`dept_state.status == "asking"`）

`chat_service.chat_once` 在原有字段基础上增加：

```json
{
  "reply": "为更准确推荐科室，请选择您是否有以下情况：",
  "awaiting_dept_choice": true,
  "dept_choices": [
    {"id": "c1", "label": "活动后加重", "target_departments": ["骨科"]},
    {"id": "c2", "label": "多关节疼痛", "target_departments": ["风湿免疫科"]},
    {"id": "none", "label": "都没有", "target_departments": []}
  ]
}
```

- **单选**；客户端回传 `id` 或 `label`（严格匹配 + 少量别名，见 §6.2）
- 非选项文本：**不**进入 `decision`；固定提示「请从下列选项中选择」并**原样重发**当前 `dept_choices`（不递增 round）
- CLI：检测 `awaiting_dept_choice` 后使用 `Prompt.ask(..., choices=[c["label"] for c in dept_choices])`，将选中 label 作为下一条 user message

### 4.2 展示文案

- `reply` 使用固定模板，**不由 LLM 生成**
- `messages` 中 AIMessage 可附带枚举列表（供纯文本客户端），格式与 `dept_choices` 一致

---

## 5. 选项抽取（`app/triage/dept_choices.py`）

### 5.1 输入

- `rag_chunk`（含 `department_recommendation`、`accompanying_symptom_keywords`、`emergency_flag`）
- `round_num`（1~3，沿用 `_pick_pair_by_round` 的 top-2 轴）
- `asked_choice_ids`（已展示过的选项 id，避免重复）

### 5.2 算法

1. 按 round 取 top-2 科室及其 `condition` 文本
2. **候选池** = `accompanying_symptom_keywords` 中作为子串出现在**任一** top-2 `condition` 内的词条
3. **排除：**
   - `emergency_flag.condition` 中出现的词/短语（整句拆词后子串匹配）
   - 疾病/诊断倾向片段：不在 accompanying 白名单内的长短语；含「炎」「病」「症」「损伤」「骨折」「筋膜炎」「骨刺」等且非纯症状描述的分片（实现时用 denylist + 白名单交集，宁可少选不可多选）
   - 与 `canonical_symptom` 完全重复的主症名（避免「足跟痛」再作为选项）
4. **鉴别性：** 保留能区分 top-2 的项——主要只出现在一侧 `condition`，或两侧关联科室不同
5. 去重；过滤 `asked_choice_ids` 已问过的 id
6. **append 固定项：** `{id: "none", label: "都没有", target_departments: []}`
7. **结果：**
   - 除 `none` 外有效选项 ≥ 1 → 返回 choices，`status=asking`
   - 除 `none` 外有效选项 = 0 → **不进入 asking**，直接 `locked_department = fallback_department(depts)`（priority=1）

### 5.3 问句模板

```
为更准确推荐科室，请选择您是否有以下情况：
```

---

## 6. 回答解析与打分

### 6.1 替换 LLM 路径

- **移除** `dept_disambiguation` 对 `generate_dept_question` / `parse_dept_answer` 的 LLM 调用
- 新增 `resolve_dept_choice(user_reply, choices) -> DeptChoice | None`
- 新增 `choice_score_boosts(choice) -> dict[str, float]`

### 6.2 匹配规则

| 输入 | 行为 |
|------|------|
| 等于某 choice `id` | 命中 |
| 等于某 choice `label` | 命中 |
| 别名表命中（如「关节痛」→「多关节疼痛」） | 命中 |
| 其他自由文本 | `None` → 重选提示 |

### 6.3 加分

| 选择 | 效果 |
|------|------|
| 普通选项 | 对 `target_departments` 各 +`CHOICE_BOOST`（默认 2.0） |
| `都没有` | 调用现有 `apply_negation_boosts(scores, "都没有")` |

解析后重新 `score_departments` + `try_lock_department`；未锁定且 round < 3 则生成下一轮 choices（新 round 轴）。

---

## 7. State 与模型

### 7.1 新增模型

```python
class DeptChoice(BaseModel):
    id: str
    label: str
    target_departments: list[str] = Field(default_factory=list)
```

### 7.2 扩展 `DeptDisambiguationState`

```python
last_choices: list[DeptChoice] = Field(default_factory=list)
asked_choice_ids: list[str] = Field(default_factory=list)
last_question: str | None = None  # 固定模板文案
```

`dept_state` 在 `triage_state_reset_patch` / session 结束时一并清空（与 session-reset spec 兼容）。

---

## 8. 图与路由

### 8.1 `dept_disambiguation_node` 流程

```
emergency 短路 → 急诊
打分 → try_lock → locked
     → build_dept_choices
          → 无有效选项 → 直接 P1 locked
          → 有选项 → asking + last_choices → END（awaiting choice）
用户回复（同 thread）→ resolve_dept_choice
     → 未命中 → 重选 AIMessage（不增 round）→ END
     → 命中 → 更新分数 → try_lock 或下一轮 choices
round ≥ 3 仍歧义 → fallback P1
```

### 8.2 `route_after_trim`

当 `dept_state.status == "asking"` 且 `last_choices` 非空：

- 用户消息 → 始终进入 `dept_disambiguation`（含重选分支）
- **不**因自由文本误路由到 `decision`

`_is_dept_followup_reply` 可简化为：上一条为 AIMessage 且 `dept_state.status == "asking"`。

---

## 9. 文件改动清单

| 文件 | 改动 |
|------|------|
| `app/triage/dept_choices.py` | **新建** — 选项抽取、问句模板、choice 解析 |
| `app/domain/dept_disambiguation.py` | 增加 `DeptChoice`、`last_choices`、`asked_choice_ids` |
| `app/graph/nodes/dept_disambiguation.py` | 用 choices 流程替换 LLM 反问 |
| `app/triage/dept_llm.py` | 删除或废弃 LLM ask/parse（保留 `llm_score_boosts` 的可选移除，改由 choice boosts 替代） |
| `app/services/chat_service.py` | 透出 `awaiting_dept_choice`、`dept_choices` |
| `cli.py` | `Prompt.ask` 菜单选择 |
| `tests/test_dept_choices.py` | **新建** — 单元与集成测试 |

---

## 10. 测试计划

| ID | 用例 | 断言 |
|----|------|------|
| T1 | RK0013「脚后跟疼怎么办」 | choices 含「活动后加重」类；**不含**「畸形」「不能负重」 |
| T2 | 选「多关节疼痛」 | 风湿免疫科 boost，最终锁定或分数领先 |
| T3 | 选「都没有」 | 否定加权；round 递增；选项轴切换 |
| T4 | 无可选症状（mock chunk） | 0 轮 asking，直接 P1 |
| T5 | 自由文本「我还行吧」 | 不匹配；`awaiting_dept_choice` 仍为 true；round 不变 |
| T6 | CLI | `dept_choices` labels 与 `Prompt.ask` choices 一致 |
| T7 | golden eval subset F | 多轮选择题路径仍达 `expect_dept` |

---

## 11. 错误处理

| 场景 | 行为 |
|------|------|
| 用户输入非选项 | 固定提示 + 重发同一 `dept_choices` |
| 抽取异常 | 记录日志；fallback priority=1，不 asking |
| round ≥ 3 仍歧义 | 现有 `fallback_department` + fallback 模板 |

---

## 12. 非目标（本期不做）

- 修改 `rag_knowledge.jsonl` 结构或新增 chunk
- 多选（multi-select）
- 反问环节 LLM 问句润色
- 前端 Web UI（仅 API + CLI）

---

## 13. 与 session-reset 的关系

本设计仅改科室反问 UX。同 thread 会话结束清空 state（`symptom_query=None`、清 messages）属独立 spec，实现时可并行，互不冲突。`dept_state.last_choices` 在 session reset 时一并清空。
