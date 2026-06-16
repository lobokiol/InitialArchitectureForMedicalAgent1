# 导诊 CLI 交互问题修复 设计

**日期：** 2026-06-11  
**状态：** 已实现  
**前置：** `docs/superpowers/specs/2026-06-09-triage-slot-gate-design.md`  
**触发：** CLI 手动验证（user 342/111/1233）暴露的回归与体验问题

---

## 1. 背景

槽位门禁 + RAG 科室消歧（Task 1–8）已落地，但同 thread 多轮 CLI 交互中出现：

- 急诊话术不明确、错 chunk
- 多轮否定回答无法推进消歧
- 历史消息污染当前轮打分/急诊判断
- 疾病链结果未被 `answer_generate` 消费
- `POST /users` 在无 Redis 时 500
- CLI 与同 thread 换题体验差

**已确认范围：选项 B — P0（T1–T6）+ P1（T7–T9）。**

---

## 2. 问题清单与根因

| ID | 现象 | 根因 | 优先级 |
|----|------|------|--------|
| E1 | 急诊只贴 `condition` 长句，无「急诊」科室 | `answer_generate` 急诊分支未用 `suggestion` + 固定模板 | P0 |
| E2 | `脚脖子肿，不能动，皮发紫` 召回 RK0012 非 RK0001 | RAG 仅用 `primary_symptom`；无 alliance 精确重排 | P0 |
| E3 | 同 thread 先发急诊词再发 `脚脖子肿` 仍走急诊 | `_user_text()` 拼接全部历史 HumanMessage | P0 |
| E4 | `都没有` ×3 重复同一外伤反问 | 否定未解析；round 未切换鉴别轴；无去重 | P0 |
| E5 | `脚气怎么办` 走 LLM 长文，非疾病链 | `answer_generate` 忽略 `disease_dept_result` | P0 |
| E6 | CLI `POST /users` 500 | `users.py` 未处理 `redis_client is None` | P0 |
| E7 | 同 thread 换题（脚气/脱皮）串话、错 RAG | messages 保留；短答脱离语境 | P1 |
| E8 | 输入 `exit` 未退出 | CLI 只认 `/exit` | P1 |
| E9 | 元问题 `什么科室` 直接拒答 | 无 meta 意图兜底（本期仅文档提示） | P1 |

---

## 3. 设计原则

1. **当前轮隔离**：急诊/打分/消歧仅看 `ner_result.query`（或 slot_table + 最新一条用户消息），不看全 thread 历史。
2. **RAG 稳召回**：查询以完整用户句为主；alliance 子串精确命中优先于纯向量分。
3. **模板优先**：急诊、锁定科室、疾病链、fallback 均走固定模板；LLM 仅用于生成/解析科室反问。
4. **否定可推进**：规则层处理「都没有/没有/不是」等否定，配合 round 切换 top-2 鉴别轴。
5. **不扩语料**：本期不新增 `rag_knowledge` chunk（脚气/足癣误路由 Document 为已知限制）。

---

## 4. 方案（已选：A + 当前轮隔离，不做 full intake_turn_id）

不引入 `intake_turn_id` 新字段；通过 `_current_turn_text()` 与 RAG 查询修正达到同等效果。换题仍建议 CLI `/new`，并在新 intake 时可选一行提示。

---

## 5. 组件改动

### 5.1 当前轮文本（T1）

**文件：** `app/graph/nodes/dept_disambiguation.py`

```python
def _current_turn_text(state: AppState) -> str:
    """slot_table 字段 + 最新用户句；不含历史 HumanMessage。"""
```

- `_user_text` 重命名为 `_current_turn_text` 或替换实现
- 优先 `state.ner_result.query`；否则 `messages[-1]`（HumanMessage）+ slot 可选字段

### 5.2 RAG 查询增强（T2）

**文件：** `app/graph/nodes/rag_symptom_recall.py`, `app/infra/opensearch_rag.py`

- 查询串：`ner.query` 或 `primary_symptom + companion_symptoms + trigger/emergency 槽`
- `search_rag_knowledge(query, k=3)` 取 top3
- **重排：** alliance 任一词条为 query 子串 → 该 hit 置顶；否则保持 OpenSearch 分

### 5.3 急诊回复模板（T3）

**文件：** `app/graph/nodes/answer.py`

```
根据您描述的情况（{canonical}），建议尽快就诊：**急诊**。
{emergency_flag.suggestion 或 condition 一句}
```

### 5.4 否定回答与 round 轴切换（T4）

**文件：** `app/triage/dept_llm.py`, `app/triage/dept_scoring.py`, `app/graph/nodes/dept_disambiguation.py`

**规则否定（先于 LLM）：**

| 用户答 | 效果 |
|--------|------|
| 都没有/没有/不是/无 | `trauma=false` → 骨科 -2，风湿免疫科/血管外科 +2 |
| 摔过/扭过/有 | `trauma=true` → 骨科 +2 |

**round 轴：**

| round | 反问 top-2 来源 |
|-------|----------------|
| 1 | priority 1 vs 2（骨科 vs 风湿免疫科） |
| 2 | priority 2 vs 3（风湿免疫科 vs 血管外科） |
| 3 | 同 round 2 或 fallback |

`generate_dept_question(canonical, depts, round=...)` 传入 round，prompt 要求不与 `last_question` 重复。

### 5.5 疾病链回复（T5）

**文件：** `app/graph/nodes/answer.py`

在 `locked_department` 判断之后、LLM 之前：

```python
if state.disease_dept_result and state.disease_dept_result.departments:
    depts = state.disease_dept_result.departments
    diseases = state.disease_dept_result.diseases
    # 模板：根据您提到的「{diseases}」，建议就诊科室：**{depts[0]}**
```

查不到科室时：`建议到导诊台进一步分诊`。

### 5.6 users API 回退（T6）

**文件：** `app/api/routers/users.py`

- `redis_client is None` 时使用模块级 `_memory_users: dict` 存 meta
- 行为与 Redis 版一致：POST upsert，GET 404

### 5.7 CLI 体验（T7–T8）

**文件：** `cli.py`

- `exit` / `quit`（无 `/`）等同 `/exit`
- `_render_intent` 增加：若响应可扩展字段含 `locked_department` / `dept_state.status`（需 `chat_service` 透出可选 debug 字段，或从 reply 模式推断）
- **简化：** `_ask_chat` 后在 Panel 显示 `triage_route`；若 reply 含 `建议就诊科室` / `急诊` 高亮提示
- 新 intake 后首条回复若与上轮科室不同，无需额外 API 字段

**T8 换题提示：** `decision_node` 检测到 `messages` 长度 > 2 且上轮 `dept_state.status in (locked, fallback, emergency)` 时，在 `answer_generate` 非急诊/锁定路径不处理；锁定路径模板前缀加：`（已按您本轮新描述重新评估）` — 仅当 `ner.query` 与历史主症不同。

实现：**`answer_generate`** 比较 `state.ner_result.query` 与 `messages[-3]` 若不同则加前缀（轻量，不新字段）。

### 5.8 测试（T9）

**文件：** `scripts/test_dept_scoring.py`, `scripts/test_chat_api.py`, 新建 `scripts/test_turn_text.py`（可选单元）

| 用例 | 期望 |
|------|------|
| 急诊 E2E | `脚脖子肿，不能动，皮发紫` → reply 含 `急诊` 且 canonical 含 `踝` |
| 否定单元 | `都没有` + boosts → 风湿免疫科或血管外科分数上升 |
| 历史不污染 | `_current_turn_text` 仅含末条用户句 |
| 疾病链 | `我有胃炎` → reply 含科室名（非空 LLM 散文） |
| users | `POST /users` 无 Redis 时 200 |

---

## 6. 不在本期

- 脚气/足癣专用 rag chunk 与 NER 消歧
- `intake_turn_id` 持久化字段
- 自动 topic-shift 检测（强制 `/new`）
- 元问题 `什么科室` 智能答复

---

## 7. 验证清单（CLI 手动）

```
/new
脚脖子肿，不能动，皮发紫     → 急诊 + 踝关节肿胀语境

/new
脚脖子肿 → 都没有 → 都没有 → 都没有  → fallback 骨科，反问轴有变化

/new
我有胃炎                    → 疾病链科室模板

/new
脚脖子肿 → 摔过             → 骨科
```

---

## 8. 依赖文件

| 文件 | 变更 |
|------|------|
| `app/graph/nodes/dept_disambiguation.py` | 当前轮文本、round 传参 |
| `app/graph/nodes/rag_symptom_recall.py` | 完整 query + k=3 |
| `app/infra/opensearch_rag.py` | alliance 重排 helper |
| `app/graph/nodes/answer.py` | 急诊/疾病/换题前缀 |
| `app/triage/dept_llm.py` | 否定规则、round prompt |
| `app/triage/dept_scoring.py` | 否定 boost |
| `app/api/routers/users.py` | 内存回退 |
| `cli.py` | exit 别名、展示增强 |
| `scripts/test_*.py` | T9 |

---

## Spec Self-Review

- [x] 无 TBD/占位
- [x] E1–E9 均有对应 Task
- [x] 与 2026-06-09 spec 不冲突（收窄 LLM、模板回复）
- [x] 范围 B 已标注；脚气误路由已 Document 为 known limit
