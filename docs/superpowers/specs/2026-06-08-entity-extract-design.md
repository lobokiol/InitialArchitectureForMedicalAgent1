# Decision 实体提取设计（四字段 + 严格子串）

**日期：** 2026-06-08  
**状态：** 已实现（核心逻辑 + 离线测试通过）  
**范围：** `decision` 节点 NER 层——仅做实体提取与主/伴拆分，不做归一化

---

## 1. 背景与目标

导诊 agent 的 `decision` 节点需在用户输入上完成 NER，并输出**四项结构化内容**供下游路由与槽位/科室链使用：

| 字段 | 类型 | 说明 |
|------|------|------|
| `primary_symptom` | `str \| null` | 唯一主症 |
| `companion_symptoms` | `list[str]` | 伴随症状 |
| `primary_disease` | `str \| null` | 唯一主病 |
| `companion_diseases` | `list[str]` | 伴随疾病 |

### 核心约束（已确认）

1. **本步只做提取**，不做归一化/标准化（别名映射、去虚词、canonical、slot 等放到后续节点）。
2. **严格子串**：每个非空实体必须为用户输入的连续子串（`entity in user_query`）。
3. **守恒**：不得凭空新增或丢弃用户已提及的实体；主/伴仅为对同一批实体的分组。
4. **单类缺省**：只提症状时疾病侧为 `null`/`[]`；只提疾病时症状侧为 `null`/`[]`。
5. **双类并存**：有病又有症状时，`primary_symptom` 与 `primary_disease` **均必须有值**。
6. **主项规则**：确定性规则，按「用户先提到的 / 句首出现优先」；同起始位置时取**更长 span**。
7. **路由**：有病（primary 或 companion 任一非空）→ `disease` 链；无病有症状 → `symptom` 链；皆无 → `reject`（固定回复「请输入症状。」）。

---

## 2. 方案选型

**采用：LLM 抽 span + 子串校验 + 规则选主项（方案 1）**

| 步骤 | 职责 |
|------|------|
| LLM | 从原句摘 `symptom_spans[]`、`disease_spans[]`，禁止改写 |
| 后处理 | 子串校验、去重、重叠 span 合并、按位置选 primary |
| 兜底 | LLM 失败时，纯 `entity_list` 子串扫描（仍严格 `in query`） |

**本步移除/旁路：**

- `compose.normalize_output`、别名表
- `output_from_hints` 规则补词
- `chief_select.URGENCY_PRIORITY` 危急度选主（留给后续导诊节点）
- `to_canonical_chief`、`resolve_slot_table_code`（移至 `symptom_slot` 等下游）

---

## 3. 数据模型

### 3.1 `EntityExtractResult`（重构 `NERExtractResult`）

```python
class EntityExtractResult(BaseModel):
    query: str
    primary_symptom: str | None = None
    companion_symptoms: list[str] = Field(default_factory=list)
    primary_disease: str | None = None
    companion_diseases: list[str] = Field(default_factory=list)
```

保留只读派生属性（供路由/兼容）：

- `has_symptom`: `primary_symptom is not None or len(companion_symptoms) > 0`
- `has_disease`: `primary_disease is not None or len(companion_diseases) > 0`
- `all_symptoms`: `[primary_symptom, *companion_symptoms]`（过滤 None）
- `all_diseases`: `[primary_disease, *companion_diseases]`（过滤 None）

### 3.2 LLM 中间结构

```python
class NERExtractOutput(BaseModel):
    symptom_spans: list[str]   # 原句子串，未分组
    disease_spans: list[str]
```

---

## 4. 处理流程

```
user_query
  → LLM 抽取 symptom_spans, disease_spans
  → validate_substrings(query)     # 丢弃不在 query 中的项
  → dedupe_by_first_occurrence
  → resolve_overlapping_spans      # 长 span 吞并被完全包含的短 span
  → select_primary_by_position     # 症状、疾病各自独立
  → EntityExtractResult
  → resolve_triage_route()
```

### 4.1 子串校验

```python
def is_valid_span(span: str, query: str) -> bool:
    s = span.strip()
    return bool(s) and s in query
```

### 4.2 去重

相同字符串只保留一次，顺序按**首次在 query 中出现的位置**。

### 4.3 重叠 span 处理

若 span A 完全包含于 span B（且均为 query 子串），仅保留 **B（更长）**。

示例：query=`肚脐上方疼痛`，同时命中 `疼痛` 与 `肚脐上方疼痛` → 只保留 `肚脐上方疼痛`。

### 4.4 主项选取 `select_primary_by_position(candidates, query)`

1. 过滤：`candidate in query`
2. 计算 `(start_index, -len(candidate), candidate)`，按元组升序排序
3. 第一项 → `primary`，其余按首次出现顺序 → `companion`

症状、疾病**分别**调用，互不影响。

---

## 5. 路由规则

```python
def resolve_triage_route(result: EntityExtractResult) -> TriageRoute:
    if result.has_disease:
        return "disease"      # 有病必有 symptom 时仍走 disease 链
    if result.has_symptom:
        return "symptom"
    return "reject"
```

| route | 条件 | 图路径 |
|-------|------|--------|
| `disease` | 任一疾病实体 | `decision → disease_dept → answer_generate` |
| `symptom` | 无疾病、有症状 | `decision → symptom_slot → answer_generate` |
| `reject` | 皆无 | `decision → reject`（「请输入症状。」） |

---

## 6. 与现有图集成

### `decision_node`

- 调用 `extract_entities(query)` → `EntityExtractResult`
- 写入 `state.ner_result`（或重命名为 `entity_extract_result`）
- 写入 `intent_result.triage_route`

### 下游节点职责迁移

| 节点 | 读取 | 本步新增职责 |
|------|------|--------------|
| `disease_dept` | `primary_disease`, `companion_diseases` | 疾病名归一化、科室查询 |
| `symptom_slot` | `primary_symptom`, `companion_symptoms` | canonical、slot_table_code |
| `reject` | route=`reject` | 不变 |

---

## 7. LLM Prompt 要点

- 角色：医疗导诊实体抽取，**只摘原句连续子串**
- 禁止：改写、同义词替换、合并描述、解释性补充
- 分类：症状 vs 疾病；同一 span 不重复归类
- 不输出主/伴（由规则层完成）
- 示例 few-shot 覆盖：纯症状、纯疾病、病+症并存

---

## 8. 错误处理

| 情况 | 处理 |
|------|------|
| LLM 调用失败 | 降级：`entity_list.json` 子串扫描，仍校验 `in query` |
| 校验后实体皆空 | `triage_route = reject` |
| LLM 返回非子串 | 丢弃并打 warn 日志 |
| 病+症输入但只抽出一类 | 以校验后结果为准，不强行补全 |

---

## 9. 测试用例

| 输入 | primary_symptom | companion | primary_disease | companion | route |
|------|-----------------|-----------|-----------------|-----------|-------|
| `最近心慌还手抖` | `心慌` | `['手抖']` | null | [] | symptom |
| `我有胃炎` | null | [] | `胃炎` | [] | disease |
| `胃炎还肚脐上方疼` | `肚脐上方疼` | [] | `胃炎` | [] | disease |
| `手抖心慌` | `手抖` | `['心慌']` | null | [] | symptom |
| `你好` | null | [] | null | [] | reject |

**不变量测试：**

- 所有非空实体 `in query`
- `set(all_symptoms + all_diseases)` 等于校验后提取全集
- 重叠 span 不重复计数
- 同位置取更长 span

---

## 10. 不在本 spec 范围

- 槽位追问、科室 KG、RAG
- 症状/疾病归一化与同义词表
- 危急症识别与 emergency agent
- 多轮对话指代消解

---

## 11. 确认记录

| 决策 | 选择 |
|------|------|
| 实体保真 | 严格子串，不做本步归一化 |
| 单类缺省 | A：另一类 null/[] |
| 主项选取 | B：先提到/句首优先；同位置更长 span |
| 双类路由 | 走 `disease` 链 |
| 实现方案 | LLM span + 校验 + 规则选主 |
