# 导诊槽位门禁 + RAG 科室消歧 设计

**日期：** 2026-06-09  
**状态：** 已确认，待实现  
**范围：** 实体提取 → 全局槽位表 → 门禁 → OpenSearch 召回 → 科室锁定（仅症状链，疾病链沿用既有逻辑）

---

## 1. 背景与目标

在现有「实体提取（原文字串四字段）+ 三路由」之上，增加：

1. **全局槽位表**：将提取结果及可选信息填入统一 intake 表单  
2. **槽位门禁**：主要症状或主要疾病至少一项，否则拒答并重置导诊状态  
3. **RAG 科室消歧**：召回 `rag_knowledge`（OpenSearch，12 条）后，通过规则打分 + LLM 反问锁定最优科室  

### 已确认决策

| 决策 | 选择 |
|------|------|
| 拒答后会话 | **同 thread**，保留 messages，清空槽位与导诊状态 |
| 科室反问轮次 | **能一轮就一轮**，上限 3 轮 |
| 触顶仍歧义 | **fallback `priority=1`** 科室 |
| 可选槽位抽取 | **混合**：四实体保真 + trigger/duration/emergency 规则/LLM 补抽 |
| 拒答文案 | `请输入症状？` |

---

## 2. 整体链路

```
用户输入
  → trim_history
  → decision（实体提取 → EntityExtractResult）
  → slot_fill（填 TriageSlotTable + 抽 trigger/duration/emergency）
  → slot_gate
       ├─ fail → reject（请输入症状？）→ session_reset → END
       └─ pass → rag_symptom_recall（OpenSearch hybrid）
            → emergency_check
                 ├─ 命中 emergency_flag → 输出急诊建议 → END
                 └─ 否 → dept_disambiguation（0~3 轮）
                      → 锁定科室 + 生成回复 → END
```

**与旧 reject 合并**：无病无名（slot_gate fail）与旧 `triage_route=reject` 走同一路径，避免双重规则。

---

## 3. 全局槽位表（`TriageSlotTable`）

### 3.1 字段定义

| 字段 | 类型 | 必填 | 默认值 | 首轮来源 |
|------|------|------|--------|----------|
| `gender` | `str` | 否 | `男` | 用户提及则覆盖 |
| `age` | `str` | 否 | `30岁` | 用户提及则覆盖 |
| `primary_symptom` | `str \| null` | **二选一** | `null` | `EntityExtractResult` |
| `companion_symptoms` | `list[str]` | 否 | `[]` | `EntityExtractResult` |
| `primary_disease` | `str \| null` | **二选一** | `null` | `EntityExtractResult` |
| `companion_diseases` | `list[str]` | 否 | `[]` | `EntityExtractResult` |
| `trigger` | `str \| null` | 否 | `null` | 同句规则/LLM 抽取 |
| `duration` | `str \| null` | 否 | `null` | 同句规则/LLM 抽取 |
| `emergency` | `str \| null` | 否 | `null` | 同句规则/LLM 抽取 |

配置文件建议：`demo/data/triage_intake_slots.json`（schema + 默认值）；运行时实例存于 `AppState.slot_table`。

### 3.2 填充规则（方案 3：混合）

- **症状/疾病四字段**：仅来自 `EntityExtractResult`，严格子串，禁止改写  
- **gender/age**：默认 `男` / `30岁`；句中出现「女」「5岁」「60岁」等则覆盖  
- **trigger / duration / emergency**：尽力从同句抽取；抽不到保持 `null`，**不阻断**导诊  

---

## 4. 槽位门禁（`slot_gate`）

```python
def slot_gate_passes(table: TriageSlotTable) -> bool:
    return bool(table.primary_symptom or table.primary_disease)
```

| 结果 | 行为 |
|------|------|
| **fail** | 固定回复 `请输入症状？`（不调 LLM）→ `session_reset` → END |
| **pass** | `slot_gate_passed=True`，进入 RAG / 科室消歧 |

### 4.1 示例

| 输入 | gate | 说明 |
|------|------|------|
| `你好` | fail | 无主要症状/疾病 |
| `脚脖子肿` | pass | 有 `primary_symptom` |
| `我有胃炎` | pass | 有 `primary_disease` |
| `胃炎还脚肿` | pass | 两者都有 |

---

## 5. 会话重置（拒答后，方案 A）

**同 `thread_id`**，**保留 `messages`**，清空导诊相关 state：

| 清空 | 保留 |
|------|------|
| `slot_table` | `messages` |
| `ner_result` | `user_id` / `thread_id` |
| `dept_state`（round、scores、last_question） | |
| `rag_chunk_id`、锁定科室 | |
| `intent_result`（或置 idle） | |

下一条用户消息视为**新一轮 intake**，重新 `decision → slot_fill → slot_gate`。

---

## 6. RAG 召回（`rag_symptom_recall`）

### 6.1 语料

- 索引：`rag_knowledge`（OpenSearch，12 条，`demo/data/rag_knowledge.jsonl`）  
- 检索：复用 `demo/opensearch_rag_kb.py` 的 **hybrid**（BM25 + kNN）  
- 查询：优先 `primary_symptom` 原文字串；可拼接 `companion_symptoms`  

### 6.2 输出

- `rag_chunk_id`（如 RK0001）  
- `canonical_symptom`  
- `department_recommendation[]`（2~3 个候选科室）  
- `emergency_flag`、`accompanying_symptom_keywords`  

### 6.3 无命中

- 无 chunk 或分数低于阈值 → 可降级提示「请补充症状描述」或走 generic 导诊（实现阶段二选一，默认：提示补充）

---

## 7. 科室消歧（`dept_disambiguation`）

### 7.1 方案：规则打分 + LLM 反问/解析（方案 3）

**规则层**定结论；**LLM 层**仅在 margin 不足时生成 1 个反问并解析回答。

### 7.2 急诊短路

`emergency_flag.condition` 与用户累计描述（含 `emergency` 槽、`trigger` 等）匹配 → 直接输出 **急诊**，不走科室反问。

### 7.3 科室打分

对每个 `department_recommendation` 项 `d`：

```
score(d) =
  w0 × (4 - priority)                    # P1=3, P2=2, P3=1
+ w1 × keyword_hits(d.condition, user_text)
+ w2 × keyword_hits(accompanying_symptom_keywords, user_text)
+ w3 × slot_table_features(d)            # trigger/duration/companion 加分
+ w4 × LLM_parsed_features(d)            # 反问回答后
```

**锁定条件**（任一）：

- `top1_score - top2_score ≥ MARGIN`（建议 MARGIN=2）  
- `top1_score ≥ LOCK_THRESHOLD`（建议 6）且 top1 唯一  

### 7.4 交互策略

| 条件 | 行为 |
|------|------|
| 置信度达标 | **0 轮**直接锁定科室 |
| margin 不足且 round < 3 | LLM 生成 **1 个**反问（基于 top-2 的 `condition` 差异） |
| round ≥ 3 仍歧义 | **fallback `priority=1`** 科室 + 提示可补充 |

### 7.5 LLM 职责（收窄）

- **生成反问**：输入 canonical_symptom、top-2 科室及 condition；输出 `{ question, targets, discriminative_axis }`  
- **解析回答**：输出 `{ features, supported_departments }` → 回写规则打分  
- **禁止**：从全院科室自由挑选；科室必须来自当前 chunk 的 `department_recommendation`  

### 7.6 LangGraph 多轮

- 需反问时：本轮 END + interrupt，用户下一条消息在同一 thread 继续 `dept_disambiguation`  
- `dept_state.round` 递增  

---

## 8. AppState 扩展

```python
slot_table: TriageSlotTable | None = None
slot_gate_passed: bool = False
rag_chunk_id: str | None = None
rag_chunk: dict | None = None          # 或 RetrievedDoc 结构
dept_state: DeptDisambiguationState | None = None
locked_department: str | None = None
session_intake_round: int = 0          # 可选统计
```

```python
class DeptDisambiguationState(BaseModel):
    candidate_departments: list[dict]
    dept_scores: dict[str, float]
    round: int = 0                       # 0~3
    status: Literal["scoring", "asking", "locked", "emergency", "fallback"]
    last_question: str | None = None
    margin: float | None = None
```

---

## 9. 节点划分

| 节点 | 职责 |
|------|------|
| `decision` | 实体提取（已有） |
| `slot_fill` | 填 `TriageSlotTable` |
| `slot_gate` | 门禁 + 分支 |
| `reject` | 固定文案 + 触发 reset |
| `session_reset` | 清 state（可内联于 reject 后） |
| `rag_symptom_recall` | OpenSearch hybrid |
| `dept_disambiguation` | 打分 / 反问 / 锁定 |
| `answer_generate` | 最终自然语言回复（可消费 locked_department） |

---

## 10. 错误处理

| 情况 | 处理 |
|------|------|
| OpenSearch 不可用 | 日志 + 降级：仅基于 slot_table 提示用户线下导诊 |
| LLM 反问/解析失败 | 回退 priority=1 或跳过反问直接 fallback |
| 多 chunk 召回接近 | 取 top1 chunk，在其内部消歧 |
| 用户仅提疾病无症状 | slot_gate pass → 走既有 `disease_dept` 链（本期科室消歧可仅绑 symptom 链） |

---

## 11. 测试用例

### 槽位门禁

| 输入 | 期望 |
|------|------|
| `你好` | reject + reset |
| `脚脖子肿` | pass，默认男/30岁 |
| reject 后再发 `心慌` | 新 intake，槽位不受污染 |

### 科室消歧（RK0001 踝肿胀）

| 输入 | 期望 |
|------|------|
| `脚脖子肿，昨天扭了` | 0 轮 → 骨科 |
| `脚脖子肿` | 1 轮反问 → 据答锁定 |
| `脚脖子肿，不能动，皮发紫` | 0 轮 → 急诊 |
| 3 轮仍模糊 | fallback 骨科（priority=1） |

---

## 12. 不在本 spec 范围

- 疾病链 OpenSearch 召回（`disease_kb.jsonl`）  
-  per-symptom 槽位表追问（`palpitation_01.json` 细槽）  
- 多主症 chunk 合并策略  
- 前端 UI  

---

## 13. 依赖与参考

- 实体提取：`docs/superpowers/specs/2026-06-08-entity-extract-design.md`  
- 语料：`demo/data/rag_knowledge.jsonl`  
- OpenSearch：`demo/opensearch_rag_kb.py`、`demo/opensearch_mappings.py`  
- 拒答常量：可统一至 `app/domain/triage_intent.py`（`请输入症状？`）
