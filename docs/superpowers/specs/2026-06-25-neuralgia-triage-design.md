# Neuralgia Symptom Clarify + Department Rules

**Date:** 2026-06-25  
**Status:** Approved  
**Scope:** Add `CL0015`（神经痛）to `rag_knowledge.jsonl`, five matching rules in `rag_department_rules.jsonl`, sync `gen_triage_body_kb.py`, and refresh OpenSearch indices so「神经疼」recalls the correct clarify chunk and completes triage through location → differential → department lock.

---

## 1. Problem

User input「神经疼」has no matching `symptomClarify` entry in `rag_knowledge.jsonl`. `rag_symptom_recall_node` falls through `_prefer_symptom_clarify` to the first `symptomClarify` hit in low-score hybrid search results — observed as **CL0007（脚痛）**, causing wrong slot questions and department scoring.

**Goal:** Full triage path:

```
神经疼 → CL0015 → age/sex/pain_location → RK016x → differential → dept lock
```

---

## 2. Decisions (confirmed)

| Topic | Choice |
|-------|--------|
| Scope | `rag_knowledge.jsonl` + `rag_department_rules.jsonl` + index refresh |
| `pain_location` partition | **A — 5 options:** 头面部 / 胸背部 / 腰臀下肢 / 四肢 / 说不清 |
| Index strategy | **Incremental for CL0015** (`upsert_doc`); **full rebuild** for `rag_department_rules` |
| Code changes | **None** to recall/clarify nodes — data-only fix |
| Generator sync | Update `scripts/gen_triage_body_kb.py` so jsonl remains regenerable |

---

## 3. Data: CL0015 (`rag_knowledge.jsonl`)

Insert after `CL0014`, before `CLB001`:

```json
{
  "id": "CL0015",
  "symptom_id": "神经痛",
  "aliases": ["神经疼", "神经痛", "神经性疼痛", "神经抽痛", "放电样痛", "烧灼神经痛"],
  "required_slots": ["age", "sex", "pain_location"],
  "questions": {
    "age": {
      "text": "请问您的年龄？",
      "options": ["0-3个月", "3个月-1岁", "2-4岁", "5-11岁", "12-18岁", "19-35岁", "35-59岁", "60岁及以上"]
    },
    "sex": {
      "text": "请问您的性别？",
      "options": ["男", "女"]
    },
    "pain_location": {
      "text": "你感觉神经痛主要在哪个部位？",
      "options": ["头面部", "胸背部", "腰臀下肢", "四肢", "说不清"]
    }
  },
  "type": "symptomClarify"
}
```

**Recall acceptance:** query `神经疼` → top `symptomClarify` id = `CL0015` (alias substring match in `_prefer_symptom_clarify`).

---

## 4. Data: RK0160–RK0164 (`rag_department_rules.jsonl`)

Append five rules. `symptom_id` = `神经痛`. `location` strings must match CL0015 `pain_location` options **literally** (existing framework — no normalization).

### RK0160 — 头面部

- **candidate_departments:** 神经内科, 疼痛科, 口腔科, 急诊科
- **differential_questions:**
  - 单侧面部电击样短暂发作 → 神经内科 5
  - 带状疱疹后遗痛 → 疼痛科 5
  - 突发剧烈伴意识改变 → 急诊科 5
  - 慢性反复发作 → 神经内科 4

### RK0161 — 胸背部

- **candidate_departments:** 疼痛科, 神经内科, 皮肤科, 呼吸内科
- **differential_questions:**
  - 沿肋骨单侧水疱皮疹 → 皮肤科 5, 疼痛科 4
  - 深呼吸或咳嗽时加重 → 呼吸内科 5
  - 慢性隐痛反复发作 → 疼痛科 5
  - 外伤后出现 → 神经内科 4

### RK0162 — 腰臀下肢

- **candidate_departments:** 骨科, 神经内科, 疼痛科, 康复医学科
- **differential_questions:**
  - 腰痛放射至小腿或足 → 骨科 5
  - 坐骨神经通路麻木无力 → 神经内科 5
  - 慢性顽固术后或外伤后 → 疼痛科 5
  - 双侧下肢麻木 → 神经内科 5

### RK0163 — 四肢

- **candidate_departments:** 神经内科, 骨科, 疼痛科, 皮肤科
- **differential_questions:**
  - 手套袜套样麻木刺痛 → 神经内科 5
  - 单侧肢体放射麻木 → 神经内科 5
  - 沿肢体皮疹水疱 → 皮肤科 5
  - 外伤后局部痛 → 骨科 5

### RK0164 — 说不清

- **candidate_departments:** 神经内科, 疼痛科, 全科医学科, 急诊科
- **differential_questions:**
  - 进行性加重 → 神经内科 5
  - 慢性顽固难缓解 → 疼痛科 5
  - 突发剧烈难以忍受 → 急诊科 5
  - 偶发短暂可缓解 → 全科医学科 4

---

## 5. Generator sync (`scripts/gen_triage_body_kb.py`)

Add to `kb` list:

```python
cl("CL0015", "神经痛",
   ["神经疼", "神经痛", "神经性疼痛", "神经抽痛", "放电样痛", "烧灼神经痛"],
   "你感觉神经痛主要在哪个部位？",
   ["头面部", "胸背部", "腰臀下肢", "四肢"]),
```

Add to `rules` list (five `rule(...)` entries mirroring §4, ids RK0160–RK0164).

Re-run generator only if verifying parity; hand-edited jsonl is source of truth for this change.

---

## 6. Index refresh

### 6.1 `rag_knowledge` — incremental

```bash
python demo/opensearch_rag_kb.py --doc CL0015
```

Uses `upsert_doc`: reads line from jsonl, `enrich_doc` (copies `aliases` → `alliance`), embeds, indexes into `rag_knowledge`.

### 6.2 `rag_department_rules` — full rebuild

```bash
python demo/opensearch_dept_rules.py
```

Recreates index and bulk-loads all rules (~79 docs). No embeddings. Production code also falls back to local jsonl via `search_dept_rule` if OpenSearch misses.

### 6.3 Acceptance

Extend `demo/opensearch_rag_kb.py` `ACCEPTANCE_QUERIES`:

```python
"神经疼": "CL0015",
```

Run:

```bash
python demo/opensearch_rag_kb.py --acceptance
```

Manual CLI smoke: `神经疼` → age → sex → pick each `pain_location` → differential → expect 神经内科 or 疼痛科 for typical neuralgia choices.

---

## 7. Runtime flow (unchanged code)

```mermaid
flowchart LR
    A[神经疼] --> B[rag_symptom_recall]
    B -->|CL0015| C[symptom_clarify]
    C --> D[pain_location]
    D --> E[search_dept_rule 神经痛+location]
    E --> F[dept_rules_disambiguation]
    F --> G[dept_confidence]
```

No changes to `app/graph/nodes/rag_symptom_recall.py` required: alias `神经疼` ∈ CL0015 `aliases` satisfies `_prefer_symptom_clarify` exact pass.

---

## 8. Out of scope

- Recall fallback when no alias match (separate hardening)
- `red_flags` questions for 神经痛
- Merging overlapping aliases into existing CL chunks (头痛/腰痛 etc.)

---

## 9. Test plan

1. `ACCEPTANCE_QUERIES["神经疼"] == "CL0015"`
2. `tests/test_dept_rules_scoring.py` or ad-hoc: load RK0160, apply differential scores, `try_lock_department` returns expected dept
3. CLI end-to-end: `神经疼` + `35-59岁` + `男` + `头面部` + differential「单侧面部电击样」→ 神经内科
