# Symptom Clarify + Department Rules Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement a reusable triage flow: `symptomClarify` (CL) slot questionnaire → `rag_department_rules` multi-select differential → LLM confidence gate (≥60) → optional `red_flags` collection → department answer.

**Architecture:** Extend LangGraph with `symptom_clarify`, `dept_rules_disambiguation`, `dept_confidence`, and `low_confidence_reject` nodes. New OpenSearch index `rag_department_rules` for exact `symptom_id`+`location` lookup. Existing foot RK path unchanged except it also passes through `dept_confidence`.

**Tech Stack:** Python 3.10+, FastAPI, LangGraph, OpenSearch (`opensearch-py`), Pydantic v2, LangChain structured output (`with_structured_output`), DashScope LLM, pytest.

## Global Constraints

- Multi-select input: numbered only — `1,3` / `1 3` / `1、3`
- Slot order: `age` → `sex` → `pain_location` → `differential` → `red_flags` (red_flags **after** confidence ≥60)
- `location` literal match between CL options and `rag_department_rules.jsonl`; **no** `location_normalize.py`
- Male (`sex=男`): remove 妇科 from `candidate_departments` and gynecology-only differential options
- Dept scoring: base score by `candidate_departments` index + multi-select `scores` sum → `try_lock_department` (`MARGIN=2.0`, `LOCK_THRESHOLD=6.0`) → fallback/tiebreak by `candidate_departments` order
- LLM confidence: inputs = `locked_department` + `TriageSlotTable` + `filled_slots` subset (`age`, `sex`, `pain_location`, differential only); **exclude `red_flags`**
- Confidence threshold: `score >= 60` → proceed; `< 60` → `low_confidence_reject` (no dept in reply)
- `red_flags`: collect only; does not affect dept scoring or LLM confidence

**Spec:** `docs/superpowers/specs/2026-06-24-symptom-clarify-dept-rules-design.md`

---

## File Map

| File | Responsibility |
|------|----------------|
| `app/domain/symptom_clarify.py` | `ClarifyChoice`, `SymptomClarifyState` |
| `app/domain/dept_confidence.py` | `DeptConfidenceResult` |
| `app/domain/models.py` | `AppState` new fields |
| `app/domain/dept_disambiguation.py` | `choice_mode`, `multi_select` on `DeptDisambiguationState` |
| `app/domain/routing.py` | `route_after_rag`, `route_after_confidence`, extend `route_after_trim` |
| `app/triage/multi_choice.py` | `resolve_multi_choice`, `parse_choice_indices` |
| `app/triage/dept_rules_scoring.py` | `filter_rule_by_sex`, `build_base_scores`, `score_from_selections`, `lock_department_from_totals` |
| `app/triage/session_reset.py` | Clear `clarify_state`, confidence fields |
| `app/infra/opensearch_dept_rules.py` | Index search by `symptom_id`+`location` |
| `app/graph/nodes/symptom_clarify.py` | age/sex/pain_location/red_flags phases |
| `app/graph/nodes/dept_rules_disambiguation.py` | differential multi-select + lock |
| `app/graph/nodes/dept_confidence.py` | LLM structured scoring |
| `app/graph/nodes/low_confidence_reject.py` | Reject reply with score |
| `app/graph/builder.py` | Wire nodes and conditional edges |
| `app/graph/nodes/rag_symptom_recall.py` | Prefer `symptomClarify` hits when alias matches |
| `sourceData/opensearch_mappings.py` | `rag_department_rules_index_body()` |
| `sourceData/opensearch_dept_rules.py` | Index + bulk load JSONL |
| `sourceData/data/rag_knowledge.jsonl` | CL0001 full `questions` |
| `app/core/config.py` | `RAG_DEPT_RULES_INDEX` env |
| `app/services/chat_service.py` | Response fields |
| `app/api/routers/chat.py` | `ChatResponse` schema |
| `cli.py` | `awaiting_clarify` + multi-select loop |
| `tests/test_multi_choice.py` | Unit tests |
| `tests/test_dept_rules_scoring.py` | Unit tests |
| `tests/test_symptom_clarify_routing.py` | Routing unit tests |

---

### Task 1: Domain models and session reset

**Files:**
- Create: `app/domain/symptom_clarify.py`
- Create: `app/domain/dept_confidence.py`
- Modify: `app/domain/models.py`
- Modify: `app/domain/dept_disambiguation.py`
- Modify: `app/triage/session_reset.py`

**Interfaces:**
- Produces: `ClarifyChoice`, `SymptomClarifyState`, `DeptConfidenceResult`
- Produces on `AppState`: `clarify_state`, `dept_confidence_result`, `dept_confidence_passed: bool | None`

- [ ] **Step 1: Create `app/domain/symptom_clarify.py`**

```python
from typing import Literal
from pydantic import BaseModel, Field, ConfigDict

class ClarifyChoice(BaseModel):
    id: str
    label: str
    slot: str | None = None
    scores: dict[str, float] | None = None
    model_config = ConfigDict(extra="ignore")

class SymptomClarifyState(BaseModel):
    status: Literal["asking", "done"] = "asking"
    clarify_chunk_id: str | None = None
    symptom_id: str | None = None
    phase: Literal["age", "sex", "pain_location", "red_flags", "done"] = "age"
    filled_slots: dict[str, str] = Field(default_factory=dict)
    last_question: str | None = None
    last_choices: list[ClarifyChoice] = Field(default_factory=list)
    multi_select: bool = False
    dept_rule_id: str | None = None
    dept_rule_chunk: dict | None = None
    model_config = ConfigDict(extra="ignore")
```

- [ ] **Step 2: Create `app/domain/dept_confidence.py`**

```python
from pydantic import BaseModel, ConfigDict

class DeptConfidenceResult(BaseModel):
    score: float
    reason: str = ""
    slot_alignment: str = ""
    model_config = ConfigDict(extra="ignore")
```

- [ ] **Step 3: Extend `AppState` in `app/domain/models.py`**

Add fields after `locked_department`:

```python
clarify_state: Optional[Any] = None
dept_confidence_result: Optional[Any] = None
dept_confidence_passed: bool | None = None
```

- [ ] **Step 4: Extend `DeptDisambiguationState` in `app/domain/dept_disambiguation.py`**

```python
multi_select: bool = False
choice_mode: Literal["accompany", "differential"] = "accompany"
```

- [ ] **Step 5: Update `app/triage/session_reset.py`**

```python
"clarify_state": None,
"dept_confidence_result": None,
"dept_confidence_passed": None,
```

- [ ] **Step 6: Commit**

```bash
git add app/domain/symptom_clarify.py app/domain/dept_confidence.py app/domain/models.py app/domain/dept_disambiguation.py app/triage/session_reset.py
git commit -m "feat: add symptom clarify and dept confidence domain models"
```

---

### Task 2: Multi-choice parser (TDD)

**Files:**
- Create: `app/triage/multi_choice.py`
- Create: `tests/test_multi_choice.py`
- Modify: `app/triage/dept_choices.py` (re-export `NONE_CHOICE_ID`, `NONE_CHOICE_LABEL` if needed)

**Interfaces:**
- Produces: `parse_choice_indices(text: str) -> list[int]`, `resolve_multi_choice(text: str, choices: list[DeptChoice]) -> tuple[list[DeptChoice], bool]` where second value is `none_selected`

- [ ] **Step 1: Write failing tests in `tests/test_multi_choice.py`**

```python
from app.domain.dept_disambiguation import DeptChoice
from app.triage.multi_choice import parse_choice_indices, resolve_multi_choice

def _choices():
    return [
        DeptChoice(id="c1", label="发热、恶心呕吐", target_departments=["普外科"]),
        DeptChoice(id="c2", label="腹泻、腹胀", target_departments=["消化内科"]),
        DeptChoice(id="none", label="都没有", target_departments=[]),
    ]

def test_parse_choice_indices_comma():
    assert parse_choice_indices("1,3") == [0, 2]

def test_parse_choice_indices_space():
    assert parse_choice_indices("1 3") == [0, 2]

def test_parse_choice_indices_cn_comma():
    assert parse_choice_indices("1、3") == [0, 2]

def test_resolve_multi_choice_picks_two():
    picked, none_sel = resolve_multi_choice("1,2", _choices())
    assert len(picked) == 2
    assert not none_sel

def test_resolve_multi_choice_none_only():
    picked, none_sel = resolve_multi_choice("3", _choices())
    assert picked == []
    assert none_sel

def test_resolve_multi_choice_invalid():
    picked, none_sel = resolve_multi_choice("9", _choices())
    assert picked is None  # signal invalid — use None vs empty list
    assert none_sel is False
```

- [ ] **Step 2: Run tests — expect FAIL**

```bash
cd e:\InitialArchitectureForMedicalAgent1-main\InitialArchitectureForMedicalAgent1-main
.\.venv\Scripts\python.exe -m pytest tests/test_multi_choice.py -v
```

- [ ] **Step 3: Implement `app/triage/multi_choice.py`**

```python
from __future__ import annotations
import re
from app.domain.dept_disambiguation import DeptChoice
from app.triage.dept_choices import NONE_CHOICE_ID, NONE_CHOICE_LABEL

def parse_choice_indices(text: str) -> list[int]:
    raw = (text or "").strip().replace("、", ",").replace(" ", ",")
    parts = [p for p in re.split(r"[,，]+", raw) if p.strip()]
    return [int(p) - 1 for p in parts if p.strip().isdigit()]

def resolve_multi_choice(
    text: str, choices: list[DeptChoice]
) -> tuple[list[DeptChoice] | None, bool]:
    t = (text or "").strip()
    if t == NONE_CHOICE_LABEL:
        return [], True
    if not t:
        return None, False
    if not any(ch.isdigit() for ch in t):
        return None, False
    indices = parse_choice_indices(t)
    if not indices:
        return None, False
    picked: list[DeptChoice] = []
    for idx in indices:
        if idx < 0 or idx >= len(choices):
            return None, False
        c = choices[idx]
        if c.id == NONE_CHOICE_ID:
            return [], True
        picked.append(c)
    return picked, False
```

- [ ] **Step 4: Run tests — expect PASS**

```bash
.\.venv\Scripts\python.exe -m pytest tests/test_multi_choice.py -v
```

- [ ] **Step 5: Commit**

```bash
git add app/triage/multi_choice.py tests/test_multi_choice.py
git commit -m "feat: add multi-choice index parser for differential questions"
```

---

### Task 3: Department rules scoring + gender filter (TDD)

**Files:**
- Create: `app/triage/dept_rules_scoring.py`
- Create: `tests/test_dept_rules_scoring.py`

**Interfaces:**
- Produces: `GYNECOLOGY_DEPT = "妇科"`, `filter_rule_by_sex(chunk: dict, sex: str) -> dict`, `build_base_scores(candidate_departments: list[str]) -> dict[str, float]`, `accumulate_scores(base: dict, selections: list[dict], active_depts: list[str]) -> dict[str, float]`, `lock_department_from_totals(totals: dict, candidate_departments: list[str], active_depts: list[str], none_selected: bool) -> tuple[str, dict[str, float], float, bool]`

Returns `(locked_dept, totals, margin, used_tie_break)`.

- [ ] **Step 1: Write failing tests using RK0025 fixture**

```python
import json
from pathlib import Path
from app.triage.dept_rules_scoring import (
    filter_rule_by_sex, build_base_scores, accumulate_scores, lock_department_from_totals,
)

RK0025 = json.loads(Path("sourceData/data/rag_department_rules.jsonl").read_text(encoding="utf-8").splitlines()[0])

def test_filter_rule_by_sex_male_removes_gynecology():
    filtered = filter_rule_by_sex(RK0025, "男")
    assert "妇科" not in filtered["candidate_departments"]
    assert all("妇科" not in (q.get("scores") or {}) or not set(q["scores"]) <= {"妇科"} for q in filtered["differential_questions"])

def test_build_base_scores_order():
    scores = build_base_scores(["消化内科", "普外科", "泌尿外科"])
    assert scores["消化内科"] > scores["普外科"] > scores["泌尿外科"]

def test_lock_none_selected_fallback_first_in_order():
    active = ["消化内科", "普外科", "泌尿外科"]
    base = build_base_scores(active)
    dept, totals, margin, tie = lock_department_from_totals(base, RK0025["candidate_departments"], active, none_selected=True)
    assert dept == "消化内科"
```

- [ ] **Step 2: Run — expect FAIL**

```bash
.\.venv\Scripts\python.exe -m pytest tests/test_dept_rules_scoring.py -v
```

- [ ] **Step 3: Implement `app/triage/dept_rules_scoring.py`**

```python
from __future__ import annotations
from app.triage.dept_scoring import try_lock_department

GYNECOLOGY_DEPT = "妇科"

def filter_rule_by_sex(chunk: dict, sex: str) -> dict:
    out = dict(chunk)
    depts = list(chunk.get("candidate_departments") or [])
    questions = list(chunk.get("differential_questions") or [])
    if sex.strip() == "男":
        depts = [d for d in depts if d != GYNECOLOGY_DEPT]
        questions = [
            q for q in questions
            if not (set((q.get("scores") or {}).keys()) == {GYNECOLOGY_DEPT})
        ]
    out["candidate_departments"] = depts
    out["differential_questions"] = questions
    return out

def build_base_scores(candidate_departments: list[str]) -> dict[str, float]:
    n = len(candidate_departments)
    return {dept: float(n - i) for i, dept in enumerate(candidate_departments)}

def accumulate_scores(
    base: dict[str, float],
    selections: list[dict],
    active_depts: list[str],
) -> dict[str, float]:
    totals = dict(base)
    for sel in selections:
        for dept, pts in (sel.get("scores") or {}).items():
            if dept in active_depts:
                totals[dept] = totals.get(dept, 0.0) + float(pts)
    return totals

def lock_department_from_totals(
    totals: dict[str, float],
    candidate_departments: list[str],
    active_depts: list[str],
    none_selected: bool,
) -> tuple[str, dict[str, float], float, bool]:
    locked, dept, margin = try_lock_department(totals)
    used_tie_break = False
    if locked and dept:
        return dept, totals, margin, used_tie_break
    best_score = max((totals.get(d, 0.0) for d in active_depts), default=0.0)
    tied = [d for d in candidate_departments if d in active_depts and totals.get(d, 0.0) == best_score]
    if len(tied) > 1:
        used_tie_break = True
    fallback = next((d for d in candidate_departments if d in active_depts and totals.get(d, 0.0) == best_score), active_depts[0])
    return fallback, totals, margin, used_tie_break
```

- [ ] **Step 4: Run — expect PASS**

- [ ] **Step 5: Commit**

```bash
git add app/triage/dept_rules_scoring.py tests/test_dept_rules_scoring.py
git commit -m "feat: dept rules scoring with gender filter and fallback"
```

---

### Task 4: OpenSearch department rules index + CL0001 data

**Files:**
- Modify: `sourceData/opensearch_mappings.py`
- Create: `sourceData/opensearch_dept_rules.py`
- Create: `app/infra/opensearch_dept_rules.py`
- Modify: `app/core/config.py`
- Modify: `sourceData/data/rag_knowledge.jsonl`
- Modify: `sourceData/opensearch_rag_kb.py` (`enrich_doc` — index CL `aliases` as `alliance`, `symptom_id`, `questions`)

**Interfaces:**
- Produces: `search_dept_rule(symptom_id: str, location: str) -> dict | None`

- [ ] **Step 1: Add mapping `rag_department_rules_index_body()`**

```python
def rag_department_rules_index_body() -> dict[str, Any]:
    return {
        "settings": {"index": {"number_of_shards": 1, "number_of_replicas": 0}},
        "mappings": {
            "properties": {
                "id": {"type": "keyword"},
                "symptom_id": {"type": "keyword"},
                "location": {"type": "keyword"},
                "candidate_departments": {"type": "keyword"},
                "differential_questions": {"type": "object", "enabled": False},
                "raw_json": {"type": "text", "index": False},
            }
        },
    }
```

- [ ] **Step 2: Create `sourceData/opensearch_dept_rules.py`** — mirror `sourceData/opensearch_rag_kb.py` bulk loader for `sourceData/data/rag_department_rules.jsonl`, index name from `RAG_DEPT_RULES_INDEX` default `rag_department_rules`.

- [ ] **Step 3: Create `app/infra/opensearch_dept_rules.py`**

```python
def search_dept_rule(symptom_id: str, location: str) -> dict | None:
    client = get_opensearch_client()
    if not client:
        return None
    body = {
        "query": {
            "bool": {
                "must": [
                    {"term": {"symptom_id": symptom_id}},
                    {"term": {"location": location}},
                ]
            }
        },
        "size": 1,
    }
    res = client.search(index=config.RAG_DEPT_RULES_INDEX, body=body)
    hits = res["hits"]["hits"]
    return dict(hits[0]["_source"]) if hits else None
```

- [ ] **Step 4: Add to `app/core/config.py`**

```python
RAG_DEPT_RULES_INDEX: str = os.getenv("RAG_DEPT_RULES_INDEX", "rag_department_rules")
```

- [ ] **Step 5: Update CL0001 line in `sourceData/data/rag_knowledge.jsonl`** per spec §3.1 (full `questions` for age, sex, pain_location, red_flags).

- [ ] **Step 6: Update `enrich_doc` in `sourceData/opensearch_rag_kb.py` for `symptomClarify`**

```python
if doc.get("type") == "symptomClarify":
    parts.extend(doc.get("aliases") or [])
    out["alliance"] = list(doc.get("aliases") or [])
    out["symptom_id"] = doc.get("symptom_id")
```

- [ ] **Step 7: Run index scripts**

```bash
.\.venv\Scripts\python.exe sourceData/opensearch_rag_kb.py
.\.venv\Scripts\python.exe sourceData/opensearch_dept_rules.py
```

- [ ] **Step 8: Commit**

```bash
git add sourceData/ app/infra/opensearch_dept_rules.py app/core/config.py
git commit -m "feat: add rag_department_rules index and CL0001 question data"
```

---

### Task 5: `symptom_clarify` graph node

**Files:**
- Create: `app/graph/nodes/symptom_clarify.py`
- Create: `app/triage/clarify_helpers.py` (format question, advance phase, build choices from CL chunk)

**Interfaces:**
- Consumes: `SymptomClarifyState`, `ClarifyChoice`, `search_dept_rule`
- Produces: `symptom_clarify_node(state: AppState) -> dict` — sets `clarify_state`, optional `messages` with AIMessage question, `status=asking` → graph END

**Phase logic:**
- On init (from RAG CL chunk): set `symptom_id`, `clarify_chunk_id`, `phase=age`
- On user reply: validate single choice via `resolve_dept_choice` pattern (reuse from `dept_choices.resolve_dept_choice` with `ClarifyChoice` adapter or convert to `DeptChoice`)
- After `pain_location` filled: call `search_dept_rule(symptom_id, filled_slots["pain_location"])`; store in `dept_rule_chunk`; **do not** ask differential here — return patch that routes to `dept_rules_disambiguation` (set `clarify_state.phase` marker or separate flag `awaiting_differential=True`)
- After confidence pass: set `phase=red_flags`, ask red_flags (single select including 都没有)
- After red_flags: `status=done`, route to `answer_generate`

- [ ] **Step 1: Implement `app/triage/clarify_helpers.py`**

```python
def choices_for_slot(cl_chunk: dict, slot: str) -> tuple[str, list[ClarifyChoice]]:
    q = (cl_chunk.get("questions") or {}).get(slot) or {}
    text = q.get("text") or ""
    options = q.get("options") or []
    choices = [ClarifyChoice(id=f"c{i}", label=lb, slot=slot) for i, lb in enumerate(options, 1)]
    return text, choices

def format_clarify_message(question: str, choices: list[ClarifyChoice]) -> str:
    lines = [question, ""]
    for i, c in enumerate(choices, 1):
        lines.append(f"{i}. {c.label}")
    return "\n".join(lines)

def next_slot_phase(current: str, required_slots: list[str]) -> str | None:
    order = ["age", "sex", "pain_location", "red_flags"]
    slots = [s for s in order if s in required_slots]
    if current not in slots:
        return slots[0] if slots else None
    idx = slots.index(current)
    return slots[idx + 1] if idx + 1 < len(slots) else None
```

- [ ] **Step 2: Implement `app/graph/nodes/symptom_clarify.py`** — full node per phase logic above; sync `slot_table.gender` / `slot_table.age` when sex/age filled.

- [ ] **Step 3: Manual smoke** — not runnable until graph wired; defer integration to Task 7.

- [ ] **Step 4: Commit**

```bash
git add app/graph/nodes/symptom_clarify.py app/triage/clarify_helpers.py
git commit -m "feat: symptom_clarify node for CL slot questionnaire"
```

---

### Task 6: `dept_rules_disambiguation` node

**Files:**
- Create: `app/graph/nodes/dept_rules_disambiguation.py`

**Interfaces:**
- Consumes: `filter_rule_by_sex`, `build_base_scores`, `accumulate_scores`, `lock_department_from_totals`, `resolve_multi_choice`, `format_choice_message` (adapt template: `"为更准确推荐科室，请选择您是否有以下情况（可多选，输入编号如 1,3）："`)
- Produces: `dept_rules_disambiguation_node` — on ask: `dept_state.status=asking`, `choice_mode=differential`, `multi_select=True`; on lock: `locked_department`, `dept_state.status=locked`, then edge to `dept_confidence`

- [ ] **Step 1: Build differential choices from `differential_questions`**

```python
def build_differential_choices(rule_chunk: dict) -> list[DeptChoice]:
    choices = []
    for i, q in enumerate(rule_chunk.get("differential_questions") or [], 1):
        choices.append(DeptChoice(
            id=f"c{i}",
            label=q["text"],
            target_departments=list((q.get("scores") or {}).keys()),
        ))
    choices.append(DeptChoice(id="none", label="都没有", target_departments=[]))
    return choices
```

Store raw `scores` on side channel: keep `rule_chunk` in state for accumulate step; map picked `DeptChoice.id` back to `differential_questions[i-1]`.

- [ ] **Step 2: Implement node** — first visit shows choices; follow-up parses multi-select; compute totals; set `locked_department`; write `filled_slots["differential"]` as comma-joined labels in `clarify_state`.

- [ ] **Step 3: Commit**

```bash
git add app/graph/nodes/dept_rules_disambiguation.py
git commit -m "feat: dept_rules_disambiguation multi-select scoring node"
```

---

### Task 7: LLM confidence + reject nodes

**Files:**
- Create: `app/graph/nodes/dept_confidence.py`
- Create: `app/graph/nodes/low_confidence_reject.py`
- Create: `app/triage/dept_confidence_prompt.py`

**Interfaces:**
- Produces: `dept_confidence_node`, `low_confidence_reject_node`, `build_confidence_prompt(state) -> str`

- [ ] **Step 1: Implement prompt builder (exclude red_flags)**

```python
def slots_for_confidence(state: AppState) -> dict[str, str]:
    cs = state.clarify_state
    if not cs:
        return {}
    allow = {"age", "sex", "pain_location", "differential"}
    return {k: v for k, v in (cs.filled_slots or {}).items() if k in allow}
```

Include `state.slot_table`, `state.locked_department`, `cs.dept_rule_chunk` if present.

- [ ] **Step 2: Implement `dept_confidence_node`**

```python
from app.core.llm import get_chat_llm
from app.domain.dept_confidence import DeptConfidenceResult

structured = get_chat_llm().with_structured_output(DeptConfidenceResult)
result = structured.invoke([HumanMessage(content=prompt)])
passed = result.score >= 60
return {
    "dept_confidence_result": result,
    "dept_confidence_passed": passed,
}
```

On exception: `DeptConfidenceResult(score=0.0, reason="llm_error")`, `passed=False`.

- [ ] **Step 3: Implement `low_confidence_reject_node`**

```python
score = state.dept_confidence_result.score if state.dept_confidence_result else 0.0
reply = f"根据目前信息暂无法准确推荐科室（置信度 {score:.0f} 分，需 ≥60 分）。建议补充症状或到医院分诊台咨询。"
return {"messages": [AIMessage(content=reply)], "locked_department": None}
```

- [ ] **Step 4: Commit**

```bash
git add app/graph/nodes/dept_confidence.py app/graph/nodes/low_confidence_reject.py app/triage/dept_confidence_prompt.py
git commit -m "feat: LLM dept confidence gate and low-confidence reject"
```

---

### Task 8: Graph wiring + routing

**Files:**
- Modify: `app/graph/builder.py`
- Modify: `app/domain/routing.py`
- Modify: `app/graph/nodes/rag_symptom_recall.py`
- Modify: `app/graph/nodes/dept_disambiguation.py` (after lock → route to confidence, not answer)
- Create: `tests/test_symptom_clarify_routing.py`

**Graph edges (target):**

```
rag_symptom_recall → conditional route_after_rag → symptom_clarify | dept_disambiguation
symptom_clarify → conditional:
  - asking → END
  - pain_location_done → dept_rules_disambiguation
  - red_flags_done → answer_generate
dept_rules_disambiguation → conditional:
  - asking → END
  - locked → dept_confidence
dept_disambiguation → conditional (replace direct answer):
  - asking → END
  - locked | fallback | emergency → dept_confidence
dept_confidence → conditional route_after_confidence:
  - passed → symptom_clarify (red_flags phase) OR answer_generate if red_flags already done / not required
  - failed → low_confidence_reject
low_confidence_reject → END
```

- [ ] **Step 1: Add `is_clarify_followup` to `app/domain/routing.py`**

```python
def is_clarify_followup_reply(state: AppState) -> bool:
    cs = state.clarify_state
    return bool(cs and cs.status == "asking" and cs.last_choices)
```

Extend `route_after_trim` priority: clarify followup before dept followup.

- [ ] **Step 2: Add `route_after_rag`, `route_after_clarify`, `route_after_dept_rules`, `route_after_confidence`, `route_after_dept_locked`**

- [ ] **Step 3: Update `app/graph/builder.py`** — register all new nodes and conditional edges.

- [ ] **Step 4: Update `rag_symptom_recall.py`** — when top hit `type != symptomClarify`, scan hits for first `symptomClarify` with alliance match in query (so「肚子疼」 prefers CL0001 over unrelated RK).

- [ ] **Step 5: Write routing tests**

```python
def test_route_after_rag_symptom_clarify():
    state = AppState(rag_chunk={"type": "symptomClarify", "id": "CL0001"})
    assert route_after_rag(state) == "symptom_clarify"
```

- [ ] **Step 6: Run tests**

```bash
.\.venv\Scripts\python.exe -m pytest tests/test_symptom_clarify_routing.py tests/test_multi_choice.py tests/test_dept_rules_scoring.py -v
```

- [ ] **Step 7: Commit**

```bash
git add app/graph/builder.py app/domain/routing.py app/graph/nodes/
git commit -m "feat: wire symptom clarify and confidence into LangGraph"
```

---

### Task 9: API, chat_service, CLI

**Files:**
- Modify: `app/services/chat_service.py`
- Modify: `app/api/routers/chat.py`
- Modify: `cli.py`
- Modify: `app/domain/state_debug.py` (optional debug fields)

- [ ] **Step 1: Extend `chat_service.chat_once` return dict**

```python
cs = state.clarify_state
awaiting_clarify = bool(cs and cs.status == "asking" and cs.phase in ("age", "sex", "pain_location", "red_flags"))
# differential uses existing awaiting_dept_choice with multi_select from dept_state

return {
    ...
    "awaiting_clarify": awaiting_clarify,
    "clarify_phase": cs.phase if cs else None,
    "clarify_choices": [c.model_dump() for c in cs.last_choices] if awaiting_clarify and cs else [],
    "multi_select": bool(ds and ds.multi_select) if ds else False,
    "dept_confidence": (state.dept_confidence_result.score if state.dept_confidence_result else None),
    "dept_confidence_passed": state.dept_confidence_passed,
    "dept_confidence_reason": (state.dept_confidence_result.reason if state.dept_confidence_result else None),
    "locked_department": state.locked_department,
}
```

- [ ] **Step 2: Extend `ChatResponse` in `app/api/routers/chat.py`** with same fields.

- [ ] **Step 3: Update `cli.py` `_ask_chat`**

```python
while data.get("awaiting_clarify") and data.get("clarify_choices"):
    labels = [c["label"] for c in data["clarify_choices"]]
    # Prompt.ask single select
    ...

while data.get("awaiting_dept_choice") and data.get("dept_choices"):
    if data.get("multi_select"):
        pick = Prompt.ask("您的选择（可多选，如 1,3）", console=console)
    else:
        pick = Prompt.ask("您的选择", choices=labels, console=console)
```

- [ ] **Step 4: Commit**

```bash
git add app/services/chat_service.py app/api/routers/chat.py cli.py
git commit -m "feat: expose clarify and confidence fields in chat API and CLI"
```

---

### Task 10: End-to-end verification

**Files:**
- Create: `scripts/test_abdominal_clarify_flow.py` (optional integration script)

- [ ] **Step 1: Re-index OpenSearch** (Task 4 commands)

- [ ] **Step 2: Start API** and run manual CLI flow:

```
肚子疼 → age → sex → 右下腹 → multi differential → red_flags → 科室 + confidence
```

Male path: verify no 妇科 option in differential.

- [ ] **Step 3: Run existing regression**

```bash
.\.venv\Scripts\python.exe scripts/test_dept_choices.py
.\.venv\Scripts\python.exe -m pytest tests/ -v
```

- [ ] **Step 4: Commit any test script**

```bash
git add scripts/test_abdominal_clarify_flow.py
git commit -m "test: abdominal clarify e2e script"
```

---

## Spec Coverage Check

| Spec § | Task |
|--------|------|
| §1.2 flow | Tasks 5–8 |
| §2.4 indexes | Task 4 |
| §3.1 CL0001 data | Task 4 |
| §3.3–3.4 models | Task 1 |
| §4 routing | Task 8 |
| §5.1 gender filter | Task 3, 6 |
| §5.2 scoring | Task 3, 6 |
| §5.3 red_flags after confidence | Tasks 5, 8 |
| §5.4 LLM confidence | Task 7 |
| §6 API/CLI | Task 9 |
| §7 error handling | Tasks 4, 6, 7 (null checks + reprompt) |
| §8 tests | Tasks 2, 3, 8, 10 |
| Foot RK + confidence | Task 8 (`dept_disambiguation` → `dept_confidence`) |

## Placeholder Scan

No TBD/TODO placeholders. All task files and signatures are named explicitly.

---

**Plan complete and saved to `docs/superpowers/plans/2026-06-24-symptom-clarify-dept-rules.md`. Two execution options:**

**1. Subagent-Driven (recommended)** — dispatch a fresh subagent per task, review between tasks, fast iteration

**2. Inline Execution** — execute tasks in this session using executing-plans, batch execution with checkpoints

**Which approach?**
