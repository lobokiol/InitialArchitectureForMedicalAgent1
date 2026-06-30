# 急诊 / 红旗症状硬规则门禁 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** On new intake, match crisis keywords from `rag_knowledge.jsonl` `EM*` entries after `slot_fill`; on hit, lock 急诊 and emit a fixed reply without RAG, clarify, disambiguation, or LLM confidence.

**Architecture:** New module `app/triage/emergency_rules.py` loads `EM*` alliance keywords from JSONL at import time. New LangGraph node `emergency_gate` runs before `slot_gate`. Hit → `answer_generate`; miss → existing flow. Legacy scattered `_EMERGENCY_*` constants removed; `dept_disambiguation` reuses `match_emergency()` for defense-in-depth.

**Tech Stack:** Python 3.11+, FastAPI, LangGraph, Pydantic v2, pytest.

## Global Constraints

- Gate placement: `slot_fill` → **`emergency_gate`** → `slot_gate` (gate **before** slot_gate so bare「昏迷」still routes to ER)
- Detection scope: **new intake only** (`decision → slot_fill` path); clarify/dept follow-ups **do not** re-check
- Keyword source: `sourceData/data/rag_knowledge.jsonl`, `type=emergency`, field `alliance`; runtime **direct JSONL read**, no OpenSearch
- Match rule: substring on `ner_result.query`; on multiple hits pick **longest** keyword
- On hit: `locked_department="急诊"`, `dept_state.status="emergency"`, skip RAG / clarify / disambiguation / `dept_confidence` / `fetch_oncall`
- Reply: fixed template in `answer_generate`; use EM entry `emergency_reply` or default「您描述的情况可能存在急危重症风险，请立即前往急诊或拨打 120。」
- reject unchanged: existing `triage_route=reject` / `slot_gate` for non-medical input
- `default_department` on EM0001: **「急诊」** (not「急症」)

**Spec:** `docs/superpowers/specs/2026-06-30-emergency-red-flag-gate-design.md`

---

## File Map

| File | Responsibility |
|------|----------------|
| `sourceData/data/rag_knowledge.jsonl` | Enrich `EM0001` alliance (~16 keywords) + `emergency_reply` |
| `scripts/gen_triage_body_kb.py` | Sync EM0001 template with jsonl |
| `app/triage/emergency_rules.py` | **New:** load EM*, `match_emergency()`, `DEFAULT_EMERGENCY_REPLY` |
| `app/graph/nodes/emergency_gate.py` | **New:** gate node |
| `app/domain/routing.py` | **New:** `route_after_emergency_gate` |
| `app/graph/builder.py` | Insert node + conditional edges |
| `app/domain/models.py` | `emergency_gate_passed`, `emergency_match` on `AppState` |
| `app/triage/session_reset.py` | Clear new fields on new intake |
| `app/graph/nodes/answer.py` | Read `emergency_reply` from `rag_chunk` |
| `app/triage/slot_fill.py` | Remove `_EMERGENCY_RE` / `table.emergency` |
| `app/graph/nodes/rag_symptom_recall.py` | Remove `table.emergency` query boost |
| `app/triage/turn_text.py` | Remove `table.emergency` append |
| `app/graph/nodes/dept_disambiguation.py` | `_is_emergency` → `match_emergency()` |
| `tests/test_emergency_gate.py` | **New:** unit + node + routing tests |

---

### Task 1: `emergency_rules` matcher (TDD core)

**Files:**
- Create: `app/triage/emergency_rules.py`
- Create: `tests/test_emergency_gate.py`

**Interfaces:**
- Produces: `@dataclass EmergencyMatch(keyword: str, em_id: str, entry: dict)`
- Produces: `DEFAULT_EMERGENCY_REPLY: str`
- Produces: `load_emergency_entries(path: Path | None = None) -> list[dict]`
- Produces: `reload_emergency_entries(entries: list[dict]) -> None` (test helper)
- Produces: `match_emergency(text: str) -> EmergencyMatch | None`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_emergency_gate.py
from __future__ import annotations

import pytest

from app.triage.emergency_rules import (
    DEFAULT_EMERGENCY_REPLY,
    EmergencyMatch,
    match_emergency,
    reload_emergency_entries,
)


@pytest.fixture(autouse=True)
def _mock_em_entries():
    reload_emergency_entries([
        {
            "id": "EM0001",
            "type": "emergency",
            "alliance": ["大出血", "大量出血", "不能动", "昏迷", "剧烈"],
            "default_department": "急诊",
            "emergency_reply": "请立即前往急诊或拨打 120。",
        }
    ])
    yield


def test_match_emergency_hits():
    hit = match_emergency("脚脖子肿，不能动，皮发紫")
    assert hit is not None
    assert hit.keyword == "不能动"
    assert hit.em_id == "EM0001"


def test_match_emergency_misses():
    assert match_emergency("我有胃炎") is None


def test_match_emergency_longest_keyword():
    hit = match_emergency("患者大量出血")
    assert hit is not None
    assert hit.keyword == "大量出血"


def test_match_emergency_bare_crisis_word():
    hit = match_emergency("昏迷")
    assert hit is not None
    assert hit.keyword == "昏迷"


def test_default_reply_constant():
    assert "120" in DEFAULT_EMERGENCY_REPLY
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_emergency_gate.py -v`
Expected: FAIL — `app.triage.emergency_rules` not found

- [ ] **Step 3: Implement `app/triage/emergency_rules.py`**

```python
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

_DEFAULT_PATH = (
    Path(__file__).resolve().parents[2] / "sourceData" / "data" / "rag_knowledge.jsonl"
)
DEFAULT_EMERGENCY_REPLY = (
    "您描述的情况可能存在急危重症风险，请立即前往急诊或拨打 120。"
)

_entries: list[dict] = []


@dataclass(frozen=True)
class EmergencyMatch:
    keyword: str
    em_id: str
    entry: dict


def load_emergency_entries(path: Path | None = None) -> list[dict]:
    p = path or _DEFAULT_PATH
    out: list[dict] = []
    if not p.is_file():
        return out
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or not line.startswith("{"):
            continue
        try:
            doc = json.loads(line)
        except json.JSONDecodeError:
            continue
        if doc.get("type") == "emergency":
            out.append(doc)
    return out


def reload_emergency_entries(entries: list[dict]) -> None:
    global _entries
    _entries = list(entries)


def _ensure_loaded() -> None:
    global _entries
    if not _entries:
        _entries = load_emergency_entries()


def match_emergency(text: str) -> EmergencyMatch | None:
    if not text or not text.strip():
        return None
    _ensure_loaded()
    blob = text.strip()
    best: EmergencyMatch | None = None
    for entry in _entries:
        em_id = str(entry.get("id") or "")
        for kw in entry.get("alliance") or []:
            if not isinstance(kw, str) or not kw:
                continue
            if kw in blob and (best is None or len(kw) > len(best.keyword)):
                best = EmergencyMatch(keyword=kw, em_id=em_id, entry=entry)
    return best


# Load on import
_entries = load_emergency_entries()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_emergency_gate.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add app/triage/emergency_rules.py tests/test_emergency_gate.py
git commit -m "feat: add emergency_rules matcher from EM* jsonl entries"
```

---

### Task 2: Enrich EM0001 source data

**Files:**
- Modify: `sourceData/data/rag_knowledge.jsonl` (line 2, EM0001)
- Modify: `scripts/gen_triage_body_kb.py` (EM0001 dict ~lines 79-87)

**Interfaces:**
- Consumes: none
- Produces: EM0001 jsonl line with 16 alliance terms + `emergency_reply` + `default_department: 急诊`

- [ ] **Step 1: Update EM0001 in `rag_knowledge.jsonl`**

Replace line 2 with (single line JSON):

```json
{"id": "EM0001", "body_part": "全身", "gender": ["男", "女"], "age": ["0-3个月", "3个月-1岁", "2-4岁", "5-11岁", "12-18岁", "19-35岁", "35-59岁", "60岁及以上"], "type": "emergency", "alliance": ["大出血", "大量出血", "吐血", "咯血", "不能动", "动不了", "畸形", "不能负重", "发紫", "皮肤发黑", "意识不清", "昏迷", "剧烈", "受不了", "喘不过气", "胸痛憋气"], "default_department": "急诊", "emergency_reply": "您描述的情况可能存在急危重症风险，请立即前往急诊或拨打 120。"}
```

- [ ] **Step 2: Sync `scripts/gen_triage_body_kb.py` EM0001 block**

```python
        {
            "id": "EM0001",
            "body_part": "全身",
            "gender": ["男", "女"],
            "age": AGE_OPTS,
            "type": "emergency",
            "alliance": [
                "大出血", "大量出血", "吐血", "咯血",
                "不能动", "动不了", "畸形", "不能负重",
                "发紫", "皮肤发黑", "意识不清", "昏迷",
                "剧烈", "受不了", "喘不过气", "胸痛憋气",
            ],
            "default_department": "急诊",
            "emergency_reply": "您描述的情况可能存在急危重症风险，请立即前往急诊或拨打 120。",
        },
```

- [ ] **Step 3: Verify loader picks up real file**

Add to `tests/test_emergency_gate.py`:

```python
def test_load_real_em0001_from_jsonl():
    from app.triage.emergency_rules import load_emergency_entries, _DEFAULT_PATH

    entries = load_emergency_entries(_DEFAULT_PATH)
    em = next(e for e in entries if e.get("id") == "EM0001")
    assert "不能动" in em.get("alliance", [])
    assert em.get("default_department") == "急诊"
```

Run: `python -m pytest tests/test_emergency_gate.py::test_load_real_em0001_from_jsonl -v`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add sourceData/data/rag_knowledge.jsonl scripts/gen_triage_body_kb.py tests/test_emergency_gate.py
git commit -m "data: enrich EM0001 crisis keywords and emergency_reply"
```

---

### Task 3: AppState fields + session reset

**Files:**
- Modify: `app/domain/models.py`
- Modify: `app/triage/session_reset.py`
- Modify: `tests/test_emergency_gate.py`

**Interfaces:**
- Produces on `AppState`: `emergency_gate_passed: bool | None = None`, `emergency_match: dict | None = None`

- [ ] **Step 1: Write failing test**

```python
# append to tests/test_emergency_gate.py
from app.domain.models import AppState

def test_app_state_emergency_fields_defaults():
    state = AppState()
    assert state.emergency_gate_passed is None
    assert state.emergency_match is None
```

- [ ] **Step 2: Run test — expect FAIL**

Run: `python -m pytest tests/test_emergency_gate.py::test_app_state_emergency_fields_defaults -v`

- [ ] **Step 3: Add fields to `AppState` in `app/domain/models.py`**

After `dept_confidence_passed`:

```python
    emergency_gate_passed: bool | None = None
    emergency_match: dict | None = None
```

In `app/triage/session_reset.py`, add to return dict:

```python
        "emergency_gate_passed": None,
        "emergency_match": None,
```

- [ ] **Step 4: Run test — expect PASS**

Run: `python -m pytest tests/test_emergency_gate.py::test_app_state_emergency_fields_defaults -v`

- [ ] **Step 5: Commit**

```bash
git add app/domain/models.py app/triage/session_reset.py tests/test_emergency_gate.py
git commit -m "feat: add emergency_gate fields to AppState and session reset"
```

---

### Task 4: `emergency_gate` node + routing + graph wiring

**Files:**
- Create: `app/graph/nodes/emergency_gate.py`
- Modify: `app/domain/routing.py`
- Modify: `app/graph/builder.py`
- Modify: `tests/test_emergency_gate.py`

**Interfaces:**
- Consumes: `match_emergency(text: str) -> EmergencyMatch | None`
- Produces: `emergency_gate_node(state: AppState) -> dict`
- Produces: `route_after_emergency_gate(state: AppState) -> str` → `"answer_generate"` | `"slot_gate"`

- [ ] **Step 1: Write failing node/routing tests**

```python
# append to tests/test_emergency_gate.py
from app.domain.models import AppState, IntentResult
from app.domain.routing import route_after_emergency_gate
from app.graph.nodes.emergency_gate import emergency_gate_node
from app.ner.models import EntityExtractResult

def test_emergency_gate_node_hit():
    state = AppState(
        ner_result=EntityExtractResult(query="昏迷", primary_symptom=None),
    )
    patch = emergency_gate_node(state)
    assert patch["emergency_gate_passed"] is False
    assert patch["locked_department"] == "急诊"
    assert patch["emergency_match"]["keyword"] == "昏迷"
    assert patch["dept_state"].status == "emergency"


def test_emergency_gate_node_miss():
    state = AppState(
        ner_result=EntityExtractResult(query="我有胃炎", primary_disease="胃炎"),
    )
    patch = emergency_gate_node(state)
    assert patch == {"emergency_gate_passed": True}


def test_route_after_emergency_gate_hit():
    state = AppState(emergency_gate_passed=False, locked_department="急诊")
    assert route_after_emergency_gate(state) == "answer_generate"


def test_route_after_emergency_gate_miss():
    state = AppState(emergency_gate_passed=True)
    assert route_after_emergency_gate(state) == "slot_gate"
```

- [ ] **Step 2: Run tests — expect FAIL**

Run: `python -m pytest tests/test_emergency_gate.py -k "emergency_gate_node or route_after" -v`

- [ ] **Step 3: Create `app/graph/nodes/emergency_gate.py`**

```python
from __future__ import annotations

from app.core.logging import logger
from app.domain.dept_disambiguation import DeptDisambiguationState
from app.domain.models import AppState
from app.triage.emergency_rules import match_emergency


def emergency_gate_node(state: AppState) -> dict:
    logger.info(">>> Enter node: emergency_gate")
    text = (state.ner_result.query if state.ner_result else "") or ""
    hit = match_emergency(text)
    if hit is None:
        logger.info("emergency_gate: no match")
        return {"emergency_gate_passed": True}
    logger.info("emergency_gate: hit keyword=%r em_id=%s", hit.keyword, hit.em_id)
    return {
        "emergency_gate_passed": False,
        "locked_department": "急诊",
        "dept_state": DeptDisambiguationState(
            status="emergency",
            candidate_departments=[{"department": "急诊"}],
        ),
        "emergency_match": {"keyword": hit.keyword, "em_id": hit.em_id},
        "rag_chunk": hit.entry,
    }
```

- [ ] **Step 4: Add routing in `app/domain/routing.py`**

```python
def route_after_emergency_gate(state: AppState) -> str:
    if state.emergency_gate_passed is False and state.locked_department == "急诊":
        logger.info(">>> route_after_emergency_gate: emergency -> answer_generate")
        return "answer_generate"
    logger.info(">>> route_after_emergency_gate: pass -> slot_gate")
    return "slot_gate"
```

- [ ] **Step 5: Wire graph in `app/graph/builder.py`**

Add import:

```python
from app.graph.nodes.emergency_gate import emergency_gate_node
from app.domain.routing import route_after_emergency_gate
```

(add `route_after_emergency_gate` to existing routing import)

Add node:

```python
    graph.add_node("emergency_gate", emergency_gate_node)
```

Replace edge `graph.add_edge("slot_fill", "slot_gate")` with:

```python
    graph.add_edge("slot_fill", "emergency_gate")
    graph.add_conditional_edges(
        "emergency_gate",
        route_after_emergency_gate,
        {
            "answer_generate": "answer_generate",
            "slot_gate": "slot_gate",
        },
    )
```

- [ ] **Step 6: Run tests — expect PASS**

Run: `python -m pytest tests/test_emergency_gate.py -k "emergency_gate_node or route_after" -v`

- [ ] **Step 7: Commit**

```bash
git add app/graph/nodes/emergency_gate.py app/domain/routing.py app/graph/builder.py tests/test_emergency_gate.py
git commit -m "feat: wire emergency_gate node before slot_gate in LangGraph"
```

---

### Task 5: `answer_generate` emergency reply from EM entry

**Files:**
- Modify: `app/graph/nodes/answer.py`
- Modify: `tests/test_emergency_gate.py`

**Interfaces:**
- Consumes: `state.rag_chunk["emergency_reply"]`, `DEFAULT_EMERGENCY_REPLY` fallback
- Produces: fixed Markdown reply, no LLM call

- [ ] **Step 1: Write failing test**

```python
# append to tests/test_emergency_gate.py
from langchain_core.messages import AIMessage
from app.domain.dept_disambiguation import DeptDisambiguationState
from app.graph.nodes.answer import answer_generate_node

def test_answer_generate_emergency_uses_emergency_reply():
    state = AppState(
        locked_department="急诊",
        dept_state=DeptDisambiguationState(status="emergency"),
        rag_chunk={
            "emergency_reply": "请立即前往急诊或拨打 120。",
        },
        slot_table=None,
    )
    patch = answer_generate_node(state)
    msg = patch["messages"][0]
    assert isinstance(msg, AIMessage)
    assert "**急诊**" in msg.content
    assert "请立即前往急诊或拨打 120。" in msg.content
```

- [ ] **Step 2: Run test — expect FAIL** (old branch uses `emergency_flag`)

Run: `python -m pytest tests/test_emergency_gate.py::test_answer_generate_emergency_uses_emergency_reply -v`

- [ ] **Step 3: Update emergency branch in `app/graph/nodes/answer.py`**

Replace the `if dept == "急诊":` block:

```python
        if dept == "急诊":
            from app.triage.emergency_rules import DEFAULT_EMERGENCY_REPLY

            chunk = state.rag_chunk or {}
            detail = chunk.get("emergency_reply") or DEFAULT_EMERGENCY_REPLY
            canonical = chunk.get("canonical_symptom") or (
                state.slot_table.primary_symptom if state.slot_table else ""
            )
            symptom_part = f"（{canonical}）" if canonical else ""
            full_content = (
                f"{prefix}根据您描述的情况{symptom_part}，建议尽快就诊：**急诊**。\n{detail}"
            )
```

- [ ] **Step 4: Run test — expect PASS**

Run: `python -m pytest tests/test_emergency_gate.py::test_answer_generate_emergency_uses_emergency_reply -v`

- [ ] **Step 5: Commit**

```bash
git add app/graph/nodes/answer.py tests/test_emergency_gate.py
git commit -m "feat: answer_generate uses EM emergency_reply for crisis path"
```

---

### Task 6: Remove legacy emergency scatter + unify dept_disambiguation

**Files:**
- Modify: `app/triage/slot_fill.py`
- Modify: `app/graph/nodes/rag_symptom_recall.py`
- Modify: `app/triage/turn_text.py`
- Modify: `app/graph/nodes/dept_disambiguation.py`

**Interfaces:**
- Consumes: `match_emergency(text: str)` in dept_disambiguation

- [ ] **Step 1: Clean `app/triage/slot_fill.py`**

Remove `_EMERGENCY_RE` and the block:

```python
    if _EMERGENCY_RE.search(q):
        table.emergency = _EMERGENCY_RE.search(q).group(0)
```

- [ ] **Step 2: Clean `app/graph/nodes/rag_symptom_recall.py`**

Remove lines:

```python
    if table.emergency:
        q = f"{q} {table.emergency}".strip()
```

- [ ] **Step 3: Clean `app/triage/turn_text.py`**

Remove:

```python
        if table.emergency:
            parts.append(table.emergency)
```

- [ ] **Step 4: Unify `app/graph/nodes/dept_disambiguation.py`**

Replace `_EMERGENCY_KW` and `_is_emergency` with:

```python
from app.triage.emergency_rules import match_emergency

def _is_emergency(chunk: dict, user_text: str) -> bool:
    return match_emergency(user_text) is not None
```

Remove unused `_EMERGENCY_KW` tuple. Keep `chunk` param for call-site compatibility.

- [ ] **Step 5: Run full emergency test suite + existing tests**

Run: `python -m pytest tests/test_emergency_gate.py tests/test_fetch_oncall.py tests/test_triage_recorder.py -q`
Expected: all PASS

- [ ] **Step 6: Commit**

```bash
git add app/triage/slot_fill.py app/graph/nodes/rag_symptom_recall.py app/triage/turn_text.py app/graph/nodes/dept_disambiguation.py
git commit -m "refactor: consolidate emergency detection into emergency_rules"
```

---

### Task 7: Regression + manual acceptance

**Files:**
- Modify: `scripts/integration_triage_db.py` (optional assertion on emergency reply containing「急诊」)

- [ ] **Step 1: Run full pytest**

Run: `python -m pytest tests/ -q --ignore=tests/e2e 2>/dev/null || python -m pytest tests/test_emergency_gate.py tests/test_fetch_oncall.py tests/test_triage_recorder.py -q`
Expected: PASS

- [ ] **Step 2: Manual smoke (if backend running)**

```bash
# new thread
curl -s -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"user_id":"smoke-em","message":"脚脖子肿，不能动，皮发紫"}' | jq -r '.reply'
```

Expected: reply contains `**急诊**` and `120`; no `dept_choices` / `awaiting_clarify`.

- [ ] **Step 3: Verify non-emergency unchanged**

```bash
curl -s -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"user_id":"smoke-em","message":"我有胃炎"}' | jq -r '.intent_result.triage_route'
```

Expected: `disease`

- [ ] **Step 4: Update spec status**

In `docs/superpowers/specs/2026-06-30-emergency-red-flag-gate-design.md`, change `状态: 待实现` → `状态: 已实现`.

- [ ] **Step 5: Commit**

```bash
git add docs/superpowers/specs/2026-06-30-emergency-red-flag-gate-design.md
git commit -m "docs: mark emergency red-flag gate spec as implemented"
```

---

## Spec Coverage Checklist

| Spec § | Task |
|--------|------|
| §2.2 graph placement before slot_gate | Task 4 |
| §3 EM0001 alliance + emergency_reply | Task 2 |
| §3.2 longest match | Task 1 |
| §4 emergency_gate_node | Task 4 |
| §4.3 fixed answer template | Task 5 |
| §6 legacy cleanup | Task 6 |
| §7 tests | Tasks 1, 4, 5, 7 |
| §8 non-goals (follow-up re-check, OS search) | not implemented (by design) |

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-06-30-emergency-red-flag-gate.md`.

**Two execution options:**

1. **Subagent-Driven (recommended)** — dispatch a fresh subagent per task, review between tasks, fast iteration
2. **Inline Execution** — implement tasks in this session with checkpoints

Which approach?
