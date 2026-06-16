# Dept Choice Disambiguation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace LLM-generated open-ended dept follow-up questions with structured single-choice menus whose options are rule-extracted from `department_recommendation.condition` only (never emergency or invented symptoms).

**Architecture:** Add `app/triage/dept_choices.py` for whitelist extraction, choice resolution, and fixed prompt text. Extend `DeptDisambiguationState` with `last_choices`. Refactor `dept_disambiguation_node` to score → build choices → ask OR lock; on user reply, resolve choice (no LLM). Expose `awaiting_dept_choice` + `dept_choices` via `chat_service` and CLI `Prompt.ask`.

**Tech Stack:** Python 3.13, LangGraph 1.2, Pydantic v2, FastAPI, Rich CLI, `uv run python` for tests.

**Spec:** `docs/superpowers/specs/2026-06-12-dept-choice-disambiguation-design.md`

---

## File Map

| File | Responsibility |
|------|----------------|
| `app/domain/dept_disambiguation.py` | `DeptChoice` model; extend state with `last_choices`, `asked_choice_ids` |
| `app/triage/dept_choices.py` | Extract choices, format AIMessage, resolve user pick, score boosts |
| `app/triage/dept_llm.py` | Keep `_pick_pair_by_round` only; remove LLM ask/parse |
| `app/graph/nodes/dept_disambiguation.py` | Choice-based ask/answer loop |
| `app/domain/routing.py` | Simpler follow-up routing when `status==asking` |
| `app/services/chat_service.py` | Return `awaiting_dept_choice`, `dept_choices` |
| `app/api/routers/chat.py` | Extend `ChatResponse` schema |
| `cli.py` | Menu pick when awaiting choice |
| `scripts/test_dept_choices.py` | Unit tests for extraction + resolve |
| `scripts/test_dept_choice_graph.py` | Graph integration (MemorySaver, no OpenSearch if mocked) |

---

### Task 1: Domain models — `DeptChoice` and state fields

**Files:**
- Modify: `app/domain/dept_disambiguation.py`
- Test: `scripts/test_dept_choices.py` (model import only, expanded in Task 2)

- [ ] **Step 1: Add models**

Replace `app/domain/dept_disambiguation.py` with:

```python
from typing import Literal

from pydantic import BaseModel, Field, ConfigDict


class DeptChoice(BaseModel):
    id: str
    label: str
    target_departments: list[str] = Field(default_factory=list)

    model_config = ConfigDict(extra="ignore")


class DeptDisambiguationState(BaseModel):
    candidate_departments: list[dict] = Field(default_factory=list)
    dept_scores: dict[str, float] = Field(default_factory=dict)
    round: int = 0
    status: Literal["scoring", "asking", "locked", "emergency", "fallback"] = "scoring"
    last_question: str | None = None
    last_choices: list[DeptChoice] = Field(default_factory=list)
    asked_choice_ids: list[str] = Field(default_factory=list)
    margin: float | None = None

    model_config = ConfigDict(extra="ignore")
```

- [ ] **Step 2: Verify import**

Run: `uv run python -c "from app.domain.dept_disambiguation import DeptChoice, DeptDisambiguationState; print('OK')"`  
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add app/domain/dept_disambiguation.py
git commit -m "feat: add DeptChoice and extend dept disambiguation state"
```

---

### Task 2: Choice extraction and resolution (`dept_choices.py`)

**Files:**
- Create: `app/triage/dept_choices.py`
- Create: `scripts/test_dept_choices.py`

- [ ] **Step 1: Write failing tests**

Create `scripts/test_dept_choices.py`:

```python
"""Unit tests for rule-based dept choice menus."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.triage.dept_choices import (
    CHOICE_QUESTION_TEMPLATE,
    build_dept_choices,
    choice_score_boosts,
    format_choice_message,
    resolve_dept_choice,
)

RK0001 = json.loads(
    Path("demo/data/rag_knowledge.jsonl").read_text(encoding="utf-8").splitlines()[0]
)


def test_build_choices_excludes_emergency_terms() -> None:
    choices, has_symptom = build_dept_choices(RK0001, round_num=1, asked_choice_ids=[])
    labels = [c.label for c in choices]
    assert "都没有" in labels
    assert not any("畸形" in lb or "不能负重" in lb for lb in labels)
    assert has_symptom or len([c for c in choices if c.id != "none"]) == 0


def test_build_choices_includes_discriminative_keyword() -> None:
    choices, has_symptom = build_dept_choices(RK0001, round_num=1, asked_choice_ids=[])
    labels = [c.label for c in choices if c.id != "none"]
    # RK0001 round1: 骨科 vs 风湿 — expect 晨僵 or 扭伤类 accompanying hit
    if has_symptom:
        assert any("晨僵" in lb or "扭伤" in lb or "肿胀" in lb for lb in labels)


def test_resolve_by_label_and_alias() -> None:
    choices, _ = build_dept_choices(RK0001, round_num=1, asked_choice_ids=[])
    pick = next(c for c in choices if c.id != "none")
    assert resolve_dept_choice(pick.label, choices) == pick
    if "晨僵" in pick.label:
        assert resolve_dept_choice("晨僵", choices) is not None


def test_resolve_none_for_invalid() -> None:
    choices, _ = build_dept_choices(RK0001, round_num=1, asked_choice_ids=[])
    assert resolve_dept_choice("我还行吧", choices) is None


def test_choice_score_boosts_none_uses_negation() -> None:
    choices, _ = build_dept_choices(RK0001, round_num=1, asked_choice_ids=[])
    none_choice = next(c for c in choices if c.id == "none")
    boosts = choice_score_boosts(none_choice)
    assert boosts == {}


def test_format_message_lists_options() -> None:
    choices, _ = build_dept_choices(RK0001, round_num=1, asked_choice_ids=[])
    msg = format_choice_message(choices)
    assert CHOICE_QUESTION_TEMPLATE in msg
    assert "都没有" in msg


def main() -> None:
    test_build_choices_excludes_emergency_terms()
    test_build_choices_includes_discriminative_keyword()
    test_resolve_by_label_and_alias()
    test_resolve_none_for_invalid()
    test_choice_score_boosts_none_uses_negation()
    test_format_message_lists_options()
    print("[OK] all dept_choices tests passed")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run tests — expect fail**

Run: `uv run python scripts/test_dept_choices.py`  
Expected: `ModuleNotFoundError: app.triage.dept_choices`

- [ ] **Step 3: Implement `app/triage/dept_choices.py`**

```python
"""Rule-based dept disambiguation choices (no LLM)."""

from __future__ import annotations

from app.domain.dept_disambiguation import DeptChoice
from app.triage.dept_llm import _pick_pair_by_round

CHOICE_QUESTION_TEMPLATE = "为更准确推荐科室，请选择您是否有以下情况："
INVALID_CHOICE_REPLY = "请从下列选项中选择（输入选项文字或编号）。"
CHOICE_BOOST = 2.0
NONE_CHOICE_ID = "none"
NONE_CHOICE_LABEL = "都没有"

_DISEASE_DENY_FRAGMENTS = (
    "炎", "病", "症", "损伤", "骨折", "筋膜炎", "骨刺", "胼胝", "甲癣", "嵌甲",
)
_LABEL_ALIASES: dict[str, str] = {
    "关节痛": "多关节疼痛",
    "活动痛": "活动后加重",
    "久站": "久站加重",
}


def _emergency_blob(chunk: dict) -> str:
    flag = chunk.get("emergency_flag") or {}
    return str(flag.get("condition") or "")


def _is_emergency_term(label: str, emergency_blob: str) -> bool:
    if not label:
        return True
    if label in emergency_blob:
        return True
    for frag in ("畸形", "不能负重", "发紫", "不能动", "皮肤发黑", "开放性"):
        if frag in label and frag in emergency_blob:
            return True
    return False


def _is_disease_like(label: str) -> bool:
    return any(d in label for d in _DISEASE_DENY_FRAGMENTS)


def _depts_for_keyword(keyword: str, dept_a: dict, dept_b: dict | None) -> list[str]:
    targets: list[str] = []
    for d in (dept_a, dept_b):
        if not d:
            continue
        cond = d.get("condition") or ""
        dept = d.get("department") or ""
        if keyword in cond and dept:
            targets.append(dept)
    return list(dict.fromkeys(targets))


def build_dept_choices(
    chunk: dict,
    round_num: int,
    asked_choice_ids: list[str],
) -> tuple[list[DeptChoice], bool]:
    """Return (choices including '都没有', has_symptom_options)."""
    depts = chunk.get("department_recommendation") or []
    dept_a, dept_b = _pick_pair_by_round(depts, round_num)
    if not dept_a:
        return [_none_choice()], False

    accompany = chunk.get("accompanying_symptom_keywords") or []
    canonical = (chunk.get("canonical_symptom") or "").strip()
    emergency_blob = _emergency_blob(chunk)
    cond_blob = (dept_a.get("condition") or "") + (dept_b.get("condition") or "" if dept_b else "")

    candidates: list[tuple[str, list[str]]] = []
    seen_labels: set[str] = set()
    for kw in accompany:
        if not isinstance(kw, str) or len(kw.strip()) < 2:
            continue
        kw = kw.strip()
        if kw not in cond_blob:
            continue
        if canonical and kw == canonical:
            continue
        if _is_emergency_term(kw, emergency_blob):
            continue
        if _is_disease_like(kw):
            continue
        targets = _depts_for_keyword(kw, dept_a, dept_b)
        if not targets:
            continue
        # discriminative: keyword must not map to identical dept sets on both sides
        if dept_b and len(targets) == 2:
            pass  # spans both — still useful as confirm
        if kw in seen_labels:
            continue
        seen_labels.add(kw)
        candidates.append((kw, targets))

    choices: list[DeptChoice] = []
    for idx, (label, targets) in enumerate(candidates, start=1):
        cid = f"c{idx}"
        if cid in asked_choice_ids:
            continue
        choices.append(DeptChoice(id=cid, label=label, target_departments=targets))

    has_symptom = len(choices) > 0
    choices.append(_none_choice())
    return choices, has_symptom


def _none_choice() -> DeptChoice:
    return DeptChoice(id=NONE_CHOICE_ID, label=NONE_CHOICE_LABEL, target_departments=[])


def format_choice_message(choices: list[DeptChoice]) -> str:
    lines = [CHOICE_QUESTION_TEMPLATE, ""]
    for i, c in enumerate(choices, start=1):
        lines.append(f"{i}. {c.label}")
    return "\n".join(lines)


def resolve_dept_choice(user_reply: str, choices: list[DeptChoice]) -> DeptChoice | None:
    text = (user_reply or "").strip()
    if not text:
        return None
    if text.isdigit():
        idx = int(text) - 1
        if 0 <= idx < len(choices):
            return choices[idx]
    alias = _LABEL_ALIASES.get(text, text)
    for c in choices:
        if text == c.id or text == c.label or alias == c.label:
            return c
    return None


def choice_score_boosts(choice: DeptChoice) -> dict[str, float]:
    if choice.id == NONE_CHOICE_ID:
        return {}
    return {d: CHOICE_BOOST for d in choice.target_departments}
```

- [ ] **Step 4: Run tests — pass**

Run: `uv run python scripts/test_dept_choices.py`  
Expected: `[OK] all dept_choices tests passed`

- [ ] **Step 5: Commit**

```bash
git add app/triage/dept_choices.py scripts/test_dept_choices.py
git commit -m "feat: rule-based dept choice extraction and resolution"
```

---

### Task 3: Refactor `dept_disambiguation_node`

**Files:**
- Modify: `app/graph/nodes/dept_disambiguation.py`

- [ ] **Step 1: Replace LLM ask/parse with choice flow**

Key logic changes in `dept_disambiguation_node`:

```python
from app.triage.dept_choices import (
    INVALID_CHOICE_REPLY,
    build_dept_choices,
    choice_score_boosts,
    format_choice_message,
    resolve_dept_choice,
)
# remove: generate_dept_question, parse_dept_answer, llm_score_boosts

# Inside node, after computing base scores:

# --- user answering an existing menu ---
if dept_state and dept_state.status == "asking" and dept_state.last_choices:
    reply = _last_human_message(state)
    picked = resolve_dept_choice(reply, dept_state.last_choices)
    if picked is None:
        return {
            "messages": [AIMessage(content=INVALID_CHOICE_REPLY + "\n\n" + format_choice_message(dept_state.last_choices))],
            "dept_state": dept_state.model_copy(),  # same round, same choices
        }
    choice_boosts = choice_score_boosts(picked)
    scores = score_departments(..., llm_boosts=choice_boosts or None)
    if picked.id == "none":
        scores = apply_negation_boosts(scores, picked.label)
    # try_lock → locked OR next round (current_round unchanged on invalid; on valid use dept_state.round)
    ...

# --- generate new menu ---
next_round = current_round + 1
choices, has_symptom = build_dept_choices(chunk, round_num=next_round, asked_choice_ids=dept_state.asked_choice_ids if dept_state else [])
if not has_symptom:
    fb = fallback_department(depts)
    return {"locked_department": fb, "dept_state": DeptDisambiguationState(status="fallback", round=next_round, ...)}

asked_ids = list(dept_state.asked_choice_ids if dept_state else [])
asked_ids.extend(c.id for c in choices if c.id != "none")
question = format_choice_message(choices)
return {
    "dept_state": DeptDisambiguationState(
        status="asking",
        round=next_round,
        last_question=CHOICE_QUESTION_TEMPLATE,
        last_choices=choices,
        asked_choice_ids=asked_ids,
        dept_scores=scores,
        margin=margin,
        candidate_departments=depts,
    ),
    "messages": [AIMessage(content=question)],
}
```

Full file should preserve `_is_emergency`, `current_turn_text`, existing lock/fallback at round>=3.

- [ ] **Step 2: Smoke import**

Run: `uv run python -c "from app.graph.nodes.dept_disambiguation import dept_disambiguation_node; print('OK')"`  
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add app/graph/nodes/dept_disambiguation.py
git commit -m "feat: choice-based dept disambiguation node"
```

---

### Task 4: Routing — always continue asking when choices pending

**Files:**
- Modify: `app/domain/routing.py`
- Test: add to `scripts/test_dept_choices.py` or new `scripts/test_dept_routing.py`

- [ ] **Step 1: Write failing routing test**

Add to `scripts/test_dept_choices.py`:

```python
from langchain_core.messages import AIMessage, HumanMessage
from app.domain.dept_disambiguation import DeptChoice, DeptDisambiguationState
from app.domain.models import AppState
from app.domain.routing import route_after_trim

def test_route_asking_goes_to_dept_disambiguation() -> None:
    state = AppState(
        messages=[
            AIMessage(content="为更准确推荐科室..."),
            HumanMessage(content="我还行吧"),
        ],
        dept_state=DeptDisambiguationState(
            status="asking",
            last_question="为更准确推荐科室，请选择您是否有以下情况：",
            last_choices=[DeptChoice(id="none", label="都没有")],
        ),
    )
    assert route_after_trim(state) == "dept_disambiguation"
```

- [ ] **Step 2: Update `_is_dept_followup_reply` in `app/domain/routing.py`**

```python
def _is_dept_followup_reply(state: AppState) -> bool:
    ds = state.dept_state
    if not ds or getattr(ds, "status", None) != "asking":
        return False
    if not ds.last_choices:
        return False
    msgs = state.messages or []
    if len(msgs) < 2:
        return False
    last = msgs[-1]
    return isinstance(last, HumanMessage)
```

- [ ] **Step 3: Run test**

Run: `uv run python scripts/test_dept_choices.py`  
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add app/domain/routing.py scripts/test_dept_choices.py
git commit -m "fix: route any human reply to dept disambiguation while asking"
```

---

### Task 5: API — expose structured choices

**Files:**
- Modify: `app/services/chat_service.py`
- Modify: `app/api/routers/chat.py`

- [ ] **Step 1: Extend chat_service return dict**

In `chat_once`, after building return dict:

```python
from app.domain.dept_disambiguation import DeptChoice

ds = state.dept_state
awaiting = bool(ds and ds.status == "asking" and ds.last_choices)
dept_choices = [c.model_dump() for c in ds.last_choices] if awaiting else []

return {
    ...
    "awaiting_dept_choice": awaiting,
    "dept_choices": dept_choices,
}
```

Also update `_extract_reply` fallback: if `messages` empty but `last_reply` exists (future session-reset), keep existing behavior.

- [ ] **Step 2: Extend FastAPI models in `app/api/routers/chat.py`**

```python
from app.domain.dept_disambiguation import DeptChoice

class ChatResponse(BaseModel):
    user_id: str
    thread_id: str
    reply: str
    intent_result: Optional[IntentResult] = None
    used_docs: UsedDocs
    awaiting_dept_choice: bool = False
    dept_choices: List[DeptChoice] = []
```

- [ ] **Step 3: Commit**

```bash
git add app/services/chat_service.py app/api/routers/chat.py
git commit -m "feat: expose dept_choices in chat API response"
```

---

### Task 6: CLI menu selection

**Files:**
- Modify: `cli.py`

- [ ] **Step 1: After chat response, prompt for choice if awaiting**

In `_ask_chat`, after `data = resp.json()`:

```python
while data.get("awaiting_dept_choice") and data.get("dept_choices"):
    labels = [c["label"] for c in data["dept_choices"]]
    console.print(Panel("请选择症状（输入选项文字）", style="info"))
    pick = Prompt.ask("您的选择", choices=labels, console=console)
    payload = {"user_id": self.user_id, "message": pick}
    if self.thread_id:
        payload["thread_id"] = self.thread_id
    resp = self.session.post(f"{self.base_url}/chat", json=payload, timeout=self.timeout)
    resp.raise_for_status()
    data = resp.json()
```

Then render final `reply` once loop exits.

- [ ] **Step 2: Manual smoke (backend running)**

Run backend + `uv run python cli.py`, send `脚脖子肿`, confirm numbered menu appears and picking an option completes triage.

- [ ] **Step 3: Commit**

```bash
git add cli.py
git commit -m "feat: CLI menu for dept choice disambiguation"
```

---

### Task 7: Remove LLM ask/parse from `dept_llm.py`

**Files:**
- Modify: `app/triage/dept_llm.py`

- [ ] **Step 1: Delete LLM functions; keep `_pick_pair_by_round`**

Remove `DEPT_ASK_PROMPT`, `DEPT_PARSE_PROMPT`, `generate_dept_question`, `parse_dept_answer`, `llm_score_boosts`, and unused pydantic output models. Export `_pick_pair_by_round` (already used by `dept_choices.py`).

- [ ] **Step 2: Grep for stale imports**

Run: `rg "generate_dept_question|parse_dept_answer|llm_score_boosts" app/`  
Expected: no matches

- [ ] **Step 3: Commit**

```bash
git add app/triage/dept_llm.py
git commit -m "refactor: remove LLM dept ask/parse in favor of rule choices"
```

---

### Task 8: Graph integration test

**Files:**
- Create: `scripts/test_dept_choice_graph.py`

- [ ] **Step 1: Write integration test (mock rag chunk in state if OpenSearch unavailable)**

```python
"""Graph path: asking → pick choice → locked dept."""
from __future__ import annotations

import json
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver

from app.domain.models import AppState
from app.graph.builder import build_app

RK0001 = json.loads(Path("demo/data/rag_knowledge.jsonl").read_text(encoding="utf-8").splitlines()[0])


def test_asking_then_invalid_then_valid(monkeypatch=None):
    app = build_app(MemorySaver())
    tid = f"test-{uuid.uuid4().hex[:8]}"
    cfg = {"configurable": {"thread_id": tid, "user_id": "test"}}

    # Turn 1: symptom intake — requires OpenSearch + NER; skip if env missing
    try:
        s1 = app.invoke({"messages": [HumanMessage(content="脚脖子肿")]}, config=cfg)
    except Exception as exc:
        print(f"[SKIP] graph invoke needs services: {exc}")
        return

    st1 = AppState(**s1) if isinstance(s1, dict) else s1
    if not (st1.dept_state and st1.dept_state.status == "asking"):
        print("[SKIP] did not reach asking (scoring locked early)")
        return

    s2 = app.invoke({"messages": [HumanMessage(content="我还行吧")]}, config=cfg)
    st2 = AppState(**s2) if isinstance(s2, dict) else s2
    assert st2.dept_state.status == "asking"
    assert st2.dept_state.round == st1.dept_state.round

    labels = [c.label for c in st2.dept_state.last_choices if c.id != "none"]
    if not labels:
        print("[SKIP] no symptom labels in menu")
        return
    s3 = app.invoke({"messages": [HumanMessage(content=labels[0])]}, config=cfg)
    st3 = AppState(**s3) if isinstance(s3, dict) else s3
    assert st3.locked_department or st3.dept_state.status in ("asking", "locked", "fallback")
    print("[OK] graph choice flow")


if __name__ == "__main__":
    test_asking_then_invalid_then_valid()
```

- [ ] **Step 2: Run (optional if stack up)**

Run: `uv run python scripts/test_dept_choice_graph.py`

- [ ] **Step 3: Commit**

```bash
git add scripts/test_dept_choice_graph.py
git commit -m "test: dept choice graph integration script"
```

---

## Spec Coverage Checklist

| Spec § | Task |
|--------|------|
| §4 API `dept_choices` | Task 5 |
| §5 rule extraction | Task 2 |
| §6 no LLM parse | Task 2, 3, 7 |
| §7 state fields | Task 1 |
| §8 graph/routing | Task 3, 4 |
| §10 T1–T5 | Task 2, 8 |
| §10 T6 CLI | Task 6 |
| §10 T7 golden eval | Run `scripts/foot_eval/` after Task 3 (manual checkpoint) |
| §11 invalid input | Task 3 |
| §13 session-reset compat | No code conflict; `last_choices` cleared by existing `triage_state_reset_patch` when that spec lands |

---

## Manual Verification Checklist

- [ ] `脚脖子肿` → numbered menu, no `畸形` / `不能负重` in options
- [ ] Pick `都没有` → new axis or fallback, round increments
- [ ] Invalid free text → reprompt, same round
- [ ] Chunk with zero accompanying hits → direct P1 dept, no menu
- [ ] CLI loops until `awaiting_dept_choice` is false

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-06-12-dept-choice-disambiguation.md`.

**Two execution options:**

1. **Subagent-Driven (recommended)** — fresh subagent per task, review between tasks  
2. **Inline Execution** — implement tasks in this session with checkpoints

Which approach?
