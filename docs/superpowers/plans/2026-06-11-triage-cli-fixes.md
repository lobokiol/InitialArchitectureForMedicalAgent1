# Triage CLI Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix P0/P1 issues found in CLI manual testing: current-turn isolation, RAG recall stability, emergency/disease templated replies, negation-aware dept disambiguation, users API without Redis, and CLI UX.

**Architecture:** Keep LangGraph topology unchanged. Replace `_user_text` history aggregation with `_current_turn_text`. Enhance RAG with full query + alliance rerank. Extend dept LLM/rules for negation and round-based axes. Template all final answers for emergency/locked/disease paths. Memory fallback for users API.

**Tech Stack:** Python 3.13, LangGraph, Pydantic v2, OpenSearch, DashScope LLM, `uv run` for tests.

**Spec:** `docs/superpowers/specs/2026-06-11-triage-cli-fixes-design.md`

---

## File Map

| File | Responsibility |
|------|----------------|
| `app/graph/nodes/dept_disambiguation.py` | `_current_turn_text`, pass `round` to question gen |
| `app/graph/nodes/rag_symptom_recall.py` | Full query, k=3, rerank |
| `app/infra/opensearch_rag.py` | `rerank_by_alliance()` |
| `app/graph/nodes/answer.py` | Emergency/disease templates, new-intake prefix |
| `app/triage/dept_llm.py` | Negation rules, round-aware question |
| `app/triage/dept_scoring.py` | `apply_negation_boosts()` |
| `app/api/routers/users.py` | In-memory fallback |
| `cli.py` | `exit` alias, reply hints |
| `scripts/test_turn_text.py` | Unit: current turn text |
| `scripts/test_dept_scoring.py` | Negation boost test |
| `scripts/test_chat_api.py` | Emergency + disease E2E |

---

### Task 1: Current-turn text helper

**Files:**
- Create: `app/triage/turn_text.py`
- Modify: `app/graph/nodes/dept_disambiguation.py`
- Test: `scripts/test_turn_text.py`

- [ ] **Step 1: Write failing test**

```python
# scripts/test_turn_text.py
from langchain_core.messages import HumanMessage, AIMessage
from app.domain.models import AppState
from app.domain.slot_table import TriageSlotTable
from app.ner.models import EntityExtractResult
from app.triage.turn_text import current_turn_text

def test_uses_ner_query_not_history():
    state = AppState(
        messages=[
            HumanMessage(content="脚脖子肿，不能动，皮发紫"),
            AIMessage(content="急诊建议..."),
            HumanMessage(content="脚脖子肿"),
        ],
        ner_result=EntityExtractResult(query="脚脖子肿", primary_symptom="脚脖子肿"),
        slot_table=TriageSlotTable(primary_symptom="脚脖子肿"),
    )
    text = current_turn_text(state)
    assert "不能动" not in text
    assert "脚脖子肿" in text
```

- [ ] **Step 2: Run test — expect fail**

Run: `uv run python scripts/test_turn_text.py`  
Expected: `ModuleNotFoundError`

- [ ] **Step 3: Implement**

```python
# app/triage/turn_text.py
def current_turn_text(state: AppState) -> str:
    parts: list[str] = []
    if state.ner_result and state.ner_result.query:
        parts.append(state.ner_result.query.strip())
    elif state.messages:
        last = state.messages[-1]
        if isinstance(last, HumanMessage) and isinstance(last.content, str):
            parts.append(last.content.strip())
    table = state.slot_table
    if table:
        if table.trigger:
            parts.append(table.trigger)
        if table.emergency:
            parts.append(table.emergency)
    return " ".join(p for p in parts if p)
```

Replace `_user_text` usage in `dept_disambiguation.py` with `current_turn_text`.

- [ ] **Step 4: Run test — pass**

Run: `uv run python scripts/test_turn_text.py`

---

### Task 2: RAG full query + alliance rerank

**Files:**
- Modify: `app/infra/opensearch_rag.py`
- Modify: `app/graph/nodes/rag_symptom_recall.py`
- Modify: `app/graph/nodes/dept_disambiguation.py` (pass `state` to recall if needed)

- [ ] **Step 1: Add rerank helper**

```python
# app/infra/opensearch_rag.py
def rerank_by_alliance(hits: list[dict], query: str) -> list[dict]:
    q = query.strip()
    if not q or not hits:
        return hits
    def score(h: dict) -> tuple:
        alliance = h.get("alliance") or []
        exact = any(isinstance(a, str) and a in q for a in alliance)
        return (0 if exact else 1, -(h.get("_score") or 0))
    return sorted(hits, key=score)
```

- [ ] **Step 2: Update recall node**

```python
# rag_symptom_recall_node
ner = state.ner_result
table = state.slot_table
q = (ner.query if ner and ner.query else "") or (table.primary_symptom if table else "")
if table and table.companion_symptoms:
    q = f"{q} {' '.join(table.companion_symptoms)}".strip()
hits = search_rag_knowledge(q, k=3)
hits = rerank_by_alliance(hits, q)
chunk = hits[0] if hits else None
```

- [ ] **Step 3: Smoke**

Run: `uv run python -c "from app.infra.opensearch_rag import search_rag_knowledge, rerank_by_alliance; hs=search_rag_knowledge('脚脖子肿，不能动，皮发紫', k=3); print(rerank_by_alliance(hs,'脚脖子肿，不能动，皮发紫')[0].get('id'))"`  
Expected: `RK0001`

---

### Task 3: Emergency answer template

**Files:**
- Modify: `app/graph/nodes/answer.py`

- [ ] **Step 1: Replace emergency branch**

```python
if dept == "急诊":
    flag = chunk.get("emergency_flag") or {}
    detail = flag.get("suggestion") or flag.get("condition") or "请尽快就医。"
    full_content = (
        f"根据您描述的情况（{canonical}），建议尽快就诊：**急诊**。\n{detail}"
    )
```

- [ ] **Step 2: E2E prep** — add to `test_chat_api.py` in Task 9

---

### Task 4: Negation boosts + round-aware questions

**Files:**
- Modify: `app/triage/dept_scoring.py`
- Modify: `app/triage/dept_llm.py`
- Modify: `app/graph/nodes/dept_disambiguation.py`

- [ ] **Step 1: Failing test**

```python
# append to scripts/test_dept_scoring.py
from app.triage.dept_scoring import apply_negation_boosts

def test_negation_boosts_rheum():
    scores = {"骨科": 5.0, "风湿免疫科": 3.0, "血管外科": 2.0}
    out = apply_negation_boosts(scores, "都没有")
    assert out["风湿免疫科"] > out["骨科"]
```

- [ ] **Step 2: Implement `apply_negation_boosts`**

```python
_NEGATION = ("都没有", "没有", "不是", "无", "否")
_AFFIRM_TRAUMA = ("摔", "扭", "外伤", "有")

def apply_negation_boosts(scores: dict[str, float], reply: str) -> dict[str, float]:
    s = dict(scores)
    r = reply.strip()
    if any(n in r for n in _NEGATION):
        s["骨科"] = s.get("骨科", 0) - 2.0
        s["风湿免疫科"] = s.get("风湿免疫科", 0) + 2.0
        s["血管外科"] = s.get("血管外科", 0) + 1.0
    if any(a in r for a in _AFFIRM_TRAUMA):
        s["骨科"] = s.get("骨科", 0) + 2.0
    return s
```

- [ ] **Step 3: Wire in dept_disambiguation**

After `parse_dept_answer` / before `score_departments`, if `reply`:
```python
from app.triage.dept_scoring import apply_negation_boosts
# after scores computed:
if reply:
    scores = apply_negation_boosts(scores, reply)
```

Also call `apply_negation_boosts` on rule path when reply matches negation without LLM.

- [ ] **Step 4: Round-aware `generate_dept_question`**

```python
def _pick_pair_by_round(depts: list[dict], round: int) -> tuple[dict, dict | None]:
    ordered = sorted(depts, key=lambda d: int(d.get("priority") or 99))
    if round <= 1 and len(ordered) >= 2:
        return ordered[0], ordered[1]
    if len(ordered) >= 3:
        return ordered[1], ordered[2]
    ...
```

Pass `round=current_round + 1` and `last_question=dept_state.last_question` into prompt: "不要重复上一问"。

- [ ] **Step 5: Run** `uv run python scripts/test_dept_scoring.py`

---

### Task 5: Disease chain answer template

**Files:**
- Modify: `app/graph/nodes/answer.py`

- [ ] **Step 1: Add branch before LLM**

```python
ddr = state.disease_dept_result
if ddr and ddr.departments:
    names = "、".join(ddr.diseases) if ddr.diseases else "您的描述"
    dept_str = "、".join(ddr.departments[:3])
    full_content = f"根据您提到的「{names}」，建议就诊科室：**{dept_str}**。"
    return {"messages": [AIMessage(content=full_content)]}
```

- [ ] **Step 2: Verify** `test_chat_api` 纯疾病 case reply contains 科室 keyword (消化/内科/胃肠 — whatever lookup returns)

---

### Task 6: Users API memory fallback

**Files:**
- Modify: `app/api/routers/users.py`

- [ ] **Step 1: Implement**

```python
_memory_users: dict[str, dict] = {}

def _store_user(user_id: str, name: str | None) -> UserInfo:
    key = f"user:{user_id}:meta"
    now = datetime.utcnow().isoformat()
    if redis_client is not None:
        created_at = redis_client.hget(key, "created_at") or now
        redis_client.hset(key, mapping={...})
        return UserInfo(...)
    rec = _memory_users.get(user_id)
    if not rec:
        rec = {"user_id": user_id, "name": name or "", "created_at": now}
        _memory_users[user_id] = rec
    else:
        if name:
            rec["name"] = name
    return UserInfo(**rec)
```

Apply same pattern to `get_user`.

- [ ] **Step 2: Smoke**

```powershell
uv run python -c "import requests; r=requests.post('http://127.0.0.1:8000/users',json={'user_id':'t1','name':'x'}); print(r.status_code, r.json())"
```
Expected: `200`

---

### Task 7: CLI exit alias + display

**Files:**
- Modify: `cli.py`

- [ ] **Step 1: Treat bare exit/quit**

```python
if raw.lower() in ("exit", "quit", "/exit", "/quit"):
    ...
```

- [ ] **Step 2: Highlight dept in reply**

After printing assistant panel, if `急诊` in reply or `建议就诊科室` in reply:
```python
console.print(Panel("已锁定导诊科室，换症状请 /new", style="info"))
```

---

### Task 8: New-intake prefix in answer

**Files:**
- Modify: `app/graph/nodes/answer.py`

- [ ] **Step 1: Helper**

```python
def _new_intake_prefix(state: AppState) -> str:
    msgs = state.messages or []
    if len(msgs) < 3:
        return ""
    prior_humans = [m.content for m in msgs[:-1] if isinstance(m, HumanMessage)]
    cur = state.ner_result.query if state.ner_result else ""
    if prior_humans and cur and cur.strip() != str(prior_humans[-1]).strip():
        return "（已按您本轮新描述重新评估）"
    return ""
```

Prepend to locked/disease template strings when non-empty.

---

### Task 9: Automated tests

**Files:**
- Create: `scripts/test_turn_text.py`
- Modify: `scripts/test_dept_scoring.py`
- Modify: `scripts/test_chat_api.py`

- [ ] **Step 1: Add chat cases**

```python
{
    "name": "急诊踝肿",
    "message": "脚脖子肿，不能动，皮发紫",
    "expect_route": "symptom",
    "expect_dept": "急诊",
},
```

- [ ] **Step 2: Run full suite**

```powershell
$env:PYTHONIOENCODING="utf-8"
uv run python scripts/test_turn_text.py
uv run python scripts/test_slot_gate.py
uv run python scripts/test_dept_scoring.py
uv run python scripts/test_chat_api.py
```

Expected: all pass (API + OpenSearch running)

- [ ] **Step 3: Update spec status**

In `2026-06-11-triage-cli-fixes-design.md` set status to `已实现` when done.

---

## Plan Self-Review

| Spec § | Task |
|--------|------|
| §5.1 当前轮文本 | Task 1 |
| §5.2 RAG | Task 2 |
| §5.3 急诊模板 | Task 3 |
| §5.4 否定/round | Task 4 |
| §5.5 疾病链 | Task 5 |
| §5.6 users | Task 6 |
| §5.7 CLI | Task 7 |
| §5.7 换题前缀 | Task 8 |
| §5.8 测试 | Task 9 |

No placeholders. Commands use `uv run` per project convention.

---

## Execution Handoff

Plan saved to `docs/superpowers/plans/2026-06-11-triage-cli-fixes.md`.

**Two execution options:**

1. **Subagent-Driven (recommended)** — fresh subagent per task, review between tasks  
2. **Inline Execution** — implement all tasks in this session with checkpoints

Which approach?
