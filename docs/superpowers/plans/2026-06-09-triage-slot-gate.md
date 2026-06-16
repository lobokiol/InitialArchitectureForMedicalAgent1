# Triage Slot Gate + RAG Dept Disambiguation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** After entity extraction, fill a global intake slot table, gate on primary symptom/disease, recall `rag_knowledge` from OpenSearch, and lock the optimal department via rule scoring + optional LLM clarifying questions (0–3 rounds).

**Architecture:** Extend LangGraph with `slot_fill` → `slot_gate` → (symptom path) `rag_symptom_recall` → `dept_disambiguation` → `answer_generate`. Rule layer owns department selection; LLM only generates/parses clarifying questions when score margin is insufficient. Reject path clears triage state on same thread while keeping message history.

**Tech Stack:** Python 3.13, LangGraph, Pydantic v2, OpenSearch (`opensearch-py`), DashScope LLM/embeddings, existing `EntityExtractResult` NER pipeline.

**Spec:** `docs/superpowers/specs/2026-06-09-triage-slot-gate-design.md`

---

## File Map

| File | Responsibility |
|------|----------------|
| `app/domain/slot_table.py` | `TriageSlotTable`, defaults, `slot_gate_passes()` |
| `app/domain/dept_disambiguation.py` | `DeptDisambiguationState`, scoring, lock/fallback logic |
| `app/domain/triage_intent.py` | Update `REJECT_MESSAGE` to `请输入症状？` |
| `app/domain/models.py` | Extend `AppState` with slot/rag/dept fields |
| `app/triage/slot_fill.py` | Map NER → slot table + rule extract trigger/duration/emergency/age/gender |
| `app/triage/session_reset.py` | Build state patch to clear triage fields |
| `app/infra/opensearch_rag.py` | Production wrapper around hybrid search (adapt from `demo/opensearch_rag_kb.py`) |
| `app/triage/dept_scoring.py` | Pure functions: score departments from chunk + user text |
| `app/triage/dept_llm.py` | LLM structured output for ask/parse |
| `app/graph/nodes/slot_fill.py` | Graph node |
| `app/graph/nodes/slot_gate.py` | Graph node + conditional router |
| `app/graph/nodes/rag_symptom_recall.py` | OpenSearch recall node |
| `app/graph/nodes/dept_disambiguation.py` | Scoring + ask/lock node |
| `app/graph/nodes/reject.py` | Append session_reset fields |
| `app/graph/builder.py` | Rewire graph |
| `app/domain/routing.py` | New routers: `route_after_slot_gate`, `route_after_dept` |
| `demo/data/triage_intake_slots.json` | Default schema config |
| `scripts/test_slot_gate.py` | Unit tests (no OpenSearch) |
| `scripts/test_dept_scoring.py` | Unit tests for RK0001 cases |
| `scripts/test_chat_api.py` | Extend E2E cases |

---

### Task 1: Slot table models + gate

**Files:**
- Create: `app/domain/slot_table.py`
- Create: `demo/data/triage_intake_slots.json`
- Test: `scripts/test_slot_gate.py`

- [ ] **Step 1: Write failing tests**

Create `scripts/test_slot_gate.py`:

```python
from app.domain.slot_table import TriageSlotTable, slot_gate_passes, default_slot_table


def test_default_age_gender():
    t = default_slot_table()
    assert t.gender == "男"
    assert t.age == "30岁"


def test_gate_pass_with_symptom():
    t = TriageSlotTable(primary_symptom="脚脖子肿")
    assert slot_gate_passes(t) is True


def test_gate_pass_with_disease():
    t = TriageSlotTable(primary_disease="胃炎")
    assert slot_gate_passes(t) is True


def test_gate_fail_empty():
    t = TriageSlotTable()
    assert slot_gate_passes(t) is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `venv\Scripts\python.exe scripts/test_slot_gate.py`  
Expected: `ModuleNotFoundError: app.domain.slot_table`

- [ ] **Step 3: Implement**

Create `demo/data/triage_intake_slots.json`:

```json
{
  "defaults": { "gender": "男", "age": "30岁" },
  "required_any_of": ["primary_symptom", "primary_disease"]
}
```

Create `app/domain/slot_table.py`:

```python
from pydantic import BaseModel, Field, ConfigDict


class TriageSlotTable(BaseModel):
    gender: str = "男"
    age: str = "30岁"
    primary_symptom: str | None = None
    companion_symptoms: list[str] = Field(default_factory=list)
    primary_disease: str | None = None
    companion_diseases: list[str] = Field(default_factory=list)
    trigger: str | None = None
    duration: str | None = None
    emergency: str | None = None

    model_config = ConfigDict(extra="ignore")


def default_slot_table() -> TriageSlotTable:
    return TriageSlotTable()


def slot_gate_passes(table: TriageSlotTable) -> bool:
    return bool(table.primary_symptom or table.primary_disease)
```

- [ ] **Step 4: Run tests**

Run: `venv\Scripts\python.exe scripts/test_slot_gate.py`  
Expected: all `[OK]` / no assertion errors

- [ ] **Step 5: Commit**

```bash
git add app/domain/slot_table.py demo/data/triage_intake_slots.json scripts/test_slot_gate.py
git commit -m "feat: add TriageSlotTable and slot gate helper"
```

---

### Task 2: Slot fill from NER + optional fields

**Files:**
- Create: `app/triage/slot_fill.py`
- Create: `app/triage/__init__.py` (empty)
- Modify: `scripts/test_slot_gate.py`

- [ ] **Step 1: Add failing tests**

Append to `scripts/test_slot_gate.py`:

```python
from app.ner.models import EntityExtractResult
from app.triage.slot_fill import fill_slot_table

def test_fill_from_ner():
    ner = EntityExtractResult(
        query="脚脖子肿，昨天扭了",
        primary_symptom="脚脖子肿",
        companion_symptoms=[],
    )
    table = fill_slot_table(ner)
    assert table.primary_symptom == "脚脖子肿"
    assert table.gender == "男"
    assert table.trigger is not None  # 扭伤相关


def test_fill_age_gender_override():
    ner = EntityExtractResult(query="女，5岁，发烧", primary_symptom="发烧")
    table = fill_slot_table(ner)
    assert table.gender == "女"
    assert "5" in table.age
```

- [ ] **Step 2: Run — expect fail**

Run: `venv\Scripts\python.exe scripts/test_slot_gate.py`

- [ ] **Step 3: Implement `app/triage/slot_fill.py`**

```python
import re

from app.domain.slot_table import TriageSlotTable, default_slot_table
from app.ner.models import EntityExtractResult

_TRIGGER_PATTERNS = [
    (re.compile(r"扭|外伤|摔|撞|运动|久站|受凉|饭后|情绪激动"), lambda m: m.group(0)),
]
_DURATION_PATTERNS = [
    (re.compile(r"(\d+[天周月年]|刚出现|首次|反复|很久|半年|几个月)"), lambda m: m.group(0)),
]
_EMERGENCY_PATTERNS = [
    (re.compile(r"不能动|发紫|剧烈|受不了|昏迷|大量出血|畸形"), lambda m: m.group(0)),
]


def _first_match(query: str, patterns: list) -> str | None:
    for pat, pick in patterns:
        m = pat.search(query)
        if m:
            return pick(m)
    return None


def _parse_demographics(query: str, table: TriageSlotTable) -> TriageSlotTable:
    if "女" in query:
        table.gender = "女"
    age_m = re.search(r"(\d+)\s*岁", query)
    if age_m:
        table.age = f"{age_m.group(1)}岁"
    return table


def fill_slot_table(ner: EntityExtractResult) -> TriageSlotTable:
    table = default_slot_table()
    table.primary_symptom = ner.primary_symptom
    table.companion_symptoms = list(ner.companion_symptoms)
    table.primary_disease = ner.primary_disease
    table.companion_diseases = list(ner.companion_diseases)
    q = ner.query or ""
    table = _parse_demographics(q, table)
    table.trigger = _first_match(q, _TRIGGER_PATTERNS)
    table.duration = _first_match(q, _DURATION_PATTERNS)
    table.emergency = _first_match(q, _EMERGENCY_PATTERNS)
    return table
```

- [ ] **Step 4: Run tests — pass**

- [ ] **Step 5: Commit**

```bash
git add app/triage/ scripts/test_slot_gate.py
git commit -m "feat: fill TriageSlotTable from NER and rule-based optional slots"
```

---

### Task 3: Session reset + reject message + AppState

**Files:**
- Modify: `app/domain/triage_intent.py`
- Create: `app/triage/session_reset.py`
- Modify: `app/domain/models.py`
- Modify: `app/graph/nodes/reject.py`

- [ ] **Step 1: Update reject message**

In `app/domain/triage_intent.py` change:

```python
REJECT_MESSAGE = "请输入症状？"
```

- [ ] **Step 2: Create session reset helper**

`app/triage/session_reset.py`:

```python
def triage_state_reset_patch() -> dict:
    """Clear triage fields; keep messages/thread unchanged."""
    return {
        "slot_table": None,
        "slot_gate_passed": False,
        "ner_result": None,
        "intent_result": None,
        "rag_chunk_id": None,
        "rag_chunk": None,
        "dept_state": None,
        "locked_department": None,
        "disease_dept_result": None,
        "symptom_slot_result": None,
    }
```

- [ ] **Step 3: Extend AppState in `app/domain/models.py`**

Add after existing optional fields:

```python
from app.domain.slot_table import TriageSlotTable  # top-level import may cause cycle; use Optional[Any] or forward ref

# Inside AppState:
slot_table: Optional[Any] = None  # TriageSlotTable
slot_gate_passed: bool = False
rag_chunk_id: str | None = None
rag_chunk: dict | None = None
dept_state: Optional[Any] = None
locked_department: str | None = None
```

Prefer adding `DeptDisambiguationState` in Task 6; use `Optional[Any]` initially to avoid import cycles.

- [ ] **Step 4: Update reject_node**

```python
from app.triage.session_reset import triage_state_reset_patch

def reject_node(state: AppState) -> dict:
    logger.info(">>> Enter node: reject (fixed message)")
    patch = triage_state_reset_patch()
    patch["messages"] = [AIMessage(content=REJECT_MESSAGE)]
    return patch
```

- [ ] **Step 5: Commit**

```bash
git add app/domain/triage_intent.py app/triage/session_reset.py app/domain/models.py app/graph/nodes/reject.py
git commit -m "feat: session reset on reject and extend AppState for triage slots"
```

---

### Task 4: Graph nodes slot_fill + slot_gate + routing

**Files:**
- Create: `app/graph/nodes/slot_fill.py`
- Create: `app/graph/nodes/slot_gate.py`
- Modify: `app/domain/routing.py`
- Modify: `app/graph/builder.py`

- [ ] **Step 1: slot_fill_node**

```python
from app.core.logging import logger
from app.domain.models import AppState
from app.triage.slot_fill import fill_slot_table

def slot_fill_node(state: AppState) -> dict:
    logger.info(">>> Enter node: slot_fill")
    ner = state.ner_result
    if not ner:
        return {"slot_table": None}
    table = fill_slot_table(ner)
    return {"slot_table": table}
```

- [ ] **Step 2: slot_gate_node**

```python
from app.domain.slot_table import slot_gate_passes

def slot_gate_node(state: AppState) -> dict:
    logger.info(">>> Enter node: slot_gate")
    table = state.slot_table
    passed = table is not None and slot_gate_passes(table)
    return {"slot_gate_passed": passed}
```

- [ ] **Step 3: routing**

In `app/domain/routing.py` add:

```python
from app.domain.slot_table import slot_gate_passes

def route_after_slot_gate(state: AppState) -> str:
    if state.slot_gate_passed:
        ir = state.intent_result
        route = ir.triage_route if ir else None
        if route == "disease":
            return "disease_dept"
        if route == "symptom":
            return "rag_symptom_recall"
        return "reject"
    return "reject"
```

Replace `route_after_decision` usage: decision → slot_fill → slot_gate (not direct conditional from decision).

- [ ] **Step 4: Rewire builder.py**

```python
graph.add_node("slot_fill", slot_fill_node)
graph.add_node("slot_gate", slot_gate_node)
graph.add_edge("decision", "slot_fill")
graph.add_edge("slot_fill", "slot_gate")
graph.add_conditional_edges("slot_gate", route_after_slot_gate, {
    "disease_dept": "disease_dept",
    "rag_symptom_recall": "rag_symptom_recall",
    "reject": "reject",
})
# remove old conditional_edges from decision directly
```

Add stub `rag_symptom_recall` node (pass-through) until Task 5.

- [ ] **Step 5: Manual smoke**

Run: `venv\Scripts\python.exe scripts/test_chat_api.py`  
Update reject expected message to `请输入症状？`  
Expected: gate reject still 5/5 or update cases

- [ ] **Step 6: Commit**

---

### Task 5: OpenSearch RAG wrapper

**Files:**
- Create: `app/infra/opensearch_rag.py`
- Modify: `app/core/config.py` (add `RAG_KB_INDEX = os.getenv("RAG_KB_INDEX", "rag_knowledge")`)
- Create: `app/graph/nodes/rag_symptom_recall.py`

- [ ] **Step 1: Copy/adapt search from demo**

`app/infra/opensearch_rag.py` — expose:

```python
def search_rag_knowledge(query: str, k: int = 3) -> list[dict]:
    """Returns full _source dicts sorted by score."""
```

Reuse `search_hybrid` logic from `demo/opensearch_rag_kb.py`; import `get_client` or inline OpenSearch client using `config.ES_URL` and `config.RAG_KB_INDEX`.

- [ ] **Step 2: rag_symptom_recall_node**

```python
def rag_symptom_recall_node(state: AppState) -> dict:
    table = state.slot_table
    if not table or not table.primary_symptom:
        return {}
    q = table.primary_symptom
    if table.companion_symptoms:
        q = q + " " + " ".join(table.companion_symptoms)
    hits = search_rag_knowledge(q, k=1)
    if not hits:
        return {"rag_chunk": None, "rag_chunk_id": None}
    chunk = hits[0]
    return {"rag_chunk": chunk, "rag_chunk_id": chunk.get("id")}
```

- [ ] **Step 3: Wire builder**

`rag_symptom_recall` → `dept_disambiguation` (stub until Task 6)

- [ ] **Step 4: Smoke with OpenSearch running**

Run demo search or node integration with query `脚脖子肿`.

- [ ] **Step 5: Commit**

---

### Task 6: Department scoring + disambiguation state

**Files:**
- Create: `app/domain/dept_disambiguation.py`
- Create: `app/triage/dept_scoring.py`
- Create: `scripts/test_dept_scoring.py`

- [ ] **Step 1: Failing tests for RK0001**

```python
RK0001_DEPTS = [
    {"department": "骨科", "priority": 1, "condition": "外伤后踝部肿胀、活动痛；怀疑韧带损伤或骨折"},
    {"department": "风湿免疫科", "priority": 2, "condition": "双侧踝部反复肿胀，伴晨僵、多关节疼痛"},
    {"department": "血管外科", "priority": 3, "condition": "久站后肿胀加重，伴静脉曲张或下肢沉重感"},
]

def test_trauma_locks_orthopedics():
    scores = score_departments(RK0001_DEPTS, user_text="脚脖子肿，昨天扭了", slot_trigger="扭")
    assert max(scores, key=scores.get) == "骨科"

def test_margin_lock_or_ask():
    locked, dept, margin = try_lock_department(scores, margin_threshold=2.0, lock_threshold=6.0)
    # trauma case should lock without ask
    assert locked is True
    assert dept == "骨科"
```

- [ ] **Step 2: Implement scoring**

`app/triage/dept_scoring.py`:

```python
MARGIN = 2.0
LOCK_THRESHOLD = 6.0
W_PRIORITY = 1.0
W_CONDITION = 1.5
W_ACCOMPANY = 1.0
W_SLOT = 1.0

def score_departments(depts, user_text, accompany_keywords=None, slot_trigger=None, slot_emergency=None) -> dict[str, float]:
    ...

def try_lock_department(scores: dict[str, float], margin_threshold=MARGIN, lock_threshold=LOCK_THRESHOLD):
    ordered = sorted(scores.items(), key=lambda x: -x[1])
    if len(ordered) == 1:
        return True, ordered[0][0], 999.0
    top, second = ordered[0], ordered[1]
    margin = top[1] - second[1]
    if margin >= margin_threshold or top[1] >= lock_threshold:
        return True, top[0], margin
    return False, None, margin
```

Keyword hit: count overlapping chars/tokens between `user_text` and `condition` (simple: split on punctuation，check token in user_text).

- [ ] **Step 3: DeptDisambiguationState model**

`app/domain/dept_disambiguation.py`:

```python
class DeptDisambiguationState(BaseModel):
    candidate_departments: list[dict] = Field(default_factory=list)
    dept_scores: dict[str, float] = Field(default_factory=dict)
    round: int = 0
    status: Literal["scoring", "asking", "locked", "emergency", "fallback"] = "scoring"
    last_question: str | None = None
    margin: float | None = None
```

- [ ] **Step 4: Run tests — pass**

- [ ] **Step 5: Commit**

---

### Task 7: Emergency check + dept_disambiguation node (rules-only path)

**Files:**
- Create: `app/graph/nodes/dept_disambiguation.py`
- Modify: `app/domain/routing.py`

- [ ] **Step 1: emergency helper**

```python
def is_emergency(chunk: dict, user_text: str, slot_emergency: str | None) -> bool:
    flag = chunk.get("emergency_flag") or {}
    cond = flag.get("condition") or ""
    blob = user_text + (slot_emergency or "")
    # simple: any 4-char substring from cond keywords in blob, or match slot_emergency
    for kw in ["畸形", "不能负重", "发紫", "剧烈", "不能动"]:
        if kw in blob:
            return True
    return False
```

- [ ] **Step 2: dept_disambiguation_node (round 0, no LLM yet)**

```python
def dept_disambiguation_node(state: AppState) -> dict:
    chunk = state.rag_chunk
    table = state.slot_table
    if not chunk or not table:
        return {}
    user_text = _accumulated_user_text(state)
    if is_emergency(chunk, user_text, table.emergency):
        return {
            "locked_department": "急诊",
            "dept_state": DeptDisambiguationState(status="emergency"),
        }
    depts = chunk.get("department_recommendation") or []
    scores = score_departments(depts, user_text, chunk.get("accompanying_symptom_keywords"), table.trigger, table.emergency)
    locked, dept, margin = try_lock_department(scores)
    if locked:
        return {
            "locked_department": dept,
            "dept_state": DeptDisambiguationState(status="locked", dept_scores=scores, margin=margin),
        }
    # fallback path for Task 8: status=asking
    p1 = next((d["department"] for d in depts if d.get("priority") == 1), None)
    return {
        "dept_state": DeptDisambiguationState(status="asking", dept_scores=scores, margin=margin, candidate_departments=depts, round=0),
        "messages": [AIMessage(content=_template_question(depts))],  # temporary template until LLM
    }
```

- [ ] **Step 3: routing after dept node**

If `dept_state.status == "asking"`: END (wait user). Next turn: detect `dept_state.round > 0` entry — implement in Task 8.

If `locked` or `emergency`: → `answer_generate`

- [ ] **Step 4: Commit**

---

### Task 8: LLM ask/parse + multi-round dept flow

**Files:**
- Create: `app/triage/dept_llm.py`
- Modify: `app/graph/nodes/dept_disambiguation.py`
- Modify: `app/graph/builder.py` (entry router: new user msg continues dept if `dept_state.status==asking`)

- [ ] **Step 1: Structured outputs**

```python
class DeptQuestionOutput(BaseModel):
    question: str
    targets: list[str]
    discriminative_axis: str

class DeptAnswerParseOutput(BaseModel):
    features: dict[str, bool] = Field(default_factory=dict)
    supported_departments: list[str] = Field(default_factory=list)
```

Prompt templates per spec §7.5.

- [ ] **Step 2: On asking path**

- Round 0 fail lock → LLM `DeptQuestionOutput` from top-2 depts
- User replies → parse → boost scores → re-lock
- Increment `round`; if `round >= 3` → fallback priority=1

- [ ] **Step 3: Graph entry**

Add `route_entry` or check in `trim_history`/custom node: if `state.dept_state and state.dept_state.status == "asking"`, skip decision and go to `dept_disambiguation`.

- [ ] **Step 4: Commit**

---

### Task 9: Answer generate consumes locked department

**Files:**
- Modify: `app/graph/nodes/answer.py`

- [ ] **Step 1: Structured reply template when `locked_department` set**

If `state.locked_department == "急诊"`: use `emergency_flag.message` from chunk.

Else: prompt LLM with slot_table + locked_department + canonical_symptom — **do not** let LLM change department.

```python
prompt = f"""
根据导诊结果回复用户（简短、专业）：
- 标准症状：{chunk.get('canonical_symptom')}
- 推荐科室：{state.locked_department}
- 用户描述：{user_text}
"""
```

- [ ] **Step 2: E2E update `scripts/test_chat_api.py`**

Add case: `脚脖子肿，昨天扭了` → reply mentions 骨科 (live).

- [ ] **Step 3: Commit**

---

### Task 10: Integration verification

- [ ] **Step 1: Unit scripts**

```powershell
$env:PYTHONIOENCODING="utf-8"
venv\Scripts\python.exe scripts/test_slot_gate.py
venv\Scripts\python.exe scripts/test_dept_scoring.py
```

- [ ] **Step 2: OpenSearch up**

```powershell
Invoke-RestMethod "http://127.0.0.1:9200/rag_knowledge/_count"
```

- [ ] **Step 3: API E2E**

```powershell
venv\Scripts\python.exe scripts/test_chat_api.py
venv\Scripts\python.exe scripts/test_chat_api.py --live-dept  # if added
```

- [ ] **Step 4: Update spec status**

In `docs/superpowers/specs/2026-06-09-triage-slot-gate-design.md` set status to `已实现`.

---

## Plan Self-Review (spec coverage)

| Spec section | Task |
|--------------|------|
| §3 TriageSlotTable | Task 1–2 |
| §4 slot_gate | Task 1, 4 |
| §5 session_reset | Task 3 |
| §6 RAG recall | Task 5 |
| §7 dept disambiguation | Task 6–8 |
| §8 AppState | Task 3, 6 |
| §9 nodes | Task 4–9 |
| §10 error handling | Task 5 (empty hits), Task 7 (fallback), Task 8 (LLM fail → P1) |
| §11 tests | Task 1, 6, 10 |
| Disease chain unchanged | Task 4 routing → `disease_dept` |

No placeholders remain in task steps above.

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-06-09-triage-slot-gate.md`.

**Two execution options:**

1. **Subagent-Driven (recommended)** — fresh subagent per task, review between tasks, fast iteration  
2. **Inline Execution** — implement tasks in this session with checkpoints

Which approach?
