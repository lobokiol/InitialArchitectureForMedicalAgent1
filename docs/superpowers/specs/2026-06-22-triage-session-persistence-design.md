# Triage Session Persistence Design

**Date:** 2026-06-22  
**Status:** Approved for implementation planning  
**Scope:** Persist complete triage cycles to SQLite for model evaluation and iteration.

---

## 1. Problem Statement

The hospital triage agent runs multi-turn conversations (department disambiguation, reject, emergency, single-turn disease routing) but only persists runtime state in Redis (LangGraph checkpoint + session metadata). There is no durable, queryable record of completed triage cycles suitable for:

- Regression testing against golden cases (`sourceData/data/foot_triage_golden.jsonl`)
- Bad-case analysis and labeling
- Exporting labeled datasets for model iteration

**Goal:** Save every complete triage cycle (regardless of outcome) to a database, one row per cycle.

---

## 2. Requirements

### 2.1 Functional

| ID | Requirement |
|----|-------------|
| R1 | Record one row per **triage cycle** (not per API call, not per entire `thread_id`) |
| R2 | Capture all terminal outcomes: `locked`, `disease`, `reject`, `emergency`, `fallback`, `unmatched` |
| R3 | Capture multi-turn department disambiguation as a single session with all turns |
| R4 | Mark abandoned cycles as `incomplete` when a new intake starts on the same thread |
| R5 | Export filtered sessions to JSONL aligned with existing golden-case fields |
| R6 | Persistence failure must not block chat responses |

### 2.2 Non-Functional

| ID | Requirement |
|----|-------------|
| N1 | Storage: SQLite (single file, no extra service) |
| N2 | Hook point: service layer (`TriageSessionRecorder` in `chat_service`), not LangGraph nodes |
| N3 | Configurable DB path via `TRIAGE_SESSION_DB_PATH` (default `data/triage_sessions.db`) |
| N4 | No query API or admin UI in v1 |

### 2.3 Out of Scope (v1)

- PostgreSQL / cloud database
- PII redaction pipeline
- REST endpoints for browsing records
- Modifying LangGraph graph structure

---

## 3. Triage Cycle Boundary

A **triage cycle** starts when the graph routes to `decision` (new intake). It ends at a **terminal outcome**.

### 3.1 Cycle Start

- `route_after_trim` returns `"decision"` (not a department follow-up reply)
- On start: if an `in_progress` draft exists for the same `thread_id`, finalize it as `outcome=incomplete`, then create a new draft

### 3.2 Cycle Continuation (Non-Terminal)

- `route_after_trim` returns `"dept_disambiguation"` because `dept_state.status == "asking"` and the latest message is a `HumanMessage`
- Append turn to the existing `in_progress` draft; increment `dept_rounds`

### 3.3 Terminal Outcomes

| `outcome` | Detection (post-invoke `AppState`) |
|-----------|-------------------------------------|
| `reject` | `slot_gate_passed == False` or `intent_result.triage_route == "reject"` with reject reply |
| `emergency` | `dept_state.status == "emergency"` or `locked_department == "急诊"` |
| `disease` | `disease_dept_result` present with departments, symptom route not taken to lock |
| `locked` | `locked_department` set, `dept_state.status == "locked"` |
| `fallback` | `dept_state.status == "fallback"` |
| `unmatched` | Terminal `answer_generate` fallback (no locked dept, no disease depts) |
| `incomplete` | New intake started while previous draft was still `in_progress` |

**Non-terminal (draft only):** `dept_state.status == "asking"` after invoke → update draft, do not finalize.

---

## 4. Architecture

```
POST /chat
  → chat_service.chat_once()
      → read checkpoint (detect dept follow-up)
      → LangGraph invoke
      → TriageSessionRecorder.record_turn(state, context)
          → TriageSessionStore (SQLite)
      → return ChatResponse
```

### 4.1 New Modules

| Module | Responsibility |
|--------|----------------|
| `app/infra/triage_session_store.py` | SQLite schema, connection, CRUD |
| `app/services/triage_recorder.py` | Cycle state machine, outcome classification, snapshot building |
| `scripts/export_triage_sessions.py` | CLI export to JSONL |

### 4.2 Integration Point

`chat_service.chat_once()` calls the recorder **after** `_app.invoke()` and **before** returning the response:

```python
recorder.record_turn(
    user_id=user_id,
    thread_id=thread_id,
    user_message=message,
    assistant_reply=reply,
    state=state,
    was_dept_followup=was_dept_followup,  # from pre-invoke checkpoint
)
```

`was_dept_followup` is determined by reading the checkpoint state and applying the same logic as `route_after_trim._is_dept_followup_reply`.

### 4.3 Error Handling

- Wrap `record_turn` in try/except; log exception at ERROR level
- Never raise to the API caller
- If SQLite file is missing or corrupt, log once and skip writes for that process lifetime (optional: retry on next call)

---

## 5. Data Model

### 5.1 Table: `triage_sessions`

| Column | Type | Notes |
|--------|------|-------|
| `id` | TEXT PK | UUID (`triage_session_id`) |
| `user_id` | TEXT NOT NULL | |
| `thread_id` | TEXT NOT NULL | |
| `status` | TEXT NOT NULL | `in_progress` \| `completed` |
| `outcome` | TEXT | Nullable while `in_progress`; required when `completed` |
| `initial_message` | TEXT NOT NULL | First user message in cycle |
| `turns_json` | TEXT NOT NULL | JSON array of turn objects |
| `turn_count` | INTEGER NOT NULL DEFAULT 0 | |
| `dept_rounds` | INTEGER NOT NULL DEFAULT 0 | Disambiguation rounds only |
| `actual_route` | TEXT | `disease` \| `symptom` \| `reject` \| null |
| `actual_dept` | TEXT | Final recommended department |
| `rag_chunk_id` | TEXT | For golden-case comparison |
| `state_snapshot_json` | TEXT | NER, slot_table, dept_scores, candidates |
| `started_at` | TEXT NOT NULL | ISO 8601 UTC |
| `completed_at` | TEXT | ISO 8601 UTC when finalized |

**Indexes:**

- `(thread_id, status)` — find in-progress draft
- `(completed_at)` — export by date
- `(outcome)` — filter by result type

### 5.2 Turn Object (`turns_json` element)

```json
{
  "round": 1,
  "user": "脚后跟疼",
  "assistant": "请问以下哪种情况更符合？\nA. ...",
  "timestamp": "2026-06-22T10:00:00Z"
}
```

### 5.3 State Snapshot (`state_snapshot_json`)

Captured at cycle end (or incomplete):

```json
{
  "ner_result": { "...": "..." },
  "slot_table": { "...": "..." },
  "rag_chunk_id": "RK0013",
  "rag_chunk": { "canonical_symptom": "..." },
  "dept_state": { "status": "locked", "dept_scores": {}, "round": 2 },
  "locked_department": "骨科",
  "disease_dept_result": null,
  "intent_result": { "triage_route": "symptom" }
}
```

Serialize via `model_dump()` where models are Pydantic; omit large binary fields.

---

## 6. Recorder State Machine

```
                    ┌─────────────────┐
                    │  new intake     │
                    │  (not followup) │
                    └────────┬────────┘
                             │
              ┌──────────────▼──────────────┐
              │ in_progress draft exists? │
              └──────┬──────────────┬───────┘
                  yes              no
                   │                │
         finalize incomplete    create draft
                   │                │
                   └────────┬───────┘
                            │
                    append turn
                            │
              ┌─────────────▼─────────────┐
              │     terminal outcome?     │
              └─────┬──────────────┬──────┘
                 yes              no (asking)
                  │                │
           finalize completed   keep in_progress
```

---

## 7. Configuration

Add to `app/core/config.py`:

```python
TRIAGE_SESSION_DB_PATH: str = os.getenv(
    "TRIAGE_SESSION_DB_PATH",
    "data/triage_sessions.db",
)
TRIAGE_SESSION_ENABLED: bool = os.getenv(
    "TRIAGE_SESSION_ENABLED", "true"
).lower() in ("1", "true", "yes")
```

Add to `.env.example`:

```
TRIAGE_SESSION_DB_PATH=data/triage_sessions.db
TRIAGE_SESSION_ENABLED=true
```

Ensure `data/` directory is created on first write; add `data/*.db` to `.gitignore` if not already covered.

---

## 8. Export Script

`scripts/export_triage_sessions.py`:

**CLI flags:**

- `--out PATH` (default `exports/triage_sessions.jsonl`)
- `--outcome locked,reject,...` (filter)
- `--since YYYY-MM-DD`
- `--status completed` (default; allow `incomplete` for abandoned)

**Export row format** (aligned with `foot_triage_golden.jsonl`):

```json
{
  "id": "<triage_session_id>",
  "message": "<initial_message>",
  "actual_route": "symptom",
  "actual_chunk_id": "RK0013",
  "actual_dept": "骨科",
  "actual_emergency": false,
  "outcome": "locked",
  "turn_count": 2,
  "dept_rounds": 1,
  "turns": [ "..."]
}
```

Golden fields (`expect_*`) are not auto-filled; export provides `actual_*` for diff tooling.

---

## 9. Testing Plan

### 9.1 Unit Tests (`tests/test_triage_recorder.py`)

| Case | Assert |
|------|--------|
| Single-turn disease | 1 completed row, `outcome=disease`, `turn_count=1` |
| Single-turn reject | `outcome=reject` |
| Emergency | `outcome=emergency`, `actual_dept=急诊` |
| 3-turn disambiguation | 1 completed row, `turn_count=3`, `dept_rounds=2` |
| New intake mid-disambiguation | Old `incomplete`, new cycle separate |
| SQLite failure | `chat_once` still returns reply |
| Export script | Produces valid JSONL from seeded DB |

### 9.2 Test Isolation

Use `:memory:` SQLite or temp file per test via dependency injection on `TriageSessionStore`.

---

## 10. Implementation Order

1. `TriageSessionStore` — schema + CRUD
2. `TriageSessionRecorder` — classification + state machine
3. Wire into `chat_service.chat_once()`
4. Config + `.env.example` + `.gitignore`
5. Unit tests
6. `export_triage_sessions.py`

---

## 11. Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| Misclassifying follow-up vs new intake | Reuse `_is_dept_followup_reply` logic; unit test both paths |
| Concurrent writes same thread | Single-writer per process; use transaction + `thread_id` unique in-progress constraint |
| Large `state_snapshot_json` | Cap snapshot fields; exclude full message history (already in `turns_json`) |
| Disk growth | Document manual pruning; export + archive workflow |

---

## 12. Success Criteria

- Every terminal triage outcome produces exactly one `completed` row in SQLite
- Multi-turn disambiguation appears as one row with full `turns_json`
- Abandoned cycles are preserved as `incomplete`
- Export JSONL can be diffed against `foot_triage_golden.jsonl` on `actual_dept` / `actual_chunk_id`
- Chat API behavior unchanged when persistence is disabled or fails
