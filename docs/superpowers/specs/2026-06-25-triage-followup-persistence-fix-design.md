# Triage Follow-up Persistence Fix

**Date:** 2026-06-25  
**Status:** Approved (design §1–§6)  
**Scope:** Fix multi-turn triage sessions split across multiple SQLite rows; archive and merge historical fragments.

---

## 1. Problem

`triage_sessions` is designed to store **one row per complete triage cycle**, with all turns accumulated in `turns_json` until a terminal outcome.

Observed behavior: multi-turn flows (e.g. CL0010 眼睛不适: clarify age → sex → dept_rules → locked) produce **multiple rows**, each with `turn_count=1`. Middle turns are stored as separate sessions with `outcome=incomplete`; `initial_message` becomes a follow-up reply (`19-35岁`, `男`, `3`) instead of the original chief complaint.

### Root cause

`chat_service.chat_once()` computes `was_dept_followup` **pre-invoke** from checkpoint state:

```python
pre_state = _read_checkpoint_state(thread_id, user_id)
was_dept_followup = bool(pre_state and is_dept_followup_reply(pre_state))
```

`is_dept_followup_reply()` (and `_is_clarify_followup`, `_is_dept_followup_reply`, `_is_dept_rules_followup`) require `messages[-1]` to be a `HumanMessage`.

| When | Last message in checkpoint | Result |
|------|---------------------------|--------|
| `route_after_trim` (post-trim) | New user message (`HumanMessage`) | Correctly continues cycle |
| `chat_service` pre-invoke | Assistant question (`AIMessage`) | `was_dept_followup=False` → new cycle |

Each follow-up turn therefore:
1. Finalizes the previous draft as `incomplete`
2. Creates a new draft with the reply as `initial_message`
3. Stores only that single turn

Unit tests pass because they manually pass `was_dept_followup=True` on turn 2+; the integration path through `chat_service` is untested.

A related fix (2026-06-24) added `clarify_state.status == "asking"` to `classify_outcome()` so turn 1 is not prematurely finalized as `unmatched`. That does not fix turn 2+ splitting.

---

## 2. Requirements

| ID | Requirement |
|----|-------------|
| R1 | Multi-turn clarify / dept / dept_rules flows finalize as **one row** with all turns in `turns_json` |
| R2 | Pre-invoke cycle continuation uses state flags (`clarify_state`, `dept_state`), not `messages[-1]` type |
| R3 | Graph routing helpers unchanged (`is_dept_followup_reply` for post-trim) |
| R4 | New intake on same thread while draft `in_progress` still finalizes old draft as `incomplete` |
| R5 | No schema changes to `triage_sessions` |
| R6 | No new `outcome` values |
| R7 | Unit tests cover pre-invoke predicate; integration script asserts multi-turn single row |
| R8 | Historical fragmented rows archived and merged via one-time repair script |

---

## 3. Design

### 3.1 New helper: `is_awaiting_triage_followup` (Approach A)

**File:** `app/domain/routing.py`

```python
def is_awaiting_triage_followup(state: AppState) -> bool:
    """Pre-invoke: system is waiting for user to reply to clarify or dept choices."""
    cs = state.clarify_state
    if cs and getattr(cs, "status", None) == "asking" and cs.last_choices:
        return True
    ds = state.dept_state
    if ds and getattr(ds, "status", None) == "asking" and ds.last_choices:
        return True
    return False
```

Covers:
- `symptom_clarify` (age / sex / pain_location)
- `dept_disambiguation` (accompany mode)
- `dept_rules_disambiguation` (differential mode)

Does **not** inspect `messages`; safe for pre-invoke checkpoint where last message is `AIMessage`.

### 3.2 Integration point

**File:** `app/services/chat_service.py`

Replace:
```python
was_dept_followup = bool(pre_state and is_dept_followup_reply(pre_state))
```

With:
```python
was_dept_followup = bool(pre_state and is_awaiting_triage_followup(pre_state))
```

Import `is_awaiting_triage_followup` from `app.domain.routing`.

**Unchanged:**
- `is_dept_followup_reply` and all `_is_*` helpers used by `route_after_trim`
- `TriageSessionRecorder.record_turn` / `classify_outcome`
- SQLite schema

### 3.3 Expected flow (CL0010)

| Round | User | Assistant | `was_dept_followup` (pre-invoke) | DB |
|-------|------|-----------|----------------------------------|-----|
| 1 | 我眼睛疼 | 请问年龄？ | False | insert draft, update (outcome None) |
| 2 | 19-35岁 | 请问性别？ | True (clarify asking) | append turn 2 |
| 3 | 男 | 鉴别选项… | True (clarify or dept asking) | append turn 3 |
| N | … | 推荐科室 | True/False | finalize, `turn_count=N` |

---

## 4. Testing

### 4.1 Unit: routing predicate

**File:** `tests/test_routing.py` (new or extend existing)

| Test | Setup | Assert |
|------|-------|--------|
| `test_awaiting_clarify_pre_invoke` | messages=[Human, AI clarify question]; clarify asking + choices | `is_awaiting_triage_followup=True`; `is_dept_followup_reply=False` |
| `test_awaiting_dept_pre_invoke` | messages=[Human, AI dept question]; dept asking + choices | same |
| `test_awaiting_dept_rules_pre_invoke` | dept asking, `choice_mode=differential` | `is_awaiting_triage_followup=True` |
| `test_not_awaiting_fresh_intake` | no clarify/dept asking | `is_awaiting_triage_followup=False` |

### 4.2 Unit: recorder with pre-invoke semantics

**File:** `tests/test_triage_recorder.py`

Add `test_recorder_pre_invoke_followup_detection`: simulate turn 1 finalize to in_progress, turn 2 with state that would satisfy `is_awaiting_triage_followup` and `was_dept_followup=True` derived from that predicate (document the contract).

Existing `test_recorder_clarify_multi_turn_single_row` and `test_recorder_multi_turn_single_row` remain; no change required.

### 4.3 Integration script

**File:** `scripts/integration_triage_db.py`

After existing multi-turn foot case (or add CL clarify case if API supports it):
- Query SQLite for the test user's sessions on the multi-turn thread
- Assert exactly **one** completed row with `turn_count >= 2` for that flow
- Fail if multiple `turn_count=1` rows appear for the same thread + time window

---

## 5. Historical data repair

### 5.1 Script

**File:** `scripts/repair_triage_fragments.py`

**Default:** `--dry-run` (print actions, no writes)

**Flags:**
- `--apply` — execute archive, merge, delete
- `--db PATH` — override `TRIAGE_SESSION_DB_PATH`

### 5.2 Archive table

Create if not exists:

```sql
CREATE TABLE IF NOT EXISTS triage_sessions_fragments (
    -- all columns from triage_sessions
    archived_at TEXT NOT NULL,
    merged_into_id TEXT  -- NULL if orphan only archived
);
```

Copy fragment rows before delete; set `merged_into_id` when merged into a new row.

### 5.3 Fragment chain detection

Group rows by `thread_id`, order by `started_at`.

A **fragment chain** is a consecutive sequence where:
1. Each row has `turn_count == 1`
2. All but the last have `outcome == 'incomplete'`, OR the chain is a run of incomplete rows with no terminal cap (orphan chain)
3. Rows are within **30 minutes** of each other (`completed_at` / `started_at` gap)
4. Optional heuristic: chain length ≥ 2 (single incomplete alone is orphan, archive only)

**Merge row construction:**
- `initial_message` — first row in chain
- `turns_json` — concatenate all turns in order, renumber `round` 1..N
- `turn_count` — total turns
- `dept_rounds` — sum of `dept_rounds` across chain
- `outcome`, `actual_dept`, `actual_route`, `rag_chunk_id`, `state_snapshot_json`, `completed_at` — from **last** row in chain
- `started_at` — first row
- `status` — `completed`
- New UUID for merged row `id`

After merge: delete original fragment rows (already archived).

### 5.4 Safety

- On `--apply`: copy DB to `{db_path}.bak.{timestamp}` before any mutation
- Dry-run prints: chains found, merge preview (initial_message, turn_count, outcome), orphan count
- Idempotent: re-run on already-repaired DB should find no new chains (merged rows have `turn_count > 1`)

### 5.5 Out of scope for repair

- Cross-`thread_id` merging
- Reconstructing turns lost when only middle fragments exist without terminal row (orphans archived, not merged)
- Modifying LangGraph checkpoint / Redis state

---

## 6. Error handling

| Scenario | Behavior |
|----------|----------|
| `pre_state` is None | `was_dept_followup=False`; normal new draft |
| Recorder failure | Existing try/except; non-fatal to chat |
| Repair script on empty DB | No-op, exit 0 |
| Repair merge ambiguous chain | Skip chain, log warning; archive individual rows |

---

## 7. Files to change

| File | Change |
|------|--------|
| `app/domain/routing.py` | Add `is_awaiting_triage_followup` |
| `app/services/chat_service.py` | Use new helper for `was_dept_followup` |
| `tests/test_routing.py` | New pre-invoke tests |
| `tests/test_triage_recorder.py` | Optional contract test |
| `scripts/integration_triage_db.py` | Assert multi-turn single row |
| `scripts/repair_triage_fragments.py` | New repair script |

---

## 8. Verification checklist

- [ ] `python tests/test_routing.py` (or pytest equivalent)
- [ ] `python tests/test_triage_recorder.py`
- [ ] Manual CL0010 via `cli.py`; `view_triage_db.py` shows one row, `turn_count >= 4`
- [ ] `python scripts/repair_triage_fragments.py --dry-run` on `data/triage_sessions.db`
- [ ] `python scripts/repair_triage_fragments.py --apply` after review
- [ ] Live integration script if server running

---

## 9. References

- `docs/superpowers/specs/2026-06-22-triage-session-persistence-design.md` — cycle boundary design
- `docs/superpowers/specs/2026-06-24-triage-clarify-persistence-design.md` — `classify_outcome` clarify fix
