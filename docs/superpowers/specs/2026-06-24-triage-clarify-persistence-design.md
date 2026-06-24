# Triage Clarify Persistence Fix

**Date:** 2026-06-24  
**Status:** Approved (design §1)  
**Scope:** Fix premature SQLite finalization during `symptom_clarify` (e.g. CL0010 眼睛不适).

---

## 1. Problem

`triage_sessions` persistence uses `classify_outcome()` to decide whether to `update_draft` (non-terminal) or `finalize` (terminal).

Today only `dept_state.status == "asking"` is treated as non-terminal. During `symptom_clarify` (age / sex / pain_location), `clarify_state.status == "asking"` but `dept_state` is often `null`. Those turns are classified as `unmatched` and the draft is finalized after the first clarify question.

**Symptom:** A multi-turn CL flow (e.g. 眼睛疼 → age → sex → dept_rules) is split across multiple SQLite rows or loses `initial_message` continuity.

**Root cause:** Post-invoke “waiting for user to pick clarify options” is not recognized as in-progress.

---

## 2. Requirements

| ID | Requirement |
|----|-------------|
| R1 | While `clarify_state.status == "asking"` and `last_choices` is non-empty, `classify_outcome` returns `None` |
| R2 | All turns of one triage cycle append to the same `in_progress` draft until a true terminal outcome |
| R3 | No schema changes to `triage_sessions` |
| R4 | No new `outcome` values (low_confidence deferred) |
| R5 | Add unit test for clarify asking → `None` |

---

## 3. Design

### 3.1 Change location

`app/services/triage_recorder.py` — function `classify_outcome()`.

Add early check **before** terminal outcome logic (alongside existing `dept_state` check):

```python
cs = state.clarify_state
if cs and getattr(cs, "status", None) == "asking" and cs.last_choices:
    return None
```

### 3.2 Why not reuse `_is_clarify_followup`

`app/domain/routing.py` `_is_clarify_followup` is used **pre-invoke** to detect that the user is replying to a clarify question (requires last message `HumanMessage`).

`classify_outcome` runs **post-invoke** when the assistant has just asked a clarify question (last message `AIMessage`, `clarify_state.last_choices` populated). Different predicate; do not share the same helper without a third “mode” parameter.

### 3.3 Unchanged behavior

- `dept_state.status == "asking"` — still non-terminal (dept disambiguation / dept_rules differential).
- `is_dept_followup_reply` / `record_turn` cycle boundaries — unchanged.
- Terminal outcomes: `locked`, `disease`, `reject`, `emergency`, `fallback`, `unmatched`, `incomplete`.

### 3.4 Expected flow (CL0010)

| Round | User | Assistant | `classify_outcome` |
|-------|------|-----------|-------------------|
| 1 | 我眼睛疼 | 请问年龄？ | `None` (clarify asking) |
| 2 | 19-35岁 | 请问性别？ | `None` (clarify asking) |
| 3 | 男 | 鉴别选项… | `None` (dept_rules asking) |
| N | … | 推荐 / 拒绝 | terminal outcome |

One `triage_sessions` row on finalize; `turns_json` contains all rounds.

---

## 4. Testing

**Unit:** `tests/test_triage_recorder.py`

- `test_classify_clarify_asking_returns_none`: `SymptomClarifyState(status="asking", last_choices=[ClarifyChoice(...)])` → `classify_outcome(state) is None`.

**Manual:** Run CL0010 flow; verify single completed row with `turn_count` matching conversation length (`scripts/view_triage_db.py`).

---

## 5. Out of scope

- `outcome: low_confidence` for confidence gate failures
- Shared `is_triage_in_progress()` helper (option C)
- `build_state_snapshot` including `clarify_state`
