# Medical Data Batch Triage Test

**Date:** 2026-06-25  
**Status:** Approved for implementation planning  
**Scope:** Extract 100 questions from `小医疗数据.json`, run them through live `/chat` with fixed persona choices, export full session records from `triage_sessions.db`.

---

## 1. Problem

The project has ~6000 real patient questions in `sourceData/data/医疗数据.json` (and a 100-row sample in `小医疗数据.json`). There is no automated way to:

1. Use these questions as triage test queries
2. Walk through multi-turn clarify / dept disambiguation with consistent answers
3. Persist results to `triage_sessions.db` and export them for review

Existing tooling only partially covers this:

| Tool | Gap |
|------|-----|
| `tests/run_eval.py` | NER + route only; no full `/chat` multi-turn |
| `scripts/integration_triage_db.py` | Handful of hard-coded cases |
| `front_Web/scripts/e2e_chat_test.py` | Single symptom; always picks first clarify option (wrong age) |
| `scripts/export_triage_sessions.py` | No `--user-id` filter |

**Goal:** One-command batch run of 100 sample questions with fixed persona, then JSONL export of complete triage sessions.

---

## 2. Decisions (confirmed)

| Topic | Choice |
|-------|--------|
| Data source | `sourceData/data/小医疗数据.json` (100 rows); not full 6000 yet |
| Field used | `questions` per line (JSONL) |
| Approach | **方案 1** — single script `scripts/run_medical_data_batch.py` |
| Export format | Full session JSONL (same shape as `export_triage_sessions.py`) |
| Thread isolation | **One new thread per case** |
| Batch user | Dedicated `user_id`: `batch-med-small-100` |
| Sex | Fixed `男` |
| Age | Fixed `19-35岁` (represents 30-year-old male) |
| Other clarify / dept choices | First option (`choices[0]`) |
| Multi-select dept | First index (`1`) |
| Golden comparison | Out of scope (source has no `expect` labels) |
| Query filtering | Run all 100 as-is (includes reject / non-symptom) |

---

## 3. Architecture

```text
小医疗数据.json
    │  read questions[]
    ▼
run_medical_data_batch.py
    │  POST /users  (batch-med-small-100)
    │  loop 100×:
    │    POST /threads  (new thread)
    │    POST /chat     (initial query)
    │    auto-reply loop (clarify → dept)
    ▼
triage_sessions.db  (via triage_recorder in chat_service)
    │
    ▼
export_triage_sessions.py --user-id batch-med-small-100
    │
    ▼
exports/medical_small_100_sessions.jsonl
```

---

## 4. Auto-reply rules

Mirror `cli.py` multi-turn loop; override only age and sex.

| API state | Reply |
|-----------|-------|
| `awaiting_clarify` + `clarify_phase=age` | `19-35岁` |
| `awaiting_clarify` + `clarify_phase=sex` | `男` |
| `awaiting_clarify` + `clarify_phase=pain_location` | `clarify_choices[0].label` |
| `awaiting_dept_choice` + `multi_select=false` | `dept_choices[0].label` |
| `awaiting_dept_choice` + `multi_select=true` | `1` |

Loop until neither `awaiting_clarify` nor `awaiting_dept_choice` is true, or `max_steps` (default 15) exceeded.

On `max_steps` exceeded: log error row to `exports/batch_med_100_errors.jsonl`, continue next case.

---

## 5. Components

### 5.1 `scripts/run_medical_data_batch.py` (new)

**Responsibilities:**

- Parse args: `--input`, `--base-url`, `--user-id`, `--limit`, `--max-steps`, `--dry-run`
- Defaults:
  - `--input sourceData/data/小医疗数据.json`
  - `--user-id batch-med-small-100`
  - `--base-url http://127.0.0.1:8000`
- Health check: `GET /healthz` before starting
- Register user via `POST /users`
- For each line in input JSONL:
  1. `POST /threads` with title `med-{index:03d}`
  2. Send `questions` text as first message
  3. Run auto-reply loop
  4. Print progress `[{i}/{n}] outcome={...} dept={...}`
- Write run summary to `exports/batch_med_100_summary.json` (counts by outcome, errors, elapsed)

**Error handling:**

- Single case failure does not abort batch
- HTTP errors and timeouts recorded in errors JSONL
- `trust_env=False` on requests session (match `cli.py` / `integration_triage_db.py`)

### 5.2 `app/infra/triage_session_store.py` (extend)

Add optional `user_id` filter to `list_sessions()`:

```python
def list_sessions(..., user_id: str | None = None) -> list[dict]:
```

### 5.3 `scripts/export_triage_sessions.py` (extend)

Add CLI flag:

```text
--user-id batch-med-small-100
```

Pass through to `list_sessions(user_id=...)`.

Default output when user-id set: `exports/medical_small_100_sessions.jsonl`.

### 5.4 Shared auto-reply helper (inline in batch script)

Extract logic from `cli.py` `_ask_chat` clarify/dept loops into functions inside the batch script (no new package module — YAGNI for 100-case tool).

---

## 6. Export record shape

Unchanged from existing exporter:

```json
{
  "id": "uuid",
  "message": "原始 questions 文本",
  "actual_route": "symptom",
  "actual_chunk_id": "CL0001",
  "actual_dept": "消化内科",
  "actual_emergency": false,
  "outcome": "locked",
  "turn_count": 4,
  "dept_rounds": 1,
  "turns": [...]
}
```

---

## 7. Usage

```powershell
# 1. Ensure API + OpenSearch + Redis running
.\scripts\start-api.ps1

# 2. Run batch (100 cases, ~10–30 min depending on LLM latency)
python scripts/run_medical_data_batch.py

# 3. Export sessions for this batch user
python scripts/export_triage_sessions.py --user-id batch-med-small-100 --out exports/medical_small_100_sessions.jsonl
```

Optional dry-run (parse input only, no API calls):

```powershell
python scripts/run_medical_data_batch.py --dry-run
```

---

## 8. Out of scope (v1)

- Running full `医疗数据.json` (6000 rows)
- Accuracy scoring against doctor `answers` field
- Filtering pediatric / gynecology questions before run
- REST API for batch trigger
- Parallel concurrent requests (sequential only to avoid rate limits)

---

## 9. Future extension (6000 rows)

When scaling to full dataset:

- Add `--offset` / `--limit` to batch script (already planned via `--limit`)
- Consider splitting into `--user-id batch-med-{chunk}` per 500 rows
- Or refactor to 方案 3 (extract cases JSON + generic runner) if reuse needed

---

## 10. Verification

| Check | Command / criterion |
|-------|---------------------|
| Input parses | `--dry-run` prints 100 queries |
| Sessions written | `SELECT COUNT(*) FROM triage_sessions WHERE user_id='batch-med-small-100'` = 100 |
| Export count | JSONL line count = completed sessions for user |
| Persona applied | Sample `turns_json` shows `19-35岁` and `男` in user turns |
| No crash on reject | Cases like「你好」finalize with `outcome=reject` |

---

## 11. Test plan (implementation)

1. Unit: auto-reply picker returns correct label per `clarify_phase` / `multi_select`
2. Integration: run 3-case subset against live API (`--limit 3`)
3. Export: `--user-id` filter returns only batch rows
4. Manual: spot-check 5 exported rows for turn sequence and dept outcome
