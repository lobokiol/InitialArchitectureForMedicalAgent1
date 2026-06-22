"""Record complete triage cycles to SQLite for evaluation."""
from __future__ import annotations

import json
from datetime import datetime, timezone

from app.core import config
from app.core.logging import logger
from app.domain.models import AppState
from app.infra.triage_session_store import TriageSessionStore


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def classify_outcome(state: AppState) -> str | None:
    ds = state.dept_state
    status = getattr(ds, "status", None) if ds else None

    if status == "asking":
        return None
    if status == "emergency" or state.locked_department == "急诊":
        return "emergency"
    if status == "fallback":
        return "fallback"
    if state.locked_department and status == "locked":
        return "locked"
    if state.locked_department:
        return "locked"
    if state.disease_dept_result and state.disease_dept_result.departments:
        return "disease"
    ir = state.intent_result
    if ir and ir.triage_route == "reject":
        return "reject"
    if state.slot_table is not None and not state.slot_gate_passed:
        return "reject"
    return "unmatched"


def build_state_snapshot(state: AppState) -> dict:
    snap: dict = {
        "rag_chunk_id": state.rag_chunk_id,
        "locked_department": state.locked_department,
    }
    if state.ner_result is not None and hasattr(state.ner_result, "model_dump"):
        snap["ner_result"] = state.ner_result.model_dump()
    if state.slot_table is not None and hasattr(state.slot_table, "model_dump"):
        snap["slot_table"] = state.slot_table.model_dump()
    if state.dept_state is not None:
        snap["dept_state"] = state.dept_state.model_dump()
    if state.disease_dept_result is not None:
        snap["disease_dept_result"] = state.disease_dept_result.model_dump()
    if state.intent_result is not None:
        snap["intent_result"] = state.intent_result.model_dump()
    if state.rag_chunk:
        snap["rag_chunk"] = {
            "id": state.rag_chunk.get("id"),
            "canonical_symptom": state.rag_chunk.get("canonical_symptom"),
        }
    return snap


def _actual_dept(state: AppState) -> str | None:
    if state.locked_department:
        return state.locked_department
    ddr = state.disease_dept_result
    if ddr and ddr.departments:
        first = ddr.departments[0]
        if isinstance(first, dict):
            return first.get("dept") or first.get("department")
        return str(first)
    return None


def _actual_route(state: AppState) -> str | None:
    ir = state.intent_result
    return ir.triage_route if ir else None


class TriageSessionRecorder:
    def __init__(self, store: TriageSessionStore | None = None) -> None:
        path = config.TRIAGE_SESSION_DB_PATH
        self._store = store or TriageSessionStore(path)
        self._store.init_schema()
        self._enabled = config.TRIAGE_SESSION_ENABLED

    def record_turn(
        self,
        *,
        user_id: str,
        thread_id: str,
        user_message: str,
        assistant_reply: str,
        state: AppState,
        was_dept_followup: bool,
    ) -> None:
        if not self._enabled:
            return
        try:
            self._record_turn_impl(
                user_id=user_id,
                thread_id=thread_id,
                user_message=user_message,
                assistant_reply=assistant_reply,
                state=state,
                was_dept_followup=was_dept_followup,
            )
        except Exception:
            logger.exception("triage session record_turn failed thread_id=%s", thread_id)

    def _record_turn_impl(
        self,
        *,
        user_id: str,
        thread_id: str,
        user_message: str,
        assistant_reply: str,
        state: AppState,
        was_dept_followup: bool,
    ) -> None:
        now = _utc_now_iso()
        draft = self._store.get_in_progress(thread_id)

        if not was_dept_followup:
            if draft:
                self._finalize_incomplete(draft, state, now)
            self._store.insert_draft(
                {
                    "user_id": user_id,
                    "thread_id": thread_id,
                    "initial_message": user_message,
                    "turns_json": json.dumps([]),
                    "turn_count": 0,
                    "dept_rounds": 0,
                    "started_at": now,
                }
            )
            draft = self._store.get_in_progress(thread_id)
        elif not draft:
            self._store.insert_draft(
                {
                    "user_id": user_id,
                    "thread_id": thread_id,
                    "initial_message": user_message,
                    "turns_json": json.dumps([]),
                    "turn_count": 0,
                    "dept_rounds": 0,
                    "started_at": now,
                }
            )
            draft = self._store.get_in_progress(thread_id)

        if not draft:
            return

        session_id = draft["id"]
        turns = json.loads(draft["turns_json"])
        turn_num = len(turns) + 1
        turns.append(
            {
                "round": turn_num,
                "user": user_message,
                "assistant": assistant_reply,
                "timestamp": now,
            }
        )
        dept_rounds = int(draft.get("dept_rounds") or 0)
        if was_dept_followup:
            dept_rounds += 1

        outcome = classify_outcome(state)
        base_update = {
            "turns_json": json.dumps(turns, ensure_ascii=False),
            "turn_count": len(turns),
            "dept_rounds": dept_rounds,
            "actual_route": _actual_route(state),
            "actual_dept": _actual_dept(state),
            "rag_chunk_id": state.rag_chunk_id,
            "state_snapshot_json": json.dumps(build_state_snapshot(state), ensure_ascii=False),
        }

        if outcome is None:
            self._store.update_draft(session_id, base_update)
            return

        self._store.finalize(
            session_id,
            {
                **base_update,
                "outcome": outcome,
                "completed_at": now,
            },
        )

    def _finalize_incomplete(self, draft: dict, state: AppState, now: str) -> None:
        self._store.finalize(
            draft["id"],
            {
                "outcome": "incomplete",
                "turns_json": draft["turns_json"],
                "turn_count": draft["turn_count"],
                "dept_rounds": draft["dept_rounds"],
                "actual_route": _actual_route(state),
                "actual_dept": _actual_dept(state),
                "rag_chunk_id": state.rag_chunk_id,
                "state_snapshot_json": json.dumps(build_state_snapshot(state), ensure_ascii=False),
                "completed_at": now,
            },
        )
