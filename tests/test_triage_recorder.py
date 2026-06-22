"""Tests for triage session SQLite persistence."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.domain.dept_disambiguation import DeptDisambiguationState
from app.domain.models import AppState, IntentResult
from app.infra.triage_session_store import TriageSessionStore
from app.services.triage_recorder import TriageSessionRecorder, classify_outcome


def test_insert_and_get_in_progress() -> None:
    store = TriageSessionStore(":memory:")
    store.init_schema()
    sid = store.insert_draft(
        {
            "user_id": "u1",
            "thread_id": "u1:s:abc",
            "initial_message": "脚后跟疼",
            "turns_json": json.dumps([]),
            "turn_count": 0,
            "dept_rounds": 0,
            "started_at": "2026-06-22T10:00:00Z",
        }
    )
    row = store.get_in_progress("u1:s:abc")
    assert row is not None
    assert row["id"] == sid
    assert row["status"] == "in_progress"
    print("[OK] insert_and_get_in_progress")


def test_finalize_session() -> None:
    store = TriageSessionStore(":memory:")
    store.init_schema()
    sid = store.insert_draft(
        {
            "user_id": "u1",
            "thread_id": "t1",
            "initial_message": "心慌",
            "turns_json": json.dumps(
                [{"round": 1, "user": "心慌", "assistant": "骨科", "timestamp": "2026-06-22T10:00:00Z"}]
            ),
            "turn_count": 1,
            "dept_rounds": 0,
            "started_at": "2026-06-22T10:00:00Z",
        }
    )
    store.finalize(
        sid,
        {
            "outcome": "locked",
            "actual_route": "symptom",
            "actual_dept": "骨科",
            "rag_chunk_id": "RK0013",
            "state_snapshot_json": json.dumps({"locked_department": "骨科"}),
            "completed_at": "2026-06-22T10:00:01Z",
        },
    )
    assert store.get_in_progress("t1") is None
    rows = store.list_sessions(status="completed", outcome="locked")
    assert len(rows) == 1
    assert rows[0]["actual_dept"] == "骨科"
    print("[OK] finalize_session")


def test_classify_reject() -> None:
    state = AppState(slot_gate_passed=False, intent_result=IntentResult(triage_route="reject"))
    assert classify_outcome(state) == "reject"
    print("[OK] classify_reject")


def test_classify_emergency() -> None:
    state = AppState(
        locked_department="急诊",
        dept_state=DeptDisambiguationState(status="emergency"),
    )
    assert classify_outcome(state) == "emergency"
    print("[OK] classify_emergency")


def test_classify_asking_returns_none() -> None:
    state = AppState(dept_state=DeptDisambiguationState(status="asking", last_choices=[]))
    assert classify_outcome(state) is None
    print("[OK] classify_asking_returns_none")


def test_recorder_multi_turn_single_row() -> None:
    store = TriageSessionStore(":memory:")
    store.init_schema()
    recorder = TriageSessionRecorder(store)

    state1 = AppState(
        intent_result=IntentResult(triage_route="symptom"),
        dept_state=DeptDisambiguationState(status="asking"),
    )
    recorder.record_turn(
        user_id="u1",
        thread_id="t1",
        user_message="脚后跟疼",
        assistant_reply="请选择 A/B",
        state=state1,
        was_dept_followup=False,
    )
    assert store.get_in_progress("t1") is not None

    state2 = AppState(
        locked_department="骨科",
        dept_state=DeptDisambiguationState(status="locked"),
        rag_chunk_id="RK0013",
    )
    recorder.record_turn(
        user_id="u1",
        thread_id="t1",
        user_message="A",
        assistant_reply="建议就诊：骨科",
        state=state2,
        was_dept_followup=True,
    )
    assert store.get_in_progress("t1") is None
    rows = store.list_sessions(status="completed")
    assert len(rows) == 1
    assert rows[0]["turn_count"] == 2
    assert rows[0]["outcome"] == "locked"
    print("[OK] recorder_multi_turn_single_row")


def test_new_intake_finalizes_incomplete() -> None:
    store = TriageSessionStore(":memory:")
    store.init_schema()
    recorder = TriageSessionRecorder(store)

    asking = AppState(dept_state=DeptDisambiguationState(status="asking"))
    recorder.record_turn(
        user_id="u1",
        thread_id="t1",
        user_message="脚疼",
        assistant_reply="选 A/B",
        state=asking,
        was_dept_followup=False,
    )
    reject = AppState(
        slot_gate_passed=False,
        intent_result=IntentResult(triage_route="reject"),
    )
    recorder.record_turn(
        user_id="u1",
        thread_id="t1",
        user_message="你好",
        assistant_reply="无法导诊",
        state=reject,
        was_dept_followup=False,
    )
    rows = store.list_sessions(status="completed")
    outcomes = {r["outcome"] for r in rows}
    assert "incomplete" in outcomes
    assert "reject" in outcomes
    print("[OK] new_intake_finalizes_incomplete")


def test_export_data_ready() -> None:
    store = TriageSessionStore(":memory:")
    store.init_schema()
    sid = store.insert_draft(
        {
            "user_id": "u1",
            "thread_id": "t1",
            "initial_message": "脚后跟疼",
            "turns_json": "[]",
            "turn_count": 0,
            "dept_rounds": 0,
            "started_at": "2026-06-22T10:00:00Z",
        }
    )
    store.finalize(
        sid,
        {
            "outcome": "locked",
            "actual_route": "symptom",
            "actual_dept": "骨科",
            "rag_chunk_id": "RK0013",
            "turn_count": 1,
            "turns_json": '[{"round":1,"user":"脚后跟疼","assistant":"骨科","timestamp":"2026-06-22T10:00:00Z"}]',
            "state_snapshot_json": "{}",
            "completed_at": "2026-06-22T10:00:01Z",
        },
    )
    rows = store.list_sessions(status="completed")
    assert rows[0]["actual_dept"] == "骨科"
    print("[OK] export_data_ready")


if __name__ == "__main__":
    test_insert_and_get_in_progress()
    test_finalize_session()
    test_classify_reject()
    test_classify_emergency()
    test_classify_asking_returns_none()
    test_recorder_multi_turn_single_row()
    test_new_intake_finalizes_incomplete()
    test_export_data_ready()
    print("All triage recorder tests passed.")
