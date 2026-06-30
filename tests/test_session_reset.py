from app.domain.models import OnCallDoctor
from app.triage.session_reset import triage_state_reset_patch


def test_reset_clears_oncall_fields():
    patch = triage_state_reset_patch()
    assert patch["oncall_appointments"] == []
    assert patch["oncall_fetch_error"] is None
    assert patch["tool_call_result"] is None
    assert patch["last_recommended_department"] is None
    assert patch["locked_department"] is None


def test_fetch_oncall_clears_when_not_triggered():
    from app.graph.nodes.fetch_oncall import fetch_oncall_node
    from app.domain.models import AppState

    state = AppState(
        locked_department="骨科",
        oncall_appointments=[OnCallDoctor(name="张医生", time="14:00", slots=3)],
        oncall_fetch_error="旧错误",
    )
    result = fetch_oncall_node(state)
    assert result["oncall_appointments"] == []
    assert result["oncall_fetch_error"] is None
