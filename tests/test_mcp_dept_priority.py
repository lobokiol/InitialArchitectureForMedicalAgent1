from langchain_core.messages import HumanMessage

from app.domain.models import AppState
from app.mcp.followup import resolve_followup_department, resolve_recommended_department


def test_explicit_dept_overrides_last_recommended_for_mcp():
    state = AppState(
        last_recommended_department="消化内科",
        locked_department="消化内科",
        messages=[HumanMessage(content="骨科怎么走？")],
    )
    assert resolve_followup_department(state) == "骨科"


def test_recommended_department_prefers_user_input():
    state = AppState(
        last_recommended_department="消化内科",
        locked_department="消化内科",
        dept_confidence_passed=True,
        messages=[HumanMessage(content="骨科怎么走？")],
    )
    assert resolve_recommended_department(state) == "骨科"


def test_recommended_department_falls_back_to_triage():
    state = AppState(
        last_recommended_department="消化内科",
        locked_department="消化内科",
        dept_confidence_passed=True,
        messages=[HumanMessage(content="怎么走？")],
    )
    assert resolve_recommended_department(state) == "消化内科"
