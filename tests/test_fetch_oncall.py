from app.domain.models import AppState, DiseaseDeptResult, OnCallDoctor


def test_oncall_doctor_model():
    doc = OnCallDoctor(name="张医生", time="14:00-18:00", slots=3)
    assert doc.name == "张医生"
    assert doc.slots == 3


def test_should_fetch_symptom_chain():
    from app.graph.nodes.fetch_oncall import resolve_department, should_fetch

    state = AppState(locked_department="骨科", dept_confidence_passed=True)
    assert should_fetch(state) is True
    assert resolve_department(state) == "骨科"


def test_should_fetch_skips_emergency():
    from app.graph.nodes.fetch_oncall import should_fetch

    state = AppState(locked_department="急诊", dept_confidence_passed=True)
    assert should_fetch(state) is False


def test_should_fetch_disease_chain():
    from app.graph.nodes.fetch_oncall import should_fetch

    state = AppState(
        disease_dept_result=DiseaseDeptResult(
            diseases=["骨折"],
            departments=[{"dept": "骨科"}],
        )
    )
    assert should_fetch(state) is True


def test_resolve_department_disease_first():
    from app.graph.nodes.fetch_oncall import resolve_department

    state = AppState(
        disease_dept_result=DiseaseDeptResult(
            diseases=["骨折"],
            departments=[{"dept": "骨科"}, {"dept": "康复科"}],
        )
    )
    assert resolve_department(state) == "骨科"


import pytest
from app.core import config
from app.mcp.client import fetch_oncall_appointments_sync


@pytest.mark.skipif(not config.MCP_ENABLED, reason="MCP disabled")
def test_mcp_client_returns_three_doctors():
    doctors = fetch_oncall_appointments_sync("骨科")
    assert len(doctors) == 3
    assert doctors[0].name.startswith("骨科")
