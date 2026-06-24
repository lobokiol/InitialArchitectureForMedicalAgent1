from app.domain.models import AppState
from app.domain.routing import route_after_rag, route_after_confidence, route_after_trim


def test_route_after_rag_symptom_clarify():
    state = AppState(rag_chunk={"type": "symptomClarify", "id": "CL0001"})
    assert route_after_rag(state) == "symptom_clarify"


def test_route_after_rag_symptom_rk():
    state = AppState(rag_chunk={"type": "symptom", "id": "RK0001"})
    assert route_after_rag(state) == "dept_disambiguation"


def test_route_after_confidence_fail():
    state = AppState(dept_confidence_passed=False, locked_department="普外科")
    assert route_after_confidence(state) == "low_confidence_reject"


def test_route_after_confidence_pass_no_red_flags():
    state = AppState(dept_confidence_passed=True, rag_chunk={"required_slots": ["age"]})
    assert route_after_confidence(state) == "answer_generate"
