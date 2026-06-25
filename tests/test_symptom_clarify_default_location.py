from langchain_core.messages import HumanMessage

from app.domain.models import AppState
from app.domain.routing import route_after_clarify
from app.domain.symptom_clarify import ClarifyChoice, SymptomClarifyState
from app.graph.nodes.symptom_clarify import symptom_clarify_node


def _eye_chunk() -> dict:
    return {
        "id": "CL0010",
        "symptom_id": "眼睛不适",
        "aliases": ["眼睛疼"],
        "required_slots": ["age", "sex"],
        "default_location": "眼睛",
        "questions": {
            "age": {"text": "请问您的年龄？", "options": ["19-35岁"]},
            "sex": {"text": "请问您的性别？", "options": ["男", "女"]},
        },
        "type": "symptomClarify",
    }


def test_default_location_binds_dept_rule_after_sex():
    chunk = _eye_chunk()
    cs = SymptomClarifyState(
        status="asking",
        symptom_id="眼睛不适",
        phase="sex",
        filled_slots={"age": "19-35岁"},
        last_question="请问您的性别？",
        last_choices=[ClarifyChoice(id="c1", label="男", slot="sex")],
    )
    state = AppState(
        rag_chunk=chunk,
        clarify_state=cs,
        messages=[HumanMessage(content="男")],
    )
    out = symptom_clarify_node(state)
    updated = out["clarify_state"]
    assert updated.status == "done"
    assert updated.dept_rule_chunk is not None
    assert updated.dept_rule_chunk.get("symptom_id") == "眼睛不适"
    assert updated.dept_rule_chunk.get("location") == "眼睛"
    assert updated.filled_slots["pain_location"] == "眼睛"

    state.clarify_state = updated
    assert route_after_clarify(state) == "dept_rules_disambiguation"
