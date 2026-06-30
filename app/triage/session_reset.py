"""Clear triage state on same thread (keep messages)."""


def triage_state_reset_patch() -> dict:
    return {
        "slot_table": None,
        "slot_gate_passed": False,
        "ner_result": None,
        "intent_result": None,
        "rag_chunk_id": None,
        "rag_chunk": None,
        "dept_state": None,
        "locked_department": None,
        "clarify_state": None,
        "dept_confidence_result": None,
        "dept_confidence_passed": None,
        "emergency_gate_passed": None,
        "emergency_match": None,
        "disease_dept_result": None,
        "symptom_slot_result": None,
        "oncall_appointments": [],
        "oncall_fetch_error": None,
        "tool_call_result": None,
        "last_recommended_department": None,
    }
