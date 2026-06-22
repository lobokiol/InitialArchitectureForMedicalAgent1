from langchain_core.messages import HumanMessage

from app.core.logging import logger
from app.domain.models import AppState


def is_dept_followup_reply(state: AppState) -> bool:
    """Public helper for triage session recorder."""
    return _is_dept_followup_reply(state)


def _is_dept_followup_reply(state: AppState) -> bool:
    """科室反问待选：任意 HumanMessage 回复都继续消歧。"""
    ds = state.dept_state
    if not ds or getattr(ds, "status", None) != "asking":
        return False
    if not ds.last_choices:
        return False
    msgs = state.messages or []
    if len(msgs) < 2:
        return False
    last = msgs[-1]
    return isinstance(last, HumanMessage)


def route_after_trim(state: AppState) -> str:
    """多轮科室消歧：待选时跳过 decision；否则新一轮 intake。"""
    if _is_dept_followup_reply(state):
        logger.info(">>> route_after_trim: continue dept_disambiguation")
        return "dept_disambiguation"
    return "decision"


def route_after_slot_gate(state: AppState) -> str:
    if not state.slot_gate_passed:
        logger.info(">>> route_after_slot_gate: reject")
        return "reject"

    ir = state.intent_result
    route = ir.triage_route if ir else None
    logger.info(">>> route_after_slot_gate: triage_route=%s", route)
    if route == "disease":
        return "disease_dept"
    if route == "symptom":
        return "rag_symptom_recall"
    return "reject"


def route_after_dept(state: AppState) -> str:
    ds = state.dept_state
    status = getattr(ds, "status", None) if ds else None
    logger.info(">>> route_after_dept: status=%s locked=%s", status, state.locked_department)
    if status == "asking":
        return "end_ask"
    return "answer_generate"


def route_after_decision(state: AppState) -> str:
    """Legacy: kept for tests; production uses route_after_slot_gate."""
    ir = state.intent_result
    route = ir.triage_route if ir else None
    logger.info(">>> route_after_decision: %s", route)
    if route == "disease":
        return "disease_dept"
    if route == "symptom":
        return "symptom_slot"
    return "reject"
