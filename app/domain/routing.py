from langchain_core.messages import HumanMessage

from app.core.logging import logger
from app.domain.models import AppState


def is_dept_followup_reply(state: AppState) -> bool:
    """Public helper for triage session recorder."""
    return _is_dept_followup_reply(state) or _is_dept_rules_followup(state) or _is_clarify_followup(state)


def _is_clarify_followup(state: AppState) -> bool:
    cs = state.clarify_state
    if not cs or getattr(cs, "status", None) != "asking":
        return False
    if not cs.last_choices:
        return False
    msgs = state.messages or []
    if len(msgs) < 2:
        return False
    return isinstance(msgs[-1], HumanMessage)


def _is_dept_rules_followup(state: AppState) -> bool:
    ds = state.dept_state
    if not ds or getattr(ds, "status", None) != "asking":
        return False
    if getattr(ds, "choice_mode", "accompany") != "differential":
        return False
    if not ds.last_choices:
        return False
    msgs = state.messages or []
    if len(msgs) < 2:
        return False
    return isinstance(msgs[-1], HumanMessage)


def _is_dept_followup_reply(state: AppState) -> bool:
    """科室反问待选（accompany 单选）：任意 HumanMessage 回复都继续消歧。"""
    ds = state.dept_state
    if not ds or getattr(ds, "status", None) != "asking":
        return False
    if getattr(ds, "choice_mode", "accompany") == "differential":
        return False
    if not ds.last_choices:
        return False
    msgs = state.messages or []
    if len(msgs) < 2:
        return False
    last = msgs[-1]
    return isinstance(last, HumanMessage)


def route_after_trim(state: AppState) -> str:
    if _is_clarify_followup(state):
        logger.info(">>> route_after_trim: continue symptom_clarify")
        return "symptom_clarify"
    if _is_dept_rules_followup(state):
        logger.info(">>> route_after_trim: continue dept_rules_disambiguation")
        return "dept_rules_disambiguation"
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


def route_after_rag(state: AppState) -> str:
    chunk = state.rag_chunk or {}
    if not chunk:
        logger.info(">>> route_after_rag: rag_miss_reject")
        return "rag_miss_reject"
    if chunk.get("type") == "symptomClarify":
        logger.info(">>> route_after_rag: symptom_clarify")
        return "symptom_clarify"
    return "dept_disambiguation"


def route_after_clarify(state: AppState) -> str:
    cs = state.clarify_state
    if cs and cs.status == "asking" and cs.last_choices:
        return "end_ask"
    if (
        cs
        and cs.dept_rule_chunk
        and not state.locked_department
        and state.dept_confidence_passed is not True
    ):
        logger.info(">>> route_after_clarify: dept_rules_disambiguation")
        return "dept_rules_disambiguation"
    if cs and cs.phase == "done" and state.dept_confidence_passed:
        logger.info(">>> route_after_clarify: answer_generate")
        return "answer_generate"
    return "end_ask"


def route_after_dept_rules(state: AppState) -> str:
    ds = state.dept_state
    status = getattr(ds, "status", None) if ds else None
    if status == "asking":
        return "end_ask"
    if state.locked_department:
        return "dept_confidence"
    return "end_ask"


def route_after_dept(state: AppState) -> str:
    ds = state.dept_state
    status = getattr(ds, "status", None) if ds else None
    logger.info(">>> route_after_dept: status=%s locked=%s", status, state.locked_department)
    if status == "asking":
        return "end_ask"
    if state.locked_department:
        return "dept_confidence"
    return "answer_generate"


def route_after_confidence(state: AppState) -> str:
    if not state.dept_confidence_passed:
        return "low_confidence_reject"
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
