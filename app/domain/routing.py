from langchain_core.messages import HumanMessage

from app.core.logging import logger
from app.domain.models import AppState


def is_awaiting_triage_followup(state: AppState) -> bool:
    """Pre-invoke: system is waiting for user to reply to clarify or dept choices."""
    cs = state.clarify_state
    if cs and getattr(cs, "status", None) == "asking" and cs.last_choices:
        return True
    ds = state.dept_state
    if ds and getattr(ds, "status", None) == "asking" and ds.last_choices:
        return True
    return False


def is_dept_followup_reply(state: AppState) -> bool:
    """Post-trim: user message is a reply to clarify or dept choices."""
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


def _last_human_text(state: AppState) -> str:
    msgs = state.messages or []
    for msg in reversed(msgs):
        if isinstance(msg, HumanMessage) and isinstance(msg.content, str):
            return msg.content.strip()
    return ""


def is_mcp_followup_reply(state: AppState) -> bool:
    from app.core import config
    from app.mcp.followup import (
        looks_like_dept_info_query,
        looks_like_new_triage,
        match_department_in_text,
    )

    if not config.MCP_ENABLED or not config.MCP_FOLLOWUP_ENABLED:
        return False
    if is_dept_followup_reply(state):
        return False
    text = _last_human_text(state)
    if not text:
        return False
    if looks_like_new_triage(text) and not looks_like_dept_info_query(text):
        return False
    explicit_dept = match_department_in_text(text)
    if looks_like_dept_info_query(text) and explicit_dept:
        return True
    if not state.last_recommended_department:
        return False
    return looks_like_dept_info_query(text) or bool(explicit_dept)


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
    if is_mcp_followup_reply(state):
        logger.info(">>> route_after_trim: mcp_followup")
        return "mcp_followup"
    return "decision"


def route_after_emergency_gate(state: AppState) -> str:
    if state.emergency_gate_passed is False and state.locked_department == "急诊":
        logger.info(">>> route_after_emergency_gate: emergency -> answer_generate")
        return "answer_generate"
    logger.info(">>> route_after_emergency_gate: pass -> slot_gate")
    return "slot_gate"


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
        return "rag_symptom_recall"
    return "reject"
