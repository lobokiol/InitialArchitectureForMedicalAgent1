from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage

from app.core.logging import logger
from app.domain.dept_disambiguation import DeptChoice
from app.domain.models import AppState
from app.domain.symptom_clarify import SymptomClarifyState
from app.infra.opensearch_dept_rules import search_dept_rule
from app.triage.clarify_helpers import (
    choices_for_slot,
    format_clarify_message,
    next_slot_phase,
    ordered_required_slots,
)
from app.triage.dept_choices import resolve_dept_choice


def _last_human(state: AppState) -> str:
    for msg in reversed(state.messages):
        if isinstance(msg, HumanMessage) and isinstance(msg.content, str):
            return msg.content.strip()
    return ""


def _as_dept_choices(choices) -> list[DeptChoice]:
    return [DeptChoice(id=c.id, label=c.label, target_departments=[]) for c in choices]


def _init_clarify(chunk: dict) -> SymptomClarifyState:
    required = chunk.get("required_slots") or []
    slots = ordered_required_slots(required)
    phase = slots[0] if slots else "done"
    return SymptomClarifyState(
        status="asking",
        clarify_chunk_id=chunk.get("id"),
        symptom_id=chunk.get("symptom_id"),
        phase=phase,  # type: ignore[arg-type]
    )


def _sync_slot_table(state: AppState, slot: str, value: str) -> None:
    table = state.slot_table
    if not table:
        return
    if slot == "age":
        table.age = value
    elif slot == "sex":
        table.gender = value


def _ask_slot(cs: SymptomClarifyState, chunk: dict, slot: str) -> dict:
    text, choices = choices_for_slot(chunk, slot)
    asking = cs.model_copy(deep=True)
    asking.phase = slot  # type: ignore[assignment]
    asking.status = "asking"
    asking.last_question = text
    asking.last_choices = choices
    return {
        "clarify_state": asking,
        "messages": [AIMessage(content=format_clarify_message(text, choices))],
    }


def symptom_clarify_node(state: AppState) -> dict:
    logger.info(">>> Enter node: symptom_clarify phase=%s", getattr(state.clarify_state, "phase", None))
    chunk = state.rag_chunk
    if not chunk or chunk.get("type") != "symptomClarify":
        return {}

    cs = state.clarify_state or _init_clarify(chunk)
    required = chunk.get("required_slots") or []

    if cs.phase in ("age", "sex", "pain_location") and cs.last_choices:
        reply = _last_human(state)
        if not reply:
            return _ask_slot(cs, chunk, cs.phase)
        picked = resolve_dept_choice(reply, _as_dept_choices(cs.last_choices))
        if picked is None:
            msg = "请从下列选项中选择（输入选项文字或编号）。\n\n" + format_clarify_message(
                cs.last_question or "", cs.last_choices
            )
            return {"messages": [AIMessage(content=msg)], "clarify_state": cs.model_copy(deep=True)}

        filled = dict(cs.filled_slots)
        filled[cs.phase] = picked.label
        _sync_slot_table(state, cs.phase, picked.label)

        if cs.phase == "pain_location":
            sid = cs.symptom_id or chunk.get("symptom_id") or ""
            rule = search_dept_rule(sid, picked.label)
            if not rule:
                return {
                    "messages": [AIMessage(content=f"暂无「{picked.label}」对应的导诊规则，请换部位或联系分诊台。")],
                }
            updated = cs.model_copy(deep=True)
            updated.filled_slots = filled
            updated.dept_rule_id = rule.get("id")
            updated.dept_rule_chunk = rule
            updated.last_choices = []
            updated.status = "done"
            updated.phase = "done"
            return {"clarify_state": updated}

        nxt = next_slot_phase(cs.phase, required)
        updated = cs.model_copy(deep=True)
        updated.filled_slots = filled
        updated.last_choices = []
        if nxt:
            updated.phase = nxt  # type: ignore[assignment]
            return _ask_slot(updated, chunk, nxt)
        updated.status = "done"
        updated.phase = "done"
        return {"clarify_state": updated}

    # first ask for current phase
    if cs.phase in ("age", "sex", "pain_location"):
        return _ask_slot(cs, chunk, cs.phase)

    return {}
