from __future__ import annotations

from langchain_core.messages import AIMessage

from app.core.logging import logger
from app.domain.models import AppState
from app.infra.opensearch_dept_rules import search_dept_rule
from app.triage.clarify_flow import (
    ask_slot,
    as_dept_choices,
    bind_dept_rule,
    finish_with_default_location,
    init_clarify,
    sync_slot_table,
)
from app.triage.clarify_helpers import format_clarify_message, next_slot_phase
from app.triage.dept_choices import resolve_dept_choice
from app.triage.turn_text import last_human_text


def symptom_clarify_node(state: AppState) -> dict:
    logger.info(">>> Enter node: symptom_clarify phase=%s", getattr(state.clarify_state, "phase", None))
    chunk = state.rag_chunk
    if not chunk or chunk.get("type") != "symptomClarify":
        return {}

    cs = state.clarify_state or init_clarify(chunk)
    required = chunk.get("required_slots") or []

    if cs.phase in ("age", "sex", "pain_location") and cs.last_choices:
        reply = last_human_text(state)
        if not reply:
            return ask_slot(cs, chunk, cs.phase)
        picked = resolve_dept_choice(reply, as_dept_choices(cs.last_choices))
        if picked is None:
            msg = "请从下列选项中选择（输入选项文字或编号）。\n\n" + format_clarify_message(
                cs.last_question or "", cs.last_choices
            )
            return {"messages": [AIMessage(content=msg)], "clarify_state": cs.model_copy(deep=True)}

        filled = dict(cs.filled_slots)
        filled[cs.phase] = picked.label
        sync_slot_table(state, cs.phase, picked.label)

        if cs.phase == "pain_location":
            sid = cs.symptom_id or chunk.get("symptom_id") or ""
            rule = search_dept_rule(sid, picked.label)
            if not rule:
                return {
                    "messages": [AIMessage(content=f"暂无「{picked.label}」对应的导诊规则，请换部位或联系分诊台。")],
                }
            updated = bind_dept_rule(cs, chunk, filled, picked.label)
            if updated is None:
                return {
                    "messages": [AIMessage(content=f"暂无「{picked.label}」对应的导诊规则，请换部位或联系分诊台。")],
                }
            return {"clarify_state": updated}

        nxt = next_slot_phase(cs.phase, required)
        updated = cs.model_copy(deep=True)
        updated.filled_slots = filled
        updated.last_choices = []
        if nxt:
            updated.phase = nxt  # type: ignore[assignment]
            return ask_slot(updated, chunk, nxt)
        auto = finish_with_default_location(updated, chunk, filled)
        if auto:
            return auto
        updated.status = "done"
        updated.phase = "done"
        return {"clarify_state": updated}

    # first ask for current phase
    if cs.phase in ("age", "sex", "pain_location"):
        return ask_slot(cs, chunk, cs.phase)

    return {}
