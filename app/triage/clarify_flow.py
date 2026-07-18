"""Pure helpers for symptom clarify slot flow."""

from __future__ import annotations

from langchain_core.messages import AIMessage

from app.core.logging import logger
from app.domain.dept_disambiguation import DeptChoice
from app.domain.models import AppState
from app.domain.symptom_clarify import SymptomClarifyState
from app.infra.opensearch_dept_rules import search_dept_rule
from app.triage.clarify_helpers import (
    choices_for_slot,
    format_clarify_message,
    ordered_required_slots,
)


def as_dept_choices(choices) -> list[DeptChoice]:
    return [DeptChoice(id=c.id, label=c.label, target_departments=[]) for c in choices]


def init_clarify(chunk: dict) -> SymptomClarifyState:
    required = chunk.get("required_slots") or []
    slots = ordered_required_slots(required)
    phase = slots[0] if slots else "done"
    return SymptomClarifyState(
        status="asking",
        clarify_chunk_id=chunk.get("id"),
        symptom_id=chunk.get("symptom_id"),
        phase=phase,  # type: ignore[arg-type]
    )


def sync_slot_table(state: AppState, slot: str, value: str) -> None:
    table = state.slot_table
    if not table:
        return
    if slot == "age":
        table.age = value
    elif slot == "sex":
        table.gender = value


def ask_slot(cs: SymptomClarifyState, chunk: dict, slot: str) -> dict:
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


def bind_dept_rule(
    cs: SymptomClarifyState,
    chunk: dict,
    filled: dict[str, str],
    location: str,
) -> SymptomClarifyState | None:
    sid = cs.symptom_id or chunk.get("symptom_id") or ""
    rule = search_dept_rule(sid, location)
    if not rule:
        return None
    slots = dict(filled)
    slots.setdefault("pain_location", location)
    updated = cs.model_copy(deep=True)
    updated.filled_slots = slots
    updated.dept_rule_id = rule.get("id")
    updated.dept_rule_chunk = rule
    updated.last_choices = []
    updated.status = "done"
    updated.phase = "done"
    return updated


def finish_with_default_location(
    cs: SymptomClarifyState,
    chunk: dict,
    filled: dict[str, str],
) -> dict | None:
    loc = chunk.get("default_location")
    if not isinstance(loc, str) or not loc.strip():
        return None
    location = loc.strip()
    updated = bind_dept_rule(cs, chunk, filled, location)
    if updated is None:
        return {
            "messages": [AIMessage(content=f"暂无「{location}」对应的导诊规则，请联系分诊台。")],
        }
    logger.info(
        "symptom_clarify: default_location=%r bound rule_id=%s",
        location,
        updated.dept_rule_id,
    )
    return {"clarify_state": updated}
