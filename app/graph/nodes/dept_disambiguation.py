from __future__ import annotations

from langchain_core.messages import AIMessage

from app.core.logging import logger
from app.domain.dept_disambiguation import DeptDisambiguationState
from app.domain.models import AppState
from app.triage.dept_choices import (
    CHOICE_QUESTION_TEMPLATE,
    INVALID_CHOICE_REPLY,
    build_dept_choices,
    choice_score_boosts,
    format_choice_message,
    lock_department_for_explicit_choice,
    resolve_dept_choice,
)
from app.triage.dept_scoring import (
    apply_negation_boosts,
    fallback_department,
    score_departments,
    try_lock_department,
)
from app.triage.dept_state_patches import locked_department_patch
from app.triage.emergency_rules import match_emergency
from app.triage.turn_text import current_turn_text, last_human_text


def _is_emergency(chunk: dict, user_text: str) -> bool:
    return match_emergency(user_text) is not None


def dept_disambiguation_node(state: AppState) -> dict:
    logger.info(">>> Enter node: dept_disambiguation")
    chunk = state.rag_chunk
    table = state.slot_table
    if not chunk or not table:
        return {}

    user_text = current_turn_text(state)
    dept_state = state.dept_state
    current_round = dept_state.round if dept_state else 0
    depts = chunk.get("department_recommendation") or []

    if _is_emergency(chunk, user_text):
        return locked_department_patch("急诊", {}, 0.0, current_round, depts, "emergency")

    choice_boosts: dict[str, float] | None = None
    picked = None
    reply = ""
    if dept_state and dept_state.status == "asking" and dept_state.last_choices:
        reply = last_human_text(state)
        if reply:
            picked = resolve_dept_choice(reply, dept_state.last_choices)
            if picked is None:
                reprompt = INVALID_CHOICE_REPLY + "\n\n" + format_choice_message(dept_state.last_choices)
                return {
                    "messages": [AIMessage(content=reprompt)],
                    "dept_state": dept_state.model_copy(deep=True),
                }
            choice_boosts = choice_score_boosts(picked)
            if picked.id == "none":
                reply = picked.label

    scores = score_departments(
        depts,
        user_text,
        chunk.get("accompanying_symptom_keywords"),
        table.trigger,
        table.emergency,
        llm_boosts=choice_boosts,
    )
    if picked and picked.id != "none":
        explicit_dept = lock_department_for_explicit_choice(picked, scores)
        if explicit_dept:
            top_score = scores.get(explicit_dept, 0.0)
            others = [v for k, v in scores.items() if k != explicit_dept]
            margin = top_score - max(others, default=0.0)
            logger.info(
                "dept_disambiguation: explicit choice %r -> lock %s",
                picked.label,
                explicit_dept,
            )
            return locked_department_patch(
                explicit_dept, scores, margin, current_round, depts, "locked"
            )
    if reply and choice_boosts is not None and not choice_boosts:
        scores = apply_negation_boosts(scores, reply, depts)
    locked, dept, margin = try_lock_department(scores)

    if locked and dept:
        return locked_department_patch(dept, scores, margin, current_round, depts, "locked")

    if current_round >= 3:
        fb = fallback_department(depts)
        return locked_department_patch(fb, scores, margin, current_round, depts, "fallback")

    next_round = current_round + 1
    asked_ids = list(dept_state.asked_choice_ids if dept_state else [])
    choices, has_symptom = build_dept_choices(
        chunk,
        round_num=next_round,
        asked_choice_ids=asked_ids,
    )
    if not has_symptom:
        fb = fallback_department(depts)
        return locked_department_patch(fb, scores, margin, next_round, depts, "fallback")

    asked_ids.extend(c.id for c in choices if c.id != "none")
    question = format_choice_message(choices)
    return {
        "dept_state": DeptDisambiguationState(
            status="asking",
            dept_scores=scores,
            margin=margin,
            round=next_round,
            last_question=CHOICE_QUESTION_TEMPLATE,
            last_choices=choices,
            asked_choice_ids=asked_ids,
            candidate_departments=depts,
        ),
        "messages": [AIMessage(content=question)],
    }
