from __future__ import annotations

from langchain_core.messages import AIMessage

from app.core.logging import logger
from app.domain.dept_disambiguation import DeptDisambiguationState
from app.domain.models import AppState
from app.triage.dept_choices import (
    build_differential_choices,
    differential_selection_dicts,
    format_choice_message,
)
from app.triage.dept_rules_scoring import (
    accumulate_scores,
    apply_pediatric_boost,
    build_base_scores,
    filter_rule_by_sex,
    lock_department_from_totals,
)
from app.triage.multi_choice import resolve_multi_choice
from app.triage.turn_text import last_human_text

DIFFERENTIAL_QUESTION = "为更准确推荐科室，请选择您是否有以下情况（可多选，输入编号如 1,3）："
INVALID_MULTI_REPLY = "请从下列选项中选择（输入编号，可多选如 1,3）。"


def dept_rules_disambiguation_node(state: AppState) -> dict:
    logger.info(">>> Enter node: dept_rules_disambiguation")
    cs = state.clarify_state
    if not cs or not cs.dept_rule_chunk:
        return {}

    sex = cs.filled_slots.get("sex", "女")
    rule = filter_rule_by_sex(cs.dept_rule_chunk, sex)
    active_depts = list(rule.get("candidate_departments") or [])
    dept_state = state.dept_state

    if dept_state and dept_state.status == "asking" and dept_state.last_choices:
        reply = last_human_text(state)
        if reply:
            picked, none_selected = resolve_multi_choice(reply, dept_state.last_choices)
            if picked is None:
                reprompt = INVALID_MULTI_REPLY + "\n\n" + format_choice_message(dept_state.last_choices)
                reprompt = reprompt.replace(
                    "为更准确推荐科室，请选择您是否有以下情况：", DIFFERENTIAL_QUESTION
                )
                return {
                    "messages": [AIMessage(content=reprompt)],
                    "dept_state": dept_state.model_copy(deep=True),
                }
            selections = differential_selection_dicts(rule, picked or [])
            base = build_base_scores(active_depts)
            totals = accumulate_scores(base, selections, active_depts)
            age = cs.filled_slots.get("age")
            totals = apply_pediatric_boost(totals, age, active_depts)
            locked_dept, totals, margin, _tie = lock_department_from_totals(
                totals,
                rule.get("candidate_departments") or [],
                active_depts,
                none_selected=none_selected,
                age_label=age,
            )
            filled = dict(cs.filled_slots)
            if none_selected:
                filled["differential"] = "都没有"
            else:
                filled["differential"] = "、".join(c.label for c in picked or [])
            updated_cs = cs.model_copy(deep=True)
            updated_cs.filled_slots = filled
            return {
                "clarify_state": updated_cs,
                "locked_department": locked_dept,
                "dept_state": DeptDisambiguationState(
                    status="locked",
                    dept_scores=totals,
                    margin=margin,
                    candidate_departments=[{"department": d} for d in active_depts],
                    multi_select=True,
                    choice_mode="differential",
                ),
            }

    choices = build_differential_choices(rule)
    question = DIFFERENTIAL_QUESTION + "\n\n" + "\n".join(
        f"{i}. {c.label}" for i, c in enumerate(choices, 1)
    )
    return {
        "dept_state": DeptDisambiguationState(
            status="asking",
            last_question=DIFFERENTIAL_QUESTION,
            last_choices=choices,
            multi_select=True,
            choice_mode="differential",
            candidate_departments=[{"department": d} for d in active_depts],
        ),
        "messages": [AIMessage(content=question)],
    }
