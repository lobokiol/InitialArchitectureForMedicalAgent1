from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage

from app.core.logging import logger
from app.domain.dept_disambiguation import DeptChoice, DeptDisambiguationState
from app.domain.models import AppState
from app.triage.dept_choices import NONE_CHOICE_ID, format_choice_message, resolve_dept_choice
from app.triage.dept_rules_scoring import (
    accumulate_scores,
    build_base_scores,
    filter_rule_by_sex,
    lock_department_from_totals,
)
from app.triage.multi_choice import resolve_multi_choice
from app.triage.turn_text import current_turn_text

DIFFERENTIAL_QUESTION = "为更准确推荐科室，请选择您是否有以下情况（可多选，输入编号如 1,3）："
INVALID_MULTI_REPLY = "请从下列选项中选择（输入编号，可多选如 1,3）。"


def _last_human(state: AppState) -> str:
    for msg in reversed(state.messages):
        if isinstance(msg, HumanMessage) and isinstance(msg.content, str):
            return msg.content.strip()
    return ""


def build_differential_choices(rule_chunk: dict) -> list[DeptChoice]:
    choices: list[DeptChoice] = []
    for i, q in enumerate(rule_chunk.get("differential_questions") or [], 1):
        choices.append(
            DeptChoice(
                id=f"c{i}",
                label=q["text"],
                target_departments=list((q.get("scores") or {}).keys()),
            )
        )
    choices.append(DeptChoice(id=NONE_CHOICE_ID, label="都没有", target_departments=[]))
    return choices


def _selection_dicts(rule_chunk: dict, picked: list[DeptChoice]) -> list[dict]:
    questions = rule_chunk.get("differential_questions") or []
    out: list[dict] = []
    for c in picked:
        if c.id.startswith("c") and c.id[1:].isdigit():
            idx = int(c.id[1:]) - 1
            if 0 <= idx < len(questions):
                out.append(questions[idx])
    return out


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
        reply = _last_human(state)
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
            selections = _selection_dicts(rule, picked or [])
            base = build_base_scores(active_depts)
            totals = accumulate_scores(base, selections, active_depts)
            locked_dept, totals, margin, _tie = lock_department_from_totals(
                totals,
                rule.get("candidate_departments") or [],
                active_depts,
                none_selected=none_selected,
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
