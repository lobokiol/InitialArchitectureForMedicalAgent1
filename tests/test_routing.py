"""Tests for graph routing and pre-invoke triage follow-up detection."""
from __future__ import annotations

import sys
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.domain.dept_disambiguation import DeptChoice, DeptDisambiguationState
from app.domain.models import AppState
from app.domain.routing import is_awaiting_triage_followup, is_dept_followup_reply
from app.domain.symptom_clarify import ClarifyChoice, SymptomClarifyState


def _post_clarify_state() -> AppState:
    return AppState(
        messages=[
            HumanMessage(content="我眼睛疼"),
            AIMessage(content="请问您的年龄？\n\n1. 19-35岁"),
        ],
        clarify_state=SymptomClarifyState(
            status="asking",
            phase="age",
            last_choices=[ClarifyChoice(id="c1", label="19-35岁", slot="age")],
        ),
    )


def test_awaiting_clarify_pre_invoke() -> None:
    state = _post_clarify_state()
    assert is_awaiting_triage_followup(state) is True
    assert is_dept_followup_reply(state) is False
    print("[OK] awaiting_clarify_pre_invoke")


def test_awaiting_dept_pre_invoke() -> None:
    state = AppState(
        messages=[
            HumanMessage(content="脚后跟疼"),
            AIMessage(content="请选择伴随症状\n\n1. A\n2. B"),
        ],
        dept_state=DeptDisambiguationState(
            status="asking",
            choice_mode="accompany",
            last_choices=[DeptChoice(id="a", label="A"), DeptChoice(id="b", label="B")],
        ),
    )
    assert is_awaiting_triage_followup(state) is True
    assert is_dept_followup_reply(state) is False
    print("[OK] awaiting_dept_pre_invoke")


def test_awaiting_dept_rules_pre_invoke() -> None:
    state = AppState(
        messages=[
            HumanMessage(content="我左下肢疼"),
            AIMessage(content="请选择鉴别选项\n\n1. 深静脉\n2. 动脉"),
        ],
        dept_state=DeptDisambiguationState(
            status="asking",
            choice_mode="differential",
            last_choices=[DeptChoice(id="a", label="深静脉"), DeptChoice(id="b", label="动脉")],
        ),
    )
    assert is_awaiting_triage_followup(state) is True
    assert is_dept_followup_reply(state) is False
    print("[OK] awaiting_dept_rules_pre_invoke")


def test_not_awaiting_fresh_intake() -> None:
    state = AppState(messages=[HumanMessage(content="我有胃炎")])
    assert is_awaiting_triage_followup(state) is False
    assert is_dept_followup_reply(state) is False
    print("[OK] not_awaiting_fresh_intake")


def test_dept_followup_post_trim() -> None:
    state = AppState(
        messages=[
            HumanMessage(content="脚后跟疼"),
            AIMessage(content="请选择"),
            HumanMessage(content="A"),
        ],
        dept_state=DeptDisambiguationState(
            status="asking",
            choice_mode="accompany",
            last_choices=[DeptChoice(id="a", label="A")],
        ),
    )
    assert is_awaiting_triage_followup(state) is True
    assert is_dept_followup_reply(state) is True
    print("[OK] dept_followup_post_trim")


if __name__ == "__main__":
    test_awaiting_clarify_pre_invoke()
    test_awaiting_dept_pre_invoke()
    test_awaiting_dept_rules_pre_invoke()
    test_not_awaiting_fresh_intake()
    test_dept_followup_post_trim()
    print("All routing tests passed.")
