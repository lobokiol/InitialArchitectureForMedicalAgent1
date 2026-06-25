"""Unit tests for rule-based dept choice menus."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from langchain_core.messages import AIMessage, HumanMessage

from app.domain.dept_disambiguation import DeptChoice, DeptDisambiguationState
from app.domain.models import AppState
from app.domain.routing import route_after_trim
from app.triage.dept_scoring import score_departments
from app.triage.dept_choices import (
    CHOICE_QUESTION_TEMPLATE,
    build_dept_choices,
    choice_score_boosts,
    format_choice_message,
    lock_department_for_explicit_choice,
    resolve_dept_choice,
)

RK0001 = json.loads(
    Path("sourceData/data/rag_knowledge.jsonl").read_text(encoding="utf-8").splitlines()[0]
)


def test_build_choices_excludes_emergency_terms() -> None:
    choices, has_symptom = build_dept_choices(RK0001, round_num=1, asked_choice_ids=[])
    labels = [c.label for c in choices]
    assert "都没有" in labels
    assert not any("畸形" in lb or "不能负重" in lb for lb in labels)
    assert has_symptom or len([c for c in choices if c.id != "none"]) == 0


def test_build_choices_includes_discriminative_keyword() -> None:
    choices, has_symptom = build_dept_choices(RK0001, round_num=1, asked_choice_ids=[])
    labels = [c.label for c in choices if c.id != "none"]
    if has_symptom:
        assert any("晨僵" in lb or "扭伤" in lb or "肿胀" in lb for lb in labels)


def test_resolve_by_label_and_alias() -> None:
    choices, _ = build_dept_choices(RK0001, round_num=1, asked_choice_ids=[])
    pick = next(c for c in choices if c.id != "none")
    assert resolve_dept_choice(pick.label, choices) == pick
    if "晨僵" in pick.label:
        assert resolve_dept_choice("晨僵", choices) is not None


def test_resolve_none_for_invalid() -> None:
    choices, _ = build_dept_choices(RK0001, round_num=1, asked_choice_ids=[])
    assert resolve_dept_choice("我还行吧", choices) is None


def test_choice_score_boosts_none_uses_negation() -> None:
    choices, _ = build_dept_choices(RK0001, round_num=1, asked_choice_ids=[])
    none_choice = next(c for c in choices if c.id == "none")
    boosts = choice_score_boosts(none_choice)
    assert boosts == {}


def test_format_message_lists_options() -> None:
    choices, _ = build_dept_choices(RK0001, round_num=1, asked_choice_ids=[])
    msg = format_choice_message(choices)
    assert CHOICE_QUESTION_TEMPLATE in msg
    assert "都没有" in msg


def test_explicit_choice_locks_rheumatology_for_chenjiang() -> None:
    depts = RK0001["department_recommendation"]
    choices, _ = build_dept_choices(RK0001, round_num=1, asked_choice_ids=[])
    chen = next(c for c in choices if c.label == "晨僵")
    scores = score_departments(
        depts,
        "脚脖子肿",
        RK0001["accompanying_symptom_keywords"],
        llm_boosts=choice_score_boosts(chen),
    )
    locked = lock_department_for_explicit_choice(chen, scores)
    assert locked == "风湿免疫科"


def test_route_asking_goes_to_dept_disambiguation() -> None:
    state = AppState(
        messages=[
            AIMessage(content="为更准确推荐科室..."),
            HumanMessage(content="我还行吧"),
        ],
        dept_state=DeptDisambiguationState(
            status="asking",
            last_question="为更准确推荐科室，请选择您是否有以下情况：",
            last_choices=[DeptChoice(id="none", label="都没有")],
        ),
    )
    assert route_after_trim(state) == "dept_disambiguation"


def main() -> None:
    test_build_choices_excludes_emergency_terms()
    test_build_choices_includes_discriminative_keyword()
    test_resolve_by_label_and_alias()
    test_resolve_none_for_invalid()
    test_choice_score_boosts_none_uses_negation()
    test_format_message_lists_options()
    test_explicit_choice_locks_rheumatology_for_chenjiang()
    test_route_asking_goes_to_dept_disambiguation()
    print("[OK] all dept_choices tests passed")


if __name__ == "__main__":
    main()
