from app.domain.dept_disambiguation import DeptChoice
from app.triage.multi_choice import parse_choice_indices, resolve_multi_choice


def _choices():
    return [
        DeptChoice(id="c1", label="发热、恶心呕吐", target_departments=["普外科"]),
        DeptChoice(id="c2", label="腹泻、腹胀", target_departments=["消化内科"]),
        DeptChoice(id="none", label="都没有", target_departments=[]),
    ]


def test_parse_choice_indices_comma():
    assert parse_choice_indices("1,3") == [0, 2]


def test_parse_choice_indices_space():
    assert parse_choice_indices("1 3") == [0, 2]


def test_parse_choice_indices_cn_comma():
    assert parse_choice_indices("1、3") == [0, 2]


def test_resolve_multi_choice_picks_two():
    picked, none_sel = resolve_multi_choice("1,2", _choices())
    assert len(picked) == 2
    assert not none_sel


def test_resolve_multi_choice_none_only():
    picked, none_sel = resolve_multi_choice("3", _choices())
    assert picked == []
    assert none_sel


def test_resolve_multi_choice_invalid():
    picked, none_sel = resolve_multi_choice("9", _choices())
    assert picked is None
    assert none_sel is False
