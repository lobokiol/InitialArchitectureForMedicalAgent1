import json
from pathlib import Path

from app.triage.dept_rules_scoring import (
    PEDIATRIC_BOOST,
    PEDIATRIC_DEPT,
    accumulate_scores,
    apply_pediatric_boost,
    build_base_scores,
    filter_rule_by_sex,
    is_pediatric_age,
    lock_department_from_totals,
)


def _load_rule(rid: str) -> dict:
    for line in Path("sourceData/data/rag_department_rules.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            doc = json.loads(line)
            if doc.get("id") == rid:
                return doc
    raise KeyError(rid)


RK0025 = _load_rule("RK0025")


def test_filter_rule_by_sex_male_removes_gynecology():
    filtered = filter_rule_by_sex(RK0025, "男")
    assert "妇科" not in filtered["candidate_departments"]
    for q in filtered["differential_questions"]:
        scores = set((q.get("scores") or {}).keys())
        assert scores != {"妇科"}


def test_build_base_scores_order():
    scores = build_base_scores(["消化内科", "普外科", "泌尿外科"])
    assert scores["消化内科"] > scores["普外科"] > scores["泌尿外科"]


def test_lock_none_selected_fallback_first_in_order():
    active = ["消化内科", "普外科", "泌尿外科"]
    base = build_base_scores(active)
    dept, totals, margin, tie = lock_department_from_totals(
        base, RK0025["candidate_departments"], active, none_selected=True
    )
    assert dept == "消化内科"
    assert not tie


def test_accumulate_scores_adds_selection():
    active = ["消化内科", "普外科", "泌尿外科"]
    base = build_base_scores(active)
    sel = {"text": "发热、恶心呕吐", "scores": {"普外科": 4}}
    totals = accumulate_scores(base, [sel], active)
    assert totals["普外科"] > base["普外科"]


def test_is_pediatric_age_buckets():
    assert is_pediatric_age("5-11岁")
    assert is_pediatric_age("2-4岁")
    assert not is_pediatric_age("12-18岁")
    assert not is_pediatric_age(None)


def test_apply_pediatric_boost_only_when_candidate():
    active = ["消化内科", "普外科", "儿科"]
    base = build_base_scores(active)
    boosted = apply_pediatric_boost(base, "5-11岁", active)
    assert boosted[PEDIATRIC_DEPT] == base[PEDIATRIC_DEPT] + PEDIATRIC_BOOST
    unchanged = apply_pediatric_boost(base, "5-11岁", ["消化内科", "普外科"])
    assert unchanged == base
    assert apply_pediatric_boost(base, "12-18岁", active) == base


def test_pediatric_boost_wins_on_none_selected():
    active = ["消化内科", "普外科", "妇科", "儿科"]
    rule = _load_rule("RK0029")
    base = build_base_scores(active)
    totals = apply_pediatric_boost(base, "5-11岁", active)
    dept, _, _, _ = lock_department_from_totals(
        totals, rule["candidate_departments"], active, none_selected=True, age_label="5-11岁"
    )
    assert dept == PEDIATRIC_DEPT


def test_pediatric_tie_break_prefers_pediatrics():
    active = ["消化内科", "儿科"]
    base = {"消化内科": 3.0, "儿科": 3.0}
    dept, _, _, tie = lock_department_from_totals(
        base, ["消化内科", "儿科"], active, none_selected=True, age_label="2-4岁"
    )
    assert tie
    assert dept == PEDIATRIC_DEPT
