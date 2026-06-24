import json
from pathlib import Path

from app.triage.dept_rules_scoring import (
    accumulate_scores,
    build_base_scores,
    filter_rule_by_sex,
    lock_department_from_totals,
)

RK0025 = json.loads(Path("demo/data/rag_department_rules.jsonl").read_text(encoding="utf-8"))


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
