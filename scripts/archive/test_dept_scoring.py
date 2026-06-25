"""Dept scoring unit tests."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.triage.dept_scoring import apply_negation_boosts, score_departments, try_lock_department

RK0001_DEPTS = [
    {"department": "骨科", "priority": 1, "condition": "外伤后踝部肿胀、活动痛；怀疑韧带损伤或骨折"},
    {"department": "风湿免疫科", "priority": 2, "condition": "双侧踝部反复肿胀，伴晨僵、多关节疼痛"},
    {"department": "血管外科", "priority": 3, "condition": "久站后肿胀加重，伴静脉曲张或下肢沉重感"},
]


def test_trauma_locks_orthopedics() -> None:
    scores = score_departments(RK0001_DEPTS, "脚脖子肿，昨天扭了", slot_trigger="扭")
    locked, dept, _ = try_lock_department(scores)
    assert locked and dept == "骨科", (scores, dept)
    print("[OK] trauma -> 骨科")


def test_negation_boosts_rheum() -> None:
    """Legacy behavior: without depts param, boosts rheum over ortho."""
    scores = {"骨科": 5.0, "风湿免疫科": 3.0, "血管外科": 2.0}
    out = apply_negation_boosts(scores, "都没有")
    assert out["风湿免疫科"] > out["骨科"]
    print("[OK] negation (legacy) -> 风湿免疫科 up")


def test_negation_with_depts_selects_highest_priority() -> None:
    """New behavior: with depts param, selects highest priority dept."""
    scores = score_departments(RK0001_DEPTS, "脚脖子肿")
    # Before negation: should NOT be locked (margin < 2.0)
    locked, _, margin = try_lock_department(scores)
    assert not locked, f"Should need disambiguation, scores={scores}"
    
    # Apply negation boost with depts
    out = apply_negation_boosts(scores, "都没有", RK0001_DEPTS)
    
    # Should now be locked to 骨科 (priority 1)
    locked, dept, margin = try_lock_department(out)
    assert locked and dept == "骨科", f"Expected 骨科, got {dept}, scores={out}"
    assert out["骨科"] > 10.0, f"骨科 should have high score, got {out['骨科']}"
    print("[OK] negation with depts -> highest priority (骨科)")


def main() -> None:
    test_trauma_locks_orthopedics()
    test_negation_boosts_rheum()
    test_negation_with_depts_selects_highest_priority()


if __name__ == "__main__":
    main()
