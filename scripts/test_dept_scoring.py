"""Dept scoring unit tests."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
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
    scores = {"骨科": 5.0, "风湿免疫科": 3.0, "血管外科": 2.0}
    out = apply_negation_boosts(scores, "都没有")
    assert out["风湿免疫科"] > out["骨科"]
    print("[OK] negation -> 风湿免疫科 up")


def main() -> None:
    test_trauma_locks_orthopedics()
    test_negation_boosts_rheum()


if __name__ == "__main__":
    main()
