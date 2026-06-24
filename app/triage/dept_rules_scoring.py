from __future__ import annotations

from app.triage.dept_scoring import try_lock_department

GYNECOLOGY_DEPT = "妇科"
PEDIATRIC_DEPT = "儿科"
PEDIATRIC_BOOST = 3.0
PEDIATRIC_AGE_BUCKETS = frozenset({
    "0-3个月",
    "3个月-1岁",
    "2-4岁",
    "5-11岁",
})


def is_pediatric_age(age_label: str | None) -> bool:
    return bool(age_label and age_label.strip() in PEDIATRIC_AGE_BUCKETS)


def apply_pediatric_boost(
    totals: dict[str, float],
    age_label: str | None,
    active_depts: list[str],
) -> dict[str, float]:
    if not is_pediatric_age(age_label) or PEDIATRIC_DEPT not in active_depts:
        return totals
    out = dict(totals)
    out[PEDIATRIC_DEPT] = out.get(PEDIATRIC_DEPT, 0.0) + PEDIATRIC_BOOST
    return out


def filter_rule_by_sex(chunk: dict, sex: str) -> dict:
    out = dict(chunk)
    depts = list(chunk.get("candidate_departments") or [])
    questions = list(chunk.get("differential_questions") or [])
    if sex.strip() == "男":
        depts = [d for d in depts if d != GYNECOLOGY_DEPT]
        questions = [
            q
            for q in questions
            if not (set((q.get("scores") or {}).keys()) == {GYNECOLOGY_DEPT})
        ]
    out["candidate_departments"] = depts
    out["differential_questions"] = questions
    return out


def build_base_scores(candidate_departments: list[str]) -> dict[str, float]:
    n = len(candidate_departments)
    return {dept: float(n - i) for i, dept in enumerate(candidate_departments)}


def accumulate_scores(
    base: dict[str, float],
    selections: list[dict],
    active_depts: list[str],
) -> dict[str, float]:
    totals = dict(base)
    for sel in selections:
        for dept, pts in (sel.get("scores") or {}).items():
            if dept in active_depts:
                totals[dept] = totals.get(dept, 0.0) + float(pts)
    return totals


def lock_department_from_totals(
    totals: dict[str, float],
    candidate_departments: list[str],
    active_depts: list[str],
    none_selected: bool,
    age_label: str | None = None,
) -> tuple[str, dict[str, float], float, bool]:
    del none_selected  # fallback path uses same tie-break regardless
    locked, dept, margin = try_lock_department(totals)
    used_tie_break = False
    if locked and dept:
        return dept, totals, margin, used_tie_break
    best_score = max((totals.get(d, 0.0) for d in active_depts), default=0.0)
    tied = [
        d
        for d in candidate_departments
        if d in active_depts and totals.get(d, 0.0) == best_score
    ]
    if len(tied) > 1:
        used_tie_break = True
    if len(tied) > 1 and is_pediatric_age(age_label) and PEDIATRIC_DEPT in tied:
        fallback = PEDIATRIC_DEPT
    else:
        fallback = next(
            (
                d
                for d in candidate_departments
                if d in active_depts and totals.get(d, 0.0) == best_score
            ),
            active_depts[0],
        )
    return fallback, totals, margin, used_tie_break
