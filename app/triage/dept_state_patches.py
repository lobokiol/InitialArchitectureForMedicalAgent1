"""Shared dept-state patch builders for disambiguation nodes."""

from __future__ import annotations

from app.domain.dept_disambiguation import DeptDisambiguationState


def locked_department_patch(
    dept: str,
    scores: dict[str, float],
    margin: float,
    current_round: int,
    depts: list[dict],
    status: str,
) -> dict:
    return {
        "locked_department": dept,
        "dept_state": DeptDisambiguationState(
            status=status,
            dept_scores=scores,
            margin=margin,
            round=current_round,
            candidate_departments=depts,
        ),
    }
