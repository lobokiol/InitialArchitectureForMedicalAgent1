from __future__ import annotations

import re

from app.domain.dept_disambiguation import DeptChoice
from app.triage.dept_choices import NONE_CHOICE_ID, NONE_CHOICE_LABEL


def parse_choice_indices(text: str) -> list[int]:
    raw = (text or "").strip().replace("、", ",").replace(" ", ",")
    parts = [p for p in re.split(r"[,，]+", raw) if p.strip()]
    return [int(p) - 1 for p in parts if p.strip().isdigit()]


def resolve_multi_choice(
    text: str, choices: list[DeptChoice]
) -> tuple[list[DeptChoice] | None, bool]:
    t = (text or "").strip()
    if t == NONE_CHOICE_LABEL:
        return [], True
    if not t:
        return None, False
    if not any(ch.isdigit() for ch in t):
        return None, False
    indices = parse_choice_indices(t)
    if not indices:
        return None, False
    picked: list[DeptChoice] = []
    for idx in indices:
        if idx < 0 or idx >= len(choices):
            return None, False
        c = choices[idx]
        if c.id == NONE_CHOICE_ID:
            return [], True
        picked.append(c)
    return picked, False
