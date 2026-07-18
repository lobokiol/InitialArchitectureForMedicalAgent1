"""Rule-based dept disambiguation choices (no LLM)."""

from __future__ import annotations

from app.domain.dept_disambiguation import DeptChoice

CHOICE_QUESTION_TEMPLATE = "为更准确推荐科室，请选择您是否有以下情况："
INVALID_CHOICE_REPLY = "请从下列选项中选择（输入选项文字或编号）。"
CHOICE_BOOST = 2.0
NONE_CHOICE_ID = "none"
NONE_CHOICE_LABEL = "都没有"


def _pick_pair_by_round(depts: list[dict], round_num: int) -> tuple[dict, dict | None]:
    """Select top-2 department pair for the given disambiguation round."""
    ordered = sorted(depts, key=lambda d: int(d.get("priority") or 99))
    if not ordered:
        return {}, None
    if len(ordered) == 1:
        return ordered[0], None
    if round_num <= 1:
        return ordered[0], ordered[1]
    if len(ordered) >= 3:
        return ordered[1], ordered[2]
    return ordered[0], ordered[1]

_DISEASE_DENY_FRAGMENTS = (
    "炎",
    "病",
    "症",
    "损伤",
    "骨折",
    "筋膜炎",
    "骨刺",
    "胼胝",
    "甲癣",
    "嵌甲",
)
_LABEL_ALIASES: dict[str, str] = {
    "关节痛": "多关节疼痛",
    "活动痛": "活动后加重",
    "久站": "久站加重",
}


def _emergency_blob(chunk: dict) -> str:
    flag = chunk.get("emergency_flag") or {}
    return str(flag.get("condition") or "")


def _is_emergency_term(label: str, emergency_blob: str) -> bool:
    if not label:
        return True
    if label in emergency_blob:
        return True
    for frag in ("畸形", "不能负重", "发紫", "不能动", "皮肤发黑", "开放性"):
        if frag in label and frag in emergency_blob:
            return True
    return False


def _is_disease_like(label: str) -> bool:
    return any(d in label for d in _DISEASE_DENY_FRAGMENTS)


def _depts_for_keyword(keyword: str, dept_a: dict, dept_b: dict | None) -> list[str]:
    targets: list[str] = []
    for d in (dept_a, dept_b):
        if not d:
            continue
        cond = d.get("condition") or ""
        dept = d.get("department") or ""
        if keyword in cond and dept:
            targets.append(dept)
    return list(dict.fromkeys(targets))


def build_dept_choices(
    chunk: dict,
    round_num: int,
    asked_choice_ids: list[str],
) -> tuple[list[DeptChoice], bool]:
    """Return (choices including '都没有', has_symptom_options)."""
    depts = chunk.get("department_recommendation") or []
    dept_a, dept_b = _pick_pair_by_round(depts, round_num)
    if not dept_a:
        return [_none_choice()], False

    accompany = chunk.get("accompanying_symptom_keywords") or []
    canonical = (chunk.get("canonical_symptom") or "").strip()
    emergency_blob = _emergency_blob(chunk)
    cond_blob = (dept_a.get("condition") or "") + (
        dept_b.get("condition") or "" if dept_b else ""
    )

    candidates: list[tuple[str, list[str]]] = []
    seen_labels: set[str] = set()
    for kw in accompany:
        if not isinstance(kw, str) or len(kw.strip()) < 2:
            continue
        kw = kw.strip()
        if kw not in cond_blob:
            continue
        if canonical and kw == canonical:
            continue
        if _is_emergency_term(kw, emergency_blob):
            continue
        if _is_disease_like(kw):
            continue
        targets = _depts_for_keyword(kw, dept_a, dept_b)
        if not targets:
            continue
        if kw in seen_labels:
            continue
        seen_labels.add(kw)
        candidates.append((kw, targets))

    choices: list[DeptChoice] = []
    for idx, (label, targets) in enumerate(candidates, start=1):
        cid = f"c{idx}"
        if cid in asked_choice_ids:
            continue
        choices.append(DeptChoice(id=cid, label=label, target_departments=targets))

    has_symptom = len(choices) > 0
    choices.append(_none_choice())
    return choices, has_symptom


def _none_choice() -> DeptChoice:
    return DeptChoice(id=NONE_CHOICE_ID, label=NONE_CHOICE_LABEL, target_departments=[])


def format_choice_message(choices: list[DeptChoice]) -> str:
    lines = [CHOICE_QUESTION_TEMPLATE, ""]
    for i, c in enumerate(choices, start=1):
        lines.append(f"{i}. {c.label}")
    return "\n".join(lines)


def resolve_dept_choice(user_reply: str, choices: list[DeptChoice]) -> DeptChoice | None:
    text = (user_reply or "").strip()
    if not text:
        return None
    if text.isdigit():
        idx = int(text) - 1
        if 0 <= idx < len(choices):
            return choices[idx]
    alias = _LABEL_ALIASES.get(text, text)
    for c in choices:
        if text == c.id or text == c.label or alias == c.label:
            return c
    return None


def choice_score_boosts(choice: DeptChoice) -> dict[str, float]:
    if choice.id == NONE_CHOICE_ID:
        return {}
    return {d: CHOICE_BOOST for d in choice.target_departments}


def lock_department_for_explicit_choice(
    choice: DeptChoice,
    scores: dict[str, float],
) -> str | None:
    """User picked a symptom option — lock the best matching target dept."""
    if choice.id == NONE_CHOICE_ID or not choice.target_departments:
        return None
    return max(choice.target_departments, key=lambda d: scores.get(d, 0.0))


def build_differential_choices(rule_chunk: dict) -> list[DeptChoice]:
    """Build multi-select differential choices from a dept-rule chunk."""
    choices: list[DeptChoice] = []
    for i, q in enumerate(rule_chunk.get("differential_questions") or [], 1):
        choices.append(
            DeptChoice(
                id=f"c{i}",
                label=q["text"],
                target_departments=list((q.get("scores") or {}).keys()),
            )
        )
    choices.append(DeptChoice(id=NONE_CHOICE_ID, label=NONE_CHOICE_LABEL, target_departments=[]))
    return choices


def differential_selection_dicts(rule_chunk: dict, picked: list[DeptChoice]) -> list[dict]:
    """Map selected choices back to differential_questions entries."""
    questions = rule_chunk.get("differential_questions") or []
    out: list[dict] = []
    for c in picked:
        if c.id.startswith("c") and c.id[1:].isdigit():
            idx = int(c.id[1:]) - 1
            if 0 <= idx < len(questions):
                out.append(questions[idx])
    return out
