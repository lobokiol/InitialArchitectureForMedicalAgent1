"""Rule-based department scoring from rag_knowledge chunk."""

from __future__ import annotations

MARGIN = 2.0
LOCK_THRESHOLD = 6.0

# trigger 槽位 → condition 鉴别词（如「扭」对应「外伤」「扭伤」）
_TRIGGER_CONDITION_HINTS: dict[str, list[str]] = {
    "扭": ["外伤", "扭伤", "韧带", "骨折", "活动痛"],
    "外伤": ["外伤", "骨折", "韧带"],
    "摔": ["外伤", "骨折"],
    "撞": ["外伤", "骨折"],
    "运动": ["劳损", "韧带", "活动痛"],
    "久站": ["久站", "静脉曲张", "沉重感"],
    "受凉": ["晨僵", "反复"],
    "寒冷": ["发凉", "苍白", "发紫"],
    "冻": ["冻伤", "水疱", "麻木"],
}

# 用户描述与 condition 的短词对齐（子串级）
_SYMPTOM_OVERLAP: list[tuple[str, str]] = [
    ("肿", "肿胀"),
    ("痛", "疼痛"),
    ("扭", "扭伤"),
    ("扭", "外伤"),
    ("僵", "晨僵"),
    ("紫", "发紫"),
    ("久站", "久站"),
    ("曲张", "静脉曲张"),
]


def _keyword_hits(text: str, blob: str) -> float:
    if not text or not blob:
        return 0.0
    hits = 0.0
    for part in text.replace("；", "，").replace("、", "，").split("，"):
        token = part.strip()
        if len(token) >= 2 and token in blob:
            hits += 1.0
    return hits


def _symptom_overlap(user_text: str, condition: str) -> float:
    hits = 0.0
    for user_frag, cond_frag in _SYMPTOM_OVERLAP:
        if user_frag in user_text and cond_frag in condition:
            hits += 1.0
    return hits


def _slot_feature_bonus(
    condition: str,
    slot_trigger: str | None,
    slot_emergency: str | None,
) -> float:
    bonus = 0.0
    if slot_trigger:
        if slot_trigger in condition:
            bonus += 2.0
        for hint in _TRIGGER_CONDITION_HINTS.get(slot_trigger, []):
            if hint in condition:
                bonus += 1.0
    if slot_emergency and slot_emergency in condition:
        bonus += 1.5
    return bonus


def _accompany_bonus(
    condition: str,
    user_text: str,
    accompany_keywords: list[str] | None,
    slot_trigger: str | None,
) -> float:
    bonus = 0.0
    for kw in accompany_keywords or []:
        in_user = kw in user_text
        if not in_user and slot_trigger and slot_trigger in kw:
            in_user = True
        if not in_user:
            continue
        if kw in condition:
            bonus += 1.5
            continue
        hints = _TRIGGER_CONDITION_HINTS.get(slot_trigger or "", [])
        if any(h in condition for h in hints):
            bonus += 1.0
    return bonus


def score_departments(
    depts: list[dict],
    user_text: str,
    accompany_keywords: list[str] | None = None,
    slot_trigger: str | None = None,
    slot_emergency: str | None = None,
    llm_boosts: dict[str, float] | None = None,
) -> dict[str, float]:
    blob = user_text
    if slot_trigger:
        blob += " " + slot_trigger
    if slot_emergency:
        blob += " " + slot_emergency

    scores: dict[str, float] = {}
    for d in depts:
        dept = d.get("department") or ""
        if not dept:
            continue
        priority = int(d.get("priority") or 3)
        condition = d.get("condition") or ""
        score = (4 - priority) * 1.0
        score += _keyword_hits(condition, blob) * 1.5
        score += _symptom_overlap(user_text, condition) * 1.0
        score += _slot_feature_bonus(condition, slot_trigger, slot_emergency)
        score += _accompany_bonus(condition, user_text, accompany_keywords, slot_trigger)
        if llm_boosts and dept in llm_boosts:
            score += llm_boosts[dept]
        scores[dept] = score
    return scores


def try_lock_department(
    scores: dict[str, float],
    margin_threshold: float = MARGIN,
    lock_threshold: float = LOCK_THRESHOLD,
) -> tuple[bool, str | None, float]:
    if not scores:
        return False, None, 0.0
    ordered = sorted(scores.items(), key=lambda x: -x[1])
    if len(ordered) == 1:
        return True, ordered[0][0], 999.0
    top, second = ordered[0], ordered[1]
    margin = top[1] - second[1]
    if margin >= margin_threshold or top[1] >= lock_threshold:
        return True, top[0], margin
    return False, None, margin


_NEGATION = ("都没有", "没有", "不是", "无", "否")
_AFFIRM_TRAUMA = ("摔过", "扭过", "扭伤", "外伤", "摔了", "扭了")


def apply_negation_boosts(scores: dict[str, float], reply: str) -> dict[str, float]:
    s = dict(scores)
    r = reply.strip()
    if not r:
        return s
    if any(n in r for n in _NEGATION):
        s["骨科"] = s.get("骨科", 0) - 2.0
        s["风湿免疫科"] = s.get("风湿免疫科", 0) + 2.0
        s["血管外科"] = s.get("血管外科", 0) + 1.0
    elif any(a in r for a in _AFFIRM_TRAUMA):
        s["骨科"] = s.get("骨科", 0) + 2.0
    return s


def fallback_department(depts: list[dict]) -> str | None:
    for d in sorted(depts, key=lambda x: int(x.get("priority") or 99)):
        if d.get("department"):
            return d["department"]
    return None
