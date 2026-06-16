"""Round-based top-2 department pair selection for disambiguation."""


def _pick_pair_by_round(depts: list[dict], round_num: int) -> tuple[dict, dict | None]:
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
