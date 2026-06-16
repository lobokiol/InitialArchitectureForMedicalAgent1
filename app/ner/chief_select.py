"""
从所有提取词中选出唯一主症：优先危急/核心症状，排除诱因/性质/频次等修饰词。
"""

# 不能单独作为主症的修饰词（诱因、频次、性质等）
MODIFIER_TERMS: set[str] = {
    "饭后",
    "进食后",
    "餐前",
    "受凉后",
    "一阵一阵",
    "阵发性",
    "绞痛",
    "隐痛",
    "灼痛",
    "刺痛",
}

# 危急/核心症状优先级（分值越高越优先作为主症）
URGENCY_PRIORITY: dict[str, int] = {
    # 危急
    "胸痛": 100,
    "呼吸困难": 100,
    "意识障碍": 100,
    "大量出血": 100,
    "剧烈头痛": 95,
    # 急性警示
    "心慌": 90,
    "心悸": 90,
    "高热": 85,
    "发热": 80,
    "寒战": 78,
    # 定位明确的疼痛主诉
    "肚脐上方疼痛": 75,
    "腹痛": 75,
    "头痛": 72,
    # 其他明显症状
    "手抖": 70,
    "饭后腹胀": 65,
    "腹胀": 60,
    "腹部瘙痒": 55,
    "腹泻": 55,
    "越吃越瘦": 40,
    "消瘦": 40,
    "脾气差": 30,
}


def _score(term: str, query: str) -> int:
    base = URGENCY_PRIORITY.get(term, 50)
    # 用户在句首/强调处提到的词略加分（简单启发）
    if query.find(term) != -1 and query.find(term) < max(len(query) // 3, 8):
        base += 3
    # 组合词（含部位/场景）略优先于单字症状
    if len(term) >= 4:
        base += 2
    return base


def select_primary_chief(candidates: list[str], query: str = "") -> str | None:
    """
    从候选主症词中选出唯一最紧急、最关键的一个。
    """
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]

    # 先排除修饰词；若全为修饰词则保留原列表兜底
    eligible = [c for c in candidates if c not in MODIFIER_TERMS]
    pool = eligible if eligible else list(candidates)

    scored = [(term, _score(term, query)) for term in pool]
    scored.sort(key=lambda x: (-x[1], -len(x[0]), x[0]))
    return scored[0][0]


def split_companion(candidates: list[str], primary: str | None) -> list[str]:
    if not primary:
        return list(candidates)
    return [c for c in candidates if c != primary]
