"""Infer clarify-slot replies (age/sex) from query text for batch triage runs."""
from __future__ import annotations

import re

FEMALE_KEYWORDS = (
    "月经",
    "大姨妈",
    "经期",
    "例假",
    "怀孕",
    "孕妇",
    "孕周",
    "胎儿",
    "胎心",
    "流产",
    "清宫",
    "外阴",
    "白带",
    "子宫",
    "卵巢",
    "宫颈",
    "备孕",
    "女生",
    "少女",
    "老婆",
    "宝妈",
    "女儿",
    "女友",
    "输卵管",
    "子宫内膜",
    "自然流产",
    "人流",
    "性孕",
)

MALE_KEYWORDS = (
    "阴茎",
    "睾丸",
    "包皮",
    "勃起",
    "老公",
    "男友",
    "父亲",
    "我爸",
    "男性",
    "男科",
)

PEDIATRIC_KEYWORDS = ("宝宝", "婴儿", "幼儿", "小孩", "孩子", "儿童", "小儿", "初生", "宝妈")


def infer_sex(query: str) -> str:
    q = query or ""
    female = any(k in q for k in FEMALE_KEYWORDS)
    male = any(k in q for k in MALE_KEYWORDS)
    if female and not male:
        return "女"
    if male and not female:
        return "男"
    return "男"


def infer_age(query: str) -> str:
    q = query or ""
    m = re.search(r"(\d+)\s*个月", q)
    if m:
        months = int(m.group(1))
        return "0-3个月" if months <= 3 else "3个月-1岁"
    m = re.search(r"(\d+)\s*岁", q)
    if m:
        age = int(m.group(1))
        if age <= 4:
            return "2-4岁"
        if age <= 11:
            return "5-11岁"
        if age <= 18:
            return "12-18岁"
        if age < 35:
            return "19-35岁"
        if age < 60:
            return "35-59岁"
        return "60岁及以上"
    if "怀孕" in q and "周" in q:
        return "19-35岁"
    if any(k in q for k in ("初生", "新生儿", "两个月", "一周两个月")):
        return "0-3个月"
    if any(k in q for k in ("宝宝", "婴儿", "幼儿")):
        return "3个月-1岁"
    if any(k in q for k in ("小孩", "孩子", "儿童", "小儿", "磨牙", "夜醒")):
        return "5-11岁"
    if any(k in q for k in ("老人家", "老人", "父亲")):
        return "60岁及以上"
    return "19-35岁"


def pick_clarify_reply(data: dict, query: str) -> str | None:
    if data.get("awaiting_clarify") and data.get("clarify_choices"):
        phase = data.get("clarify_phase")
        if phase == "age":
            return infer_age(query)
        if phase == "sex":
            return infer_sex(query)
        return data["clarify_choices"][0]["label"]
    if data.get("awaiting_dept_choice") and data.get("dept_choices"):
        return "1" if data.get("multi_select") else data["dept_choices"][0]["label"]
    return None
