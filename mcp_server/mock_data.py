from __future__ import annotations

BASE_DOCTORS = [
    {"name": "张医生", "time": "14:00-18:00", "slots": 3},
    {"name": "李医生", "time": "08:00-12:00", "slots": 5},
    {"name": "王医生", "time": "全天", "slots": 0},
]


def doctors_for_department(department: str) -> list[dict]:
    prefix = department.strip() or "综合"
    return [{**doc, "name": f"{prefix}·{doc['name']}"} for doc in BASE_DOCTORS]
