from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

_MOCK_DIR = Path(__file__).resolve().parents[1] / "mock"

BASE_DOCTORS = [
    {"name": "张医生", "time": "14:00-18:00", "slots": 3},
    {"name": "李医生", "time": "08:00-12:00", "slots": 5},
    {"name": "王医生", "time": "全天", "slots": 0},
]

NOT_FOUND = "department_not_found"


@lru_cache(maxsize=1)
def _load_json(name: str) -> dict:
    path = _MOCK_DIR / name
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def list_departments() -> list[str]:
    return list(_load_json("departments.json").keys())


def _resolve_department_key(department: str) -> str | None:
    dept = (department or "").strip()
    if not dept:
        return None
    data = _load_json("departments.json")
    if dept in data:
        return dept
    for key in data:
        if key in dept or dept in key:
            return key
    return None


def doctors_for_department(department: str) -> list[dict]:
    key = _resolve_department_key(department)
    prefix = key or (department.strip() or "综合")
    return [{**doc, "name": f"{prefix}·{doc['name']}"} for doc in BASE_DOCTORS]


def intro_for_department(department: str) -> dict:
    key = _resolve_department_key(department)
    if not key:
        return {"error": NOT_FOUND, "department": department}
    return dict(_load_json("departments.json")[key])


def route_for_department(department: str, from_location: str = "导诊台") -> dict:
    key = _resolve_department_key(department)
    if not key:
        return {"error": NOT_FOUND, "department": department}
    row = dict(_load_json("routes.json")[key])
    if from_location:
        row["from"] = from_location
    return row
