from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
GOLDEN_PATH = ROOT / "demo" / "data" / "foot_triage_golden.jsonl"
LEGACY_PATH = ROOT / "demo" / "data" / "foot_symptom_eval.jsonl"

FOOT_DISEASE_CASES: list[dict] = [
    {"message": "脚气怎么办", "expect_dept": "皮肤科"},
    {"message": "香港脚看什么科", "expect_dept": "皮肤科"},
    {"message": "灰指甲挂什么科", "expect_dept": "皮肤科"},
    {"message": "大脚趾痛风", "expect_dept": "风湿免疫科"},
    {"message": "脚底疣治疗", "expect_dept": "皮肤科"},
    {"message": "大脚骨疼", "expect_dept": "骨科"},
]

REJECT_CASES: list[dict] = [
    {"message": ""},
    {"message": "你好"},
    {"message": "怎么挂号"},
    {"message": "怎么预约"},
    {"message": "今天天气怎么样"},
    {"message": "帮我查报告"},
    {"message": "医院几点开门"},
    {"message": "谢谢"},
    {"message": "在吗"},
    {"message": "请问一下"},
]

EMERGENCY_EXTRA: list[dict] = [
    {"message": "脚脖子肿，不能动，皮发紫"},
    {"message": "外伤后脚畸形不能走路"},
    {"message": "脚又红又肿越来越疼还发烧"},
]


def build_golden_from_legacy() -> None:
    """One-time migration: legacy eval + disease/reject/emergency seeds."""
    if not LEGACY_PATH.exists():
        raise FileNotFoundError(f"Missing legacy eval: {LEGACY_PATH}")

    rows: list[dict] = []
    for line in LEGACY_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        c = json.loads(line)
        rows.append(
            {
                "id": c["id"],
                "subset": "A",
                "message": c["message"],
                "expect_route": c.get("expect_route", "symptom"),
                "expect_chunk_id": c["expect_chunk_id"],
                "expect_dept": c.get("expect_dept"),
                "expect_emergency": c.get("expect_dept") == "急诊",
                "tags": ["migrated"],
            }
        )

    a_rows = list(rows)
    for i, c in enumerate(a_rows, start=1):
        if c.get("expect_dept") == "急诊":
            continue
        rows.append(
            {
                "id": f"FTB{i:03d}",
                "subset": "B",
                "message": c["message"],
                "expect_route": c["expect_route"],
                "expect_chunk_id": c.get("expect_chunk_id"),
                "expect_dept": c["expect_dept"],
                "expect_emergency": False,
                "tags": ["migrated", "dept"],
            }
        )

    for i, c in enumerate(FOOT_DISEASE_CASES, start=1):
        rows.append(
            {
                "id": f"FTC{i:03d}",
                "subset": "C",
                "message": c["message"],
                "expect_route": "disease",
                "expect_dept": c["expect_dept"],
                "expect_emergency": False,
                "tags": ["disease"],
            }
        )

    for i, c in enumerate(REJECT_CASES, start=1):
        rows.append(
            {
                "id": f"FTD{i:03d}",
                "subset": "D",
                "message": c["message"],
                "expect_route": "reject",
                "expect_emergency": False,
                "tags": ["reject"],
            }
        )

    emergency_msgs = {c["message"] for c in EMERGENCY_EXTRA}
    for c in rows:
        if c.get("expect_dept") == "急诊" and c["subset"] == "A":
            emergency_msgs.add(c["message"])

    for i, msg in enumerate(sorted(emergency_msgs), start=1):
        rows.append(
            {
                "id": f"FTE{i:03d}",
                "subset": "E",
                "message": msg,
                "expect_route": "symptom",
                "expect_dept": "急诊",
                "expect_emergency": True,
                "tags": ["emergency"],
            }
        )

    lines = [json.dumps(r, ensure_ascii=False) for r in rows]
    GOLDEN_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_cases(subsets: set[str] | None = None) -> list[dict]:
    if not GOLDEN_PATH.exists():
        build_golden_from_legacy()
    cases: list[dict] = []
    for line in GOLDEN_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        c = json.loads(line)
        if subsets and c.get("subset") not in subsets:
            continue
        cases.append(c)
    return cases
