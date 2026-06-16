"""
症状知识库：RAG chunk 元数据 + 槽位表代号解析。
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

_KB_PATH = Path(__file__).resolve().parent / "symptom_kb.json"

# 候选词/别名 → 标准主症（归一后再查 slot_table_code）
_PALPITATION_ALIASES = (
    "心慌", "心里发慌", "心跳厉害", "心里难受", "心累", "落空感", "心跳快",
    "心跳加速", "心跳不齐", "心律不齐", "心扑通扑通跳", "心咚咚跳", "心突突跳",
    "心乱", "心焦", "心跳声大", "能听见心跳", "早搏感", "漏跳感",
)

CHIEF_TO_CANONICAL: dict[str, str] = {
    **{a: "心悸" for a in _PALPITATION_ALIASES},
    "肚脐上方疼痛": "腹痛",
    "肚脐上面": "腹痛",
    "肚脐上方": "腹痛",
    "上腹部": "腹痛",
    "饭后腹胀": "腹胀",
}


@lru_cache(maxsize=1)
def load_symptom_kb() -> list[dict]:
    with _KB_PATH.open(encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("symptom_kb.json must be a JSON array")
    return data


def _build_lookup() -> dict[str, dict]:
    """alias / canonical_term / key_word → chunk record"""
    lookup: dict[str, dict] = {}
    for record in load_symptom_kb():
        keys = {record.get("canonical_term"), record.get("key_word"), *record.get("alias", [])}
        for k in keys:
            if k:
                lookup[str(k).strip()] = record
    return lookup


@lru_cache(maxsize=1)
def symptom_lookup_table() -> dict[str, dict]:
    return _build_lookup()


def to_canonical_chief(chief: str | None) -> str | None:
    if not chief:
        return None
    return CHIEF_TO_CANONICAL.get(chief.strip(), chief.strip())


def resolve_slot_table_code(chief: str | None) -> str | None:
    """
    根据唯一主症（标准词）解析槽位表代号。
    例：心慌/心悸 → palpitation_v1；腹痛/肚脐上方疼痛 → abdominal_pain_v1
    """
    if not chief:
        return None
    canonical = to_canonical_chief(chief)
    if not canonical:
        return None
    record = symptom_lookup_table().get(canonical)
    if not record:
        return None
    code = record.get("slot_table_code")
    return str(code) if code else None


def get_symptom_chunk(chief: str | None) -> dict | None:
    """返回匹配到的 RAG 知识库 chunk（测试/展示用）。"""
    if not chief:
        return None
    canonical = to_canonical_chief(chief)
    if not canonical:
        return None
    return symptom_lookup_table().get(canonical)
