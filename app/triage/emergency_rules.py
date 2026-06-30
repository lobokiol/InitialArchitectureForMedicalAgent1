from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

_DEFAULT_PATH = (
    Path(__file__).resolve().parents[2] / "sourceData" / "data" / "rag_knowledge.jsonl"
)
DEFAULT_EMERGENCY_REPLY = (
    "您描述的情况可能存在急危重症风险，请立即前往急诊或拨打 120。"
)

_entries: list[dict] = []


@dataclass(frozen=True)
class EmergencyMatch:
    keyword: str
    em_id: str
    entry: dict


def load_emergency_entries(path: Path | None = None) -> list[dict]:
    p = path or _DEFAULT_PATH
    out: list[dict] = []
    if not p.is_file():
        return out
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or not line.startswith("{"):
            continue
        try:
            doc = json.loads(line)
        except json.JSONDecodeError:
            continue
        if doc.get("type") == "emergency":
            out.append(doc)
    return out


def reload_emergency_entries(entries: list[dict]) -> None:
    global _entries
    _entries = list(entries)


def _ensure_loaded() -> None:
    global _entries
    if not _entries:
        _entries = load_emergency_entries()


def match_emergency(text: str) -> EmergencyMatch | None:
    if not text or not text.strip():
        return None
    _ensure_loaded()
    blob = text.strip()
    best: EmergencyMatch | None = None
    for entry in _entries:
        em_id = str(entry.get("id") or "")
        for kw in entry.get("alliance") or []:
            if not isinstance(kw, str) or not kw:
                continue
            if kw in blob and (best is None or len(kw) > len(best.keyword)):
                best = EmergencyMatch(keyword=kw, em_id=em_id, entry=entry)
    return best


_entries = load_emergency_entries()
