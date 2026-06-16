"""Fill global TriageSlotTable from NER + rule-based optional slots."""

from __future__ import annotations

import re

from app.domain.slot_table import TriageSlotTable, default_slot_table
from app.ner.models import EntityExtractResult

_TRIGGER_RE = re.compile(r"扭|外伤|摔|撞|运动|久站|受凉|饭后|情绪激动|寒冷|冻")
_DURATION_RE = re.compile(r"\d+[天周月年]|刚出现|首次|反复|很久|半年|几个月|昨天|今天")
_EMERGENCY_RE = re.compile(r"不能动|发紫|剧烈|受不了|昏迷|大量出血|畸形|不能负重")


def _parse_demographics(query: str, table: TriageSlotTable) -> None:
    if "女" in query:
        table.gender = "女"
    age_m = re.search(r"(\d+)\s*岁", query)
    if age_m:
        table.age = f"{age_m.group(1)}岁"


def fill_slot_table(ner: EntityExtractResult) -> TriageSlotTable:
    table = default_slot_table()
    table.primary_symptom = ner.primary_symptom
    table.companion_symptoms = list(ner.companion_symptoms)
    table.primary_disease = ner.primary_disease
    table.companion_diseases = list(ner.companion_diseases)
    q = ner.query or ""
    _parse_demographics(q, table)
    if _TRIGGER_RE.search(q):
        table.trigger = _TRIGGER_RE.search(q).group(0)  # type: ignore[union-attr]
    if _DURATION_RE.search(q):
        table.duration = _DURATION_RE.search(q).group(0)  # type: ignore[union-attr]
    if _EMERGENCY_RE.search(q):
        table.emergency = _EMERGENCY_RE.search(q).group(0)  # type: ignore[union-attr]
    return table
