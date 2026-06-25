"""Unit tests for triage slot table defaults and gate."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.domain.slot_table import (
    TriageSlotTable,
    default_slot_table,
    slot_gate_passes,
)

SLOTS_JSON = ROOT / "sourceData" / "data" / "triage_intake_slots.json"


def test_default_age_gender() -> None:
    table = default_slot_table()
    assert table.gender == "男"
    assert table.age == "30岁"
    print("[OK] default age/gender")


def test_intake_json_defaults() -> None:
    data = json.loads(SLOTS_JSON.read_text(encoding="utf-8"))
    assert data["defaults"]["gender"] == "男"
    assert data["defaults"]["age"] == "30岁"
    assert data["required_any_of"] == ["primary_symptom", "primary_disease"]
    print("[OK] triage_intake_slots.json")


def test_gate_pass_with_symptom() -> None:
    table = TriageSlotTable(primary_symptom="心慌")
    assert slot_gate_passes(table)
    print("[OK] gate pass with symptom")


def test_gate_pass_with_disease() -> None:
    table = TriageSlotTable(primary_disease="胃炎")
    assert slot_gate_passes(table)
    print("[OK] gate pass with disease")


def test_gate_fail_empty() -> None:
    table = default_slot_table()
    assert not slot_gate_passes(table)
    print("[OK] gate fail empty")


from app.ner.models import EntityExtractResult
from app.triage.slot_fill import fill_slot_table


def test_fill_from_ner() -> None:
    ner = EntityExtractResult(
        query="脚脖子肿，昨天扭了",
        primary_symptom="脚脖子肿",
    )
    table = fill_slot_table(ner)
    assert table.primary_symptom == "脚脖子肿"
    assert table.trigger is not None
    print("[OK] fill from ner")


def test_fill_age_gender() -> None:
    ner = EntityExtractResult(query="女，5岁，发烧", primary_symptom="发烧")
    table = fill_slot_table(ner)
    assert table.gender == "女"
    assert "5" in table.age
    print("[OK] fill age/gender")


def main() -> None:
    test_default_age_gender()
    test_intake_json_defaults()
    test_gate_pass_with_symptom()
    test_gate_pass_with_disease()
    test_gate_fail_empty()
    test_fill_from_ner()
    test_fill_age_gender()


if __name__ == "__main__":
    main()
