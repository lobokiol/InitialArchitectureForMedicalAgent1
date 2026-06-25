"""Unit tests for current-turn text isolation."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from langchain_core.messages import AIMessage, HumanMessage

from app.domain.models import AppState
from app.domain.slot_table import TriageSlotTable
from app.ner.models import EntityExtractResult
from app.triage.turn_text import current_turn_text


def test_uses_ner_query_not_history() -> None:
    state = AppState(
        messages=[
            HumanMessage(content="脚脖子肿，不能动，皮发紫"),
            AIMessage(content="急诊建议..."),
            HumanMessage(content="脚脖子肿"),
        ],
        ner_result=EntityExtractResult(query="脚脖子肿", primary_symptom="脚脖子肿"),
        slot_table=TriageSlotTable(primary_symptom="脚脖子肿"),
    )
    text = current_turn_text(state)
    assert "不能动" not in text
    assert "脚脖子肿" in text
    print("[OK] ner query excludes history")


def main() -> None:
    test_uses_ner_query_not_history()


if __name__ == "__main__":
    main()
