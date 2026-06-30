"""Current-turn user text for scoring/emergency (no thread history)."""

from __future__ import annotations

from langchain_core.messages import HumanMessage

from app.domain.models import AppState


def current_turn_text(state: AppState) -> str:
    parts: list[str] = []
    if state.ner_result and state.ner_result.query:
        parts.append(state.ner_result.query.strip())
    elif state.messages:
        last = state.messages[-1]
        if isinstance(last, HumanMessage) and isinstance(last.content, str):
            parts.append(last.content.strip())
    table = state.slot_table
    if table:
        if table.trigger:
            parts.append(table.trigger)
    return " ".join(p for p in parts if p)
