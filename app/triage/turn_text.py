"""User-turn text helpers for triage scoring and follow-up routing."""

from __future__ import annotations

from langchain_core.messages import HumanMessage

from app.domain.models import AppState


def last_human_text(state: AppState) -> str:
    """Most recent HumanMessage content (reverse scan). Empty if none."""
    msgs = state.messages or []
    for msg in reversed(msgs):
        if isinstance(msg, HumanMessage) and isinstance(msg.content, str):
            return msg.content.strip()
    return ""


def current_turn_text(state: AppState) -> str:
    """Current-turn user text for scoring/emergency (no full thread history)."""
    parts: list[str] = []
    if state.ner_result and state.ner_result.query:
        parts.append(state.ner_result.query.strip())
    elif state.messages:
        last = state.messages[-1]
        if isinstance(last, HumanMessage) and isinstance(last.content, str):
            parts.append(last.content.strip())
    table = state.slot_table
    if table and table.trigger:
        parts.append(table.trigger)
    return " ".join(p for p in parts if p)
