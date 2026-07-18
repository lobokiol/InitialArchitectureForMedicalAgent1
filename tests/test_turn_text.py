"""Lock shared turn-text helpers used by routing and graph nodes."""

from langchain_core.messages import AIMessage, HumanMessage

from app.domain.models import AppState
from app.triage.turn_text import current_turn_text, last_human_text


def test_last_human_text_returns_latest_human():
    state = AppState(
        messages=[
            HumanMessage(content="第一次"),
            AIMessage(content="回复"),
            HumanMessage(content="第二次"),
        ]
    )
    assert last_human_text(state) == "第二次"


def test_last_human_text_empty_without_human():
    state = AppState(messages=[AIMessage(content="只有助手")])
    assert last_human_text(state) == ""


def test_current_turn_text_prefers_ner_query():
    from app.ner.models import EntityExtractResult

    state = AppState(
        messages=[HumanMessage(content="忽略")],
        ner_result=EntityExtractResult(query="头疼发烧"),
    )
    assert "头疼发烧" in current_turn_text(state)
