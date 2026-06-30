from langchain_core.messages import HumanMessage

from app.domain.models import AppState
from app.domain.routing import is_mcp_followup_reply


def test_mcp_followup_route_with_keywords():
    state = AppState(
        last_recommended_department="消化内科",
        messages=[HumanMessage(content="消化内科怎么走？")],
    )
    assert is_mcp_followup_reply(state) is True


def test_mcp_followup_skips_without_last_dept():
    state = AppState(messages=[HumanMessage(content="怎么走？")])
    assert is_mcp_followup_reply(state) is False


def test_mcp_followup_first_turn_with_explicit_dept():
    state = AppState(messages=[HumanMessage(content="骨科怎么走？")])
    assert is_mcp_followup_reply(state) is True


def test_mcp_followup_skips_new_symptom():
    state = AppState(
        last_recommended_department="消化内科",
        messages=[HumanMessage(content="我肚子疼")],
    )
    assert is_mcp_followup_reply(state) is False
