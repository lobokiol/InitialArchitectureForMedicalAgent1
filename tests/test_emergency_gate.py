from __future__ import annotations

import pytest

from app.domain.dept_disambiguation import DeptDisambiguationState
from app.domain.models import AppState
from app.domain.routing import route_after_emergency_gate
from app.graph.nodes.answer import answer_generate_node
from app.graph.nodes.emergency_gate import emergency_gate_node
from app.ner.models import EntityExtractResult
from app.triage.emergency_rules import (
    DEFAULT_EMERGENCY_REPLY,
    load_emergency_entries,
    match_emergency,
    reload_emergency_entries,
    _DEFAULT_PATH,
)
from langchain_core.messages import AIMessage, HumanMessage


@pytest.fixture(autouse=True)
def _mock_em_entries():
    reload_emergency_entries([
        {
            "id": "EM0001",
            "type": "emergency",
            "alliance": ["大出血", "大量出血", "不能动", "昏迷", "剧烈"],
            "default_department": "急诊",
            "emergency_reply": "请立即前往急诊或拨打 120。",
        }
    ])
    yield


def test_match_emergency_hits():
    hit = match_emergency("脚脖子肿，不能动，皮发紫")
    assert hit is not None
    assert hit.keyword == "不能动"
    assert hit.em_id == "EM0001"


def test_match_emergency_misses():
    assert match_emergency("我有胃炎") is None


def test_match_emergency_longest_keyword():
    hit = match_emergency("患者大量出血")
    assert hit is not None
    assert hit.keyword == "大量出血"


def test_match_emergency_bare_crisis_word():
    hit = match_emergency("昏迷")
    assert hit is not None
    assert hit.keyword == "昏迷"


def test_default_reply_constant():
    assert "120" in DEFAULT_EMERGENCY_REPLY


def test_load_real_em0001_from_jsonl():
    entries = load_emergency_entries(_DEFAULT_PATH)
    em = next(e for e in entries if e.get("id") == "EM0001")
    assert "不能动" in em.get("alliance", [])
    assert em.get("default_department") == "急诊"


def test_app_state_emergency_fields_defaults():
    state = AppState()
    assert state.emergency_gate_passed is None
    assert state.emergency_match is None


def test_emergency_gate_node_hit():
    state = AppState(
        ner_result=EntityExtractResult(query="昏迷", primary_symptom=None),
    )
    patch = emergency_gate_node(state)
    assert patch["emergency_gate_passed"] is False
    assert patch["locked_department"] == "急诊"
    assert patch["emergency_match"]["keyword"] == "昏迷"
    assert patch["dept_state"].status == "emergency"


def test_emergency_gate_node_miss():
    state = AppState(
        ner_result=EntityExtractResult(query="我有胃炎", primary_disease="胃炎"),
    )
    patch = emergency_gate_node(state)
    assert patch == {"emergency_gate_passed": True}


def test_route_after_emergency_gate_hit():
    state = AppState(emergency_gate_passed=False, locked_department="急诊")
    assert route_after_emergency_gate(state) == "answer_generate"


def test_route_after_emergency_gate_miss():
    state = AppState(emergency_gate_passed=True)
    assert route_after_emergency_gate(state) == "slot_gate"


def test_answer_generate_emergency_uses_emergency_reply():
    state = AppState(
        locked_department="急诊",
        dept_state=DeptDisambiguationState(status="emergency"),
        rag_chunk={
            "emergency_reply": "您描述的情况可能存在急危重症风险，请立即前往急诊或拨打 120。",
        },
        slot_table=None,
        messages=[HumanMessage(content="旧消息"), AIMessage(content="旧回复"), HumanMessage(content="皮肤发黑")],
        ner_result=EntityExtractResult(query="皮肤发黑", primary_symptom="肚子疼"),
    )
    patch = answer_generate_node(state)
    msg = patch["messages"][0]
    assert isinstance(msg, AIMessage)
    assert msg.content == (
        "建议尽快就诊：**急诊**。\n"
        "您描述的情况可能存在急危重症风险，请立即前往急诊或拨打 120。"
    )
    assert "重新评估" not in msg.content
    assert "肚子疼" not in msg.content
