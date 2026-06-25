"""
验证三意图路由：disease | symptom | reject

用法:
  python scripts/archive/test_decision_routing.py
  python scripts/archive/test_decision_routing.py --live
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, START, StateGraph

from app.domain.models import AppState
from app.domain.triage_intent import REJECT_MESSAGE
from app.domain.routing import route_after_decision
from app.graph.nodes.decision import decision_node
from app.graph.nodes.disease_dept import disease_dept_node
from app.graph.nodes.rag_symptom_recall import rag_symptom_recall_node
from app.graph.nodes.reject import reject_node
from app.ner.models import EntityExtractResult
from app.ner.triage_route import resolve_triage_route


def test_resolve_triage_route() -> None:
    cases = [
        (
            EntityExtractResult(
                query="a",
                primary_disease="胃炎",
                companion_symptoms=["腹痛"],
            ),
            "disease",
        ),
        (
            EntityExtractResult(query="b", primary_symptom="心悸"),
            "symptom",
        ),
        (EntityExtractResult(query="c"), "reject"),
        (
            EntityExtractResult(
                query="d",
                primary_disease="胃炎",
                primary_symptom="肚子疼",
            ),
            "disease",
        ),
    ]
    for ner, expected in cases:
        assert resolve_triage_route(ner) == expected
    print("[OK] resolve_triage_route")


def build_test_graph():
    graph = StateGraph(AppState)
    graph.add_node("decision", decision_node)
    graph.add_node("disease_dept", disease_dept_node)
    graph.add_node("rag_symptom_recall", rag_symptom_recall_node)
    graph.add_node("reject", reject_node)
    graph.add_edge(START, "decision")
    graph.add_conditional_edges(
        "decision",
        route_after_decision,
        {
            "disease_dept": "disease_dept",
            "rag_symptom_recall": "rag_symptom_recall",
            "reject": "reject",
        },
    )
    graph.add_edge("disease_dept", END)
    graph.add_edge("rag_symptom_recall", END)
    graph.add_edge("reject", END)
    return graph.compile()


LIVE_CASES = [
    {"query": "我有胃炎", "expect_route": "disease"},
    {"query": "最近有点心慌手抖", "expect_route": "symptom"},
    {"query": "你好", "expect_route": "reject", "expect_message": REJECT_MESSAGE},
    {"query": "怎么预约", "expect_route": "reject", "expect_message": REJECT_MESSAGE},
]


def run_live_tests() -> None:
    app = build_test_graph()
    for case in LIVE_CASES:
        out = app.invoke({"messages": [HumanMessage(content=case["query"])]})
        ir = out.get("intent_result")
        route = ir.triage_route if ir else None
        last_msg = out["messages"][-1] if out.get("messages") else None
        text = last_msg.content if isinstance(last_msg, AIMessage) else None

        payload = {
            "query": case["query"],
            "triage_route": route,
            "reply": text,
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        assert route == case["expect_route"]
        if case.get("expect_message"):
            assert text == case["expect_message"]
    print("[OK] live routing tests")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true")
    args = parser.parse_args()
    test_resolve_triage_route()
    if args.live:
        run_live_tests()
    else:
        print("Skip live tests (use --live)")


if __name__ == "__main__":
    main()
