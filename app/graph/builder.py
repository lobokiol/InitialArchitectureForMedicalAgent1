from langgraph.graph import StateGraph, START, END

from app.domain.models import AppState
from app.domain.routing import (
    route_after_clarify,
    route_after_confidence,
    route_after_dept,
    route_after_dept_rules,
    route_after_emergency_gate,
    route_after_rag,
    route_after_slot_gate,
    route_after_trim,
)
from app.graph.nodes.decision import decision_node
from app.graph.nodes.emergency_gate import emergency_gate_node
from app.graph.nodes.dept_confidence import dept_confidence_node, low_confidence_reject_node
from app.graph.nodes.dept_disambiguation import dept_disambiguation_node
from app.graph.nodes.dept_rules_disambiguation import dept_rules_disambiguation_node
from app.graph.nodes.disease_dept import disease_dept_node
from app.graph.nodes.fetch_oncall import fetch_oncall_node
from app.graph.nodes.mcp_followup import mcp_followup_node
from app.graph.nodes.reject import reject_node
from app.graph.nodes.answer import answer_generate_node
from app.graph.nodes.trim_history import trim_history_node
from app.graph.nodes.slot_fill import slot_fill_node
from app.graph.nodes.slot_gate import slot_gate_node
from app.graph.nodes.rag_miss_reject import rag_miss_reject_node
from app.graph.nodes.rag_symptom_recall import rag_symptom_recall_node
from app.graph.nodes.symptom_clarify import symptom_clarify_node


def build_graph() -> StateGraph:
    """
    导诊主图（槽位门禁 + CL 澄清 + department_rules 多选 + LLM 置信度门禁）：
    """
    graph = StateGraph(AppState)

    graph.add_node("trim_history", trim_history_node)
    graph.add_node("decision", decision_node)
    graph.add_node("slot_fill", slot_fill_node)
    graph.add_node("emergency_gate", emergency_gate_node)
    graph.add_node("slot_gate", slot_gate_node)
    graph.add_node("disease_dept", disease_dept_node)
    graph.add_node("rag_symptom_recall", rag_symptom_recall_node)
    graph.add_node("symptom_clarify", symptom_clarify_node)
    graph.add_node("dept_rules_disambiguation", dept_rules_disambiguation_node)
    graph.add_node("dept_disambiguation", dept_disambiguation_node)
    graph.add_node("dept_confidence", dept_confidence_node)
    graph.add_node("low_confidence_reject", low_confidence_reject_node)
    graph.add_node("fetch_oncall", fetch_oncall_node)
    graph.add_node("mcp_followup", mcp_followup_node)
    graph.add_node("reject", reject_node)
    graph.add_node("rag_miss_reject", rag_miss_reject_node)
    graph.add_node("answer_generate", answer_generate_node)

    graph.add_edge(START, "trim_history")
    graph.add_conditional_edges(
        "trim_history",
        route_after_trim,
        {
            "decision": "decision",
            "symptom_clarify": "symptom_clarify",
            "dept_rules_disambiguation": "dept_rules_disambiguation",
            "dept_disambiguation": "dept_disambiguation",
            "mcp_followup": "mcp_followup",
        },
    )

    graph.add_edge("decision", "slot_fill")
    graph.add_edge("slot_fill", "emergency_gate")
    graph.add_conditional_edges(
        "emergency_gate",
        route_after_emergency_gate,
        {
            "answer_generate": "answer_generate",
            "slot_gate": "slot_gate",
        },
    )
    graph.add_conditional_edges(
        "slot_gate",
        route_after_slot_gate,
        {
            "disease_dept": "disease_dept",
            "rag_symptom_recall": "rag_symptom_recall",
            "reject": "reject",
        },
    )

    graph.add_edge("disease_dept", "fetch_oncall")
    graph.add_conditional_edges(
        "rag_symptom_recall",
        route_after_rag,
        {
            "symptom_clarify": "symptom_clarify",
            "dept_disambiguation": "dept_disambiguation",
            "rag_miss_reject": "rag_miss_reject",
        },
    )
    graph.add_conditional_edges(
        "symptom_clarify",
        route_after_clarify,
        {
            "end_ask": END,
            "dept_rules_disambiguation": "dept_rules_disambiguation",
            "answer_generate": "fetch_oncall",
        },
    )
    graph.add_conditional_edges(
        "dept_rules_disambiguation",
        route_after_dept_rules,
        {
            "end_ask": END,
            "dept_confidence": "dept_confidence",
        },
    )
    graph.add_conditional_edges(
        "dept_disambiguation",
        route_after_dept,
        {
            "end_ask": END,
            "dept_confidence": "dept_confidence",
            "answer_generate": "fetch_oncall",
        },
    )
    graph.add_conditional_edges(
        "dept_confidence",
        route_after_confidence,
        {
            "answer_generate": "fetch_oncall",
            "low_confidence_reject": "low_confidence_reject",
        },
    )

    graph.add_edge("fetch_oncall", "answer_generate")
    graph.add_edge("mcp_followup", END)
    graph.add_edge("reject", END)
    graph.add_edge("rag_miss_reject", END)
    graph.add_edge("low_confidence_reject", END)
    graph.add_edge("answer_generate", END)

    return graph


def build_app(checkpointer):
    graph = build_graph()
    return graph.compile(checkpointer=checkpointer)
