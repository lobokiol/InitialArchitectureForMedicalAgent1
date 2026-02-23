from langgraph.graph import StateGraph, START, END

from app.domain.models import AppState
from app.domain.routing import (
    route_after_decision,
    route_after_es,
    route_after_docs,
)
from app.graph.nodes.decision import decision_node
from app.graph.nodes.es_rag import es_rag_node
from app.graph.nodes.milvus_rag import milvus_rag_node
from app.graph.nodes.check_docs import check_docs_node
from app.graph.nodes.rewrite import rewrite_question
from app.graph.nodes.answer import answer_generate_node
from app.graph.nodes.trim_history import trim_history_node
from app.graph.nodes.tool_calling import tool_calling_node  # 新增


def build_graph() -> StateGraph:
    graph = StateGraph(AppState)

    graph.add_node("decision", decision_node)
    graph.add_node("tool_calling", tool_calling_node)
    graph.add_node("es_rag", es_rag_node)
    graph.add_node("milvus_rag", milvus_rag_node)
    graph.add_node("check_docs", check_docs_node)
    graph.add_node("rewrite_question", rewrite_question)
    graph.add_node("answer_generate", answer_generate_node)
    graph.add_node("trim_history", trim_history_node) 
    graph.add_edge(START, "trim_history")
    graph.add_edge("trim_history", "decision")

    graph.add_conditional_edges("decision", route_after_decision, {
        "answer_generate": "answer_generate",
        "es_rag": "es_rag",
        "tool_calling": "tool_calling",
    })
    graph.add_edge("tool_calling", "es_rag")  # 工具调用后继续检索
    # graph.add_edge("tool_calling", "answer_generate")   #  工具调用后生成答案
    graph.add_conditional_edges("es_rag", route_after_es, {
        "milvus_rag": "milvus_rag",
        "check_docs": "check_docs",
    })

    graph.add_edge("milvus_rag", "check_docs")

    graph.add_conditional_edges("check_docs", route_after_docs, {
        "answer_generate": "answer_generate",
        "rewrite_question": "rewrite_question",
    })

    graph.add_edge("rewrite_question", "es_rag")
    graph.add_edge("answer_generate", END)
    
    return graph


def build_app(checkpointer):
    graph = build_graph()
    return graph.compile(checkpointer=checkpointer)
