"""Serialize AppState for debug / CLI display."""

from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from app.domain.models import AppState, IntentResult, RetrievedDoc


def _dump_message(msg: BaseMessage) -> dict[str, str]:
    if isinstance(msg, HumanMessage):
        role = "human"
    elif isinstance(msg, AIMessage):
        role = "ai"
    else:
        role = "system"
    content = msg.content if isinstance(msg.content, str) else str(msg.content)
    return {"role": role, "content": content}


def _dump_optional(obj: Any) -> Any:
    if obj is None:
        return None
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if isinstance(obj, (dict, list, str, int, float, bool)):
        return obj
    return str(obj)


def _dump_docs(docs: list[RetrievedDoc]) -> list[dict]:
    return [d.model_dump() for d in docs]


def _dump_rag_chunk(chunk: dict | None) -> dict | None:
    if not chunk:
        return None
    out = dict(chunk)
    out.pop("embedding", None)
    return out


def dump_app_state(state: AppState) -> dict[str, Any]:
    return {
        "messages": [_dump_message(m) for m in (state.messages or [])],
        "intent_result": (
            state.intent_result.model_dump()
            if isinstance(state.intent_result, IntentResult)
            else _dump_optional(state.intent_result)
        ),
        "ner_result": _dump_optional(state.ner_result),
        "disease_dept_result": _dump_optional(state.disease_dept_result),
        "symptom_slot_result": _dump_optional(state.symptom_slot_result),
        "medical_docs": _dump_docs(state.medical_docs),
        "process_docs": _dump_docs(state.process_docs),
        "relevance_result": _dump_optional(state.relevance_result),
        "rewrite_attempts": state.rewrite_attempts,
        "need_tool_call": state.need_tool_call,
        "tool_call_result": _dump_optional(state.tool_call_result),
        "slot_table": _dump_optional(state.slot_table),
        "slot_gate_passed": state.slot_gate_passed,
        "rag_chunk_id": state.rag_chunk_id,
        "rag_chunk": _dump_rag_chunk(state.rag_chunk),
        "dept_state": _dump_optional(state.dept_state),
        "locked_department": state.locked_department,
    }
