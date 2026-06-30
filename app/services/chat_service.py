from typing import Optional, Dict, Any

from langchain_core.messages import HumanMessage, AIMessage, BaseMessage

from app.core.logging import logger
from app.domain.models import AppState, IntentResult, RetrievedDoc
from app.domain.state_debug import dump_app_state
from app.domain.routing import is_awaiting_triage_followup
from app.graph.builder import build_app
from app.graph.nodes.fetch_oncall import resolve_department
from app.infra.redis_client import checkpointer
from app.services.triage_recorder import TriageSessionRecorder
from app.sessions.manager import SessionManager


_app = build_app(checkpointer)
_session_manager = SessionManager()
_recorder = TriageSessionRecorder()


def _extract_reply(messages: list[BaseMessage]) -> str:
    if not messages:
        return ""
    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            return msg.content
    return messages[-1].content if messages else ""


def _ensure_thread(user_id: str, thread_id: Optional[str]) -> str:
    if thread_id:
        _session_manager.set_current_thread(user_id, thread_id)
        return thread_id
    existing = _session_manager.get_current_thread(user_id)
    if existing:
        return existing
    return _session_manager.create_thread(user_id, title="默认对话")


def _read_checkpoint_state(thread_id: str, user_id: str) -> AppState | None:
    cfg = {"configurable": {"thread_id": thread_id, "user_id": user_id}}
    try:
        snap = _app.get_state(cfg)
        if snap and snap.values:
            return AppState(**snap.values)
    except Exception:
        logger.debug("checkpoint read failed thread_id=%s", thread_id, exc_info=True)
    return None


def chat_once(
    user_id: str,
    thread_id: Optional[str],
    message: str,
) -> Dict[str, Any]:
    """
    Synchronous entry for chat; backend LangGraph uses synchronous invoke.
    """
    logger.info("chat_once called (user_id=%s, thread_id=%s) message=%r", user_id, thread_id, message)

    thread_id = _ensure_thread(user_id, thread_id)

    pre_state = _read_checkpoint_state(thread_id, user_id)
    was_dept_followup = bool(pre_state and is_awaiting_triage_followup(pre_state))

    inputs = {"messages": [HumanMessage(content=message)]}
    config = {
        "configurable": {
            "thread_id": thread_id,
            "user_id": user_id,
        }
    }

    node_trace: list[str] = []
    for chunk in _app.stream(inputs, config=config, stream_mode="updates"):
        for node_name in chunk:
            node_trace.append(node_name)

    snap = _app.get_state(config)
    if not snap or not snap.values:
        raise RuntimeError("LangGraph finished without checkpoint state")
    state = AppState(**snap.values)

    reply = _extract_reply(state.messages)
    trace_line = " → ".join(node_trace) if node_trace else "(empty)"
    logger.info("--- 节点流转 ---")
    logger.info("%s", trace_line)
    logger.info("--- 回复 --- %s", reply)

    try:
        _recorder.record_turn(
            user_id=user_id,
            thread_id=thread_id,
            user_message=message,
            assistant_reply=reply,
            state=state,
            was_dept_followup=was_dept_followup,
        )
    except Exception:
        logger.exception("triage recorder failed (non-fatal)")

    _session_manager.touch_thread(thread_id)

    def _dump_docs(docs: list[RetrievedDoc]):
        return [d.model_dump() for d in docs]

    intent_dict = state.intent_result.model_dump() if isinstance(state.intent_result, IntentResult) else None

    ds = state.dept_state
    cs = state.clarify_state
    awaiting_dept = bool(ds and ds.status == "asking" and ds.last_choices)
    awaiting_clarify = bool(
        cs and cs.status == "asking" and cs.last_choices and cs.phase in ("age", "sex", "pain_location")
    )
    dept_choices = [c.model_dump() for c in ds.last_choices] if awaiting_dept and ds else []
    clarify_choices = [c.model_dump() for c in cs.last_choices] if awaiting_clarify and cs else []
    multi_select = bool(ds and ds.multi_select) if ds else False
    conf = state.dept_confidence_result

    return {
        "user_id": user_id,
        "thread_id": thread_id,
        "reply": reply,
        "intent_result": intent_dict,
        "used_docs": {
            "medical": _dump_docs(state.medical_docs),
            "process": _dump_docs(state.process_docs),
        },
        "awaiting_dept_choice": awaiting_dept,
        "dept_choices": dept_choices,
        "awaiting_clarify": awaiting_clarify,
        "clarify_phase": cs.phase if cs else None,
        "clarify_choices": clarify_choices,
        "multi_select": multi_select,
        "dept_confidence": conf.score if conf else None,
        "dept_confidence_passed": state.dept_confidence_passed,
        "dept_confidence_reason": conf.reason if conf else None,
        "locked_department": state.locked_department,
        "recommended_department": resolve_department(state),
        "oncall_appointments": [d.model_dump() for d in state.oncall_appointments],
        "oncall_fetch_error": state.oncall_fetch_error,
        "node_trace": node_trace,
        "app_state": dump_app_state(state),
    }


def get_session_manager() -> SessionManager:
    return _session_manager
