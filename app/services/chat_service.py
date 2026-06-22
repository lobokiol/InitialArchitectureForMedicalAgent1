from typing import Optional, Dict, Any

from langchain_core.messages import HumanMessage, AIMessage, BaseMessage

from app.core.logging import logger
from app.domain.models import AppState, IntentResult, RetrievedDoc
from app.domain.routing import is_dept_followup_reply
from app.graph.builder import build_app
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
    logger.info("chat_once called (user_id=%s, thread_id=%s)", user_id, thread_id)

    thread_id = _ensure_thread(user_id, thread_id)

    pre_state = _read_checkpoint_state(thread_id, user_id)
    was_dept_followup = bool(pre_state and is_dept_followup_reply(pre_state))

    inputs = {"messages": [HumanMessage(content=message)]}
    config = {
        "configurable": {
            "thread_id": thread_id,
            "user_id": user_id,
        }
    }

    state = _app.invoke(inputs, config=config)
    if isinstance(state, dict):
        state = AppState(**state)

    reply = _extract_reply(state.messages)

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
    awaiting = bool(ds and ds.status == "asking" and ds.last_choices)
    dept_choices = [c.model_dump() for c in ds.last_choices] if awaiting and ds else []

    return {
        "user_id": user_id,
        "thread_id": thread_id,
        "reply": reply,
        "intent_result": intent_dict,
        "used_docs": {
            "medical": _dump_docs(state.medical_docs),
            "process": _dump_docs(state.process_docs),
        },
        "awaiting_dept_choice": awaiting,
        "dept_choices": dept_choices,
    }


def get_session_manager() -> SessionManager:
    return _session_manager
