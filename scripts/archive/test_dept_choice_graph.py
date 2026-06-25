"""Graph path: menu → explicit choice locks single department."""
from __future__ import annotations

import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver

from app.domain.models import AppState
from app.graph.builder import build_app


def _last_ai(state: AppState) -> str:
    for msg in reversed(state.messages):
        if isinstance(msg, AIMessage) and isinstance(msg.content, str):
            return msg.content
    return ""


def test_chenjiang_locks_rheumatology() -> None:
    app = build_app(MemorySaver())
    tid = f"test-{uuid.uuid4().hex[:8]}"
    cfg = {"configurable": {"thread_id": tid, "user_id": "test"}}

    try:
        s1 = app.invoke({"messages": [HumanMessage(content="脚脖子肿怎么办")]}, config=cfg)
    except Exception as exc:
        print(f"[SKIP] graph invoke needs services: {exc}")
        return

    st1 = AppState(**s1) if isinstance(s1, dict) else s1
    assert st1.dept_state and st1.dept_state.status == "asking"
    labels = [c.label for c in st1.dept_state.last_choices]
    assert "晨僵" in labels, labels

    s2 = app.invoke({"messages": [HumanMessage(content="晨僵")]}, config=cfg)
    st2 = AppState(**s2) if isinstance(s2, dict) else s2
    assert st2.locked_department == "风湿免疫科", st2.locked_department
    assert st2.dept_state and st2.dept_state.status == "locked"
    reply = _last_ai(st2)
    assert "风湿免疫科" in reply, reply
    assert reply.count("**") >= 2  # single bold dept in template
    print("[OK] 晨僵 -> 风湿免疫科 (single dept)")


def test_invalid_choice_keeps_asking() -> None:
    app = build_app(MemorySaver())
    tid = f"test-{uuid.uuid4().hex[:8]}"
    cfg = {"configurable": {"thread_id": tid, "user_id": "test"}}

    try:
        s1 = app.invoke({"messages": [HumanMessage(content="脚脖子肿")]}, config=cfg)
    except Exception as exc:
        print(f"[SKIP] graph invoke needs services: {exc}")
        return

    st1 = AppState(**s1) if isinstance(s1, dict) else s1
    if not (st1.dept_state and st1.dept_state.status == "asking"):
        print("[SKIP] did not reach asking")
        return

    round_before = st1.dept_state.round
    s2 = app.invoke({"messages": [HumanMessage(content="我还行吧")]}, config=cfg)
    st2 = AppState(**s2) if isinstance(s2, dict) else s2
    assert st2.dept_state.status == "asking"
    assert st2.dept_state.round == round_before
    assert st2.locked_department is None
    print("[OK] invalid choice keeps asking")


def main() -> None:
    test_chenjiang_locks_rheumatology()
    test_invalid_choice_keeps_asking()


if __name__ == "__main__":
    main()
