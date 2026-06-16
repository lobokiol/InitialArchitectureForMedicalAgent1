from __future__ import annotations

import uuid

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver

from app.domain.models import AppState
from app.domain.triage_intent import REJECT_MESSAGE
from app.graph.builder import build_app

_MAX_TURNS = 4


def _extract_reply(messages: list) -> str:
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and isinstance(msg.content, str):
            return msg.content
    return ""


def _dept_status(state: AppState) -> str | None:
    ds = state.dept_state
    return getattr(ds, "status", None) if ds else None


def invoke_once(app, thread_id: str, message: str) -> AppState:
    config = {"configurable": {"thread_id": thread_id, "user_id": "foot-eval"}}
    out = app.invoke({"messages": [HumanMessage(content=message)]}, config=config)
    if isinstance(out, dict):
        return AppState(**out)
    return out


def run_single_case(app, case: dict) -> tuple[AppState, str]:
    thread_id = f"eval-{case['id']}-{uuid.uuid4().hex[:6]}"
    state = invoke_once(app, thread_id, case["message"])
    return state, _extract_reply(state.messages)


def run_multiturn_case(app, case: dict) -> tuple[AppState, str]:
    thread_id = f"eval-{case['id']}-{uuid.uuid4().hex[:6]}"
    turns = case.get("turns") or []
    state: AppState | None = None
    reply = ""
    for turn in turns:
        role = turn.get("role")
        if role == "user":
            msg = turn.get("message", "")
            state = invoke_once(app, thread_id, msg)
            reply = _extract_reply(state.messages)
        elif role == "assistant" and turn.get("expect_asking") and state:
            if _dept_status(state) != "asking":
                break
    if state is None:
        state = AppState()
    return state, reply


def _first_disease_dept(state: AppState) -> str | None:
    ddr = state.disease_dept_result
    if not ddr or not ddr.departments:
        return None
    item = ddr.departments[0]
    if isinstance(item, dict):
        return item.get("dept") or item.get("department")
    return str(item) if item else None


def evaluate_graph_case(app, case: dict) -> dict:
    subset = case.get("subset")
    if subset == "F":
        state, reply = run_multiturn_case(app, case)
    else:
        state, reply = run_single_case(app, case)

    route = state.intent_result.triage_route if state.intent_result else None
    locked = state.locked_department
    ok = False
    got: str | None = None

    if subset == "B":
        got = locked
        ok = locked == case.get("expect_dept")
    elif subset == "C":
        got = _first_disease_dept(state)
        ok = got == case.get("expect_dept") and route == "disease"
    elif subset == "D":
        got = route
        ok = (
            route == "reject"
            and reply.strip() == REJECT_MESSAGE
            and locked is None
        )
    elif subset == "E":
        got = locked
        ok = locked == "急诊"
    elif subset == "F":
        got = locked
        ok = locked == case.get("expect_dept")

    return {
        "id": case["id"],
        "subset": subset,
        "message": case.get("message") or str(case.get("turns", ""))[:80],
        "expect": case.get("expect_dept") or case.get("expect_route"),
        "got": got,
        "route": route,
        "locked_department": locked,
        "reply_preview": reply[:120] if reply else "",
        "dept_status": _dept_status(state),
        "ok": ok,
    }


def run_graph_eval(cases: list[dict]) -> tuple[list[dict], int, int]:
    app = build_app(MemorySaver())
    results: list[dict] = []
    passed = 0
    for c in cases:
        r = evaluate_graph_case(app, c)
        results.append(r)
        if r["ok"]:
            passed += 1
        mark = "OK" if r["ok"] else "FAIL"
        print(
            f"[{mark}] {r['id']} subset={r['subset']} "
            f"expect={r['expect']!r} got={r['got']!r} status={r.get('dept_status')}"
        )
    return results, passed, len(cases)
