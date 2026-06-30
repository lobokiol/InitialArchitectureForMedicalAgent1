from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from langchain_core.messages import HumanMessage
from pydantic import BaseModel, ConfigDict

from app.core import config
from app.core.logging import logger
from app.domain.models import AppState
from app.mcp.client import (
    fetch_department_intro_sync,
    fetch_department_route_sync,
    get_mcp_client,
)
from app.ner.catalog_scan import load_entity_catalog, scan_catalog_substrings

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEPT_INFO_KEYWORDS = (
    "怎么走",
    "怎么去",
    "在哪",
    "在哪里",
    "哪里",
    "路线",
    "介绍",
    "看什么",
    "擅长",
    "什么科",
    "楼层",
    "电话",
    "怎么去",
    "如何走",
)

NO_DEPT_CLARIFY = "请先完成导诊推荐科室，或告诉我您要查询的科室名称。"
ASK_INTRO_OR_ROUTE = "请问您想了解科室介绍，还是来院路线？"
MCP_UNAVAILABLE = "暂无法获取科室信息，请稍后重试或到医院导诊台咨询。"


class FollowupToolPick(BaseModel):
    tool_name: Literal["get_department_intro", "get_department_route", "none"]
    reason: str = ""

    model_config = ConfigDict(extra="ignore")


def known_departments() -> list[str]:
    path = _PROJECT_ROOT / "hospital_mcp" / "mock" / "departments.json"
    with path.open(encoding="utf-8") as f:
        return list(json.load(f).keys())


def _last_human_text(state: AppState) -> str:
    msgs = state.messages or []
    for msg in reversed(msgs):
        if isinstance(msg, HumanMessage) and isinstance(msg.content, str):
            return msg.content.strip()
    return ""


def looks_like_new_triage(text: str) -> bool:
    cat = load_entity_catalog()
    scanned = scan_catalog_substrings(text, cat["主症"], cat["疾病"])
    return bool(scanned.symptom_spans or scanned.disease_spans)


def looks_like_dept_info_query(text: str) -> bool:
    return any(kw in text for kw in _DEPT_INFO_KEYWORDS)


def match_department_in_text(text: str) -> str | None:
    for dept in sorted(known_departments(), key=len, reverse=True):
        if dept in text:
            return dept
    return None


def resolve_followup_department(state: AppState) -> str | None:
    """用户句中明确科室优先，否则用上一轮推荐科室。"""
    text = _last_human_text(state)
    explicit = match_department_in_text(text)
    if explicit:
        return explicit
    return state.last_recommended_department


def resolve_recommended_department(state: AppState) -> str | None:
    """API/前端展示用：当前轮用户指定科室 > 导诊锁定科室 > last_recommended。"""
    explicit = match_department_in_text(_last_human_text(state))
    if explicit:
        return explicit
    from app.graph.nodes.fetch_oncall import resolve_department

    return resolve_department(state) or state.last_recommended_department


def _llm_pick_tool(user_text: str, department: str, *, user_specified_dept: bool) -> FollowupToolPick:
    from app.core.llm import get_chat_llm

    tools = asyncio_run_list_tools()
    tool_lines = "\n".join(f"- {t.name}: {t.description or ''}" for t in tools)
    dept_line = (
        f"用户本次指定的科室：{department}（以该科室为准）"
        if user_specified_dept
        else f"上一轮推荐科室：{department}"
    )
    prompt = f"""你是医院导诊助手。用户正在追问科室相关信息。
{dept_line}
用户问题：{user_text}

可用 MCP 工具：
{tool_lines}

请选择最合适的一个工具名；若与科室介绍、来院路线无关则选 none。
只返回 JSON。"""
    structured = get_chat_llm().with_structured_output(FollowupToolPick)
    result = structured.invoke([HumanMessage(content=prompt)])
    if not isinstance(result, FollowupToolPick):
        result = FollowupToolPick.model_validate(result)
    return result


def asyncio_run_list_tools():
    import asyncio

    return asyncio.run(get_mcp_client().list_tools(followup_only=True))


def _llm_format_reply(user_text: str, tool_name: str, payload: dict) -> str:
    from app.core.llm import get_chat_llm

    prompt = f"""根据医院 MCP 工具返回的 JSON，用简洁中文 Markdown 回答用户。
用户问题：{user_text}
工具：{tool_name}
数据：
{json.dumps(payload, ensure_ascii=False, indent=2)}

要求：
- 路线类请用有序列表展示 steps
- 介绍类请涵盖 summary、scope、visit_tips、floor、phone
- 若 JSON 含 error=department_not_found，礼貌提示未找到科室
- 不要编造 JSON 中没有的信息"""
    llm = get_chat_llm()
    result = llm.invoke([HumanMessage(content=prompt)])
    return str(result.content)


def run_mcp_followup(state: AppState) -> dict:
    if not config.MCP_ENABLED or not config.MCP_FOLLOWUP_ENABLED:
        return {}

    dept = resolve_followup_department(state)
    if not dept:
        from langchain_core.messages import AIMessage

        return {"messages": [AIMessage(content=NO_DEPT_CLARIFY)]}

    user_text = _last_human_text(state)
    explicit_dept = match_department_in_text(user_text)
    try:
        pick = _llm_pick_tool(user_text, dept, user_specified_dept=bool(explicit_dept))
        if pick.tool_name == "none":
            from langchain_core.messages import AIMessage

            return {"messages": [AIMessage(content=ASK_INTRO_OR_ROUTE)]}

        if pick.tool_name == "get_department_intro":
            data = fetch_department_intro_sync(dept)
        else:
            data = fetch_department_route_sync(dept)

        payload = data.model_dump() if hasattr(data, "model_dump") else dict(data)
        reply = _llm_format_reply(user_text, pick.tool_name, payload)
        from langchain_core.messages import AIMessage

        patch = {
            "messages": [AIMessage(content=reply)],
            "tool_call_result": {"tool": pick.tool_name, "department": dept, "data": payload},
            "last_recommended_department": dept,
        }
        return patch
    except Exception:
        logger.exception("mcp_followup failed dept=%s", dept)
        from langchain_core.messages import AIMessage

        return {"messages": [AIMessage(content=MCP_UNAVAILABLE)]}
