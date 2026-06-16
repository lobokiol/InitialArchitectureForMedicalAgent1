
from typing import List

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

from app.core.logging import logger
from app.domain.models import AppState, RetrievedDoc


ANSWER_PROMPT = """
你是一个专业的医疗导诊助手。

下面是你和用户目前为止的对话历史：
{history_block}

---

用户当前问题：{user_query}

【医学文档】
{medical_block}

【流程文档】
{process_block}

【工具调用结果】
{tool_result_block}

回答要求：
- 优先利用医学文档和流程文档中的信息作答
- 如果文档中没有涉及、但从对话历史中可以推断（例如用户的自我介绍、之前提过的偏好等），也可以基于历史回答
- 如果仍然无法回答，要老实说明：“根据现有资料无法确定”
"""


def _fmt_docs(docs: List[RetrievedDoc], max_docs: int = 8) -> str:
    '''
        这段代码的功能是格式化文档列表并返回字符串：

    1. 若输入为空，返回"（无结果）"
    2. 按分数降序排序文档（None值排在最后）
    3. 取前max_docs个文档
    4. 为每个文档生成带编号和分数的字符串
    5. 用换行符连接所有文档字符串并返回
    '''
    if not docs:
        return "（无结果）"

    docs_sorted = sorted(
        docs,
        key=lambda d: (d.score is None, -(d.score or 0.0))
    )
    selected = docs_sorted[:max_docs]

    out = []
    for i, d in enumerate(selected, 1):
        score_str = f"(score={d.score:.3f})" if d.score is not None else ""
        out.append(f"- 文档{i}{score_str}: {d.content}")
    return "\n".join(out)

def _fmt_tool_result(tool_result) -> str:
    """格式化工具调用结果"""
    if not tool_result:
        return "（无工具调用结果）"
    
    # 直接返回字符串（如果已经是字符串）
    if isinstance(tool_result, str):
        return tool_result
    
    # 如果是字典且包含 messages
    if isinstance(tool_result, dict) and "messages" in tool_result:
        parts = []
        for msg in tool_result["messages"]:
            if hasattr(msg, 'content'):
                parts.append(msg.content)
        return "\n".join(parts)
    
    return str(tool_result)
def format_history(messages: list[BaseMessage]) -> str:
    if not messages:
        return "（无历史对话）"

    lines: List[str] = []
    for m in messages:
        if m is None:
            continue
        if isinstance(m, HumanMessage):
            role = "用户"
        elif isinstance(m, AIMessage):
            role = "助手"
        else:
            role = "系统"
        lines.append(f"{role}：{m.content}")

    return "\n".join(lines)


def _new_intake_prefix(state: AppState) -> str:
    msgs = state.messages or []
    if len(msgs) < 3:
        return ""
    prior_humans = [
        m.content for m in msgs[:-1] if isinstance(m, HumanMessage) and isinstance(m.content, str)
    ]
    cur = (state.ner_result.query if state.ner_result else "") or ""
    if prior_humans and cur and cur.strip() != str(prior_humans[-1]).strip():
        return "（已按您本轮新描述重新评估）"
    return ""


def answer_generate_node(state: AppState) -> dict:
    logger.info(">>> Enter node: answer_generate")
    prefix = _new_intake_prefix(state)
    if prefix:
        prefix = prefix + "\n"

    if state.locked_department:
        chunk = state.rag_chunk or {}
        canonical = chunk.get("canonical_symptom") or (state.slot_table.primary_symptom if state.slot_table else "")
        dept = state.locked_department
        ds = state.dept_state
        status = getattr(ds, "status", None) if ds else None
        if dept == "急诊":
            flag = chunk.get("emergency_flag") or {}
            detail = flag.get("suggestion") or flag.get("condition") or "请尽快就医。"
            full_content = (
                f"{prefix}根据您描述的情况（{canonical}），建议尽快就诊：**急诊**。\n{detail}"
            )
        elif status == "fallback":
            full_content = (
                f"{prefix}根据您描述的症状（{canonical}），建议首选就诊：**{dept}**。"
                "如与实际情况不符，请补充更多细节以便更准确推荐。"
            )
        else:
            full_content = f"{prefix}根据您描述的症状（{canonical}），建议就诊科室：**{dept}**。"
        return {"messages": [AIMessage(content=full_content)]}

    ddr = state.disease_dept_result
    if ddr and ddr.departments:
        names = "、".join(ddr.diseases) if ddr.diseases else "您的描述"
        dept_names: list[str] = []
        for item in ddr.departments[:3]:
            if isinstance(item, dict):
                d = item.get("dept") or item.get("department") or ""
                if d:
                    dept_names.append(str(d))
            elif item:
                dept_names.append(str(item))
        dept_str = "、".join(dept_names) if dept_names else "导诊台"
        full_content = f"{prefix}根据您提到的「{names}」，建议就诊科室：**{dept_str}**。"
        return {"messages": [AIMessage(content=full_content)]}

    logger.info(
        "answer_generate_node: no locked dept / disease depts, using fixed fallback (LLM disabled)"
    )

    # Legacy Milvus/RAG Q&A — not used in production triage graph.
    # user_query = state.messages[-1].content
    # history_block = format_history(state.messages)
    # prompt = ANSWER_PROMPT.format(
    #     history_block=history_block,
    #     user_query=user_query,
    #     medical_block=_fmt_docs(state.medical_docs),
    #     process_block=_fmt_docs(state.process_docs),
    #     tool_result_block=_fmt_tool_result(state.tool_call_result),
    # )
    # try:
    #     result = get_chat_llm().invoke([HumanMessage(content=prompt)])
    #     full_content = result.content if isinstance(result, AIMessage) else getattr(result, "content", "")
    # except Exception:
    #     logger.exception("answer_generate_node LLM 调用失败，返回兜底回答")
    #     full_content = "抱歉，当前系统生成答案时出现了问题，请稍后再试。"

    full_content = (
        f"{prefix}根据现有资料无法匹配导诊结果，建议先到**导诊台**咨询，或补充更具体的症状描述。"
    )

    return {"messages": [AIMessage(content=full_content)]}

# ----------------------------
# from typing import List, Optional

# from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

# from app.core.logging import logger
# from app.core.llm import get_chat_llm
# from app.domain.models import AppState, RetrievedDoc


# DEPARTMENT_PROMPT = """
# 你是一个医疗导诊助手。

# 根据用户的问题推断应该就诊的科室。

# 规则：
# - 只返回一个科室名称，如"内分泌科"、"消化内科"、"妇科"等
# - 如果问题是纯咨询、不需要就医，返回"无需就医"
# - 如果不确定，返回"导诊台"

# 用户问题：{user_query}
# """


# def _get_department(user_query: str) -> Optional[str]:
#     try:
#         query_str = str(user_query) if user_query else ""
#         prompt = DEPARTMENT_PROMPT.format(user_query=query_str)
#         result = get_chat_llm().invoke([HumanMessage(content=prompt)])
#         content = getattr(result, "content", "") or ""
#         try:
#             if isinstance(content, list):
#                 content = content[0]
#             if hasattr(content, "text"):
#                 content = content.text
#             dept = str(content).strip()
#         except Exception:
#             dept = str(content).strip() if content else ""
#         logger.info("get_department: query=%s, dept=%s", query_str, dept)
#         return dept if dept else None
#     except Exception:
#         logger.exception("get_department 调用失败")
#         return None


# ANSWER_PROMPT = """
# 你是一个专业的医疗导诊助手。

# 下面是你和用户目前为止的对话历史：
# {history_block}

# ---

# 用户当前问题：{user_query}

# 【医学文档】
# {medical_block}

# 【流程文档】
# {process_block}

# 回答要求：
# - 优先利用医学文档和流程文档中的信息作答
# - 如果文档中没有涉及、但从对话历史中可以推断（例如用户的自我介绍、之前提过的偏好等），也可以基于历史回答
# - 如果仍然无法回答，要老实说明：“根据现有资料无法确定”
# """


# def _fmt_docs(docs: List[RetrievedDoc], max_docs: int = 8) -> str:
#     if not docs:
#         return "（无结果）"

#     docs_sorted = sorted(docs, key=lambda d: (d.score is None, -(d.score or 0.0)))
#     selected = docs_sorted[:max_docs]

#     out = []
#     for i, d in enumerate(selected, 1):
#         score_str = f"(score={d.score:.3f})" if d.score is not None else ""
#         out.append(f"- 文档{i}{score_str}: {d.content}")
#     return "\n".join(out)


# def format_history(messages: list[BaseMessage]) -> str:
#     if not messages:
#         return "（无历史对话）"

#     lines: List[str] = []
#     for m in messages:
#         if m is None:
#             continue
#         if isinstance(m, HumanMessage):
#             role = "用户"
#         elif isinstance(m, AIMessage):
#             role = "助手"
#         else:
#             role = "系统"
#         lines.append(f"{role}：{m.content}")

#     return "\n".join(lines)


# def answer_generate_node(state: AppState) -> dict:
#     logger.info(">>> Enter node: answer_generate")
#     user_query = state.messages[-1].content

#     history_block = format_history(state.messages)

#     prompt = ANSWER_PROMPT.format(
#         history_block=history_block,
#         user_query=user_query,
#         medical_block=_fmt_docs(state.medical_docs),
#         process_block=_fmt_docs(state.process_docs),
#     )

#     logger.info(
#         "answer_generate_node: medical_docs=%d, process_docs=%d",
#         len(state.medical_docs),
#         len(state.process_docs),
#     )

#     try:
#         result = get_chat_llm().invoke([HumanMessage(content=prompt)])
#         full_content = (
#             result.content
#             if isinstance(result, AIMessage)
#             else getattr(result, "content", "")
#         )
#     except Exception:
#         logger.exception("answer_generate_node LLM 调用失败，返回兜底回答")
#         full_content = "抱歉，当前系统生成答案时出现了问题，请稍后再试。"

#     if state.intent_result and state.intent_result.has_process:
#         query_str = str(user_query) if user_query else ""
#         dept = _get_department(query_str)
#         if dept and dept != "无需就医":
#             full_content += f"\n\n如需就医，建议就诊科室：{dept}"

#     return {"messages": [AIMessage(content=full_content)]}
