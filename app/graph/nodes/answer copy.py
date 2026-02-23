from typing import List

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

from app.core.logging import logger
from app.core.llm import get_chat_llm
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

回答要求：
- 优先利用医学文档和流程文档中的信息作答
- 如果文档中没有涉及、但从对话历史中可以推断（例如用户的自我介绍、之前提过的偏好等），也可以基于历史回答
- 如果仍然无法回答，要老实说明：“根据现有资料无法确定”
"""


def _fmt_docs(docs: List[RetrievedDoc], max_docs: int = 8) -> str:
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


def answer_generate_node(state: AppState) -> dict:
    logger.info(">>> Enter node: answer_generate")
    user_query = state.messages[-1].content

    history_block = format_history(state.messages)

    prompt = ANSWER_PROMPT.format(
        history_block=history_block,
        user_query=user_query,
        medical_block=_fmt_docs(state.medical_docs),
        process_block=_fmt_docs(state.process_docs)
    )

    logger.info(
        "answer_generate_node: medical_docs=%d, process_docs=%d",
        len(state.medical_docs),
        len(state.process_docs),
    )

    try:
        result = get_chat_llm().invoke([HumanMessage(content=prompt)])
        full_content = result.content if isinstance(result, AIMessage) else getattr(result, "content", "")
    except Exception:
        logger.exception("answer_generate_node LLM 调用失败，返回兜底回答")
        full_content = "抱歉，当前系统生成答案时出现了问题，请稍后再试。"

    return {"messages": [AIMessage(content=full_content)]}
