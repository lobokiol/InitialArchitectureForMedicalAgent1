from typing import List

from langchain_core.messages import HumanMessage

from app.core.logging import logger
from app.core.llm import get_chat_llm
from app.domain.models import AppState, RetrievedDoc, RelevanceResult


RAG_EVAL_PROMPT = """
你是一个医疗问答系统的文档评估模块。

你必须只返回一个 json 对象（合法的 JSON，注意是小写的 json），不要输出任何额外说明。

这个 json 对象包含以下字段：
- can_answer_overall: bool
- need_rewrite_symptom: bool
- need_rewrite_process: bool
- reason: string 或 null

请判断：
- 当前文档是否足以回答 can_answer_overall
- 是否需要改写症状 query（need_rewrite_symptom）
- 是否需要改写流程 query（need_rewrite_process）

注意：本节点 **不负责生成重写后的 query**，只负责给出上述 json 字段。

用户问题：
{user_query}

symptom_query: {symptom_query}
process_query: {process_query}

【医学文档】
{medical_block}

【流程文档】
{process_block}
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


def check_docs_node(state: AppState) -> dict:
    logger.info(">>> Enter node: check_docs")
    ir = state.intent_result
    user_query = state.messages[-1].content

    prompt = RAG_EVAL_PROMPT.format(
        user_query=user_query,
        symptom_query=ir.symptom_query if ir else None,
        process_query=ir.process_query if ir else None,
        medical_block=_fmt_docs(state.medical_docs),
        process_block=_fmt_docs(state.process_docs),
    )

    logger.info(
        "check_docs_node: medical_docs=%d, process_docs=%d",
        len(state.medical_docs),
        len(state.process_docs),
    )

    try:
        structured = get_chat_llm().with_structured_output(RelevanceResult)
        rr = structured.invoke([HumanMessage(content=prompt)])
    except Exception:
        logger.exception("check_docs_node LLM 调用失败，使用兜底 relevance_result")
        rr = RelevanceResult(
            can_answer_overall=False,
            need_rewrite_symptom=False,
            need_rewrite_process=False,
            reason="LLM 调用失败，默认认为文档不足，且不再重写。",
        )

    logger.info("check_docs_node: relevance_result=%s", rr)
    return {"relevance_result": rr}
