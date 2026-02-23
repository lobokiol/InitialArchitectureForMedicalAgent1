from langchain_core.messages import HumanMessage

from app.core.logging import logger
from app.core.llm import get_chat_llm
from app.domain.models import AppState, MAX_REWRITE


SYMPTOM_REWRITE_PROMPT = """
请改写下面的医学检索 query，使其更适合医学知识向量检索：

用户问题：{user_query}
旧 query：{old_query}

要求：不超过 30 字，只输出 query。
"""

PROCESS_REWRITE_PROMPT = """
请改写下面的流程类检索 query，使其更适合流程知识库：

用户问题：{user_query}
旧 query：{old_query}

要求：不超过 30 字，只输出 query。
"""


def rewrite_question(state: AppState) -> dict:
    logger.info(">>> Enter node: rewrite_question (attempt=%d)", state.rewrite_attempts)
    ir = state.intent_result
    rr = state.relevance_result
    user_query = state.messages[-1].content

    attempts = state.rewrite_attempts

    if attempts >= MAX_REWRITE:
        logger.info("rewrite_question: attempts >= MAX_REWRITE, turn off further search")
        ir_new = ir.model_copy() if hasattr(ir, "model_copy") else ir.copy(deep=True)
        ir_new.need_symptom_search = False
        ir_new.need_process_search = False
        return {
            "intent_result": ir_new,
            "rewrite_attempts": attempts,
        }

    ir_new = ir.model_copy() if hasattr(ir, "model_copy") else ir.copy(deep=True)

    if ir_new.has_symptom and rr and rr.need_rewrite_symptom:
        old_symptom_q = ir_new.symptom_query or user_query
        logger.info("rewrite_question: rewriting symptom_query, old=%s", old_symptom_q)
        try:
            newq = get_chat_llm().invoke([
                HumanMessage(content=SYMPTOM_REWRITE_PROMPT.format(
                    user_query=user_query,
                    old_query=old_symptom_q,
                ))
            ]).content.strip()
            ir_new.symptom_query = newq
            ir_new.need_symptom_search = True
            logger.info("rewrite_question: new symptom_query=%s", newq)
        except Exception:
            logger.exception("rewrite_question 症状 query 重写失败，关闭 symptom 检索")
            ir_new.need_symptom_search = False
    else:
        ir_new.need_symptom_search = False

    if ir_new.has_process and rr and rr.need_rewrite_process:
        old_process_q = ir_new.process_query or user_query
        logger.info("rewrite_question: rewriting process_query, old=%s", old_process_q)
        try:
            newq = get_chat_llm().invoke([
                HumanMessage(content=PROCESS_REWRITE_PROMPT.format(
                    user_query=user_query,
                    old_query=old_process_q,
                ))
            ]).content.strip()
            ir_new.process_query = newq
            ir_new.need_process_search = True
            logger.info("rewrite_question: new process_query=%s", newq)
        except Exception:
            logger.exception("rewrite_question 流程 query 重写失败，关闭 process 检索")
            ir_new.need_process_search = False
    else:
        ir_new.need_process_search = False

    return {
        "intent_result": ir_new,
        "rewrite_attempts": attempts + 1,
    }
