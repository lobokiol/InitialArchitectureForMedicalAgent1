from app.core.logging import logger
from app.domain.models import AppState, MAX_REWRITE


def route_after_decision(state: AppState) -> str:
    ir = state.intent_result
    logger.info(">>> route_after_decision: intent_result=%s", ir)

    if not ir:
        logger.info("route_after_decision -> answer_generate (no intent_result)")
        return "answer_generate"
    
     # 检查是否需要工具调用
    if hasattr(ir, 'need_tool_call') and ir.need_tool_call:
        logger.info("route_after_decision -> tool_calling")
        return "tool_calling"
    
    if not ir.has_symptom and not ir.has_process:
        logger.info("route_after_decision -> answer_generate (no symptom & no process)")
        return "answer_generate"

    logger.info("route_after_decision -> es_rag")
    return "es_rag"


def route_after_es(state: AppState) -> str:
    ir = state.intent_result
    logger.info(">>> route_after_es: intent_result=%s", ir)

    if ir and ir.need_symptom_search:
        logger.info("route_after_es -> milvus_rag (need_symptom_search=True)")
        return "milvus_rag"

    logger.info("route_after_es -> check_docs (no symptom search needed)")
    return "check_docs"


def route_after_docs(state: AppState) -> str:
    r = state.relevance_result
    ir = state.intent_result
    logger.info(
        ">>> route_after_docs: relevance_result=%s, rewrite_attempts=%d",
        r,
        state.rewrite_attempts,
    )

    if state.rewrite_attempts >= MAX_REWRITE:
        logger.info("route_after_docs -> answer_generate (rewrite_attempts >= MAX_REWRITE)")
        return "answer_generate"

    if r and r.can_answer_overall:
        logger.info("route_after_docs -> answer_generate (can_answer_overall=True)")
        return "answer_generate"

    if not (r and (r.need_rewrite_symptom or r.need_rewrite_process)):
        logger.info("route_after_docs -> answer_generate (no need to rewrite)")
        return "answer_generate"

    logger.info("route_after_docs -> rewrite_question (need rewrite)")
    return "rewrite_question"
