from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage

from app.core.logging import logger
from app.domain.dept_confidence import DeptConfidenceResult
from app.domain.models import AppState
from app.triage.dept_confidence_prompt import build_confidence_prompt


def dept_confidence_node(state: AppState) -> dict:
    logger.info(">>> Enter node: dept_confidence")
    if not state.locked_department:
        return {"dept_confidence_passed": False}

    prompt = build_confidence_prompt(state)
    try:
        from app.core.llm import get_chat_llm

        structured = get_chat_llm().with_structured_output(DeptConfidenceResult)
        result = structured.invoke([HumanMessage(content=prompt)])
        if not isinstance(result, DeptConfidenceResult):
            result = DeptConfidenceResult.model_validate(result)
    except Exception:
        logger.exception("dept_confidence LLM failed")
        result = DeptConfidenceResult(score=0.0, reason="llm_error", slot_alignment="")
    passed = result.score >= 60
    logger.info("dept_confidence score=%.1f passed=%s", result.score, passed)
    return {
        "dept_confidence_result": result,
        "dept_confidence_passed": passed,
    }


def low_confidence_reject_node(state: AppState) -> dict:
    logger.info(">>> Enter node: low_confidence_reject")
    score = state.dept_confidence_result.score if state.dept_confidence_result else 0.0
    reply = (
        f"根据目前信息暂无法准确推荐科室（置信度 {score:.0f} 分，需 ≥60 分）。"
        "建议补充症状或到医院分诊台咨询。"
    )
    return {
        "messages": [AIMessage(content=reply)],
        "locked_department": None,
    }
