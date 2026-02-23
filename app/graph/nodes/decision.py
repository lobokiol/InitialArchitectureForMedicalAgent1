from langchain_core.messages import HumanMessage

from app.core.logging import logger
from app.core.llm import get_chat_llm
from app.domain.models import AppState, IntentResult


DECISION_PROMPT = """
    你是一个医疗问答系统的"意图识别"模块。

    你必须只返回一个 json 对象（合法的 JSON），不要输出任何解释或多余文字。

    字段要求与 IntentResult 一致：
    - has_symptom: bool - 用户是否在描述身体症状/不适（如"头疼"、"肚子大"）
    - has_process: bool - 用户是否在询问如何做某事（如"怎么办"、"怎么治疗"、"如何减小"）
    - main_intent: "symptom" | "process" | "mixed" | "non_medical"
    - symptom_query: string 或 null - 从问题中提取的症状描述
    - process_query: string 或 null - 从问题中提取的操作请求（如"怎么减小"、"如何治疗"）
    - need_symptom_search: bool
    - need_process_search: bool

    判断规则：
    - 如果用户问"怎么XX"、"如何XX"、"怎么办"，即使同时描述了症状，也应设置has_process=True
    - symptom_query应提取症状本身（如"肚子大"）
    - process_query应提取操作意图（如"怎么减小"）

    用户问题：{query}
"""


def decision_node(state: AppState) -> dict:
    logger.info(">>> Enter node: decision")
    user_query = state.messages[-1].content
    logger.info("decision_node user_query=%s", user_query)

    try:
        structured_llm = get_chat_llm().with_structured_output(IntentResult)
        intent = structured_llm.invoke(
            [HumanMessage(content=DECISION_PROMPT.format(query=user_query))]
        )
    except Exception:
        logger.exception("decision_node LLM 调用失败，使用兜底 intent_result")
        intent = IntentResult(
            has_symptom=False,
            has_process=False,
            main_intent="non_medical",
            symptom_query=None,
            process_query=None,
            need_symptom_search=False,
            need_process_search=False,
        )
        return {"intent_result": intent}

    intent.need_symptom_search = intent.has_symptom
    intent.need_process_search = intent.has_process

    logger.info("decision_node intent_result=%s", intent)
    return {"intent_result": intent}
