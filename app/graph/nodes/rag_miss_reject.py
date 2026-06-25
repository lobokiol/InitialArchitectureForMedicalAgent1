from langchain_core.messages import AIMessage

from app.core.logging import logger
from app.domain.models import AppState
from app.triage.session_reset import triage_state_reset_patch

RAG_MISS_MESSAGE = (
    "暂无法识别该症状，请补充具体部位或描述；"
    "也可到医院分诊台咨询。"
)


def rag_miss_reject_node(state: AppState) -> dict:
    logger.info(">>> Enter node: rag_miss_reject")
    patch = triage_state_reset_patch()
    patch["messages"] = [AIMessage(content=RAG_MISS_MESSAGE)]
    return patch
