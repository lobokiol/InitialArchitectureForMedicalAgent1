from langchain_core.messages import AIMessage

from app.core.logging import logger
from app.domain.models import AppState
from app.domain.triage_intent import REJECT_MESSAGE
from app.triage.session_reset import triage_state_reset_patch


def reject_node(state: AppState) -> dict:
    """槽位门禁失败：固定拒答 + 清空导诊状态（保留 messages）。"""
    logger.info(">>> Enter node: reject (fixed message)")
    patch = triage_state_reset_patch()
    patch["messages"] = [AIMessage(content=REJECT_MESSAGE)]
    return patch
