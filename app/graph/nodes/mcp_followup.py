from app.core.logging import logger
from app.domain.models import AppState
from app.mcp.followup import run_mcp_followup


def mcp_followup_node(state: AppState) -> dict:
    logger.info(">>> Enter node: mcp_followup_agent")
    return run_mcp_followup(state)
