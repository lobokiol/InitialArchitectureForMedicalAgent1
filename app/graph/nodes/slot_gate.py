from app.core.logging import logger
from app.domain.models import AppState
from app.domain.slot_table import slot_gate_passes


def slot_gate_node(state: AppState) -> dict:
    logger.info(">>> Enter node: slot_gate")
    table = state.slot_table
    passed = table is not None and slot_gate_passes(table)
    logger.info("slot_gate passed=%s", passed)
    return {"slot_gate_passed": passed}
