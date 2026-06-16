from app.core.logging import logger
from app.domain.models import AppState
from app.triage.slot_fill import fill_slot_table


def slot_fill_node(state: AppState) -> dict:
    logger.info(">>> Enter node: slot_fill")
    ner = state.ner_result
    if not ner:
        return {"slot_table": None}
    table = fill_slot_table(ner)
    logger.info("slot_fill result=%s", table)
    return {"slot_table": table}
