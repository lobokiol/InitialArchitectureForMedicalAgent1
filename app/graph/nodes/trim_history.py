from app.core.logging import logger
from app.domain.models import AppState, MAX_HISTORY_MSGS, TRIM_TRIGGER_MSGS


def trim_history_node(state: AppState) -> dict:
    logger.info(">>> Enter node: trim_history")

    msgs = state.messages or []
    total = len(msgs)

    if total <= TRIM_TRIGGER_MSGS:
        logger.info(
            "trim_history_node: no need to trim, total_messages=%d, trigger=%d, keep=%d",
            total,
            TRIM_TRIGGER_MSGS,
            MAX_HISTORY_MSGS,
        )
        return {"messages": msgs}

    trimmed = msgs[-MAX_HISTORY_MSGS:]

    logger.info(
        "trim_history_node: trimmed messages from %d to %d (trigger=%d, keep=%d)",
        total,
        len(trimmed),
        TRIM_TRIGGER_MSGS,
        MAX_HISTORY_MSGS,
    )

    return {"messages": trimmed}
