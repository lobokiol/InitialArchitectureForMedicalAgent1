from app.core import config
from app.core.logging import logger
from app.domain.models import AppState


def trim_history_node(state: AppState) -> dict:
    logger.info(">>> Enter node: trim_history")

    msgs = state.messages or []
    total = len(msgs)
    trigger = config.TRIM_TRIGGER_MSGS
    keep = config.MAX_HISTORY_MSGS

    if total <= trigger:
        logger.info(
            "trim_history_node: no need to trim, total_messages=%d, trigger=%d, keep=%d",
            total,
            trigger,
            keep,
        )
        return {"messages": msgs}

    trimmed = msgs[-keep:]

    logger.info(
        "trim_history_node: trimmed messages from %d to %d (trigger=%d, keep=%d)",
        total,
        len(trimmed),
        trigger,
        keep,
    )

    return {"messages": trimmed}
