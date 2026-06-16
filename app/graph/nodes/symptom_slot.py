from app.core.logging import logger
from app.domain.models import AppState, SymptomSlotResult
from app.ner.slot_registry import resolve_slot_table_code, to_canonical_chief


def symptom_slot_node(state: AppState) -> dict:
    """
    症状链入口：读取 primary_symptom，在本节点做 canonical / slot 解析。
    """
    logger.info(">>> Enter node: symptom_slot")
    ner = state.ner_result
    if not ner or not ner.has_symptom:
        logger.warning("symptom_slot_node: no symptoms on state")
        return {"symptom_slot_result": SymptomSlotResult(route="symptom")}

    primary = ner.primary_symptom
    canonical = to_canonical_chief(primary)
    slot_code = resolve_slot_table_code(primary)

    result = SymptomSlotResult(
        route="symptom",
        chief_symptom=primary,
        chief_symptom_canonical=canonical,
        slot_table_code=slot_code,
        symptom_candidates=ner.all_symptoms,
        companion_symptoms=ner.companion_symptoms,
    )
    logger.info("symptom_slot_node result=%s", result)
    return {"symptom_slot_result": result}
