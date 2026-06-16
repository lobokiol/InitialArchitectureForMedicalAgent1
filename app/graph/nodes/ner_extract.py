from app.core.logging import logger
from app.ner.service import extract_entity_tags


def ner_extract_node(state: dict) -> dict:
    """独立 NER 测试节点。"""
    query = (state.get("query") or "").strip()
    logger.info(">>> Enter node: ner_extract query=%s", query)
    if not query:
        return {
            "primary_symptom": None,
            "companion_symptoms": [],
            "primary_disease": None,
            "companion_diseases": [],
            "error": "empty query",
        }
    try:
        result = extract_entity_tags(query)
        return {
            "primary_symptom": result.primary_symptom,
            "companion_symptoms": result.companion_symptoms,
            "primary_disease": result.primary_disease,
            "companion_diseases": result.companion_diseases,
            "error": None,
        }
    except Exception as exc:
        logger.exception("ner_extract_node failed")
        return {
            "primary_symptom": None,
            "companion_symptoms": [],
            "primary_disease": None,
            "companion_diseases": [],
            "error": str(exc),
        }
