from app.core.logging import logger
from app.domain.models import AppState, IntentResult
from app.ner.models import EntityExtractResult
from app.ner.service import extract_entity_tags
from app.ner.triage_route import resolve_triage_route
from app.triage.session_reset import triage_state_reset_patch


def decision_node(state: AppState) -> dict:
    """
    意图识别节点（NER + 三分类路由）。

    输出 ner_result 四字段 + intent_result.triage_route
    """
    logger.info(">>> Enter node: decision")
    # 新一轮 intake：清空上轮 locked / RAG / 槽位（保留 messages）
    patch: dict = triage_state_reset_patch()
    user_query = state.messages[-1].content
    if isinstance(user_query, list):
        user_query = str(user_query)
    user_query = (user_query or "").strip()
    logger.info("decision_node user_query=%s", user_query)

    if not user_query:
        patch.update(_build_output(EntityExtractResult(query=""), "reject"))
        return patch

    try:
        ner = extract_entity_tags(user_query)
    except Exception:
        logger.exception("decision_node NER failed")
        patch.update(_build_output(EntityExtractResult(query=user_query), "reject"))
        return patch

    route = resolve_triage_route(ner)
    logger.info(
        "decision_node route=%s primary_disease=%s primary_symptom=%s",
        route,
        ner.primary_disease,
        ner.primary_symptom,
    )
    patch.update(_build_output(ner, route))
    return patch


def _build_output(ner: EntityExtractResult, route: str) -> dict:
    intent = IntentResult(
        triage_route=route,
        has_symptom=ner.has_symptom,
        has_process=False,
        main_intent="symptom" if route == "symptom" else "non_medical",
        symptom_query=ner.primary_symptom,
        need_symptom_search=route == "symptom",
        need_process_search=False,
    )
    return {
        "intent_result": intent,
        "ner_result": ner,
        "disease_dept_result": None,
        "symptom_slot_result": None,
    }
