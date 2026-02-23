from app.core.logging import logger
from app.domain.models import AppState
from app.infra.milvus_client import search_medical_docs


def milvus_rag_node(state: AppState) -> dict:
    logger.info(">>> Enter node: milvus_rag")
    ir = state.intent_result
    logger.info("milvus_rag_node intent_result=%s", ir)

    if not (ir and ir.has_symptom and ir.need_symptom_search and ir.symptom_query):
        logger.info("milvus_rag_node: no symptom search needed, skip Milvus")
        return {}

    query = ir.symptom_query.strip()
    if not query:
        logger.info("milvus_rag_node: symptom_query is empty, skip Milvus")
        return {}

    logger.info("milvus_rag_node: Milvus query=%s", query)

    try:
        docs = search_medical_docs(query)
    except Exception:
        logger.exception("Milvus 查询失败，关闭症状检索")
        new_ir = ir.model_copy() if hasattr(ir, "model_copy") else ir.copy(deep=True)
        new_ir.need_symptom_search = False
        return {"intent_result": new_ir}

    if not docs:
        logger.info("milvus_rag_node: no docs returned from Milvus")
        return {}

    return {"medical_docs": docs}
