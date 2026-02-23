from typing import List

from app.core.logging import logger
from app.domain.models import AppState, RetrievedDoc
from app.infra.es_client import search_process_docs


def es_rag_node(state: AppState) -> dict:
    logger.info(">>> Enter node: es_rag")
    ir = state.intent_result
    logger.info("es_rag_node intent_result=%s", ir)

    if not (
        ir
        and ir.has_process
        and ir.need_process_search
        and ir.process_query
        and ir.process_query.strip()
    ):
        logger.info("es_rag_node: no process search needed, skip ES")
        return {}

    query = ir.process_query.strip()
    logger.info("es_rag_node: ES query=%s", query)

    docs: List[RetrievedDoc] = search_process_docs(query, size=5)

    if not docs:
        logger.info("es_rag_node: no hits found in ES for query=%s", query)
        return {}

    logger.info("es_rag_node: got %d docs", len(docs))
    return {"process_docs": docs}
