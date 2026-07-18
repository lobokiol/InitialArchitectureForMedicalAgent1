from app.core.logging import logger
from app.domain.models import AppState
from app.infra.opensearch_rag import rerank_by_alliance, search_rag_knowledge
from app.triage.rag_clarify_select import prefer_symptom_clarify

# Backward-compatible re-export for existing tests
_prefer_symptom_clarify = prefer_symptom_clarify


def rag_symptom_recall_node(state: AppState) -> dict:
    logger.info(">>> Enter node: rag_symptom_recall")
    table = state.slot_table
    ner = state.ner_result
    if not table or not table.primary_symptom:
        return {"rag_chunk": None, "rag_chunk_id": None}

    primary = table.primary_symptom
    q = (ner.query if ner and ner.query else "") or primary
    if table.companion_symptoms:
        q = f"{q} {' '.join(table.companion_symptoms)}".strip()
    if table.trigger:
        q = f"{q} {table.trigger}".strip()

    hits = search_rag_knowledge(q, k=5)
    if primary:
        extra = search_rag_knowledge(primary, k=5)
        seen = {h.get("id") for h in hits}
        for h in extra:
            if h.get("id") not in seen:
                hits.append(h)
                seen.add(h.get("id"))
    hits = rerank_by_alliance(hits, primary or q)
    if not hits:
        logger.warning("rag_symptom_recall: no hits for %r", q)
        return {"rag_chunk": None, "rag_chunk_id": None}

    clarify = prefer_symptom_clarify(hits, q, primary)
    if clarify:
        chunk = clarify
    elif hits[0].get("type") == "symptom":
        chunk = hits[0]
    else:
        chunk = None

    if chunk is None:
        logger.warning("rag_symptom_recall: no chunk selected for %r", q)
        return {"rag_chunk": None, "rag_chunk_id": None}

    chunk_id = chunk.get("id")
    logger.info(
        "rag_symptom_recall selected id=%s type=%s",
        chunk_id,
        chunk.get("type"),
    )
    return {"rag_chunk": chunk, "rag_chunk_id": chunk_id}
