from app.core.logging import logger
from app.domain.models import AppState
from app.infra.opensearch_rag import rerank_by_alliance, search_rag_knowledge


def _prefer_symptom_clarify(hits: list[dict], query: str) -> dict | None:
    q = (query or "").strip()
    for h in hits:
        if h.get("type") != "symptomClarify":
            continue
        aliases = h.get("aliases") or h.get("alliance") or []
        if any(isinstance(a, str) and (a in q or q in a) for a in aliases):
            return h
    for h in hits:
        if h.get("type") == "symptomClarify":
            return h
    return None


def rag_symptom_recall_node(state: AppState) -> dict:
    logger.info(">>> Enter node: rag_symptom_recall")
    table = state.slot_table
    ner = state.ner_result
    if not table or not table.primary_symptom:
        return {"rag_chunk": None, "rag_chunk_id": None}

    q = (ner.query if ner and ner.query else "") or table.primary_symptom
    if table.companion_symptoms:
        q = f"{q} {' '.join(table.companion_symptoms)}".strip()
    if table.trigger:
        q = f"{q} {table.trigger}".strip()
    if table.emergency:
        q = f"{q} {table.emergency}".strip()

    hits = search_rag_knowledge(q, k=5)
    if table.primary_symptom:
        extra = search_rag_knowledge(table.primary_symptom, k=5)
        seen = {h.get("id") for h in hits}
        for h in extra:
            if h.get("id") not in seen:
                hits.append(h)
                seen.add(h.get("id"))
    hits = rerank_by_alliance(hits, table.primary_symptom or q)
    if not hits:
        logger.warning("rag_symptom_recall: no hits for %r", q)
        return {"rag_chunk": None, "rag_chunk_id": None}

    clarify = _prefer_symptom_clarify(hits, q)
    chunk = clarify if clarify else hits[0]
    chunk_id = chunk.get("id")
    logger.info(
        "rag_symptom_recall hit id=%s type=%s",
        chunk_id,
        chunk.get("type"),
    )
    return {"rag_chunk": chunk, "rag_chunk_id": chunk_id}
