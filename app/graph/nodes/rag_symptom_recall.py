from app.core import config
from app.core.logging import logger
from app.domain.models import AppState
from app.infra.opensearch_rag import rerank_by_alliance, search_rag_knowledge


def _alias_matches(aliases: list, primary: str, query: str) -> bool:
    for text in (primary, query):
        t = (text or "").strip()
        if not t:
            continue
        for a in aliases:
            if isinstance(a, str) and (a in t or t in a):
                return True
    return False


def _clarify_hits(hits: list[dict]) -> list[dict]:
    return [h for h in hits if h.get("type") == "symptomClarify"]


def _prefer_symptom_clarify(
    hits: list[dict],
    query: str,
    primary_symptom: str,
) -> dict | None:
    clarify = _clarify_hits(hits)
    if not clarify:
        return None

    for h in clarify:
        aliases = h.get("aliases") or h.get("alliance") or []
        if _alias_matches(aliases, primary_symptom, query):
            return h

    ranked = sorted(clarify, key=lambda h: -(h.get("_score") or 0))
    top = ranked[0]
    top_score = top.get("_score") or 0
    second_score = ranked[1].get("_score") if len(ranked) > 1 else 0
    margin = top_score - second_score

    if top_score >= config.RAG_CLARIFY_MIN_SCORE and margin >= config.RAG_CLARIFY_MIN_MARGIN:
        logger.info(
            "rag_symptom_recall threshold pass id=%s score=%.4f margin=%.4f",
            top.get("id"),
            top_score,
            margin,
        )
        return top

    logger.info(
        "rag_symptom_recall threshold reject top=%s score=%.4f margin=%.4f",
        top.get("id"),
        top_score,
        margin,
    )
    return None


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
    if table.emergency:
        q = f"{q} {table.emergency}".strip()

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

    clarify = _prefer_symptom_clarify(hits, q, primary)
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
        "rag_symptom_recall hit id=%s type=%s",
        chunk_id,
        chunk.get("type"),
    )
    return {"rag_chunk": chunk, "rag_chunk_id": chunk_id}
