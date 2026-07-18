"""Select symptomClarify chunks from RAG hits (pure helpers)."""

from __future__ import annotations

from app.core import config
from app.core.logging import logger


def alias_matches(aliases: list, primary: str, query: str) -> bool:
    for text in (primary, query):
        t = (text or "").strip()
        if not t:
            continue
        for a in aliases:
            if isinstance(a, str) and (a in t or t in a):
                return True
    return False


def clarify_hits(hits: list[dict]) -> list[dict]:
    return [h for h in hits if h.get("type") == "symptomClarify"]


def prefer_symptom_clarify(
    hits: list[dict],
    query: str,
    primary_symptom: str,
) -> dict | None:
    clarify = clarify_hits(hits)
    if not clarify:
        return None

    for h in clarify:
        aliases = h.get("aliases") or h.get("alliance") or []
        if alias_matches(aliases, primary_symptom, query):
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
