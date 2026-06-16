"""OpenSearch rag_knowledge hybrid search (production wrapper)."""

from __future__ import annotations

from typing import Any

from opensearchpy import OpenSearch

from app.core import config
from app.core.logging import logger

_client: OpenSearch | None = None
_unavailable = False


def get_opensearch_client() -> OpenSearch | None:
    global _client, _unavailable
    if _unavailable:
        return None
    if _client is not None:
        return _client
    try:
        url = config.ES_URL
        _client = OpenSearch(
            hosts=[url],
            use_ssl=url.startswith("https"),
            verify_certs=False,
            ssl_show_warn=False,
            timeout=30,
        )
        _client.info()
        return _client
    except Exception:
        logger.exception("OpenSearch unavailable at %s", config.ES_URL)
        _unavailable = True
        return None


def _embed_query(query: str) -> list[float]:
    from app.core.llm import get_embedding_model

    return get_embedding_model().embed_documents([query])[0]


def search_rag_knowledge_keyword(client: OpenSearch, query: str, k: int = 3) -> list[dict[str, Any]]:
    body = {
        "query": {
            "bool": {
                "should": [
                    {
                        "multi_match": {
                            "query": query,
                            "fields": [
                                "canonical_symptom.keyword^5",
                                "alliance^4",
                                "search_text^2",
                                "description",
                            ],
                            "type": "best_fields",
                        }
                    },
                ],
                "minimum_should_match": 1,
            }
        },
        "size": k,
    }
    res = client.search(index=config.RAG_KB_INDEX, body=body)
    return [_hit_source(h) for h in res["hits"]["hits"]]


def search_rag_knowledge_hybrid(client: OpenSearch, query: str, k: int = 3) -> list[dict[str, Any]]:
    try:
        vec = _embed_query(query)
    except Exception:
        logger.exception("embedding failed, fallback to keyword")
        return search_rag_knowledge_keyword(client, query, k=k)

    body = {
        "query": {
            "hybrid": {
                "queries": [
                    {
                        "bool": {
                            "should": [
                                {
                                    "multi_match": {
                                        "query": query,
                                        "fields": ["alliance^4", "search_text^2", "description"],
                                    }
                                }
                            ]
                        }
                    },
                    {"knn": {"embedding": {"vector": vec, "k": k}}},
                ]
            }
        },
        "size": k,
    }
    try:
        res = client.search(index=config.RAG_KB_INDEX, body=body)
        return [_hit_source(h) for h in res["hits"]["hits"]]
    except Exception:
        logger.exception("hybrid search failed, fallback to keyword")
        return search_rag_knowledge_keyword(client, query, k=k)


def _hit_source(hit: dict) -> dict[str, Any]:
    src = dict(hit.get("_source") or {})
    src["_score"] = hit.get("_score")
    return src


def rerank_by_alliance(hits: list[dict[str, Any]], query: str) -> list[dict[str, Any]]:
    q = query.strip()
    if not q or not hits:
        return hits

    def _sort_key(h: dict) -> tuple:
        alliance = h.get("alliance") or []
        exact = any(isinstance(a, str) and a in q for a in alliance)
        return (0 if exact else 1, -(h.get("_score") or 0))

    return sorted(hits, key=_sort_key)


def search_rag_knowledge(query: str, k: int = 1) -> list[dict[str, Any]]:
    client = get_opensearch_client()
    if client is None:
        return []
    return search_rag_knowledge_hybrid(client, query, k=k)
