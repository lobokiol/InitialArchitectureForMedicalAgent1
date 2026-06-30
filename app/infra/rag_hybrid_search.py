"""Shared rag_knowledge hybrid search: BM25 clause, pipeline, and query body."""

from __future__ import annotations

from typing import Any

from opensearchpy import OpenSearch

from app.core import config
from app.core.logging import logger

BM25_WEIGHT = 0.4
KNN_WEIGHT = 0.6

_pipeline_ensured = False


def bm25_clause(query: str) -> dict[str, Any]:
    return {
        "bool": {
            "should": [
                {
                    "multi_match": {
                        "query": query,
                        "fields": [
                            "canonical_symptom^5",
                            "alliance^4",
                            "description^2",
                            "search_text",
                        ],
                        "type": "best_fields",
                    }
                },
                {"term": {"alliance": {"value": query, "boost": 8}}},
            ],
            "minimum_should_match": 1,
        }
    }


def hybrid_pipeline_body() -> dict[str, Any]:
    return {
        "description": "Normalize BM25 + kNN scores for rag_knowledge hybrid recall",
        "phase_results_processors": [
            {
                "normalization-processor": {
                    "normalization": {"technique": "min_max"},
                    "combination": {
                        "technique": "arithmetic_mean",
                        "parameters": {"weights": [BM25_WEIGHT, KNN_WEIGHT]},
                    },
                }
            }
        ],
    }


def ensure_hybrid_pipeline(
    client: OpenSearch,
    pipeline_name: str | None = None,
) -> None:
    global _pipeline_ensured
    if _pipeline_ensured:
        return
    name = pipeline_name or config.RAG_KB_HYBRID_PIPELINE
    try:
        client.transport.perform_request(
            "PUT",
            f"/_search/pipeline/{name}",
            body=hybrid_pipeline_body(),
        )
        _pipeline_ensured = True
        logger.info("ensured hybrid search pipeline %r", name)
    except Exception:
        logger.exception("failed to ensure hybrid search pipeline %r", name)


def hybrid_search_body(query: str, vector: list[float], k: int) -> dict[str, Any]:
    return {
        "query": {
            "hybrid": {
                "queries": [
                    bm25_clause(query),
                    {"knn": {"embedding": {"vector": vector, "k": k}}},
                ]
            }
        },
        "size": k,
    }


def keyword_search_body(query: str, k: int) -> dict[str, Any]:
    return {"query": bm25_clause(query), "size": k}


def hybrid_search_params(pipeline_name: str | None = None) -> dict[str, str]:
    return {"search_pipeline": pipeline_name or config.RAG_KB_HYBRID_PIPELINE}
