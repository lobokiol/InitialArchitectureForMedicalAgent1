from typing import List

from elasticsearch import Elasticsearch

from app.core import config
from app.core.logging import logger
from app.domain.models import RetrievedDoc


es_client = Elasticsearch(config.ES_URL, request_timeout=30)


def _search_es_with_fallback(query: str, size: int = 5):
    """
    简化版 ES 检索：
    1. 先用 AND 做精确检索
    2. 如果 0 命中，再用 OR 放宽检索
    """
    for operator in ("AND", "OR"):
        must = [
            {
                "query_string": {
                    "query": query,
                    "fields": ["scene^2", "raw_text"],
                    "default_operator": operator,
                }
            }
        ]

        body = {"query": {"bool": {"must": must}}, "size": size}

        try:
            res = es_client.search(index=config.ES_INDEX_NAME, body=body)
            hits = res.get("hits", {}).get("hits", [])
            logger.info(
                "es search with operator=%s, hits=%d",
                operator,
                len(hits),
            )
        except Exception:
            logger.exception("ES 查询失败 (operator=%s)", operator)
            return []

        if hits:
            return hits

    return []


def search_process_docs(query: str, size: int = 5) -> List[RetrievedDoc]:
    hits = _search_es_with_fallback(query, size=size)
    docs: List[RetrievedDoc] = []
    for h in hits:
        src = h.get("_source", {})
        docs.append(
            RetrievedDoc(
                id=src.get("id", h.get("_id")),
                source="process",
                title=src.get("scene"),
                content=src.get("raw_text", ""),
                score=h.get("_score"),
            )
        )
    logger.info("search_process_docs: got %d docs", len(docs))
    return docs
