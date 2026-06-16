from typing import Any, List

from opensearchpy import OpenSearch

from app.core import config
from app.core.logging import logger
from app.domain.models import RetrievedDoc

_client: OpenSearch | None = None


def get_search_client() -> OpenSearch:
    """OpenSearch client (ES-compatible API, works with local OpenSearch 2.x)."""
    global _client
    if _client is None:
        url = config.ES_URL.rstrip("/")
        use_ssl = url.startswith("https")
        _client = OpenSearch(
            hosts=[url],
            use_ssl=use_ssl,
            verify_certs=False,
            ssl_show_warn=False,
            timeout=30,
        )
    return _client


def check_opensearch() -> dict[str, Any]:
    client = get_search_client()
    info = client.info()
    if hasattr(info, "body"):
        info = info.body
    version = info.get("version", {})
    return {
        "ok": True,
        "cluster_name": info.get("cluster_name"),
        "version": version.get("number"),
        "distribution": version.get("distribution", "opensearch"),
    }


def _search_hits(res: Any) -> list:
    if hasattr(res, "body"):
        res = res.body
    return res.get("hits", {}).get("hits", [])


def _search_es_with_fallback(query: str, size: int = 5):
    """
    简化版 OpenSearch 检索：
    1. 先用 AND 做精确检索
    2. 如果 0 命中，再用 OR 放宽检索
    """
    client = get_search_client()
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
            res = client.search(index=config.ES_INDEX_NAME, body=body)
            hits = _search_hits(res)
            logger.info(
                "opensearch search with operator=%s, hits=%d",
                operator,
                len(hits),
            )
        except Exception:
            logger.exception("OpenSearch 查询失败 (operator=%s)", operator)
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
