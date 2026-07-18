from typing import Any

from opensearchpy import OpenSearch

from app.core import config

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
