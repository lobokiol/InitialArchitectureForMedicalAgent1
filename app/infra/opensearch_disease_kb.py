"""OpenSearch disease_kb lookup — alias / canonical → departments."""

from __future__ import annotations

from typing import Any

from opensearchpy import OpenSearch

from app.core import config
from app.core.logging import logger
from app.infra.disease_kb_store import lookup_departments_local
from app.infra.opensearch_rag import get_opensearch_client


def lookup_disease_by_term(client: OpenSearch, term: str, index: str | None = None) -> dict[str, Any] | None:
    index_name = index or config.DISEASE_KB_INDEX
    body = {
        "query": {
            "bool": {
                "should": [
                    {"term": {"aliases": {"value": term, "boost": 4}}},
                    {"term": {"canonical_disease.keyword": {"value": term, "boost": 5}}},
                ],
                "minimum_should_match": 1,
            }
        },
        "size": 1,
    }
    res = client.search(index=index_name, body=body)
    hits = res.get("hits", {}).get("hits") or []
    if not hits:
        return None
    return dict(hits[0].get("_source") or {})


def lookup_departments(disease_terms: list[str]) -> list[dict[str, str]]:
    """疾病/别名 → 科室；优先 OpenSearch，失败回退本地 disease_kb。"""
    client = get_opensearch_client()
    if client is None:
        logger.warning("OpenSearch unavailable, disease lookup uses local disease_kb")
        return lookup_departments_local(disease_terms)

    seen_depts: set[str] = set()
    results: list[dict[str, str]] = []
    for term in disease_terms:
        row = lookup_disease_by_term(client, term.strip())
        if row:
            canonical = row.get("canonical_disease") or term
            for dept in row.get("departments") or []:
                if dept and dept not in seen_depts:
                    seen_depts.add(dept)
                    results.append({"disease": canonical, "dept": dept})
            continue
        for item in lookup_departments_local([term]):
            if item["dept"] not in seen_depts:
                seen_depts.add(item["dept"])
                results.append(item)
    return results
