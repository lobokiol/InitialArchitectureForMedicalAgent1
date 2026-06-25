"""OpenSearch rag_department_rules exact lookup (production wrapper)."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from app.core import config
from app.core.logging import logger
from app.infra.opensearch_rag import get_opensearch_client

_RULES_PATH = Path(__file__).resolve().parents[2] / config.SOURCE_DATA_DIR / "data" / "rag_department_rules.jsonl"


@lru_cache(maxsize=1)
def _load_local_rules() -> list[dict[str, Any]]:
    if not _RULES_PATH.is_file():
        return []
    docs: list[dict[str, Any]] = []
    for line in _RULES_PATH.read_text(encoding="utf-8").splitlines():
        raw = line.strip()
        if raw:
            docs.append(json.loads(raw))
    return docs


def _search_local(symptom_id: str, location: str) -> dict[str, Any] | None:
    for doc in _load_local_rules():
        if doc.get("symptom_id") == symptom_id and doc.get("location") == location:
            return doc
    return None


def search_dept_rule(symptom_id: str, location: str) -> dict[str, Any] | None:
    client = get_opensearch_client()
    if client is not None:
        body = {
            "query": {
                "bool": {
                    "must": [
                        {"term": {"symptom_id": symptom_id}},
                        {"term": {"location": location}},
                    ]
                }
            },
            "size": 1,
        }
        try:
            res = client.search(index=config.RAG_DEPT_RULES_INDEX, body=body)
            hits = res["hits"]["hits"]
            if hits:
                return dict(hits[0]["_source"])
        except Exception:
            logger.exception("search_dept_rule OpenSearch failed symptom_id=%r location=%r", symptom_id, location)

    local = _search_local(symptom_id, location)
    if local:
        logger.info("search_dept_rule: local fallback hit id=%s", local.get("id"))
    else:
        logger.warning("search_dept_rule: no rule for symptom_id=%r location=%r", symptom_id, location)
    return local
