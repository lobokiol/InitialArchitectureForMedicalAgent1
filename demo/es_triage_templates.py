"""Index demo/data/triage_templates.jsonl into Elasticsearch and test symptom recall."""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path
from typing import Any

from elasticsearch import Elasticsearch, helpers

ES_URL = "http://127.0.0.1:9200"
INDEX_NAME = "triage_templates"
DATA_PATH = Path(__file__).resolve().parent / "data" / "triage_templates.jsonl"

INDEX_MAPPING = {
    "mappings": {
        "properties": {
            "id": {"type": "keyword"},
            "type": {"type": "keyword"},
            "canonical_symptom": {"type": "keyword"},
            "raw_question": {"type": "text"},
            "description": {"type": "text"},
            "alliance": {"type": "keyword"},
            "template": {"type": "keyword"},
            "version": {"type": "keyword"},
            "search_text": {"type": "text"},
        }
    }
}


def get_es_client() -> Elasticsearch:
    es = Elasticsearch(ES_URL, request_timeout=30)
    info = es.info()
    print(
        "[ES] cluster:",
        info.get("cluster_name"),
        "version:",
        info.get("version", {}).get("number"),
    )
    return es


def wait_for_es(timeout_sec: int = 120) -> Elasticsearch:
    deadline = time.time() + timeout_sec
    last_err: Exception | None = None
    while time.time() < deadline:
        try:
            return get_es_client()
        except Exception as exc:
            last_err = exc
            time.sleep(2)
    raise RuntimeError(f"Elasticsearch not ready after {timeout_sec}s: {last_err!r}")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def enrich_doc(doc: dict[str, Any]) -> dict[str, Any]:
    parts = [
        doc.get("canonical_symptom", ""),
        doc.get("raw_question", ""),
        doc.get("description", ""),
        " ".join(doc.get("alliance") or []),
        " ".join(doc.get("template") or []),
    ]
    out = dict(doc)
    out["search_text"] = " ".join(p for p in parts if p)
    return out


def recreate_index(es: Elasticsearch, index_name: str = INDEX_NAME) -> None:
    if es.indices.exists(index=index_name):
        es.indices.delete(index=index_name)
        print(f"[ES] deleted index {index_name!r}")
    es.indices.create(index=index_name, body=INDEX_MAPPING)
    print(f"[ES] created index {index_name!r}")


def index_templates(es: Elasticsearch, data_path: Path = DATA_PATH) -> int:
    recreate_index(es)
    docs = [enrich_doc(d) for d in load_jsonl(data_path)]
    actions = [
        {"_index": INDEX_NAME, "_id": doc["id"], "_source": doc}
        for doc in docs
        if doc.get("id")
    ]
    helpers.bulk(es, actions)
    es.indices.refresh(index=INDEX_NAME)
    print(f"[ES] indexed {len(actions)} docs")
    return len(actions)


def _normalize_query(query: str) -> str:
    q = query.strip()
    q = re.sub(r"[？?。！!，,；;：:\s]+", "", q)
    for suffix in (
        "该怎么办", "怎么办", "该咋办", "咋办", "怎么搞", "怎么处理",
        "是什么", "啥情况", "什么问题",
    ):
        if q.endswith(suffix):
            q = q[: -len(suffix)]
    return q.strip("我你他她这那是有的点些很非常一直最近今天")


def _extract_query_terms(query: str) -> list[str]:
    q = _normalize_query(query)
    if not q:
        return []
    terms = [q]
    if len(q) >= 4:
        terms.append(q[-4:])
    if len(q) >= 3:
        terms.append(q[-3:])
    if len(q) >= 2:
        terms.append(q[-2:])
    out: list[str] = []
    seen: set[str] = set()
    for t in terms:
        if len(t) >= 2 and t not in seen:
            seen.add(t)
            out.append(t)
    return out


def _alliance_terms_in_query(query: str, terms: list[str]) -> list[str]:
    hits: list[str] = []
    for doc in load_jsonl(DATA_PATH):
        for alias in doc.get("alliance") or []:
            if len(alias) < 2:
                continue
            if alias in query or any(alias in t or t in alias for t in terms):
                hits.append(alias)
    return hits


def search_templates(es: Elasticsearch, query: str, k: int = 3) -> list[dict[str, Any]]:
    terms = _extract_query_terms(query)
    alliance_hits = _alliance_terms_in_query(query, terms)
    should: list[dict[str, Any]] = [
        {
            "multi_match": {
                "query": query,
                "fields": [
                    "canonical_symptom^5",
                    "alliance^4",
                    "search_text^2",
                    "raw_question",
                    "description",
                ],
                "type": "best_fields",
                "operator": "or",
            }
        }
    ]
    for term in terms:
        should.append({"term": {"alliance": {"value": term, "boost": 8}}})
        should.append({"term": {"canonical_symptom": {"value": term, "boost": 10}}})
        should.append({"match_phrase": {"search_text": {"query": term, "boost": 3}}})
        should.append({"wildcard": {"search_text": {"value": f"*{term}*", "boost": 2}}})
    for alias in alliance_hits:
        should.append({"term": {"alliance": {"value": alias, "boost": 9}}})

    body = {
        "query": {"bool": {"should": should, "minimum_should_match": 1}},
        "size": k,
    }
    res = es.search(index=INDEX_NAME, body=body)
    return [
        {
            "id": h["_source"].get("id"),
            "canonical_symptom": h["_source"].get("canonical_symptom"),
            "score": h.get("_score"),
        }
        for h in res.get("hits", {}).get("hits", [])
    ]


def run_tests(es: Elasticsearch) -> None:
    queries = ["我乏力该怎么办。", "我没力气怎么办"]
    passed = 0
    for q in queries:
        print(f"\n=== Query: {q!r} ===")
        hits = search_templates(es, q, k=3)
        for i, h in enumerate(hits, 1):
            print(f"  #{i} id={h['id']} score={h['score']:.2f} symptom={h['canonical_symptom']}")
        top_id = hits[0]["id"] if hits else None
        ok = top_id == "S0020"
        print(f"top1={top_id} recall S0020: {'PASS' if ok else 'FAIL'}")
        if ok:
            passed += 1
    print(f"\n=== Summary: {passed}/{len(queries)} passed ===")


def main() -> int:
    wait = "--no-wait" not in sys.argv
    es = wait_for_es() if wait else get_es_client()
    index_templates(es)
    run_tests(es)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
