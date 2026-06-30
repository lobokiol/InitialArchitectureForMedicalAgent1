"""Create rag_knowledge index on OpenSearch and hybrid semantic recall."""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from opensearchpy import OpenSearch, helpers

_DEMO_DIR = Path(__file__).resolve().parent
if str(_DEMO_DIR) not in sys.path:
    sys.path.insert(0, str(_DEMO_DIR))

from opensearch_mappings import EMBEDDING_DIM, rag_knowledge_index_body

_root = _DEMO_DIR.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from app.infra.rag_hybrid_search import (
    ensure_hybrid_pipeline,
    hybrid_search_body,
    hybrid_search_params,
    keyword_search_body,
)

load_dotenv()

ES_URL = os.getenv("ES_URL", "http://127.0.0.1:9200")
INDEX_NAME = os.getenv("RAG_KB_INDEX", "rag_knowledge")
HYBRID_PIPELINE = os.getenv("RAG_KB_HYBRID_PIPELINE", "rag-knowledge-hybrid-pipeline")
DATA_PATH = Path(__file__).resolve().parent / "data" / "rag_knowledge.jsonl"

ACCEPTANCE_QUERIES: dict[str, str] = {
    "肚子疼": "CL0001",
    "脚疼": "CL0007",
    "神经疼": "CL0015",
}


def get_client() -> OpenSearch:
    return OpenSearch(
        hosts=[ES_URL],
        use_ssl=ES_URL.startswith("https"),
        verify_certs=False,
        ssl_show_warn=False,
        timeout=30,
    )


def wait_for_opensearch(timeout_sec: int = 120) -> OpenSearch:
    deadline = time.time() + timeout_sec
    last_err: Exception | None = None
    while time.time() < deadline:
        try:
            client = get_client()
            info = client.info()
            print(
                "[OS] cluster:",
                info.get("cluster_name"),
                "version:",
                info.get("version", {}).get("number"),
            )
            return client
        except Exception as exc:
            last_err = exc
            time.sleep(2)
    raise RuntimeError(f"OpenSearch not ready after {timeout_sec}s: {last_err!r}")


def enrich_doc(doc: dict[str, Any], raw_line: str) -> dict[str, Any]:
    parts = [
        doc.get("canonical_symptom", ""),
        doc.get("symptom_id", ""),
        doc.get("raw_question", ""),
        doc.get("description", ""),
        " ".join(doc.get("aliases") or []),
        " ".join(doc.get("alliance") or []),
        " ".join(doc.get("accompanying_symptom_keywords") or []),
    ]
    if doc.get("type") == "symptomClarify":
        aliases = doc.get("aliases") or []
        parts.extend(aliases)
    out = dict(doc)
    if doc.get("type") == "symptomClarify":
        out["alliance"] = list(doc.get("aliases") or [])
        out["symptom_id"] = doc.get("symptom_id")
    out["raw_json"] = raw_line
    out["search_text"] = " ".join(p for p in parts if p)
    return out


def recreate_index(client: OpenSearch, index_name: str = INDEX_NAME) -> None:
    if client.indices.exists(index=index_name):
        client.indices.delete(index=index_name)
        print(f"[OS] deleted index {index_name!r}")
    client.indices.create(index=index_name, body=rag_knowledge_index_body())
    print(f"[OS] created index {index_name!r} (embedding dim={EMBEDDING_DIM})")


def embed_texts(texts: list[str], batch_size: int = 25) -> list[list[float]]:
    from app.core.llm import get_embedding_model

    model = get_embedding_model()
    vectors: list[list[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        batch_vectors = model.embed_documents(batch)
        vectors.extend(batch_vectors)
    if not vectors or len(vectors[0]) != EMBEDDING_DIM:
        raise RuntimeError(
            f"unexpected embedding dim: got {len(vectors[0]) if vectors else 0}, "
            f"expected {EMBEDDING_DIM}"
        )
    return vectors


def load_raw_lines(data_path: Path = DATA_PATH) -> list[tuple[str, dict[str, Any]]]:
    rows: list[tuple[str, dict[str, Any]]] = []
    for line in data_path.read_text(encoding="utf-8").splitlines():
        raw_line = line.strip()
        if raw_line:
            rows.append((raw_line, json.loads(raw_line)))
    return rows


def upsert_doc(
    client: OpenSearch,
    doc_id: str,
    data_path: Path = DATA_PATH,
    index_name: str = INDEX_NAME,
    with_embedding: bool = True,
) -> bool:
    """Index or overwrite a single document by id (no full rebuild)."""
    raw_line: str | None = None
    doc: dict[str, Any] | None = None
    for line in data_path.read_text(encoding="utf-8").splitlines():
        raw = line.strip()
        if not raw:
            continue
        parsed = json.loads(raw)
        if parsed.get("id") == doc_id:
            raw_line, doc = raw, parsed
            break
    if doc is None or raw_line is None:
        print(f"[OS] doc id={doc_id!r} not found in {data_path}")
        return False

    enriched = enrich_doc(doc, raw_line)
    if with_embedding:
        enriched["embedding"] = embed_texts([raw_line])[0]

    client.index(index=index_name, id=doc_id, body=enriched)
    client.indices.refresh(index=index_name)
    print(f"[OS] upserted {doc_id!r} into {index_name!r} (with_embedding={with_embedding})")
    return True


def index_rag_knowledge(
    client: OpenSearch,
    data_path: Path = DATA_PATH,
    index_name: str = INDEX_NAME,
    with_embedding: bool = True,
) -> int:
    recreate_index(client, index_name)
    if with_embedding:
        ensure_hybrid_pipeline(client, HYBRID_PIPELINE)

    raw_rows = load_raw_lines(data_path)
    docs = [enrich_doc(doc, raw_line) for raw_line, doc in raw_rows]

    if with_embedding:
        raw_lines = [raw_line for raw_line, _ in raw_rows]
        vectors = embed_texts(raw_lines)
        for doc, vec in zip(docs, vectors):
            doc["embedding"] = vec

    actions = [{"_index": index_name, "_id": doc["id"], "_source": doc} for doc in docs if doc.get("id")]
    helpers.bulk(client, actions)
    client.indices.refresh(index=index_name)
    print(f"[OS] indexed {len(actions)} docs (with_embedding={with_embedding})")
    return len(actions)


def search_keyword(
    client: OpenSearch,
    query: str,
    k: int = 3,
    index_name: str = INDEX_NAME,
) -> list[dict[str, Any]]:
    body = keyword_search_body(query, k)
    res = client.search(index=index_name, body=body)
    return [_hit_row(h) for h in res["hits"]["hits"]]


def search_semantic(
    client: OpenSearch,
    query: str,
    k: int = 3,
    index_name: str = INDEX_NAME,
) -> list[dict[str, Any]]:
    vec = embed_texts([query])[0]
    body = {
        "query": {"knn": {"embedding": {"vector": vec, "k": k}}},
        "size": k,
    }
    res = client.search(index=index_name, body=body)
    return [_hit_row(h) for h in res["hits"]["hits"]]


def search_hybrid(
    client: OpenSearch,
    query: str,
    k: int = 3,
    index_name: str = INDEX_NAME,
    pipeline_name: str = HYBRID_PIPELINE,
) -> list[dict[str, Any]]:
    vec = embed_texts([query])[0]
    body = hybrid_search_body(query, vec, k)
    params = hybrid_search_params(pipeline_name)
    res = client.search(
        index=index_name,
        body=body,
        params=params,
    )
    return [_hit_row(h) for h in res["hits"]["hits"]]


def recall(
    client: OpenSearch,
    query: str,
    k: int = 3,
    index_name: str = INDEX_NAME,
) -> dict[str, Any]:
    """Default recall entry: hybrid BM25 + kNN over rag_knowledge."""
    hits = search_hybrid(client, query, k=k, index_name=index_name)
    return {"query": query, "hits": hits}


def _hit_row(hit: dict[str, Any]) -> dict[str, Any]:
    src = hit.get("_source", {})
    return {
        "id": src.get("id"),
        "canonical_symptom": src.get("canonical_symptom"),
        "body_part": src.get("body_part"),
        "score": hit.get("_score"),
    }


def run_acceptance(client: OpenSearch) -> int:
    passed = 0
    for query, expected_id in ACCEPTANCE_QUERIES.items():
        result = recall(client, query, k=3)
        top_id = result["hits"][0]["id"] if result["hits"] else None
        ok = top_id == expected_id
        print(f"\n=== acceptance: {query!r} ===")
        for i, h in enumerate(result["hits"], 1):
            score = h.get("score")
            score_s = f"{score:.4f}" if isinstance(score, (int, float)) else str(score)
            print(
                f"  #{i} id={h['id']} score={score_s} "
                f"symptom={h['canonical_symptom']}"
            )
        print(f"top1={top_id} expected={expected_id}: {'PASS' if ok else 'FAIL'}")
        if ok:
            passed += 1
    print(f"\n=== acceptance summary: {passed}/{len(ACCEPTANCE_QUERIES)} passed ===")
    return 0 if passed == len(ACCEPTANCE_QUERIES) else 1


def main() -> int:
    skip_embed = "--no-embed" in sys.argv
    acceptance_only = "--acceptance" in sys.argv
    doc_id: str | None = None
    if "--doc" in sys.argv:
        idx = sys.argv.index("--doc")
        if idx + 1 >= len(sys.argv):
            print("usage: opensearch_rag_kb.py [--doc DOC_ID] [--no-embed] [--acceptance]")
            return 2
        doc_id = sys.argv[idx + 1]
    client = wait_for_opensearch()
    if doc_id:
        if not upsert_doc(client, doc_id, with_embedding=not skip_embed):
            return 1
    elif not acceptance_only:
        index_rag_knowledge(client, with_embedding=not skip_embed)
    return run_acceptance(client)


if __name__ == "__main__":
    raise SystemExit(main())
