from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEMO = ROOT / "demo"
RAG_JSONL = DEMO / "data" / "rag_knowledge.jsonl"
DISEASE_JSONL = DEMO / "data" / "disease_kb.jsonl"
GOLDEN_JSONL = DEMO / "data" / "foot_triage_golden.jsonl"


def file_sha8(path: Path) -> str:
    if not path.exists():
        return "missing"
    return hashlib.sha256(path.read_bytes()).hexdigest()[:8]


def count_jsonl_lines(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def collect_versions() -> dict:
    return {
        "rag_knowledge_sha": file_sha8(RAG_JSONL),
        "disease_kb_sha": file_sha8(DISEASE_JSONL),
        "golden_sha": file_sha8(GOLDEN_JSONL),
        "rag_knowledge_lines": count_jsonl_lines(RAG_JSONL),
        "disease_kb_lines": count_jsonl_lines(DISEASE_JSONL),
        "golden_lines": count_jsonl_lines(GOLDEN_JSONL),
    }


def reindex_all() -> None:
    demo_dir = str(DEMO)
    if demo_dir not in sys.path:
        sys.path.insert(0, demo_dir)
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    from opensearch_disease_kb import index_disease_kb, wait_for_opensearch
    from opensearch_rag_kb import index_rag_knowledge

    client = wait_for_opensearch()
    print("[reindex] rag_knowledge ...")
    index_rag_knowledge(client)
    print("[reindex] disease_kb ...")
    index_disease_kb(client)


def check_opensearch_indices(skip_embed_check: bool = False) -> dict:
    from app.infra.opensearch_rag import get_opensearch_client

    client = get_opensearch_client()
    if client is None:
        raise RuntimeError("OpenSearch client unavailable")

    rag_index = __import__("app.core.config", fromlist=["config"]).RAG_KB_INDEX
    disease_index = __import__("app.core.config", fromlist=["config"]).DISEASE_KB_INDEX

    rag_count = int(client.count(index=rag_index).get("count", 0))
    disease_count = int(client.count(index=disease_index).get("count", 0))
    rag_lines = count_jsonl_lines(RAG_JSONL)
    disease_lines = count_jsonl_lines(DISEASE_JSONL)

    missing_embed = 0
    if not skip_embed_check:
        res = client.search(
            index=rag_index,
            body={"query": {"match_all": {}}, "size": 500, "_source": ["embedding"]},
        )
        for hit in res.get("hits", {}).get("hits") or []:
            emb = (hit.get("_source") or {}).get("embedding")
            if not emb:
                missing_embed += 1

    info = {
        "rag_index": rag_index,
        "rag_doc_count": rag_count,
        "rag_missing_embedding": missing_embed,
        "disease_index": disease_index,
        "disease_doc_count": disease_count,
    }

    errors: list[str] = []
    if rag_count != rag_lines:
        errors.append(f"rag count {rag_count} != jsonl lines {rag_lines}")
    if disease_count != disease_lines:
        errors.append(f"disease count {disease_count} != jsonl lines {disease_lines}")
    if missing_embed > 0:
        errors.append(f"rag missing embedding: {missing_embed}")

    info["ok"] = not errors
    info["errors"] = errors
    return info
