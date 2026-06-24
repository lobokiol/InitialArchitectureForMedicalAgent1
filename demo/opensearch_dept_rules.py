"""Create rag_department_rules index on OpenSearch and bulk load JSONL."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from opensearchpy import OpenSearch, helpers

_DEMO_DIR = Path(__file__).resolve().parent
if str(_DEMO_DIR) not in sys.path:
    sys.path.insert(0, str(_DEMO_DIR))

from opensearch_mappings import rag_department_rules_index_body
from opensearch_rag_kb import get_client, wait_for_opensearch

load_dotenv()

INDEX_NAME = os.getenv("RAG_DEPT_RULES_INDEX", "rag_department_rules")
DATA_PATH = Path(__file__).resolve().parent / "data" / "rag_department_rules.jsonl"


def load_docs(data_path: Path = DATA_PATH) -> list[dict[str, Any]]:
    docs: list[dict[str, Any]] = []
    for line in data_path.read_text(encoding="utf-8").splitlines():
        raw = line.strip()
        if not raw:
            continue
        doc = json.loads(raw)
        doc["raw_json"] = raw
        docs.append(doc)
    return docs


def recreate_index(client: OpenSearch, index_name: str = INDEX_NAME) -> None:
    if client.indices.exists(index=index_name):
        client.indices.delete(index=index_name)
        print(f"[OS] deleted index {index_name!r}")
    client.indices.create(index=index_name, body=rag_department_rules_index_body())
    print(f"[OS] created index {index_name!r}")


def index_dept_rules(client: OpenSearch, data_path: Path = DATA_PATH, index_name: str = INDEX_NAME) -> int:
    recreate_index(client, index_name)
    docs = load_docs(data_path)
    actions = [
        {"_index": index_name, "_id": doc["id"], "_source": doc}
        for doc in docs
        if doc.get("id")
    ]
    helpers.bulk(client, actions)
    client.indices.refresh(index=index_name)
    print(f"[OS] indexed {len(actions)} dept rule docs")
    return len(actions)


def main() -> None:
    client = wait_for_opensearch()
    index_dept_rules(client)


if __name__ == "__main__":
    main()
