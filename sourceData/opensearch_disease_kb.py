"""Index sourceData/data/disease_kb.jsonl into OpenSearch and test foot disease recall."""
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
_REPO_ROOT = _DEMO_DIR.parent
if str(_DEMO_DIR) not in sys.path:
    sys.path.insert(0, str(_DEMO_DIR))
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from opensearch_mappings import disease_kb_index_body

load_dotenv()

ES_URL = os.getenv("ES_URL", "http://127.0.0.1:9200")
INDEX_NAME = os.getenv("DISEASE_KB_INDEX", "disease_kb")
DATA_PATH = _DEMO_DIR / "data" / "disease_kb.jsonl"

FOOT_DISEASE_ACCEPTANCE: dict[str, tuple[str, str]] = {
    "脚气怎么办": ("足癣", "皮肤科"),
    "香港脚看什么科": ("足癣", "皮肤科"),
    "灰指甲挂什么科": ("甲癣", "皮肤科"),
    "大脚趾痛风": ("痛风性关节炎", "风湿免疫科"),
    "脚底疣治疗": ("跖疣", "皮肤科"),
    "大脚骨疼": ("拇外翻", "骨科"),
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


def enrich_doc(doc: dict[str, Any]) -> dict[str, Any]:
    parts = [
        doc.get("canonical_disease", ""),
        doc.get("description", ""),
        " ".join(doc.get("aliases") or []),
    ]
    out = dict(doc)
    out["search_text"] = " ".join(p for p in parts if p)
    return out


def recreate_index(client: OpenSearch, index_name: str = INDEX_NAME) -> None:
    if client.indices.exists(index=index_name):
        client.indices.delete(index=index_name)
        print(f"[OS] deleted index {index_name!r}")
    client.indices.create(index=index_name, body=disease_kb_index_body())
    print(f"[OS] created index {index_name!r}")


def index_disease_kb(
    client: OpenSearch,
    data_path: Path = DATA_PATH,
    index_name: str = INDEX_NAME,
) -> int:
    recreate_index(client, index_name)
    docs = [
        enrich_doc(json.loads(line))
        for line in data_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    actions = [
        {"_index": index_name, "_id": doc["id"], "_source": doc}
        for doc in docs
        if doc.get("id")
    ]
    helpers.bulk(client, actions)
    client.indices.refresh(index=index_name)
    print(f"[OS] indexed {len(actions)} disease docs")
    return len(actions)


def run_acceptance() -> int:
    from app.infra.opensearch_disease_kb import lookup_departments
    from app.ner.catalog_scan import load_entity_catalog, scan_catalog_substrings
    from app.ner.extract import build_entity_result
    from app.ner.triage_route import resolve_triage_route

    passed = 0
    for query, (expected_disease, expected_dept) in FOOT_DISEASE_ACCEPTANCE.items():
        cat = load_entity_catalog()
        raw = scan_catalog_substrings(query, cat["主症"], cat["疾病"])
        ner = build_entity_result(query, raw)
        route = resolve_triage_route(ner)
        depts = lookup_departments(ner.all_diseases) if ner.all_diseases else []
        top_dept = depts[0]["dept"] if depts else None
        top_disease = depts[0]["disease"] if depts else None
        ok = (
            route == "disease"
            and ner.primary_disease is not None
            and top_disease == expected_disease
            and top_dept == expected_dept
        )
        print(f"\n=== {query!r} ===")
        print(f"  route={route} primary_disease={ner.primary_disease!r}")
        print(f"  departments={depts}")
        print(f"  expected disease={expected_disease} dept={expected_dept}: {'PASS' if ok else 'FAIL'}")
        if ok:
            passed += 1
    print(f"\n=== acceptance: {passed}/{len(FOOT_DISEASE_ACCEPTANCE)} passed ===")
    return 0 if passed == len(FOOT_DISEASE_ACCEPTANCE) else 1


def main() -> int:
    acceptance_only = "--acceptance" in sys.argv
    client = wait_for_opensearch()
    if not acceptance_only:
        index_disease_kb(client)
    return run_acceptance()


if __name__ == "__main__":
    raise SystemExit(main())
