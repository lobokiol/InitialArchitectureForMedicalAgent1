from __future__ import annotations


def _primary_symptom_from_message(msg: str) -> str:
    return msg.split("，")[0].split(",")[0].strip()


def recall_chunk_id(message: str) -> str | None:
    from app.infra.opensearch_rag import rerank_by_alliance, search_rag_knowledge

    q = message.strip()
    primary = _primary_symptom_from_message(q)
    hits = search_rag_knowledge(q, k=3)
    if primary:
        extra = search_rag_knowledge(primary, k=3)
        seen = {h.get("id") for h in hits}
        for h in extra:
            if h.get("id") not in seen:
                hits.append(h)
                seen.add(h.get("id"))
    hits = rerank_by_alliance(hits, primary or q)
    return hits[0].get("id") if hits else None


def run_offline_recall(cases: list[dict]) -> tuple[list[dict], int, int]:
    results: list[dict] = []
    passed = 0
    for c in cases:
        got = recall_chunk_id(c["message"])
        expect = c.get("expect_chunk_id")
        ok = got == expect
        if ok:
            passed += 1
        results.append(
            {
                "id": c["id"],
                "subset": c.get("subset"),
                "message": c["message"],
                "expect": expect,
                "got": got,
                "ok": ok,
            }
        )
        mark = "OK" if ok else "FAIL"
        print(f"[{mark}] {c['id']} recall expect={expect} got={got}")
    return results, passed, len(cases)
