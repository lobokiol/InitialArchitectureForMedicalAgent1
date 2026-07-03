#!/usr/bin/env python3
"""Verify 12 gyn/ped symptom+unmatched cases after RAG kb update."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.auth_helper import DEFAULT_EVAL_PASSWORD, DEFAULT_EVAL_PHONE, authed_session
from scripts.batch_clarify_persona import pick_clarify_reply

SRC = ROOT / "sourceData" / "data" / "小医疗数据.json"
BASE = "http://127.0.0.1:8000"
PROXIES = {"http": None, "https": None}

WANT_DEPT: dict[int, str] = {
    6: "妇科",
    7: "妇科",
    8: "妇科",
    10: "儿科",
    22: "妇科",
    26: "妇科",
    48: "儿科",
    81: "妇科",
    84: "妇科",
    87: "儿科",
    97: "妇科",
    98: "儿科",
}


def load_queries() -> list[str]:
    return [
        json.loads(line)["questions"]
        for line in SRC.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def run_case(sess, base: str, idx: int, query: str, want_dept: str) -> dict:
    tid = sess.post(
        f"{base}/threads", json={"title": query[:20]}, proxies=PROXIES, timeout=30
    ).json()["thread_id"]
    msg = query
    last: dict = {}
    for _ in range(15):
        last = sess.post(
            f"{base}/chat",
            json={"thread_id": tid, "message": msg},
            proxies=PROXIES,
            timeout=120,
        ).json()
        nxt = pick_clarify_reply(last, query)
        if nxt is None:
            break
        msg = nxt
    route = (last.get("intent_result") or {}).get("triage_route")
    dept = last.get("locked_department")
    return {
        "idx": idx,
        "ok": dept == want_dept,
        "want": want_dept,
        "dept": dept,
        "route": route,
        "chunk_id": last.get("rag_chunk_id"),
        "confidence_passed": last.get("dept_confidence_passed"),
        "reply_head": (last.get("reply") or "")[:80],
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", default=BASE)
    p.add_argument("--phone", default=DEFAULT_EVAL_PHONE)
    p.add_argument("--password", default=DEFAULT_EVAL_PASSWORD)
    args = p.parse_args()
    base = args.base_url.rstrip("/")

    queries = load_queries()
    sess, _, _ = authed_session(base, args.phone, args.password, proxies=PROXIES)
    sess.get(f"{base}/healthz", proxies=PROXIES, timeout=10).raise_for_status()

    results = []
    for idx, want_dept in WANT_DEPT.items():
        query = queries[idx - 1]
        r = run_case(sess, base, idx, query, want_dept)
        results.append(r)
        status = "PASS" if r["ok"] else "FAIL"
        print(
            f"#{idx} {status} want={r['want']} got={r['dept']} "
            f"route={r['route']} chunk={r['chunk_id']} conf={r['confidence_passed']}"
        )
        if not r["ok"]:
            print(f"     reply: {r['reply_head']}")

    passed = sum(1 for r in results if r["ok"])
    print(f"--- {passed}/{len(results)} passed ---")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
