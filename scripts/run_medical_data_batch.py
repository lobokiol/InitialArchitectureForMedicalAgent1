#!/usr/bin/env python3
"""Batch-run 小医疗数据.json questions through /chat with fixed persona."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "sourceData" / "data" / "小医疗数据.json"
PROXIES = {"http": None, "https": None}


def load_questions(path: Path, limit: int | None) -> list[str]:
    rows: list[str] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            q = json.loads(line).get("questions", "").strip()
            if q:
                rows.append(q)
            if limit and len(rows) >= limit:
                break
    return rows


def pick_reply(data: dict) -> str | None:
    if data.get("awaiting_clarify") and data.get("clarify_choices"):
        phase = data.get("clarify_phase")
        if phase == "age":
            return "19-35岁"
        if phase == "sex":
            return "男"
        return data["clarify_choices"][0]["label"]
    if data.get("awaiting_dept_choice") and data.get("dept_choices"):
        return "1" if data.get("multi_select") else data["dept_choices"][0]["label"]
    return None


def run_case(sess: requests.Session, base: str, user_id: str, query: str, max_steps: int) -> dict:
    r = sess.post(
        f"{base}/threads", json={"user_id": user_id, "title": query[:30]}, timeout=30, proxies=PROXIES
    )
    r.raise_for_status()
    thread_id = r.json()["thread_id"]

    msg = query
    last: dict = {}
    for step in range(max_steps):
        r = sess.post(
            f"{base}/chat",
            json={"user_id": user_id, "thread_id": thread_id, "message": msg},
            timeout=120,
            proxies=PROXIES,
        )
        r.raise_for_status()
        last = r.json()
        nxt = pick_reply(last)
        if nxt is None:
            return {"ok": True, "thread_id": thread_id, "steps": step + 1, "data": last}
        msg = nxt
    return {"ok": False, "thread_id": thread_id, "steps": max_steps, "error": "max_steps", "data": last}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    p.add_argument("--base-url", default="http://127.0.0.1:8000")
    p.add_argument("--user-id", default="batch-med-small-100")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--max-steps", type=int, default=15)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    base = args.base_url.rstrip("/")

    questions = load_questions(args.input, args.limit)
    print(f"Loaded {len(questions)} questions from {args.input}")
    if args.dry_run:
        for i, q in enumerate(questions[:5], 1):
            print(f"  [{i}] {q[:60]}...")
        if len(questions) > 5:
            print(f"  ... and {len(questions) - 5} more")
        return 0

    sess = requests.Session()
    sess.trust_env = False
    sess.get(f"{base}/healthz", timeout=10, proxies=PROXIES).raise_for_status()
    sess.post(f"{base}/users", json={"user_id": args.user_id}, timeout=10, proxies=PROXIES)

    t0 = time.time()
    errors: list[dict] = []

    for i, query in enumerate(questions, 1):
        try:
            result = run_case(sess, base, args.user_id, query, args.max_steps)
            if not result["ok"]:
                errors.append({"index": i, "query": query, **result})
                print(f"[{i}/{len(questions)}] TIMEOUT {query[:40]}...")
                continue
            data = result["data"]
            intent = data.get("intent_result") or {}
            route = intent.get("triage_route")
            dept = data.get("locked_department")
            print(f"[{i}/{len(questions)}] route={route} dept={dept} steps={result['steps']}")
        except requests.RequestException as exc:
            errors.append({"index": i, "query": query, "error": str(exc)})
            print(f"[{i}/{len(questions)}] ERROR {exc}")

    elapsed = time.time() - t0
    print(f"Done in {elapsed:.0f}s, errors={len(errors)}")
    if errors:
        err_path = ROOT / "exports" / "batch_med_100_errors.jsonl"
        err_path.parent.mkdir(parents=True, exist_ok=True)
        with err_path.open("w", encoding="utf-8") as f:
            for e in errors:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")
        print(f"Errors written to {err_path}")
    print(f"Export: python scripts/export_triage_sessions.py --user-id {args.user_id} --out exports/medical_small_100_sessions.jsonl")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
