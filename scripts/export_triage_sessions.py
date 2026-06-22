#!/usr/bin/env python3
"""Export triage sessions from SQLite to JSONL for eval diff."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core import config
from app.infra.triage_session_store import TriageSessionStore


def main() -> int:
    parser = argparse.ArgumentParser(description="Export triage sessions to JSONL")
    parser.add_argument("--out", default="exports/triage_sessions.jsonl")
    parser.add_argument("--outcome", default=None, help="Comma-separated outcomes")
    parser.add_argument("--since", default=None, help="YYYY-MM-DD")
    parser.add_argument("--status", default="completed")
    parser.add_argument("--db", default=config.TRIAGE_SESSION_DB_PATH)
    args = parser.parse_args()

    store = TriageSessionStore(args.db)
    store.init_schema()
    outcome = args.outcome if args.outcome and "," not in args.outcome else None
    rows = store.list_sessions(status=args.status, outcome=outcome, since=args.since)
    if args.outcome and "," in args.outcome:
        allowed = set(args.outcome.split(","))
        rows = [r for r in rows if r.get("outcome") in allowed]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            turns = json.loads(row.get("turns_json") or "[]")
            record = {
                "id": row["id"],
                "message": row["initial_message"],
                "actual_route": row.get("actual_route"),
                "actual_chunk_id": row.get("rag_chunk_id"),
                "actual_dept": row.get("actual_dept"),
                "actual_emergency": row.get("actual_dept") == "急诊",
                "outcome": row.get("outcome"),
                "turn_count": row.get("turn_count"),
                "dept_rounds": row.get("dept_rounds"),
                "turns": turns,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Exported {len(rows)} rows to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
