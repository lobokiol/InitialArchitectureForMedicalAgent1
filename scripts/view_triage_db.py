#!/usr/bin/env python3
"""Quick viewer for triage_sessions SQLite DB."""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core import config


def main() -> int:
    parser = argparse.ArgumentParser(description="View triage_sessions SQLite DB")
    parser.add_argument("--db", default=config.TRIAGE_SESSION_DB_PATH)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--outcome", default=None)
    parser.add_argument("--user-id", default=None)
    parser.add_argument("--schema", action="store_true", help="Show table schema only")
    args = parser.parse_args()

    conn = sqlite3.connect(args.db)
    conn.row_factory = sqlite3.Row

    if args.schema:
        print("=== triage_sessions columns ===")
        for col in conn.execute("PRAGMA table_info(triage_sessions)"):
            print(f"  {col['name']:22} {col['type']}")
        return 0

    clauses = ["1=1"]
    params: list = []
    if args.outcome:
        clauses.append("outcome = ?")
        params.append(args.outcome)
    if args.user_id:
        clauses.append("user_id = ?")
        params.append(args.user_id)

    sql = f"""
        SELECT id, user_id, outcome, actual_dept, turn_count, initial_message, completed_at
        FROM triage_sessions
        WHERE {' AND '.join(clauses)}
        ORDER BY completed_at DESC
        LIMIT ?
    """
    params.append(args.limit)

    rows = conn.execute(sql, params).fetchall()
    print(f"DB: {args.db}  rows: {len(rows)}")
    for row in rows:
        print(json.dumps(dict(row), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
