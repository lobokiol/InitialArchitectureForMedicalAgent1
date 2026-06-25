#!/usr/bin/env python3
"""Archive and merge fragmented triage_sessions rows (turn_count=1 chains)."""
from __future__ import annotations

import argparse
import json
import shutil
import sqlite3
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core import config

SESSION_COLUMNS = [
    "id",
    "user_id",
    "thread_id",
    "status",
    "outcome",
    "initial_message",
    "turns_json",
    "turn_count",
    "dept_rounds",
    "actual_route",
    "actual_dept",
    "rag_chunk_id",
    "state_snapshot_json",
    "started_at",
    "completed_at",
]

FRAGMENTS_DDL = """
CREATE TABLE IF NOT EXISTS triage_sessions_fragments (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    thread_id TEXT NOT NULL,
    status TEXT NOT NULL,
    outcome TEXT,
    initial_message TEXT NOT NULL,
    turns_json TEXT NOT NULL,
    turn_count INTEGER NOT NULL DEFAULT 0,
    dept_rounds INTEGER NOT NULL DEFAULT 0,
    actual_route TEXT,
    actual_dept TEXT,
    rag_chunk_id TEXT,
    state_snapshot_json TEXT,
    started_at TEXT NOT NULL,
    completed_at TEXT,
    archived_at TEXT NOT NULL,
    merged_into_id TEXT
);
"""

MAX_GAP_MINUTES = 30
TERMINAL_OUTCOMES = frozenset({"locked", "disease", "reject", "emergency", "fallback", "unmatched"})


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _parse_ts(value: str | None) -> datetime | None:
    if not value:
        return None
    text = value.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def _minutes_apart(a: str | None, b: str | None) -> float:
    ta, tb = _parse_ts(a), _parse_ts(b)
    if ta is None or tb is None:
        return 0.0
    return abs((tb - ta).total_seconds()) / 60.0


def _is_fragment(row: dict) -> bool:
    return int(row.get("turn_count") or 0) == 1


def _find_chains(rows: list[dict]) -> list[list[dict]]:
    """Consecutive turn_count=1 rows on same thread within time window."""
    chains: list[list[dict]] = []
    current: list[dict] = []

    for row in rows:
        if not _is_fragment(row):
            if len(current) >= 2:
                chains.append(current)
            current = []
            continue

        if not current:
            current = [row]
            continue

        prev = current[-1]
        gap = _minutes_apart(prev.get("completed_at") or prev.get("started_at"), row.get("started_at"))
        if gap <= MAX_GAP_MINUTES:
            current.append(row)
        else:
            if len(current) >= 2:
                chains.append(current)
            current = [row]

    if len(current) >= 2:
        chains.append(current)
    return chains


def _merge_chain(chain: list[dict]) -> dict:
    first, last = chain[0], chain[-1]
    turns: list[dict] = []
    for row in chain:
        turns.extend(json.loads(row["turns_json"]))
    for i, turn in enumerate(turns, start=1):
        turn["round"] = i

    dept_rounds = sum(int(r.get("dept_rounds") or 0) for r in chain)
    merged_id = uuid.uuid4().hex
    outcome = last.get("outcome") or "incomplete"
    if outcome == "incomplete" and len(chain) >= 2:
        outcome = "incomplete"

    return {
        "id": merged_id,
        "user_id": first["user_id"],
        "thread_id": first["thread_id"],
        "status": "completed",
        "outcome": outcome,
        "initial_message": first["initial_message"],
        "turns_json": json.dumps(turns, ensure_ascii=False),
        "turn_count": len(turns),
        "dept_rounds": dept_rounds,
        "actual_route": last.get("actual_route"),
        "actual_dept": last.get("actual_dept"),
        "rag_chunk_id": last.get("rag_chunk_id"),
        "state_snapshot_json": last.get("state_snapshot_json") or "{}",
        "started_at": first["started_at"],
        "completed_at": last.get("completed_at") or _utc_now_iso(),
    }


def _ensure_fragments_table(conn: sqlite3.Connection) -> None:
    conn.executescript(FRAGMENTS_DDL)
    conn.commit()


def _archive_row(conn: sqlite3.Connection, row: dict, *, archived_at: str, merged_into_id: str | None) -> None:
    cols = SESSION_COLUMNS + ["archived_at", "merged_into_id"]
    placeholders = ", ".join("?" for _ in cols)
    values = [row[c] for c in SESSION_COLUMNS] + [archived_at, merged_into_id]
    conn.execute(
        f"INSERT OR REPLACE INTO triage_sessions_fragments ({', '.join(cols)}) VALUES ({placeholders})",
        values,
    )


def repair(db_path: str, *, apply: bool) -> dict:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    cur = conn.execute(
        """
        SELECT * FROM triage_sessions
        WHERE status = 'completed'
        ORDER BY thread_id, started_at
        """
    )
    all_rows = [dict(r) for r in cur.fetchall()]

    by_thread: dict[str, list[dict]] = {}
    for row in all_rows:
        by_thread.setdefault(row["thread_id"], []).append(row)

    plans: list[dict] = []
    for thread_id, rows in by_thread.items():
        for chain in _find_chains(rows):
            merged = _merge_chain(chain)
            plans.append(
                {
                    "thread_id": thread_id,
                    "fragment_ids": [r["id"] for r in chain],
                    "merged": merged,
                }
            )

    result = {
        "db": db_path,
        "apply": apply,
        "chains_found": len(plans),
        "plans": [
            {
                "thread_id": p["thread_id"],
                "fragment_count": len(p["fragment_ids"]),
                "initial_message": p["merged"]["initial_message"][:40],
                "turn_count": p["merged"]["turn_count"],
                "outcome": p["merged"]["outcome"],
            }
            for p in plans
        ],
    }

    if not apply:
        conn.close()
        return result

    backup = f"{db_path}.bak.{datetime.now().strftime('%Y%m%d%H%M%S')}"
    shutil.copy2(db_path, backup)
    result["backup"] = backup

    _ensure_fragments_table(conn)
    archived_at = _utc_now_iso()

    for plan in plans:
        merged = plan["merged"]
        merged_id = merged["id"]
        for frag in plan["fragment_ids"]:
            row = conn.execute("SELECT * FROM triage_sessions WHERE id = ?", (frag,)).fetchone()
            if row:
                _archive_row(conn, dict(row), archived_at=archived_at, merged_into_id=merged_id)

        conn.execute(
            """
            INSERT INTO triage_sessions (
                id, user_id, thread_id, status, outcome, initial_message,
                turns_json, turn_count, dept_rounds, actual_route, actual_dept,
                rag_chunk_id, state_snapshot_json, started_at, completed_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                merged["id"],
                merged["user_id"],
                merged["thread_id"],
                merged["status"],
                merged["outcome"],
                merged["initial_message"],
                merged["turns_json"],
                merged["turn_count"],
                merged["dept_rounds"],
                merged["actual_route"],
                merged["actual_dept"],
                merged["rag_chunk_id"],
                merged["state_snapshot_json"],
                merged["started_at"],
                merged["completed_at"],
            ),
        )
        for frag_id in plan["fragment_ids"]:
            conn.execute("DELETE FROM triage_sessions WHERE id = ?", (frag_id,))

    conn.commit()
    conn.close()
    result["merged_rows"] = len(plans)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Repair fragmented triage_sessions rows")
    parser.add_argument("--db", default=config.TRIAGE_SESSION_DB_PATH)
    parser.add_argument("--apply", action="store_true", help="Execute archive+merge (default: dry-run)")
    args = parser.parse_args()

    if not Path(args.db).exists():
        print(f"DB not found: {args.db}")
        return 1

    report = repair(args.db, apply=args.apply)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["chains_found"] == 0:
        print("No fragment chains found.")
    elif not args.apply:
        print("Dry-run only. Re-run with --apply to archive, merge, and delete fragments.")
    else:
        print(f"Merged {report.get('merged_rows', 0)} chains. Backup: {report.get('backup')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
