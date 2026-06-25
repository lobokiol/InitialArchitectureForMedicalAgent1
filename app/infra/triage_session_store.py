"""SQLite persistence for triage session records."""
from __future__ import annotations

import sqlite3
import uuid
from pathlib import Path
from typing import Any

from app.core import config


_SCHEMA = """
CREATE TABLE IF NOT EXISTS triage_sessions (
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
    completed_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_triage_thread_status ON triage_sessions(thread_id, status);
CREATE INDEX IF NOT EXISTS idx_triage_completed_at ON triage_sessions(completed_at);
CREATE INDEX IF NOT EXISTS idx_triage_outcome ON triage_sessions(outcome);
"""


class TriageSessionStore:
    def __init__(self, db_path: str = ":memory:") -> None:
        self._db_path = db_path
        self._conn: sqlite3.Connection | None = None

    def _connect(self) -> sqlite3.Connection:
        if self._conn is None:
            if self._db_path != ":memory:":
                Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
        return self._conn

    def init_schema(self) -> None:
        conn = self._connect()
        conn.executescript(_SCHEMA)
        conn.commit()

    def get_in_progress(self, thread_id: str) -> dict[str, Any] | None:
        conn = self._connect()
        cur = conn.execute(
            "SELECT * FROM triage_sessions WHERE thread_id = ? AND status = 'in_progress' LIMIT 1",
            (thread_id,),
        )
        row = cur.fetchone()
        return dict(row) if row else None

    def insert_draft(self, row: dict[str, Any]) -> str:
        session_id = row.get("id") or uuid.uuid4().hex
        conn = self._connect()
        conn.execute(
            """
            INSERT INTO triage_sessions (
                id, user_id, thread_id, status, outcome, initial_message,
                turns_json, turn_count, dept_rounds, actual_route, actual_dept,
                rag_chunk_id, state_snapshot_json, started_at, completed_at
            ) VALUES (?, ?, ?, 'in_progress', NULL, ?, ?, ?, ?, NULL, NULL, NULL, NULL, ?, NULL)
            """,
            (
                session_id,
                row["user_id"],
                row["thread_id"],
                row["initial_message"],
                row["turns_json"],
                row.get("turn_count", 0),
                row.get("dept_rounds", 0),
                row["started_at"],
            ),
        )
        conn.commit()
        return session_id

    def update_draft(self, session_id: str, row: dict[str, Any]) -> None:
        conn = self._connect()
        conn.execute(
            """
            UPDATE triage_sessions SET
                turns_json = ?,
                turn_count = ?,
                dept_rounds = ?,
                actual_route = COALESCE(?, actual_route),
                actual_dept = COALESCE(?, actual_dept),
                rag_chunk_id = COALESCE(?, rag_chunk_id),
                state_snapshot_json = COALESCE(?, state_snapshot_json)
            WHERE id = ? AND status = 'in_progress'
            """,
            (
                row["turns_json"],
                row["turn_count"],
                row["dept_rounds"],
                row.get("actual_route"),
                row.get("actual_dept"),
                row.get("rag_chunk_id"),
                row.get("state_snapshot_json"),
                session_id,
            ),
        )
        conn.commit()

    def finalize(self, session_id: str, row: dict[str, Any]) -> None:
        conn = self._connect()
        conn.execute(
            """
            UPDATE triage_sessions SET
                status = 'completed',
                outcome = ?,
                turns_json = COALESCE(?, turns_json),
                turn_count = COALESCE(?, turn_count),
                dept_rounds = COALESCE(?, dept_rounds),
                actual_route = ?,
                actual_dept = ?,
                rag_chunk_id = ?,
                state_snapshot_json = ?,
                completed_at = ?
            WHERE id = ?
            """,
            (
                row["outcome"],
                row.get("turns_json"),
                row.get("turn_count"),
                row.get("dept_rounds"),
                row.get("actual_route"),
                row.get("actual_dept"),
                row.get("rag_chunk_id"),
                row.get("state_snapshot_json"),
                row["completed_at"],
                session_id,
            ),
        )
        conn.commit()

    def list_sessions(
        self,
        *,
        status: str | None = None,
        outcome: str | None = None,
        since: str | None = None,
        user_id: str | None = None,
    ) -> list[dict[str, Any]]:
        conn = self._connect()
        clauses: list[str] = []
        params: list[Any] = []
        if status:
            clauses.append("status = ?")
            params.append(status)
        if outcome:
            clauses.append("outcome = ?")
            params.append(outcome)
        if since:
            clauses.append("completed_at >= ?")
            params.append(since)
        if user_id:
            clauses.append("user_id = ?")
            params.append(user_id)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        cur = conn.execute(
            f"SELECT * FROM triage_sessions {where} ORDER BY completed_at DESC, started_at DESC",
            params,
        )
        return [dict(r) for r in cur.fetchall()]


def check_triage_session_db() -> dict[str, Any]:
    """Triage SQLite readiness for /ready."""
    if not config.TRIAGE_SESSION_ENABLED:
        return {"ok": True, "enabled": False}

    try:
        store = TriageSessionStore(config.TRIAGE_SESSION_DB_PATH)
        store.init_schema()
        conn = store._connect()
        count = conn.execute("SELECT COUNT(*) FROM triage_sessions").fetchone()[0]
        return {
            "ok": True,
            "enabled": True,
            "path": config.TRIAGE_SESSION_DB_PATH,
            "session_count": count,
        }
    except Exception as exc:
        return {
            "ok": False,
            "enabled": True,
            "path": config.TRIAGE_SESSION_DB_PATH,
            "error": str(exc),
        }
