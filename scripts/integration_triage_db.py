#!/usr/bin/env python3
"""Live API + SQLite integration check for triage session persistence."""
from __future__ import annotations

import json
import sqlite3
import sys
import uuid
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core import config

BASE = "http://127.0.0.1:8000"
DB = config.TRIAGE_SESSION_DB_PATH
PROXIES = {"http": None, "https": None}
REPORT = ROOT / "exports" / "integration_triage_report.json"


def chat(user_id: str, message: str, thread_id: str | None = None) -> dict:
    payload: dict = {"user_id": user_id, "message": message}
    if thread_id:
        payload["thread_id"] = thread_id
    r = requests.post(f"{BASE}/chat", json=payload, timeout=120, proxies=PROXIES)
    r.raise_for_status()
    return r.json()


def main() -> int:
    requests.get(f"{BASE}/healthz", timeout=10, proxies=PROXIES).raise_for_status()

    user_id = f"live-{uuid.uuid4().hex[:8]}"
    requests.post(f"{BASE}/users", json={"user_id": user_id}, timeout=10, proxies=PROXIES)

    report: dict = {"user_id": user_id, "chats": [], "sql": {}}

    # 1) reject
    r1 = chat(user_id, "你好")
    report["chats"].append({"case": "reject", "message": "你好", "reply": r1["reply"], "thread_id": r1["thread_id"]})

    # 2) disease
    r2 = chat(user_id, "我有胃炎")
    report["chats"].append(
        {
            "case": "disease",
            "message": "我有胃炎",
            "reply": r2["reply"],
            "route": (r2.get("intent_result") or {}).get("triage_route"),
            "thread_id": r2["thread_id"],
        }
    )

    # 3) emergency
    r3 = chat(user_id, "脚脖子肿，不能动，皮发紫")
    report["chats"].append(
        {
            "case": "emergency",
            "message": "脚脖子肿，不能动，皮发紫",
            "reply": r3["reply"],
            "thread_id": r3["thread_id"],
        }
    )

    # 4) multi-turn disambiguation (foot)
    r4 = chat(user_id, "脚后跟疼")
    thread = r4["thread_id"]
    report["chats"].append(
        {
            "case": "symptom_turn1",
            "message": "脚后跟疼",
            "reply": r4["reply"],
            "awaiting": r4.get("awaiting_dept_choice"),
            "choices": r4.get("dept_choices"),
            "thread_id": thread,
        }
    )
    if r4.get("awaiting_dept_choice") and r4.get("dept_choices"):
        label = r4["dept_choices"][0]["label"]
        r5 = chat(user_id, label, thread_id=thread)
        report["chats"].append(
            {
                "case": "symptom_turn2",
                "message": label,
                "reply": r5["reply"],
                "thread_id": thread,
            }
        )
        cur.execute(
            """
            SELECT id, outcome, turn_count, status
            FROM triage_sessions
            WHERE thread_id = ? AND status = 'completed'
            ORDER BY completed_at DESC
            LIMIT 5
            """,
            (thread,),
        )
        foot_sessions = [dict(r) for r in cur.fetchall()]
        report["sql"]["foot_multi_turn_sessions"] = foot_sessions
        multi_row = next((s for s in foot_sessions if s["turn_count"] >= 2), None)
        if multi_row is None:
            print(f"FAIL: foot multi-turn should be one row with turn_count>=2, got {foot_sessions}")
            print(f"Report: {REPORT}")
            return 1

    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    cur.execute(
        """
        SELECT id, status, outcome, initial_message, actual_dept, actual_route,
               turn_count, dept_rounds, started_at, completed_at
        FROM triage_sessions WHERE user_id = ? ORDER BY started_at
        """,
        (user_id,),
    )
    report["sql"]["user_sessions"] = [dict(r) for r in cur.fetchall()]

    cur.execute(
        """
        SELECT outcome, COUNT(*) AS cnt FROM triage_sessions
        WHERE status = 'completed' GROUP BY outcome ORDER BY cnt DESC
        """
    )
    report["sql"]["outcome_distribution"] = [dict(r) for r in cur.fetchall()]

    cur.execute("SELECT COUNT(*) FROM triage_sessions")
    report["sql"]["total_rows"] = cur.fetchone()[0]

    cur.execute(
        """
        SELECT initial_message, outcome, turn_count, turns_json
        FROM triage_sessions WHERE user_id = ? AND turn_count > 1 LIMIT 1
        """,
        (user_id,),
    )
    multi = cur.fetchone()
    if multi:
        d = dict(multi)
        d["turns_json"] = json.loads(d["turns_json"])
        report["sql"]["multi_turn_example"] = d

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    completed = [s for s in report["sql"]["user_sessions"] if s["status"] == "completed"]
    if len(completed) < 3:
        print(f"FAIL: expected >=3 completed sessions, got {len(completed)}")
        print(f"Report: {REPORT}")
        return 1

    print(f"OK: {len(completed)} sessions saved for user {user_id}")
    print(f"Report written to {REPORT}")
    for s in completed:
        print(f"  - {s['outcome']}: {s['initial_message'][:20]} -> dept={s['actual_dept']} turns={s['turn_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
