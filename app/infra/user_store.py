"""SQLite persistence for user accounts and WeChat bindings."""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from passlib.hash import bcrypt as bcrypt_hash

from app.core import config

_SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    phone         TEXT PRIMARY KEY,
    password_hash TEXT,
    display_name  TEXT,
    created_at    TEXT NOT NULL,
    updated_at    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS wechat_bindings (
    openid    TEXT PRIMARY KEY,
    phone     TEXT NOT NULL REFERENCES users(phone),
    unionid   TEXT,
    bound_at  TEXT NOT NULL
);
"""


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


@dataclass
class UserRecord:
    phone: str
    password_hash: str | None
    display_name: str | None
    created_at: str
    updated_at: str


class UserStore:
    def __init__(self, db_path: str | None = None) -> None:
        self._db_path = db_path if db_path is not None else config.TRIAGE_SESSION_DB_PATH
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

    def _row_to_record(self, row: sqlite3.Row) -> UserRecord:
        return UserRecord(
            phone=row["phone"],
            password_hash=row["password_hash"],
            display_name=row["display_name"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def create_user(self, phone: str, password_hash: str, display_name: str) -> UserRecord:
        now = _utc_now()
        conn = self._connect()
        try:
            conn.execute(
                """
                INSERT INTO users (phone, password_hash, display_name, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (phone, password_hash, display_name, now, now),
            )
            conn.commit()
        except sqlite3.IntegrityError:
            raise ValueError("PHONE_EXISTS") from None
        return UserRecord(
            phone=phone,
            password_hash=password_hash,
            display_name=display_name,
            created_at=now,
            updated_at=now,
        )

    def get_user(self, phone: str) -> UserRecord | None:
        conn = self._connect()
        cur = conn.execute("SELECT * FROM users WHERE phone = ?", (phone,))
        row = cur.fetchone()
        return self._row_to_record(row) if row else None

    def verify_password(self, phone: str, plain: str) -> bool:
        user = self.get_user(phone)
        if user is None or user.password_hash is None:
            return False
        return bcrypt_hash.verify(plain, user.password_hash)

    def upsert_wechat_user(
        self, phone: str, openid: str, unionid: str | None = None
    ) -> UserRecord:
        now = _utc_now()
        conn = self._connect()
        existing = self.get_user(phone)
        if existing is None:
            conn.execute(
                """
                INSERT INTO users (phone, password_hash, display_name, created_at, updated_at)
                VALUES (?, NULL, NULL, ?, ?)
                """,
                (phone, now, now),
            )
        else:
            conn.execute(
                "UPDATE users SET updated_at = ? WHERE phone = ?",
                (now, phone),
            )
        conn.execute(
            """
            INSERT INTO wechat_bindings (openid, phone, unionid, bound_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(openid) DO UPDATE SET
                phone = excluded.phone,
                unionid = excluded.unionid,
                bound_at = excluded.bound_at
            """,
            (openid, phone, unionid, now),
        )
        conn.commit()
        user = self.get_user(phone)
        assert user is not None
        return user

    def get_phone_by_openid(self, openid: str) -> str | None:
        conn = self._connect()
        cur = conn.execute(
            "SELECT phone FROM wechat_bindings WHERE openid = ?",
            (openid,),
        )
        row = cur.fetchone()
        return row["phone"] if row else None


_user_store: UserStore | None = None


def get_user_store() -> UserStore:
    global _user_store
    if _user_store is None:
        _user_store = UserStore()
        _user_store.init_schema()
    return _user_store
