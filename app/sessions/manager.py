import uuid
from datetime import datetime
from typing import Optional

from app.infra.redis_client import redis_client


USER_CURRENT_KEY = "user:{user_id}:current_thread"
USER_THREADS_KEY = "user:{user_id}:threads"
THREAD_META_KEY = "thread:{thread_id}:meta"


class SessionManager:
    """
    管理 user / thread（会话）的信息：
    - 当前会话
    - 会话列表
    - 新建 / 切换 / 删除会话
    """

    def __init__(self):
        self.client = redis_client

    @staticmethod
    def _now_ts() -> float:
        return datetime.utcnow().timestamp()

    @staticmethod
    def _now_iso() -> str:
        return datetime.utcnow().isoformat()

    def _user_current_key(self, user_id: str) -> str:
        return USER_CURRENT_KEY.format(user_id=user_id)

    def _user_threads_key(self, user_id: str) -> str:
        return USER_THREADS_KEY.format(user_id=user_id)

    def _thread_meta_key(self, thread_id: str) -> str:
        return THREAD_META_KEY.format(thread_id=thread_id)

    def get_current_thread(self, user_id: str) -> Optional[str]:
        return self.client.get(self._user_current_key(user_id))

    def set_current_thread(self, user_id: str, thread_id: str):
        self.client.set(self._user_current_key(user_id), thread_id)

    def create_thread(self, user_id: str, title: Optional[str] = None) -> str:
        session_id = uuid.uuid4().hex[:8]
        thread_id = f"{user_id}:s:{session_id}"

        now_iso = self._now_iso()
        now_ts = self._now_ts()

        if not title:
            title = f"对话 {now_iso[:10]}"

        meta_key = self._thread_meta_key(thread_id)
        self.client.hset(
            meta_key,
            mapping={
                "user_id": user_id,
                "title": title,
                "created_at": now_iso,
                "last_active_at": now_iso,
                "is_deleted": "0",
            },
        )

        self.client.zadd(self._user_threads_key(user_id), {thread_id: now_ts})
        self.set_current_thread(user_id, thread_id)
        return thread_id

    def touch_thread(self, thread_id: str):
        meta_key = self._thread_meta_key(thread_id)
        if not self.client.exists(meta_key):
            return

        now_iso = self._now_iso()
        now_ts = self._now_ts()

        self.client.hset(meta_key, "last_active_at", now_iso)

        user_id = self.client.hget(meta_key, "user_id")
        if user_id:
            self.client.zadd(self._user_threads_key(user_id), {thread_id: now_ts})

    def list_threads(self, user_id: str) -> list[dict]:
        key = self._user_threads_key(user_id)
        thread_ids = self.client.zrevrange(key, 0, -1)

        result = []
        for tid in thread_ids:
            meta = self.client.hgetall(self._thread_meta_key(tid))
            if not meta:
                continue
            if meta.get("is_deleted") == "1":
                continue
            result.append(
                {
                    "thread_id": tid,
                    "title": meta.get("title", tid),
                    "created_at": meta.get("created_at", ""),
                    "last_active_at": meta.get("last_active_at", ""),
                    "is_deleted": meta.get("is_deleted") == "1",
                }
            )
        return result

    def delete_thread(self, user_id: str, thread_id: str) -> Optional[str]:
        meta_key = self._thread_meta_key(thread_id)
        if not self.client.exists(meta_key):
            return None

        self.client.hset(meta_key, "is_deleted", "1")
        self.client.zrem(self._user_threads_key(user_id), thread_id)

        cur = self.get_current_thread(user_id)
        if cur != thread_id:
            return None

        threads = self.list_threads(user_id)
        if threads:
            new_thread_id = threads[0]["thread_id"]
            self.set_current_thread(user_id, new_thread_id)
            return new_thread_id

        new_thread_id = self.create_thread(user_id, title="新对话")
        return new_thread_id
