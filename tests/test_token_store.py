from app.core import config
from app.infra.token_store import TokenStore


class FakeRedis:
    def __init__(self):
        self._data: dict[str, str] = {}
        self._lists: dict[str, list[str]] = {}

    def setex(self, key, ttl, value):
        self._data[key] = value

    def get(self, key):
        return self._data.get(key)

    def delete(self, *keys):
        for k in keys:
            self._data.pop(k, None)
            self._lists.pop(k, None)

    def lpush(self, key, value):
        self._lists.setdefault(key, []).insert(0, value)

    def lrange(self, key, start, end):
        lst = self._lists.get(key, [])
        if end == -1:
            end = len(lst) - 1
        if not lst or start > end:
            return []
        return lst[start : end + 1]

    def llen(self, key):
        return len(self._lists.get(key, []))

    def rpop(self, key):
        lst = self._lists.get(key, [])
        if not lst:
            return None
        return lst.pop()

    def lrem(self, key, count, value):
        lst = self._lists.get(key, [])
        removed = 0
        if count == 0:
            while value in lst:
                lst.remove(value)
                removed += 1
        elif count > 0:
            for _ in range(count):
                if value in lst:
                    lst.remove(value)
                    removed += 1
                else:
                    break
        else:
            for i in range(len(lst) - 1, -1, -1):
                if count == 0:
                    break
                if lst[i] == value:
                    lst.pop(i)
                    removed += 1
                    count += 1
        return removed


def test_save_and_validate_refresh():
    store = TokenStore(FakeRedis())
    store.save_refresh("jti-1", "+8613800138000", 3600)
    assert store.is_refresh_valid("jti-1") is True
    store.revoke_refresh("jti-1")
    assert store.is_refresh_valid("jti-1") is False


def test_no_redis_fallback():
    store = TokenStore(None)
    store.save_refresh("jti-1", "+8613800138000", 3600)
    assert store.is_refresh_valid("jti-1") is True
    store.revoke_refresh("jti-1")
    assert store.is_refresh_valid("jti-1") is True


def test_max_refresh_per_phone(monkeypatch):
    monkeypatch.setattr(config, "MAX_REFRESH_PER_PHONE", 2)
    redis = FakeRedis()
    store = TokenStore(redis)
    phone = "+8613800138000"

    store.save_refresh("jti-1", phone, 3600)
    store.save_refresh("jti-2", phone, 3600)
    store.save_refresh("jti-3", phone, 3600)

    assert store.is_refresh_valid("jti-1") is False
    assert store.is_refresh_valid("jti-2") is True
    assert store.is_refresh_valid("jti-3") is True


def test_revoke_all_for_phone():
    store = TokenStore(FakeRedis())
    phone = "+8613800138000"

    store.save_refresh("jti-1", phone, 3600)
    store.save_refresh("jti-2", phone, 3600)

    store.revoke_all_for_phone(phone)

    assert store.is_refresh_valid("jti-1") is False
    assert store.is_refresh_valid("jti-2") is False
