"""Lock SessionManager thread meta accessors used by API routers."""

from app.sessions.manager import InMemorySessionStore, SessionManager


def test_get_thread_info_returns_api_shape():
    mgr = SessionManager(client=InMemorySessionStore())
    tid = mgr.create_thread("13800138000", title="导诊会话")
    info = mgr.get_thread_info(tid)
    assert info is not None
    assert info["thread_id"] == tid
    assert info["title"] == "导诊会话"
    assert info["is_deleted"] is False
    assert info["created_at"]
    assert info["last_active_at"]


def test_get_thread_info_missing_returns_none():
    mgr = SessionManager(client=InMemorySessionStore())
    assert mgr.get_thread_info("missing:s:0000") is None


def test_get_thread_meta_empty_for_missing():
    mgr = SessionManager(client=InMemorySessionStore())
    assert mgr.get_thread_meta("missing:s:0000") == {}
