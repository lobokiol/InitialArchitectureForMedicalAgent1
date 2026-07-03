from passlib.hash import bcrypt

from app.infra.user_store import UserStore


def test_create_and_get_user(tmp_path):
    store = UserStore(str(tmp_path / "t.db"))
    store.init_schema()
    pw = bcrypt.hash("secret123")
    store.create_user("+8613800138000", pw, "测试")
    u = store.get_user("+8613800138000")
    assert u is not None
    assert u.display_name == "测试"
    assert store.verify_password("+8613800138000", "secret123")


def test_wechat_binding(tmp_path):
    store = UserStore(str(tmp_path / "t.db"))
    store.init_schema()
    store.upsert_wechat_user("+8613800138000", "openid-abc")
    assert store.get_phone_by_openid("openid-abc") == "+8613800138000"
