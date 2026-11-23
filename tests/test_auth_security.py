"""账号安全：最后一位启用管理员的降级/停用保护。"""
from app.auth import AccountStore


def test_cannot_demote_last_enabled_admin(tmp_path):
    s = AccountStore(tmp_path / "auth.json")  # 启动即生成 bootstrap admin
    assert s.count_admins() == 1
    assert s.create("alice", "secret1", "user") is None

    err = s.set_role("admin", "user")
    assert err == "不能降级最后一个启用的管理员"
    assert s.get("admin").role == "admin"


def test_can_demote_non_last_admin(tmp_path):
    s = AccountStore(tmp_path / "auth.json")
    assert s.create("admin2", "secret1", "admin") is None
    assert s.count_admins() == 2

    assert s.set_role("admin2", "user") is None
    assert s.get("admin2").role == "user"
    assert s.count_admins() == 1


def test_cannot_disable_last_enabled_admin(tmp_path):
    s = AccountStore(tmp_path / "auth.json")
    assert s.count_admins() == 1

    err = s.set_enabled("admin", False)
    assert err == "不能停用最后一个启用的管理员"
    assert s.get("admin").enabled is True


def test_can_disable_non_last_admin(tmp_path):
    s = AccountStore(tmp_path / "auth.json")
    assert s.create("admin2", "secret1", "admin") is None
    assert s.count_admins() == 2

    assert s.set_enabled("admin2", False) is None
    assert s.get("admin2").enabled is False
    assert s.count_admins() == 1
