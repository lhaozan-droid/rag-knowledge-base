"""企业知识库的账号鉴权。

设计目标：不注册、管理员统一发放账号，一人一号、可单独停用。
- 账号存 data/auth.json（明文可见，密码只存 bcrypt 哈希）。
- 会话 token 用 hmac-sha256 自签，不引入额外依赖，完全离线可用。
- 角色分 admin / user：admin 可进管理页，user 只能问答。

安全要点：
- 密码永不落明文，只存 bcrypt 哈希。
- token 含过期时间戳 + 账号名 + 角色，任何字段被篡改都会导致签名校验失败。
- secret 缺失时每次启动随机生成：重启会踢掉所有会话（安全优先，可接受）。
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
import threading
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional

import bcrypt

TOKEN_TTL_SECONDS = 7 * 24 * 3600  # 7 天，管理员可自行决定长短


@dataclass
class Account:
    username: str
    password_hash: str
    role: str = "user"        # user | admin
    enabled: bool = True
    created_at: int = 0
    updated_at: int = 0

    def public(self) -> dict:
        d = asdict(self)
        d.pop("password_hash", None)
        return d


class AccountStore:
    """基于 JSON 文件的账号库。锁保护并发读写，够轻量知识库场景用。"""

    def __init__(self, path: Path):
        self._path = Path(path)
        self._lock = threading.Lock()
        self._accounts: Dict[str, Account] = {}
        self._load()

    def _load(self) -> None:
        if not self._path.exists():
            self._accounts = {}
            self._ensure_bootstrap_admin()
            self._save()
            return
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
            self._accounts = {
                u: Account(**v) for u, v in (raw.get("accounts") or {}).items()
            }
        except Exception:  # noqa: BLE001 —— 损坏时重建，不丢原始文件
            backup = self._path.with_suffix(".json.bak")
            try:
                self._path.replace(backup)
            except Exception:  # noqa: BLE001
                pass
            self._accounts = {}
            self._ensure_bootstrap_admin()
            self._save()

    def _ensure_bootstrap_admin(self) -> None:
        """首次需要鉴权时（账号文件懒加载）若没有任何账号，创建一个初始 admin，
        密码打印到日志/控制台。

        注意触发时机：不是 `serve` 启动那一刻，而是**第一次提交登录表单**
        （`POST /login`）时，此时才懒加载构造 `AccountStore`。
        启动日志里没有初始密码，打开登录页也不会触发——`GET /` 只在带有效
        cookie 时才碰 `get_auth()`，未登录访问 `/health` 在 `get_auth()` 之前就 401 了。
        """
        if self._accounts:
            return
        pwd = secrets.token_urlsafe(9)
        h = bcrypt.hashpw(pwd.encode(), bcrypt.gensalt()).decode()
        now = int(time.time())
        self._accounts["admin"] = Account(
            username="admin", password_hash=h, role="admin",
            enabled=True, created_at=now, updated_at=now,
        )
        # 初始密码只在创建账号那一刻打印一次，之后不再输出
        print(f"[auth] 初始化管理员账号 admin，初始密码：{pwd}（请登录后立即修改）")

    def _save(self) -> None:
        tmp = self._path.with_suffix(".json.tmp")
        data = {"accounts": {u: asdict(a) for u, a in self._accounts.items()}}
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(self._path)

    # ---- 查询 ----
    def get(self, username: str) -> Optional[Account]:
        return self._accounts.get(username)

    def list_public(self) -> List[dict]:
        return [a.public() for a in sorted(self._accounts.values(), key=lambda a: a.username)]

    # ---- 校验 ----
    def verify(self, username: str, password: str) -> bool:
        acct = self._accounts.get(username)
        if not acct or not acct.enabled:
            return False
        try:
            return bcrypt.checkpw(password.encode(), acct.password_hash.encode())
        except Exception:  # noqa: BLE001
            return False

    # ---- 管理操作 ----
    def create(self, username: str, password: str, role: str = "user") -> Optional[str]:
        """创建账号，返回错误信息或 None（成功）。"""
        username = (username or "").strip()
        if not username:
            return "账号不能为空"
        if len(username) > 64 or not all(c.isalnum() or c in "-_." for c in username):
            return "账号仅支持字母、数字、-、_、."
        if username in self._accounts:
            return "账号已存在"
        if len(password) < 6:
            return "密码至少 6 位"
        now = int(time.time())
        h = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
        self._accounts[username] = Account(
            username=username, password_hash=h, role=role,
            enabled=True, created_at=now, updated_at=now,
        )
        self._save()
        return None

    def set_password(self, username: str, password: str) -> Optional[str]:
        acct = self._accounts.get(username)
        if not acct:
            return "账号不存在"
        if len(password) < 6:
            return "密码至少 6 位"
        acct.password_hash = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
        acct.updated_at = int(time.time())
        self._save()
        return None

    def set_enabled(self, username: str, enabled: bool) -> Optional[str]:
        acct = self._accounts.get(username)
        if not acct:
            return "账号不存在"
        # 保护「最后一个启用的管理员」：不能停用它，否则系统将无管理员可用而锁死。
        # 其余管理员可以正常停用。
        if acct.role == "admin" and acct.enabled and not enabled:
            if self.count_admins() <= 1:
                return "不能停用最后一个启用的管理员"
        acct.enabled = enabled
        acct.updated_at = int(time.time())
        self._save()
        return None

    def set_role(self, username: str, role: str) -> Optional[str]:
        acct = self._accounts.get(username)
        if not acct:
            return "账号不存在"
        if role not in ("admin", "user"):
            return "非法角色"
        # 保护「最后一个启用的管理员」：不能把它降为普通用户，否则系统锁死。
        if acct.role == "admin" and acct.enabled and role != "admin":
            if self.count_admins() <= 1:
                return "不能降级最后一个启用的管理员"
        acct.role = role
        acct.updated_at = int(time.time())
        self._save()
        return None

    def count_admins(self) -> int:
        return sum(1 for a in self._accounts.values() if a.role == "admin" and a.enabled)


# ================= token =================

def _secret() -> bytes:
    s = os.environ.get("RAG_AUTH_SECRET", "")
    if s:
        return s.encode()
    # 无固定 secret 时用 data 目录下的持久化 secret 文件，重启不丢会话。
    return _secret_bytes


_secret_bytes: bytes = b""


def init_secret(data_dir: Path) -> None:
    """初始化 token 签名密钥。优先环境变量，其次 data/.auth_secret，最后随机生成。"""
    global _secret_bytes
    env = os.environ.get("RAG_AUTH_SECRET", "")
    if env:
        _secret_bytes = env.encode()
        return
    f = data_dir / ".auth_secret"
    if f.exists():
        _secret_bytes = f.read_bytes()
        if _secret_bytes:
            return
    _secret_bytes = secrets.token_bytes(32)
    f.write_bytes(_secret_bytes)


def sign_token(username: str, role: str, ttl: int = TOKEN_TTL_SECONDS) -> str:
    exp = int(time.time()) + ttl
    payload = f"{username}.{role}.{exp}"
    sig = hmac.new(_secret_bytes, payload.encode(), hashlib.sha256).hexdigest()
    return f"{payload}.{sig}"


def verify_token(token: str) -> Optional[dict]:
    """校验 token，返回 {username, role} 或 None。"""
    if not token:
        return None
    parts = token.split(".")
    if len(parts) != 4:
        return None
    username, role, exp, sig = parts
    payload = f"{username}.{role}.{exp}"
    expected = hmac.new(_secret_bytes, payload.encode(), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(expected, sig):
        return None
    try:
        if int(exp) < int(time.time()):
            return None
    except ValueError:
        return None
    return {"username": username, "role": role}
