"""Chunk Registry：SQLite 单一事实源。

为什么不让向量库同时存文本？
- 上千份文档时，向量库只该干一件事件（存向量 + 粗筛），payload 反复 Upsert 很慢；
- 父块/子块、审计日志、未答问题都要关系查询，SQLite 更顺手；
- 换向量库（chroma → qdrant）时文本和 ID 不受影响。
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from app.models import Chunk, Document, Turn

SCHEMA = """
PRAGMA journal_mode=WAL;
CREATE TABLE IF NOT EXISTS docs(
  doc_id TEXT PRIMARY KEY, path TEXT UNIQUE, title TEXT, source_type TEXT,
  content_hash TEXT, mtime REAL, char_count INTEGER, meta TEXT, updated_at REAL);
CREATE TABLE IF NOT EXISTS chunks(
  chunk_id TEXT PRIMARY KEY, doc_id TEXT, text TEXT, ordinal INTEGER, title TEXT,
  section TEXT, path TEXT, parent_id TEXT, is_parent INTEGER, meta TEXT);
CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id);
CREATE INDEX IF NOT EXISTS idx_chunks_parent ON chunks(parent_id);
CREATE TABLE IF NOT EXISTS turns(
  id INTEGER PRIMARY KEY AUTOINCREMENT, conversation_id TEXT, role TEXT,
  content TEXT, ts REAL, flags TEXT, username TEXT NOT NULL DEFAULT '');
CREATE INDEX IF NOT EXISTS idx_turns_conv ON turns(conversation_id, id);
CREATE TABLE IF NOT EXISTS unanswered(
  id INTEGER PRIMARY KEY AUTOINCREMENT, conversation_id TEXT, query TEXT,
  rewritten TEXT, top_score REAL, ts REAL, reason TEXT, detail TEXT);
-- 多级分类树。parent_id=0 表示根级；文档通过 docs.category_id 挂到叶子/中间节点均可。
CREATE TABLE IF NOT EXISTS categories(
  id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL,
  parent_id INTEGER NOT NULL DEFAULT 0, sort_order INTEGER NOT NULL DEFAULT 0,
  created_at REAL);
CREATE INDEX IF NOT EXISTS idx_categories_parent ON categories(parent_id);
"""

# 老库升级：unanswered 表补 reason / detail 两列。
# 不补的话，三类未答（检索空 / 模型拒答 / 引用缺失）会混在一起，
# 而它们的处理方式完全不同——检索空是知识缺口，引用缺失是工程问题。
# docs.category_id：分类是独立于内容的「管理属性」，单独一列而不是塞进 meta，
# 这样按分类筛选/统计能走索引，也不必为了改分类去重写整份 meta。
_MIGRATIONS = [
    ("unanswered", "reason", "TEXT"),
    ("unanswered", "detail", "TEXT"),
    ("docs", "category_id", "INTEGER NOT NULL DEFAULT 0"),
    # 会话按账号隔离：turns 补 username 列。老数据补成空串，
    # 空串不属于任何真实账号（账号名不允许为空），因此旧对话对新列表不可见。
    ("turns", "username", "TEXT NOT NULL DEFAULT ''"),
]


def _migrate(conn: sqlite3.Connection) -> None:
    for table, column, ctype in _MIGRATIONS:
        cols = {r["name"] for r in conn.execute(f"PRAGMA table_info({table})")}
        if cols and column not in cols:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {ctype}")
    conn.commit()


class Registry:
    def __init__(self, path: str | Path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.executescript(SCHEMA)
            self._conn.commit()
            _migrate(self._conn)

    def close(self) -> None:
        self._conn.close()

    # ---------------- docs ----------------
    def upsert_doc(self, d: Document) -> None:
        with self._lock:
            self._conn.execute(
                """INSERT INTO docs(doc_id, path, title, source_type, content_hash, mtime,
                                    char_count, meta, updated_at)
                   VALUES (?,?,?,?,?,?,?,?,?)
                   ON CONFLICT(doc_id) DO UPDATE SET
                     path=excluded.path, title=excluded.title, source_type=excluded.source_type,
                     content_hash=excluded.content_hash, mtime=excluded.mtime,
                     char_count=excluded.char_count, meta=excluded.meta,
                     updated_at=excluded.updated_at""",
                (d.doc_id, d.path, d.title, d.source_type, d.content_hash, d.mtime,
                 d.char_count, json.dumps(d.meta, ensure_ascii=False), time.time()))
            self._conn.commit()

    def get_doc(self, doc_id: str) -> Optional[Document]:
        row = self._conn.execute("SELECT * FROM docs WHERE doc_id=?", (doc_id,)).fetchone()
        return _row_to_doc(row) if row else None

    def get_doc_by_path(self, path: str) -> Optional[Document]:
        row = self._conn.execute("SELECT * FROM docs WHERE path=?", (path,)).fetchone()
        return _row_to_doc(row) if row else None

    def all_docs(self) -> List[Document]:
        rows = self._conn.execute("SELECT * FROM docs ORDER BY path").fetchall()
        return [_row_to_doc(r) for r in rows]

    def list_docs(self, q: str | None = None,
                  source_type: str | None = None,
                  category_id: int | None = None,
                  include_sub: bool = True,
                  limit: int = 0, offset: int = 0,
                  unclassified: bool = False) -> List[dict]:
        """逐条文档元数据（含子块数），供管理后台文档清单使用。

        与 all_docs() 的区别：额外返回 chunk_count / updated_at，并按最近更新倒序。
        chunk_count 只统计子块（is_parent=0），与 stats().chunks 口径一致。
        当前数据模型没有「下线/归档」状态，删除是物理删除，因此列出的文档
        status 统一为 active（该字段仅为前端展示与后续扩展预留）。

        category_id 过滤默认含子孙分类（include_sub=True）：在树里点「员工制度」，
        期待看到它下面所有子类的文档，只算直接挂靠的会让父节点看起来是空的。
        limit>0 时分页；不传则一次返回全部（保持旧行为，供脚本/测试使用）。
        """
        where, args = self._docs_filter(q, source_type, category_id, include_sub,
                                        unclassified)
        sql = (
            "SELECT d.doc_id, d.path, d.title, d.source_type, d.content_hash,"
            "       d.mtime, d.char_count, d.updated_at, d.meta, d.category_id,"
            "       COUNT(CASE WHEN c.is_parent = 0 THEN 1 END) AS chunk_count"
            " FROM docs d LEFT JOIN chunks c ON c.doc_id = d.doc_id")
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " GROUP BY d.doc_id ORDER BY d.updated_at DESC, d.path"
        if limit and limit > 0:
            sql += " LIMIT ? OFFSET ?"
            args.extend([int(limit), int(offset or 0)])
        rows = self._conn.execute(sql, args).fetchall()

        cats = {int(r["id"]): (r["name"], int(r["parent_id"]))
                for r in self._conn.execute(
                    "SELECT id, name, parent_id FROM categories")}
        out: List[dict] = []
        for r in rows:
            try:
                meta = json.loads(r["meta"] or "{}")
            except Exception:  # noqa: BLE001
                meta = {}
            cid = int(r["category_id"] or 0)
            out.append({
                "doc_id": r["doc_id"],
                "path": r["path"],
                "title": r["title"],
                "source_type": r["source_type"],
                "char_count": r["char_count"] or 0,
                "chunk_count": int(r["chunk_count"] or 0),
                "mtime": r["mtime"],
                "updated_at": r["updated_at"],
                "status": "active",
                "meta": meta,
                "category_id": cid,
                "category_path": _category_path(cid, cats),
            })
        return out

    def count_docs_filtered(self, q: str | None = None,
                            source_type: str | None = None,
                            category_id: int | None = None,
                            include_sub: bool = True,
                            unclassified: bool = False) -> int:
        """与 list_docs 同条件的总数，用于分页。条件构造必须共用，否则两处
        筛选逻辑迟早漂移，表现为「第一页有数据、翻页后变少」。"""
        where, args = self._docs_filter(q, source_type, category_id, include_sub,
                                        unclassified)
        sql = "SELECT COUNT(*) FROM docs d"
        if where:
            sql += " WHERE " + " AND ".join(where)
        return int(self._conn.execute(sql, args).fetchone()[0])

    def _docs_filter(self, q, source_type, category_id, include_sub,
                     unclassified: bool = False):
        """构造 list_docs / count_docs_filtered 共用的 WHERE 片段。

        「未分类」必须用独立的 unclassified 开关，不能复用 category_id=0：
        0 同时表示「不筛选」，两者混在一个字段里会让「查未分类文档」永远
        退化成「查全部文档」，而调用方看不出区别。
        """
        where: List[str] = []
        args: list = []
        if q:
            where.append("(d.title LIKE ? OR d.path LIKE ?)")
            args.extend([f"%{q}%", f"%{q}%"])
        if source_type:
            where.append("d.source_type = ?")
            args.append(source_type)
        if unclassified:
            where.append("d.category_id = 0")
        elif category_id:
            ids = [int(category_id)]
            if include_sub:
                ids.extend(self.category_descendants(int(category_id)))
            where.append(f"d.category_id IN ({','.join('?' * len(ids))})")
            args.extend(ids)
        return where, args

    def count_chunks_of_doc(self, doc_id: str) -> int:
        return int(self._conn.execute(
            "SELECT COUNT(*) FROM chunks WHERE doc_id=?", (doc_id,)).fetchone()[0])

    def delete_doc(self, doc_id: str) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM chunks WHERE doc_id=?", (doc_id,))
            self._conn.execute("DELETE FROM docs WHERE doc_id=?", (doc_id,))
            self._conn.commit()

    # ---------------- categories ----------------
    def list_categories(self) -> List[dict]:
        """全部分类（扁平列表，前端按 parent_id 组装树）。

        同时给出两个数：doc_count 只算直接挂靠该分类的文档；total_count 含全部
        子孙。树上展示必须用 total_count——只算直接数会让「员工制度」这种纯分组
        节点显示成 0，管理者会误以为分类没生效。
        """
        rows = self._conn.execute(
            "SELECT id, name, parent_id, sort_order, created_at FROM categories"
            " ORDER BY sort_order, id").fetchall()
        cats = [{"id": int(r["id"]), "name": r["name"],
                 "parent_id": int(r["parent_id"]),
                 "sort_order": int(r["sort_order"]),
                 "created_at": r["created_at"]} for r in rows]
        direct = {c["id"]: 0 for c in cats}
        for r in self._conn.execute(
                "SELECT category_id, COUNT(*) AS n FROM docs GROUP BY category_id"):
            cid = int(r["category_id"] or 0)
            if cid in direct:
                direct[cid] = int(r["n"])
        children: Dict[int, List[int]] = {}
        for c in cats:
            children.setdefault(c["parent_id"], []).append(c["id"])
        memo: Dict[int, int] = {}

        def total(cid: int) -> int:
            if cid in memo:
                return memo[cid]
            n = direct.get(cid, 0)
            for ch in children.get(cid, []):
                n += total(ch)
            memo[cid] = n
            return n

        for c in cats:
            c["doc_count"] = direct.get(c["id"], 0)
            c["total_count"] = total(c["id"])
        return cats

    def get_category(self, cat_id: int) -> Optional[dict]:
        row = self._conn.execute(
            "SELECT id, name, parent_id, sort_order, created_at FROM categories"
            " WHERE id=?", (int(cat_id),)).fetchone()
        return dict(row) if row else None

    def category_children(self, cat_id: int) -> List[int]:
        return [int(r["id"]) for r in self._conn.execute(
            "SELECT id FROM categories WHERE parent_id=? ORDER BY sort_order, id",
            (int(cat_id),))]

    def category_descendants(self, cat_id: int) -> List[int]:
        """全部子孙分类 id（不含自己）。带环检测，避免脏数据把调用方卡死。"""
        out: List[int] = []
        seen = {int(cat_id)}
        stack = [int(cat_id)]
        while stack:
            cur = stack.pop()
            for kid in self.category_children(cur):
                if kid in seen:
                    continue
                seen.add(kid)
                out.append(kid)
                stack.append(kid)
        return out

    def create_category(self, name: str, parent_id: int = 0) -> int:
        name = (name or "").strip()
        if not name:
            raise ValueError("分类名不能为空")
        parent_id = int(parent_id or 0)
        if parent_id and self.get_category(parent_id) is None:
            raise KeyError(f"父分类不存在：{parent_id}")
        if self._conn.execute(
                "SELECT id FROM categories WHERE parent_id=? AND name=?",
                (parent_id, name)).fetchone():
            raise ValueError(f"同级下已存在分类「{name}」")
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO categories(name, parent_id, sort_order, created_at)"
                " VALUES (?,?,?,?)",
                (name, parent_id, self._next_sort_order(parent_id), time.time()))
            self._conn.commit()
            return int(cur.lastrowid)

    def _next_sort_order(self, parent_id: int) -> int:
        row = self._conn.execute(
            "SELECT COALESCE(MAX(sort_order),0)+1 FROM categories WHERE parent_id=?",
            (int(parent_id),)).fetchone()
        return int(row[0]) if row else 1

    def rename_category(self, cat_id: int, name: str) -> None:
        name = (name or "").strip()
        if not name:
            raise ValueError("分类名不能为空")
        cat = self.get_category(cat_id)
        if cat is None:
            raise KeyError(f"分类不存在：{cat_id}")
        if self._conn.execute(
                "SELECT id FROM categories WHERE parent_id=? AND name=? AND id<>?",
                (cat["parent_id"], name, int(cat_id))).fetchone():
            raise ValueError(f"同级下已存在分类「{name}」")
        with self._lock:
            self._conn.execute("UPDATE categories SET name=? WHERE id=?",
                               (name, int(cat_id)))
            self._conn.commit()

    def move_category(self, cat_id: int, parent_id: int) -> None:
        """移动分类到另一个父级下。禁止移到自身或其子孙下，否则树会成环。"""
        cat = self.get_category(cat_id)
        if cat is None:
            raise KeyError(f"分类不存在：{cat_id}")
        parent_id = int(parent_id or 0)
        if parent_id == int(cat_id):
            raise ValueError("不能把分类移到它自己下面")
        if parent_id:
            if self.get_category(parent_id) is None:
                raise KeyError(f"目标父分类不存在：{parent_id}")
            cur, guard = parent_id, 0
            while cur and guard < 100:
                if cur == int(cat_id):
                    raise ValueError("不能把分类移到它自己的子分类下")
                row = self._conn.execute(
                    "SELECT parent_id FROM categories WHERE id=?", (cur,)).fetchone()
                cur = int(row["parent_id"]) if row else 0
                guard += 1
        with self._lock:
            self._conn.execute("UPDATE categories SET parent_id=? WHERE id=?",
                               (parent_id, int(cat_id)))
            self._conn.commit()

    def reorder_categories(self, orders: List[tuple]) -> None:
        """批量设置同级排序。orders 为 [(cat_id, sort_order), ...]。"""
        if not orders:
            return
        with self._lock:
            self._conn.executemany(
                "UPDATE categories SET sort_order=? WHERE id=?",
                [(int(so), int(cid)) for cid, so in orders])
            self._conn.commit()

    def delete_category(self, cat_id: int, force: bool = False) -> dict:
        """删除分类。

        有子分类一律拒绝——否则整棵子树会变成孤儿，再也找不回来。
        挂靠了文档时默认拒绝，force=True 才删，并把这些文档置为「未分类」：
        分类是管理属性，误删一个分类绝不能连带毁掉知识库里的文档。
        """
        cat = self.get_category(cat_id)
        if cat is None:
            raise KeyError(f"分类不存在：{cat_id}")
        kids = self.category_children(cat_id)
        if kids:
            raise ValueError(
                f"「{cat['name']}」下还有 {len(kids)} 个子分类，请先删除或移走")
        used = int(self._conn.execute(
            "SELECT COUNT(*) FROM docs WHERE category_id=?", (int(cat_id),)).fetchone()[0])
        if used and not force:
            raise ValueError(
                f"「{cat['name']}」下还有 {used} 篇文档，请先移走或确认强制删除")
        with self._lock:
            if used:
                self._conn.execute("UPDATE docs SET category_id=0 WHERE category_id=?",
                                   (int(cat_id),))
            self._conn.execute("DELETE FROM categories WHERE id=?", (int(cat_id),))
            self._conn.commit()
        return {"id": int(cat_id), "name": cat["name"], "docs_unassigned": used}

    def set_doc_category(self, doc_id: str, category_id: int) -> bool:
        """给单篇文档归类。category_id=0 表示「未分类」。"""
        with self._lock:
            cur = self._conn.execute(
                "UPDATE docs SET category_id=? WHERE doc_id=?",
                (int(category_id or 0), doc_id))
            self._conn.commit()
            return bool(cur.rowcount)

    def set_docs_category(self, doc_ids: List[str], category_id: int) -> int:
        """批量归类，返回实际更新篇数。"""
        if not doc_ids:
            return 0
        cid = int(category_id or 0)
        n = 0
        with self._lock:
            for i in range(0, len(doc_ids), 400):
                part = list(doc_ids[i:i + 400])
                cur = self._conn.execute(
                    f"UPDATE docs SET category_id=? WHERE doc_id IN "
                    f"({','.join('?' * len(part))})", [cid] + part)
                n += cur.rowcount or 0
            self._conn.commit()
        return n

    def migrate_categories_from_meta(self) -> dict:
        """把 docs.meta.category 里的历史字符串分类解析成分类树并回填 category_id。

        历史数据是以「 / 」串联的层级字符串，形如「人力资源 / 假期管理 / 年假申请」。
        分隔符必须是「 / 」（带空格的斜杠）：分类名本身可能含裸斜杠，
        例如「人力/行政 / 假期管理」，按 '/' 切会把一个名字劈成两半。
        名称尾部还常残留一个「/」（如「假期管理/ / 年假申请/」），要清掉。

        幂等：只处理 category_id=0 的文档，重复执行不会造出重复分类。
        """
        rows = self._conn.execute(
            "SELECT doc_id, meta FROM docs WHERE category_id=0").fetchall()
        if not rows:
            return {"docs": 0, "categories": 0}
        cache: dict = {}
        for r in self._conn.execute("SELECT id, name, parent_id FROM categories"):
            cache[(int(r["parent_id"]), r["name"])] = int(r["id"])
        created = 0
        updated = 0
        for r in rows:
            try:
                meta = json.loads(r["meta"] or "{}")
            except Exception:  # noqa: BLE001
                meta = {}
            raw = str(meta.get("category") or "").strip()
            if not raw:
                continue
            parent = 0
            last = 0
            for seg in raw.split(" / "):
                seg = seg.strip().strip("/").strip()
                if not seg:
                    continue
                key = (parent, seg)
                cid = cache.get(key)
                if cid is None:
                    with self._lock:
                        cur = self._conn.execute(
                            "INSERT INTO categories(name, parent_id, sort_order,"
                            " created_at) VALUES (?,?,?,?)",
                            (seg, parent, self._next_sort_order(parent), time.time()))
                    cid = int(cur.lastrowid)
                    cache[key] = cid
                    created += 1
                parent = cid
                last = cid
            if last:
                with self._lock:
                    self._conn.execute(
                        "UPDATE docs SET category_id=? WHERE doc_id=?",
                        (last, r["doc_id"]))
                updated += 1
        with self._lock:
            self._conn.commit()
        return {"docs": updated, "categories": created}

    # ---------------- chunks ----------------
    def add_chunks(self, chunks: Iterable[Chunk]) -> int:
        rows = [(c.chunk_id, c.doc_id, c.text, c.ordinal, c.title, c.section, c.path,
                 c.parent_id, c.is_parent, json.dumps(c.meta, ensure_ascii=False))
                for c in chunks]
        with self._lock:
            self._conn.executemany(
                """INSERT OR REPLACE INTO chunks(chunk_id, doc_id, text, ordinal, title,
                     section, path, parent_id, is_parent, meta)
                   VALUES (?,?,?,?,?,?,?,?,?,?)""", rows)
            self._conn.commit()
        return len(rows)

    def get_chunks(self, chunk_ids: List[str]) -> Dict[str, Chunk]:
        if not chunk_ids:
            return {}
        out: Dict[str, Chunk] = {}
        for i in range(0, len(chunk_ids), 400):
            part = chunk_ids[i:i + 400]
            q = f"SELECT * FROM chunks WHERE chunk_id IN ({','.join('?' * len(part))})"
            for row in self._conn.execute(q, part):
                ch = _row_to_chunk(row)
                out[ch.chunk_id] = ch
        return out

    def list_chunks_of_doc(self, doc_id: str) -> List[dict]:
        """按文档列出它的全部分块（子块+父块），按 ordinal 排序。

        供管理后台「查看切片」只读展示：排查「为什么检索没命中」时，
        需要看到这篇文档被切成了几块、每块正文、父块与子块的对应关系。
        """
        rows = self._conn.execute(
            "SELECT * FROM chunks WHERE doc_id=? ORDER BY ordinal",
            (doc_id,)).fetchall()
        out: List[dict] = []
        for r in rows:
            try:
                meta = json.loads(r["meta"] or "{}")
            except Exception:  # noqa: BLE001
                meta = {}
            out.append({
                "chunk_id": r["chunk_id"],
                "ordinal": r["ordinal"],
                "text": r["text"],
                "title": r["title"],
                "section": r["section"] or "",
                "is_parent": bool(r["is_parent"]),
                "parent_id": r["parent_id"] or "",
                "char_count": len(r["text"] or ""),
                "meta": meta,
            })
        return out

    def iter_children(self, batch: int = 1000):
        """流式遍历所有子块，用于构建 BM25 索引（上千份文档避免一次性载入内存）。"""
        cur = self._conn.execute(
            "SELECT chunk_id, text FROM chunks WHERE is_parent=0 ORDER BY doc_id, ordinal")
        while True:
            rows = cur.fetchmany(batch)
            if not rows:
                break
            for r in rows:
                yield r["chunk_id"], r["text"]

    def iter_index_docs(self, batch: int = 1000):
        """给 BM25 用的索引文本：正文 + 标题 + 所属分类。

        为什么把分类拼进去：用户问「休假怎么申请」，目标文档标题是
        《考勤与请假管理办法》，正文写的是「在 OA「请假申请」中提交」，
        字面几乎不重合；但它所属分类是「员工制度 / 请假与考勤 / 年假申请」，
        **分类名恰恰就是用户会用的说法**。把分类当作检索信号，
        等于给每篇文档补回了用户视角的关键词，比硬扩展 query 更稳。
        """
        import json as _json

        cats: dict[str, str] = {}
        for r in self._conn.execute("SELECT doc_id, meta FROM docs"):
            try:
                m = _json.loads(r["meta"] or "{}")
            except Exception:  # noqa: BLE001
                m = {}
            c = str(m.get("category") or "").replace(" / ", " ")
            if c:
                cats[r["doc_id"]] = c

        cur = self._conn.execute(
            "SELECT chunk_id, doc_id, title, text FROM chunks "
            "WHERE is_parent=0 ORDER BY doc_id, ordinal")
        while True:
            rows = cur.fetchmany(batch)
            if not rows:
                break
            for r in rows:
                parts = [str(r["title"] or "")]
                cat = cats.get(r["doc_id"])
                if cat:
                    parts.append(cat)
                parts.append(r["text"] or "")
                yield r["chunk_id"], "\n".join(p for p in parts if p)

    def reset_docs(self) -> None:
        """只清文档与 chunk，保留对话历史和未答问题（用于重建知识库后复盘）。"""
        with self._lock:
            self._conn.execute("DELETE FROM chunks")
            self._conn.execute("DELETE FROM docs")
            self._conn.commit()

    def find_by_content_hash(self, content_hash: str) -> Optional[Document]:
        row = self._conn.execute(
            "SELECT * FROM docs WHERE content_hash=?", (content_hash,)).fetchone()
        return _row_to_doc(row) if row else None

    def count_docs(self) -> int:
        return int(self._conn.execute("SELECT COUNT(*) FROM docs").fetchone()[0])

    def count_chunks(self) -> int:
        return int(self._conn.execute("SELECT COUNT(*) FROM chunks WHERE is_parent=0").fetchone()[0])

    # ---------------- conversation ----------------
    def append_turn(self, conversation_id: str, role: str, content: str,
                    flags: dict | None = None, username: str = "") -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO turns(conversation_id, role, content, ts, flags, username)"
                " VALUES (?,?,?,?,?,?)",
                (conversation_id, role, content, time.time(),
                 json.dumps(flags or {}, ensure_ascii=False), username or ""))
            self._conn.commit()

    def get_turns(self, conversation_id: str, limit: int = 20,
                  username: Optional[str] = None) -> List[Turn]:
        """按会话取对话轮次。username 非空时叠加账号过滤，实现按账号隔离。

        返回为空有两种情况：会话不存在、或会话不属于该账号。接口层统一按
        「无权限/不存在」处理（404），不泄露会话是否存在、归属何人的信息。
        """
        if username:
            rows = self._conn.execute(
                "SELECT role, content, ts FROM turns"
                " WHERE conversation_id=? AND username=? ORDER BY id DESC LIMIT ?",
                (conversation_id, username, limit)).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT role, content, ts FROM turns"
                " WHERE conversation_id=? ORDER BY id DESC LIMIT ?",
                (conversation_id, limit)).fetchall()
        return [Turn(role=r["role"], content=r["content"], ts=r["ts"]) for r in reversed(rows)]

    def conversation_owner(self, conversation_id: str) -> Optional[str]:
        """会话归属的账号名；会话没有任何 turn 时返回 None。

        同一 conversation_id 在隔离生效后只应属于一个账号（写路径已校验归属），
        这里取最早一条 turn 的 username 作为权威归属；老数据空串表示无主孤儿。
        """
        row = self._conn.execute(
            "SELECT username FROM turns WHERE conversation_id=? ORDER BY id LIMIT 1",
            (conversation_id,)).fetchone()
        return row["username"] if row else None

    def list_conversations(self, username: str, limit: int = 50) -> List[dict]:
        """当前账号的会话列表，供「历史对话」渲染。

        每个 conversation_id 聚合出：
        - title：该会话最新一条 user 内容的开头（截断 40 字）
        - updated_at：该会话最后一条 turn 的时间戳（无论 user/assistant）
        - turn_count：该会话轮次数

        只统计 username 归属的 turn，账号之间互不可见；按 updated_at 倒序。

        username 为空时直接返回空列表（fail-closed）：空串是迁移遗留的无主孤儿标记，
        绝不能把空串当通配把孤儿会话暴露给调用方。
        """
        if not username:
            return []
        rows = self._conn.execute(
            """SELECT t.conversation_id,
                      (SELECT content FROM turns u
                        WHERE u.conversation_id = t.conversation_id
                          AND u.role = 'user'
                          AND u.username = t.username
                        ORDER BY u.id DESC LIMIT 1) AS title,
                      MAX(t.ts) AS updated_at,
                      COUNT(*) AS turn_count
                 FROM turns t
                WHERE t.username = ?
                GROUP BY t.conversation_id
                ORDER BY updated_at DESC
                LIMIT ?""",
            (username, int(limit))).fetchall()
        out: List[dict] = []
        for r in rows:
            title = (r["title"] or "").strip()
            out.append({
                "conversation_id": r["conversation_id"],
                "title": title[:40],
                "updated_at": r["updated_at"],
                "turn_count": int(r["turn_count"] or 0),
            })
        return out

    def log_unanswered(self, conversation_id: str, query: str,
                       rewritten: str, top_score: float,
                       reason: str = "no_hit", detail: str = "") -> None:
        """记录没答好的问题。reason 区分三类失败，它们的处置方式完全不同。

        - no_hit           检索没命中 → 知识库缺内容，补文档
        - model_refused    模型说资料不足/越界 → 文档表述和员工问法对不上，改分块或补充别名
        - citation_missing 答了但没标引用 → 工程问题，查 Prompt 或分块粒度
        """
        with self._lock:
            self._conn.execute(
                "INSERT INTO unanswered(conversation_id, query, rewritten, top_score,"
                " ts, reason, detail) VALUES (?,?,?,?,?,?,?)",
                (conversation_id, query, rewritten, top_score, time.time(),
                 reason, (detail or "")[:500]))
            self._conn.commit()

    def list_unanswered(self, limit: int = 50, reason: str | None = None) -> List[dict]:
        sql = ("SELECT query, rewritten, top_score, ts, conversation_id, reason, detail"
               " FROM unanswered")
        args: list = []
        if reason:
            sql += " WHERE reason=?"
            args.append(reason)
        sql += " ORDER BY id DESC LIMIT ?"
        args.append(limit)
        return [dict(r) for r in self._conn.execute(sql, args).fetchall()]

    def unanswered_by_reason(self) -> Dict[str, int]:
        rows = self._conn.execute(
            "SELECT COALESCE(reason,'(unset)') AS r, COUNT(*) AS n"
            " FROM unanswered GROUP BY r").fetchall()
        return {r["r"]: int(r["n"]) for r in rows}

    # ---------------- stats ----------------
    def stats(self) -> dict:
        docs = self.count_docs()
        chunks = self.count_chunks()
        questions = int(self._conn.execute(
            "SELECT COUNT(*) FROM turns WHERE role='user'").fetchone()[0])
        unanswered = int(self._conn.execute("SELECT COUNT(*) FROM unanswered").fetchone()[0])
        return {"docs": docs, "chunks": chunks, "avg_chunks_per_doc":
                round(chunks / docs, 1) if docs else 0,
                "questions": questions, "unanswered": unanswered}


def _category_path(cat_id: int, cats: dict) -> str:
    """分类全路径（如「员工制度 / 请假与考勤」），未分类返回空串。

    cats 由调用方一次性查出后传入，避免逐篇文档各查一次父链（上千篇文档
    会退化成上万次查询）。带层数上限，脏数据造成环时不会死循环。
    """
    parts: List[str] = []
    cur = int(cat_id or 0)
    guard = 0
    while cur and cur in cats and guard < 50:
        name, parent = cats[cur]
        parts.append(name)
        cur = int(parent or 0)
        guard += 1
    return " / ".join(reversed(parts))


def _row_to_doc(row: sqlite3.Row) -> Document:
    return Document(
        doc_id=row["doc_id"], path=row["path"], title=row["title"],
        source_type=row["source_type"], content_hash=row["content_hash"],
        mtime=row["mtime"], char_count=row["char_count"] or 0,
        meta=json.loads(row["meta"] or "{}"))


def _row_to_chunk(row: sqlite3.Row) -> Chunk:
    return Chunk(
        chunk_id=row["chunk_id"], doc_id=row["doc_id"], text=row["text"],
        ordinal=row["ordinal"], title=row["title"], section=row["section"] or "",
        path=row["path"] or "", parent_id=row["parent_id"] or "",
        is_parent=row["is_parent"] or 0, meta=json.loads(row["meta"] or "{}"))
