"""增量入库流水线。

上千份文档的关键约束，全部在这里处理：
1. 幂等：内容哈希不变就跳过，重复跑不会产生重复 chunk；
2. 增量：只对改动文档重算向量（配合 Embedding 缓存，实际只算新增段落）；
3. 一致性：先删旧 chunk 再写新，避免"半更新"留下孤儿数据；
4. 去重：同一份文件被复制成多个副本时只保留一份；
5. 可恢复：单文件失败不影响整批，错误全部回收到 report。
"""
from __future__ import annotations

import hashlib
import logging
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from app.config import Config
from app.ingest.chunker import chunk_document, embedding_text
from app.ingest.loaders import LoadedDoc, load_file, scan_dir
from app.models import Document, IngestReport
from app.retrieval.bm25 import BM25Index

log = logging.getLogger(__name__)

Progress = Optional[Callable[[int, int, str], None]]

# 缓冲区攒到多少条向量就落盘。单次 Chroma 写入有 6 秒左右的固定开销，
# 所以不能一条一条写；但也不能攒太大，否则进程被杀时未落盘的向量会永久缺失
# （注册表里的 chunk 已经登记，续跑会被 mtime 快路径跳过）。1500 条大约对应
# 150 篇普通文档，一次写入约 6~8 秒，丢失窗口可接受。
FLUSH_CHUNK_THRESHOLD = 1500


def _sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def _rel(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.name


class IngestPipeline:
    def __init__(self, cfg: Config, registry, store, embedder,
                 bm25: BM25Index | None = None):
        self.cfg = cfg
        self.registry = registry
        self.store = store
        self.embedder = embedder
        # 注意：不能写 `bm25 or BM25Index(...)`。
        # BM25Index 定义了 __len__，空索引是 falsy，会被静默替换成另一个新对象，
        # 于是调用方手里的索引永远填不上数据（空库首次入库后稀疏路直接失效）。
        self.bm25 = bm25 if bm25 is not None else BM25Index(cfg.bm25_path)
        # 向量写入攒批。Chroma 单次 upsert 有约 6 秒固定开销，与写入条数几乎无关：
        # 逐篇写 4000 篇要 6 小时，攒成 20 批只要几分钟。
        self._pending: List[Tuple[List[str], List[List[float]], List[dict]]] = []
        # 待删除的 doc_id（更新场景）。和向量写入同批执行，避免每篇更新都触发一次
        # Chroma 写入的固定开销。
        self._pending_deletes: set[str] = set()
        # 本次入库是否走进度条回调。只影响逐篇日志的级别，不影响任何入库行为。
        self._progress_active = False

    def _pending_chunks(self) -> int:
        """缓冲区里攒了多少条向量（按 chunk 计，而不是按批次数）。"""
        return sum(len(b[0]) for b in self._pending)

    def _flush_vectors(self) -> None:
        if not self._pending and not self._pending_deletes:
            return
        # 顺序很关键：删除必须在新向量写入之前执行。
        # 同一次 flush 里如果先写后删，会把刚写进去的新向量一并删掉。
        for doc_id in self._pending_deletes:
            self.store.delete_doc(doc_id)
        self._pending_deletes.clear()
        if not self._pending:
            return
        # 按 id 去重兜底：万一有重复 id 混进来，Chroma 会直接抛
        # DuplicateIDError 让整批入库中断，这里保住后面的数据
        ids: Dict[str, List[float]] = {}
        metas: Dict[str, dict] = {}
        for batch_ids, vecs, batch_metas in self._pending:
            for cid, vec, meta in zip(batch_ids, vecs, batch_metas):
                ids[cid] = vec
                metas[cid] = meta
        self._pending.clear()
        if ids:
            self.store.add(ids=list(ids), vectors=list(ids.values()),
                           metadatas=[metas[i] for i in ids])

    # ------------------------------------------------------------------
    def ingest(self, target: str | Path, reset: bool = False,
               force: bool = False, progress: Progress = None) -> IngestReport:
        t0 = time.time()
        target = Path(target).resolve()
        root = target if target.is_dir() else target.parent
        files, rejected = scan_dir(target)
        report = IngestReport()
        self._progress_active = progress is not None
        if rejected:
            report.unsupported = [f"{_rel(p, root)}（{reason}）" for p, reason in rejected]
            log.warning("有 %d 个文件不会入库（格式不支持），详见报告", len(rejected))

        if reset:
            log.warning("reset=True：清空知识库后重新入库")
            self.store.reset()
            self.registry.reset_docs()
            self._pending.clear()
            self._pending_deletes.clear()

        existing = {d.doc_id: d for d in self.registry.all_docs()}
        # 按路径再建一份索引：doc_id 可能是内容自带的外部 ID（article_id），
        # 拿路径哈希去 existing 里查永远查不到，mtime 快路径就形同虚设，
        # 每次重跑都得把几千个文件全部读一遍、算一遍哈希。
        by_path: Dict[str, Document] = {d.path: d for d in existing.values()}
        hash_index: Dict[str, str] = {d.content_hash: d.doc_id for d in existing.values()}

        self._pending.clear()
        self._pending_deletes.clear()
        try:
            for i, path in enumerate(files, 1):
                rel = path.relative_to(root).as_posix()
                if progress:
                    progress(i, len(files), rel)
                try:
                    self._ingest_one(path, rel, existing, hash_index, report,
                                     force, by_path)
                except Exception as exc:  # noqa: BLE001
                    report.failed += 1
                    msg = f"{rel}: {type(exc).__name__}: {exc}"
                    report.errors.append(msg)
                    log.exception("入库失败 %s", rel)
                # 攒够一小批就写：既摊掉单次写入的固定开销，又把「进程被杀时
                # 缓冲区里还没落盘的向量」这个丢失窗口压到尽可能小。
                # 之前按 100 批（约 6000 条）才写，一次中断就丢了上千条向量，
                # 而注册表里 chunk 早已登记，续跑时会被 mtime 快路径跳过，
                # 只能事后再对账，代价远高于多写几次。
                if (self._pending_chunks() >= FLUSH_CHUNK_THRESHOLD
                        or len(self._pending_deletes) >= 200):
                    self._flush_vectors()
        finally:
            # 无论是异常退出还是 Ctrl-C，都把已经算好的向量落盘，别白算
            self._flush_vectors()

        report.elapsed_s = round(time.time() - t0, 2)
        if report.added or report.updated or reset:
            self.rebuild_bm25()
        report.docs_total = self.registry.count_docs()
        report.chunks_total = self.registry.count_chunks()
        return report

    # ------------------------------------------------------------------
    def _ingest_one(self, path: Path, rel: str, existing: Dict[str, Document],
                    hash_index: Dict[str, str], report: IngestReport,
                    force: bool, by_path: Dict[str, Document] | None = None) -> None:
        path_id = _sha1(rel)[:20]
        stat = path.stat()
        old = (by_path or {}).get(rel) or existing.get(path_id)

        # 快速路径：mtime 完全一致且按路径已命中，直接跳过，不读文件（上千份时能省大量 IO）
        if old and not force and abs(old.mtime - stat.st_mtime) < 1e-6:
            report.skipped += 1
            return

        loaded: LoadedDoc = load_file(path)

        # 文档身份：优先用内容自带的外部 ID（抓取来的 article_id），
        # 没有外部 ID 的自有文档仍以路径哈希为身份。
        # 这样路径只代表"位置"，调整目录、改文件名都不会留下孤儿记录。
        external_id = str(loaded.meta.get("article_id") or "").strip()
        doc_id = external_id or path_id
        old = existing.get(doc_id) or old
        if not loaded.text.strip():
            reason = str(loaded.meta.get("empty_reason") or "无可抽取文本")
            log.warning("读到了但没有可用文本: %s（%s）", rel, reason)
            report.empty_text.append(f"{rel}（{reason}）")
            return

        content_hash = _sha1(loaded.text)
        if old and old.content_hash == content_hash and old.doc_id == doc_id:
            # 内容没变，仅刷新 mtime 与路径，避免下次重复解析
            old.mtime = stat.st_mtime
            if old.path != rel:
                old.path = rel
            self.registry.upsert_doc(old)
            report.skipped += 1
            return

        dup_of = hash_index.get(content_hash)
        if dup_of and dup_of != doc_id and dup_of in existing:
            log.info("内容重复（与 %s 相同），跳过: %s", existing[dup_of].path, rel)
            report.skipped += 1
            return

        doc = Document(
            doc_id=doc_id, path=rel, title=loaded.title,
            source_type=loaded.source_type, content_hash=content_hash,
            mtime=stat.st_mtime, char_count=len(loaded.text), meta=dict(loaded.meta))

        chunks = chunk_document(doc, loaded.text, self.cfg.chunking)
        if not chunks:
            report.empty_text.append(f"{rel}（内容过短，未产生分块）")
            return

        # 部门元数据：知识库用一级子目录区分部门时，自动带上，便于权限过滤。
        # 按「来源/分类」组织目录的库必须关掉（ingest.dept_from_dir=false），
        # 否则会把「来源 / 分类」这类目录名错当成部门。
        dept = ""
        if getattr(self.cfg.ingest, "dept_from_dir", True) and "/" in rel:
            dept = rel.split("/")[0]
        children = []
        for c in chunks:
            if dept:
                c.meta.setdefault("dept", dept)
            if c.is_parent == 0:
                children.append(c)

        # 1) 清理旧数据（更新场景；doc_id 变更时按旧 id 删，避免留孤儿）
        if old:
            # 删除排队到下次 flush，和向量写入放在同一个批次里按「先删后写」执行。
            # 这里绝不能当场 flush：Chroma 单次写入有 6 秒左右的固定开销，
            # 每篇更新都 flush 一次会让「重跑一遍全库」退化成每篇 12 秒，
            # 1300 篇要跑 4 小时。排队后只在真正写向量时才付这笔固定成本。
            self._pending_deletes.add(old.doc_id)
            self.registry.delete_doc(old.doc_id)
            if old.doc_id != doc_id:
                log.info("文档身份变更 %s -> %s（%s）", old.doc_id, doc_id, rel)

        # 2) 写 chunk 元数据
        self.registry.add_chunks(chunks)

        # 3) 向量化子块（先攒着，由 ingest 统一批量写库）
        bs = max(1, self.cfg.embedding.batch_size)
        for i in range(0, len(children), bs):
            batch = children[i:i + bs]
            vectors = self.embedder.encode([embedding_text(c) for c in batch])
            self._pending.append((
                [c.chunk_id for c in batch],
                vectors,
                [{"doc_id": c.doc_id, "title": c.title,
                  "section": c.section, "path": c.path,
                  **({"dept": c.meta["dept"]} if "dept" in c.meta else {})}
                 for c in batch]))

        # 4) 写文档元数据
        self.registry.upsert_doc(doc)
        hash_index[content_hash] = doc_id
        existing[doc_id] = doc
        if by_path is not None:
            by_path[rel] = doc

        report.chunks += len(children)
        # 有进度条时逐篇明细降级为 DEBUG：CLI 的进度行和这条日志说的是同一件事，
        # 两个都往 stderr 打会互相撕行（\r 刷新的进度行被日志从中间截断）。
        # 没有进度条（管道 / CI / API 上传）时保留 INFO，一份一行，日志才可读。
        detail = logging.DEBUG if self._progress_active else logging.INFO
        if old:
            report.updated += 1
            log.log(detail, "更新 %s（%d chunks）", rel, len(children))
        else:
            report.added += 1
            log.log(detail, "新增 %s（%d chunks）", rel, len(children))

    # ------------------------------------------------------------------
    def rebuild_bm25(self) -> int:
        # 用 iter_index_docs 而不是 iter_children：索引文本里额外拼了标题和所属分类，
        # 让「分类名」也能被 BM25 命中（用户口语与文档标题不重合时的兜底信号）。
        n = self.bm25.build(self.registry.iter_index_docs())
        self.bm25.save()
        log.info("BM25 索引重建完成: %d 个 chunk（复用分词 %d，重新分词 %d）",
                 n, self.bm25.last_reused, self.bm25.last_tokenized)
        return n
