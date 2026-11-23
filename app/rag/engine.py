"""RAG 引擎：把入库、检索、生成、引用校验、兜底串成一条可观测的链路。

一次 answer() 的完整流程（也是排障时逐段打点的顺序）：
  1. 取历史 → 2. 追问改写 → 3. 向量召回 + BM25 召回 → 4. RRF 融合
  → 5. 重排 → 6. 父子块上下文拼装 → 7. PII 脱敏 → 8. 调模型
  → 9. 引用校验 → 10. 落库（含未答问题）
"""
from __future__ import annotations

import json
import logging
import re
import time
import unicodedata
import uuid
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

from app.config import Config, load_config
from app.embed import build_embedder
from app.ingest.pipeline import IngestPipeline
from app.llm.client import ChatLLM, ExtractiveLLM, LLMError, build_llm
from app.llm.prompts import build_system_prompt, build_user_prompt
from app.models import Answer, IngestReport, RetrievedChunk, Source
from app.rag.glossary import expand_for_sparse
from app.rag.correct import correct_query, llm_correct_query
from app.rag.guard import redact_blocks
from app.rag.parse import parse_answer
from app.rag.rewrite import rewrite_query
from app.rag.verdict import (REFUSAL_MARK, Verdict, judge,
                             looks_like_refusal_head)
from app.retrieval.bm25 import BM25Index
from app.retrieval.fusion import rrf_fuse
from app.retrieval.rerank import build_reranker
from app.store.base import build_store
from app.store.registry import Registry

log = logging.getLogger(__name__)

_CITE_RE = re.compile(r"\[(\d{1,2})\]")

# 首尾标点的剥离集合（给 str.strip 用，所以是字符串不是 set）。
# 含中英文标点、引号括号书名号、常见全角符号与空白。
_EDGE_PUNCT = (
    "，。、；：？！,.;:?!…·~～—-–*/\\|_+=#@$%^&<>()[]{}（）【】《》〈〉「」『』"
    "“”‘’\"'`　 \t\r\n"
)


def normalize_query(query: str) -> str:
    """检索前的 query 归一：只动标点与空白，不删词、不改词序、不做同义替换。

    为什么必须做：embedding 对句末标点敏感，BM25 反而不受影响
    （`bm25._tokenize` 本来就会把标点过滤掉）。同一个问题
    「报销多久能到账」带一个全角问号，向量就把《差旅与费用报销制度》的
    §3.3 发票要求排到前面、把真正写着「次月 15 日随工资打款」的 §3.2 挤掉，
    RRF 融合后候选里就没有答案了。问号不携带任何语义，不该改变检索结果。

    做三步：
      1. NFKC 归一，全角标点/全角字母数字转半角（避免漏剥、也让「ＶＰＮ」能命中）；
      2. 反复剥两端的标点（`str.strip(chars)` 本身就是两端反复剥）；
      3. 连续空白折叠成一个空格。

    剥完为空时**回退原 query**：用户只发了一个「？」的场合，空串送去 embedding
    会得到一个与任何文档都无关的向量，比带着问号检索更糟。
    """
    raw = query or ""
    if not raw.strip():
        return raw
    s = unicodedata.normalize("NFKC", raw).strip(_EDGE_PUNCT)
    s = re.sub(r"\s+", " ", s).strip()
    return s or raw

# 流式时先攒够这么多个字符再开始外推，用于在第一时间判断是否为拒答。
# 代价是最多几十毫秒的首字延迟，换来的是「拒答不会被当成正常回答推出半句再断掉」。
_STREAM_GUARD_CHARS = len(REFUSAL_MARK) + 4


class ConversationAccessError(PermissionError):
    """尝试读写不属于当前账号的会话。接口层据此返回 404，不泄露会话存在性/归属。"""


class RagEngine:
    def __init__(self, cfg: Optional[Config] = None, *, llm: ChatLLM | None = None,
                 embedder=None, store=None, registry: Registry | None = None,
                 bm25: BM25Index | None = None):
        self.cfg = cfg or load_config()
        self.cfg.ensure_dirs()
        self.registry = registry or Registry(self.cfg.registry_path)
        self.store = store or build_store(self.cfg)
        self.embedder = embedder or build_embedder(self.cfg)
        self.llm = llm or build_llm(self.cfg)
        self.bm25 = bm25 or BM25Index(self.cfg.bm25_path)
        self.reranker = build_reranker(self.cfg, self.llm)
        self.pipeline = IngestPipeline(self.cfg, self.registry, self.store,
                                       self.embedder, self.bm25)
        # 稀疏索引缺失/格式过期时自动重建。否则 BM25 会静默返回空，
        # 双路召回退化成单路，且日志里看不出异常。
        if len(self.bm25) == 0 and not self.bm25.load():
            if self.registry.count_chunks():
                log.warning("BM25 索引缺失或格式过期，启动时自动重建")
                self.pipeline.rebuild_bm25()
        self.company = self.cfg.app.name
        # 留存最近一次入库报告：内存一份 + 磁盘一份，重启后仍可查询。
        self._last_report: Optional[dict] = None
        self._last_report_at: Optional[float] = None
        self._report_path = self.cfg.data_path / "last_ingest_report.json"
        self._load_last_report()

    # ================= 入库 =================
    def ingest(self, target: str | Path, reset: bool = False, force: bool = False,
               progress=None) -> IngestReport:
        report = self.pipeline.ingest(target, reset=reset, force=force, progress=progress)
        if report.added or report.updated or reset:
            # 入库后 BM25 已重建，这里只需保证向量库句柄未被 reset 掉
            pass
        self._save_last_report(report)
        return report

    def _load_last_report(self) -> None:
        try:
            if self._report_path.exists():
                wrapper = json.loads(self._report_path.read_text(encoding="utf-8"))
                if isinstance(wrapper, dict):
                    self._last_report = wrapper.get("report")
                    self._last_report_at = wrapper.get("last_run_at")
        except Exception:  # noqa: BLE001
            log.warning("上次入库报告读取失败，忽略", exc_info=True)

    def _save_last_report(self, report: IngestReport) -> None:
        data = report.to_dict()
        self._last_report = data
        self._last_report_at = time.time()
        try:
            self._report_path.write_text(
                json.dumps({"report": data, "last_run_at": self._last_report_at},
                           ensure_ascii=False), encoding="utf-8")
        except Exception:  # noqa: BLE001
            log.warning("上次入库报告写入失败", exc_info=True)

    def last_ingest_report(self) -> Optional[dict]:
        """最近一次入库报告（从未入库时返回 None）。"""
        return self._last_report

    def last_ingest_at(self) -> Optional[float]:
        """最近一次入库完成的 Unix 时间戳（秒，从未入库时为 None）。"""
        return self._last_report_at

    def kb_root(self) -> Optional[str]:
        """知识库根目录（供管理后台「全部重新入库」），未配置时返回 None。"""
        root = getattr(self.cfg.ingest, "root", "") or ""
        return root or None

    # ================= 知识库维护 =================
    def list_docs(self, q: Optional[str] = None,
                  source_type: Optional[str] = None,
                  category_id: Optional[int] = None,
                  include_sub: bool = True,
                  limit: int = 0, offset: int = 0,
                  unclassified: bool = False) -> List[dict]:
        """知识库文档清单（逐条元数据）。

        除标题/路径关键字与类型筛选外，新增分类筛选与分页：四千多篇文档
        一次性渲染会卡死浏览器，后台必须能翻页。
        """
        return self.registry.list_docs(q=q, source_type=source_type,
                                       category_id=category_id,
                                       include_sub=include_sub,
                                       limit=limit, offset=offset,
                                       unclassified=unclassified)

    def count_docs(self, q: Optional[str] = None,
                   source_type: Optional[str] = None,
                   category_id: Optional[int] = None,
                   include_sub: bool = True,
                   unclassified: bool = False) -> int:
        """与 list_docs 同条件的总数（分页用）。"""
        return self.registry.count_docs_filtered(
            q=q, source_type=source_type, category_id=category_id,
            include_sub=include_sub, unclassified=unclassified)

    # ---------------- 分类 ----------------
    def list_categories(self) -> List[dict]:
        return self.registry.list_categories()

    def create_category(self, name: str, parent_id: int = 0) -> int:
        return self.registry.create_category(name, parent_id)

    def rename_category(self, cat_id: int, name: str) -> None:
        self.registry.rename_category(cat_id, name)

    def move_category(self, cat_id: int, parent_id: int) -> None:
        self.registry.move_category(cat_id, parent_id)

    def delete_category(self, cat_id: int, force: bool = False) -> dict:
        return self.registry.delete_category(cat_id, force=force)

    def set_doc_category(self, doc_id: str, category_id: int) -> bool:
        return self.registry.set_doc_category(doc_id, category_id)

    def set_docs_category(self, doc_ids: List[str], category_id: int) -> int:
        return self.registry.set_docs_category(doc_ids, category_id)

    def migrate_categories_from_meta(self) -> dict:
        return self.registry.migrate_categories_from_meta()

    def delete_docs(self, doc_ids: List[str]) -> dict:
        """批量删除文档。逐篇复用 delete_doc（各自维护向量库/注册表/BM25 一致性），
        单篇失败不影响其余，错误收集到 errors 里返回。

        BM25 不支持增量删除，每删一篇就要全量重建一次；批量删 N 篇会重建 N 次。
        这里先全部删完再统一重建一次，把 O(N) 次重建降到 1 次。
        """
        deleted = 0
        errors: List[str] = []
        for doc_id in doc_ids:
            try:
                doc = (self.registry.get_doc(doc_id)
                       or self.registry.get_doc_by_path(doc_id))
                if doc is None:
                    errors.append(f"{doc_id}: 文档不存在")
                    continue
                self.store.delete_doc(doc.doc_id)
                self.registry.delete_doc(doc.doc_id)
                deleted += 1
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{doc_id}: {type(exc).__name__}: {exc}")
                log.exception("批量删除失败 %s", doc_id)
        if deleted:
            self.pipeline.rebuild_bm25()
        return {"deleted": deleted, "errors": errors}

    def chunks_of_doc(self, doc_id: str) -> List[dict]:
        """某篇文档的全部分块（供管理后台「查看切片」只读展示）。

        doc_id 兼容 path：先按 doc_id 查，查不到再按 path 解析。
        文档不存在时抛 KeyError，由接口层转 404。
        """
        doc = self.registry.get_doc(doc_id) or self.registry.get_doc_by_path(doc_id)
        if doc is None:
            raise KeyError(f"文档不存在：{doc_id}")
        return self.registry.list_chunks_of_doc(doc.doc_id)

    def delete_doc(self, doc_id: str) -> dict:
        """按 doc_id（兼容 path）删除单篇文档及其全部 chunk。

        三处同步清理：向量库、注册表（docs+chunks）、BM25 稀疏索引。
        BM25 不支持按 doc 增量删除，删除后全量重建一次；重建本身带增量分词
        缓存，未受影响的 chunk 不会重新分词。
        """
        doc = self.registry.get_doc(doc_id) or self.registry.get_doc_by_path(doc_id)
        if doc is None:
            raise KeyError(f"文档不存在：{doc_id}")
        chunk_count = self.registry.count_chunks_of_doc(doc.doc_id)
        self.store.delete_doc(doc.doc_id)
        self.registry.delete_doc(doc.doc_id)
        self.pipeline.rebuild_bm25()
        log.info("删除文档 %s（%s，%d chunks）", doc.doc_id, doc.path, chunk_count)
        return {
            "doc_id": doc.doc_id,
            "path": doc.path,
            "title": doc.title,
            "source_type": doc.source_type,
            "chunks_deleted": chunk_count,
        }

    # ================= 检索 =================
    def retrieve(self, query: str, where: Optional[dict] = None,
                 top_k: Optional[int] = None) -> Tuple[List[RetrievedChunk], float]:
        r = self.cfg.retrieval
        top_k = top_k or r.final_top_k

        # 归一放在这里，是唯一能保证「向量路和稀疏路吃到同一份字符串」的位置：
        # /chat、/search、cli 全都经由 retrieve() 进入检索。只要有一条路拿到
        # 带标点的原文，问号就还能把 §3.2 顶出候选。
        query = normalize_query(query)

        vec = self.embedder.encode_one(query)
        dense = self.store.search(vec, r.dense_top_k, where)

        # 稀疏路没有元数据，只能先检索再过滤。若仍按 sparse_top_k 取候选，
        # 过滤后可能只剩几条（命中的全是其他部门）。带 where 时把候选池放大再截断。
        sparse_k = r.sparse_top_k
        if where:
            sparse_k = min(max(r.sparse_top_k * 20, 200), 2000, max(1, len(self.bm25)))
        # 稀疏路补同义词：用户说「休假怎么申请」，文档写「请假申请/年休假」，
        # 字面零重合时 BM25 会整条落空。这里只扩 BM25，向量路保持语义检索，
        # 两条路互补后再 RRF 融合。
        sparse_query = expand_for_sparse(query) if getattr(r, "expand_glossary", False) else query
        sparse = self.bm25.search(sparse_query, sparse_k)

        ids = {cid for cid, _ in dense} | {cid for cid, _ in sparse}
        rows = self.registry.get_chunks(list(ids))

        # BM25 没有元数据，这里按 where 过滤一次
        if where:
            sparse = [(cid, s) for cid, s in sparse
                      if cid in rows and _meta_match(rows[cid].meta, where)][:r.sparse_top_k]

        fused = rrf_fuse(dense, sparse, r.rrf_k, r.dense_weight,
                         r.sparse_weight, r.fusion_k)
        hits: List[RetrievedChunk] = []
        for f in fused:
            ch = rows.get(f.chunk_id)
            if ch is None:
                continue
            hits.append(RetrievedChunk(
                chunk=ch, score=f.score, dense_score=f.dense_score,
                sparse_score=f.sparse_score, rank_dense=f.rank_dense,
                rank_sparse=f.rank_sparse))

        best_dense = max((h.dense_score for h in hits), default=0.0)
        if not hits:
            return [], best_dense

        final = self.reranker.rerank(query, _cap_per_doc(hits, r.max_per_doc), top_k)
        return final, best_dense

    def _retrieve_with_correction(self, query: str, where: Optional[dict] = None,
                                  top_k: Optional[int] = None):
        """检索 + 错别字纠错重检。

        先做零成本的确定性纠错，再检索；若分数可疑（低于 correct_trigger_score），
        用 LLM 纠错后重检一次，显著变好才采纳，否则退回原结果。
        返回 (hits, best_dense, final_query)，final_query 是实际参与检索的查询。
        """
        q = correct_query(query)
        hits, best = self.retrieve(q, where, top_k)
        trigger = self.cfg.retrieval.correct_trigger_score
        if trigger > 0 and best < trigger and getattr(self.llm, "name", "") != "extractive":
            corrected = llm_correct_query(q, self.llm)
            if corrected and corrected != q:
                hits2, best2 = self.retrieve(corrected, where, top_k)
                if best2 > best + 0.05:
                    log.info("纠错重检提升: %.4f -> %.4f（%s -> %s）",
                             best, best2, q, corrected)
                    return hits2, best2, corrected
        return hits, best, q

    # ================= 生成 =================
    def answer(self, query: str, conversation_id: str | None = None,
               where: Optional[dict] = None,
               history: Optional[List[Dict[str, str]]] = None,
               username: str = "") -> Answer:
        t0 = time.perf_counter()
        cid = self._resolve_conversation_id(conversation_id, username)
        hist = history if history is not None else self._history(cid, username)

        rewritten = query
        if self.cfg.conversation.rewrite and hist:
            rewritten = rewrite_query(query, hist, self.llm)

        hits, best_dense, rewritten = self._retrieve_with_correction(rewritten, where)

        threshold = self.cfg.retrieval.min_top_score
        if not hits or (threshold > 0 and best_dense < threshold):
            text = self.cfg.guard.no_answer_text
            self.registry.log_unanswered(cid, query, rewritten, round(best_dense, 4))
            self.registry.append_turn(cid, "user", query, username=username)
            self.registry.append_turn(cid, "assistant", text, {"answered": False},
                                      username=username)
            conclusion, rules, notes = parse_answer(text)
            return Answer(text=text, sources=[], rewritten_query=rewritten,
                          answered=False, conversation_id=cid,
                          latency_ms=int((time.perf_counter() - t0) * 1000),
                          flags={"no_hit": True, "best_dense": round(best_dense, 4)},
                          conclusion=conclusion, rules=rules, notes=notes)

        blocks, sources = self._build_context(hits)
        redacted_n = 0
        if self.cfg.guard.redact_pii:
            blocks, redacted_n = redact_blocks(blocks)

        messages = self._messages(hist, rewritten, blocks)
        usage: Dict[str, int] = {}
        try:
            res = self.llm.chat(messages, context_blocks=blocks, query=rewritten)
            text = (res.text or "").strip()
            if not text:
                # 推理模型（如 deepseek-v4-flash）把 max_tokens 全花在思考上时 content 为空。
                # 静默返回空答案比报错更糟，这里按失败处理，走摘录兜底。
                raise LLMError("模型返回空内容（可能 max_tokens 被思考耗尽）")
            usage = {k: int(v) for k, v in (res.usage or {}).items()
                     if isinstance(v, (int, float))}
        except LLMError as exc:
            log.error("生成失败，走摘录兜底: %s", exc)
            res = ExtractiveLLM().chat(messages, context_blocks=blocks, query=rewritten)
            text = res.text
            usage = {}

        verdict = judge(text, [s.index for s in sources],
                        require_citation=self.cfg.guard.require_citation)
        flags_detail = ""
        if verdict.reason == "citation_missing":
            text += "\n\n（提示：本条回答未标注出处，请以正式制度文件为准。）"
        elif verdict.refused:
            # 拒答时优先用模型说明的「缺什么、该找谁」——这比一句通用答复有用。
            # 但空着的时候必须回落到配置里的兜底答复，不能把空字符串当答案发出去。
            detail = (verdict.text or "").strip()
            text = detail[:200] or self.cfg.guard.no_answer_text
            flags_detail = detail[:200]

        flags = self._record(verdict, cid, query, rewritten, best_dense, text,
                             username)
        flags.update({"redacted": redacted_n, "model": getattr(res, "model", "")})
        if verdict.refused and flags_detail:
            flags["refusal_detail"] = flags_detail

        conclusion, rules, notes = parse_answer(text)
        return Answer(text=text, sources=sources, rewritten_query=rewritten,
                      answered=verdict.answered, conversation_id=cid,
                      latency_ms=int((time.perf_counter() - t0) * 1000),
                      usage=usage, flags=flags,
                      conclusion=conclusion, rules=rules, notes=notes)

    def answer_stream(self, query: str, conversation_id: str | None = None,
                      where: Optional[dict] = None,
                      username: str = "") -> Iterator[dict]:
        """SSE 事件流：meta → sources → delta* → done。

        与 answer() 用完全相同的闸门（阈值、拒答判定、引用校验、未答落库）。
        区别只在于文本是边生成边推的，所以拒答必须在推之前就判定出来——
        做法是先攒够 _STREAM_GUARD_CHARS 个字符，足够识别 NO_ANSWER 标记。
        """
        t0 = time.perf_counter()
        cid = self._resolve_conversation_id(conversation_id, username)
        hist = self._history(cid, username)

        rewritten = query
        if self.cfg.conversation.rewrite and hist:
            rewritten = rewrite_query(query, hist, self.llm)

        hits, best_dense, rewritten = self._retrieve_with_correction(rewritten, where)
        yield {"type": "meta", "conversation_id": cid, "rewritten_query": rewritten}

        threshold = self.cfg.retrieval.min_top_score
        if not hits or (threshold > 0 and best_dense < threshold):
            text = self.cfg.guard.no_answer_text
            verdict = Verdict(answered=False, reason="no_hit", text=text)
            flags = self._record(verdict, cid, query, rewritten, best_dense, text,
                                 username)
            conclusion, rules, notes = parse_answer(text)
            yield {"type": "sources", "sources": []}
            yield {"type": "delta", "text": text}
            yield {"type": "done", "text": text, "flags": flags,
                   "conclusion": conclusion, "rules": rules, "notes": notes,
                   "latency_ms": int((time.perf_counter() - t0) * 1000)}
            return

        blocks, sources = self._build_context(hits)
        if self.cfg.guard.redact_pii:
            blocks, _ = redact_blocks(blocks)
        yield {"type": "sources", "sources": [s.to_dict() for s in sources]}

        messages = self._messages(hist, rewritten, blocks)
        buf: List[str] = []
        emitted = False       # 是否已开始外推
        refused = False       # 是否已判定为拒答（一旦判定，一个字都不推）

        try:
            for piece in self.llm.stream(messages, context_blocks=blocks, query=rewritten):
                buf.append(piece)
                if refused:
                    continue
                joined = "".join(buf)
                if not emitted:
                    if len(joined) < _STREAM_GUARD_CHARS:
                        continue          # 攒够再判，避免把 NO_ANSWER 推出去
                    if looks_like_refusal_head(joined):
                        refused = True
                        continue
                    emitted = True
                    yield {"type": "delta", "text": joined}
                    continue
                yield {"type": "delta", "text": piece}
        except LLMError as exc:
            log.error("流式生成失败，改用摘录兜底: %s", exc)
            fallback = ExtractiveLLM().chat(messages, context_blocks=blocks,
                                            query=rewritten).text
            buf = [fallback]
            yield {"type": "delta", "text": fallback}
            emitted = True

        text = "".join(buf).strip()
        if not text:
            log.error("流式生成返回空内容，改用摘录兜底")
            fallback = ExtractiveLLM().chat(messages, context_blocks=blocks,
                                            query=rewritten).text
            buf = [fallback]
            text = fallback
            yield {"type": "delta", "text": fallback}
            emitted = True

        verdict = judge(text, [s.index for s in sources],
                        require_citation=self.cfg.guard.require_citation)
        final = text

        if verdict.refused:
            if not emitted:
                # 拒答：此前一个字都没推，现在补推规范答复
                final = ((verdict.text or "").strip()[:200]
                         or self.cfg.guard.no_answer_text)
                yield {"type": "delta", "text": final}
        else:
            if not emitted:
                # 回答总长还没到判定阈值就结束了：现在一次性推出去。
                # 漏了这一步，短回答会一个字都不显示（只有松 sources 和 done）。
                yield {"type": "delta", "text": text}
            if verdict.reason == "citation_missing":
                hint = "\n\n（提示：本条回答未标注出处，请以正式制度文件为准。）"
                final = text + hint
                yield {"type": "delta", "text": hint}

        flags = self._record(verdict, cid, query, rewritten, best_dense, final,
                             username)
        conclusion, rules, notes = parse_answer(final)
        yield {"type": "done", "text": final, "flags": flags,
               "citations": verdict.citations,
               "conclusion": conclusion, "rules": rules, "notes": notes,
               "latency_ms": int((time.perf_counter() - t0) * 1000)}

    # ================= 内部 =================
    def _resolve_conversation_id(self, conversation_id: str | None,
                                 username: str) -> str:
        """解析会话 ID 并做写路径归属校验（读历史 + 落库共用的守门）。

        - 未传 conversation_id：新建会话（全 128bit uuid4().hex）。
        - 传了但该 cid 没有任何 turn：允许，后续 append_turn 会盖上当前 username。
        - 传了且该 cid 已有 turn，但归属他人（含迁移产生的空串孤儿）：拒绝。

        username 为空（CLI / 脚本 / 单测等未鉴权调用）时跳过归属校验，
        保持旧行为的向后兼容；线上 HTTP 路径永远带真实 username，不会走到这里。
        """
        cid = conversation_id or uuid.uuid4().hex
        if conversation_id and username:
            owner = self.registry.conversation_owner(conversation_id)
            if owner is not None and owner != username:
                raise ConversationAccessError("会话不存在")
        return cid

    def check_conversation_access(self, conversation_id: str | None,
                                  username: str) -> None:
        """接口层预检：流式响应要在一开始就拒绝非归属会话（返回 404），
        而不是先 200 再推一个 error 事件。sync 路径无需调用（answer() 内部已校验）。
        """
        if conversation_id:
            self._resolve_conversation_id(conversation_id, username)

    def _record(self, verdict: Verdict, cid: str, query: str, rewritten: str,
                best_dense: float, final_text: str, username: str = "") -> Dict[str, Any]:
        """落库 + 组装 flags。同步与流式共用，保证两边记录的是同一件事。"""
        flags: Dict[str, Any] = {
            "citations": verdict.citations,
            "citation_missing": not verdict.citations,
            "best_dense": round(best_dense, 4),
            "verdict": verdict.reason or "ok",
            "refused": verdict.refused,
        }
        if verdict.should_log_unanswered:
            self.registry.log_unanswered(cid, query, rewritten, round(best_dense, 4),
                                         reason=verdict.reason or "unknown",
                                         detail=final_text[:300])
        self.registry.append_turn(cid, "user", query, username=username)
        self.registry.append_turn(cid, "assistant", final_text,
                                  {"answered": verdict.answered,
                                   "citations": verdict.citations,
                                   "verdict": verdict.reason or "ok"},
                                  username=username)
        return flags

    def _history(self, conversation_id: str, username: str = "") -> List[Dict[str, str]]:
        turns = self.registry.get_turns(
            conversation_id, self.cfg.conversation.max_turns * 2,
            username=username or None)
        return [{"role": t.role, "content": t.content} for t in turns]

    def _messages(self, hist: List[Dict[str, str]], query: str,
                  blocks: List[dict]) -> List[Dict[str, str]]:
        msgs: List[Dict[str, str]] = [
            {"role": "system", "content": build_system_prompt(self.company)}]
        for t in hist[-self.cfg.conversation.history_turns_for_llm * 2:]:
            content = t["content"][:500]
            if t["role"] == "assistant":
                # 历史回答里的 [1] 指向上一轮的资料，而本轮资料又从 [1] 重新编号。
                # 不清掉的话，模型照抄旧编号，引用校验还会因为编号集合对得上而误判通过——
                # 结果是一条引用完整、却指向无关文档的答案。
                content = _CITE_RE.sub("", content)
            msgs.append({"role": t["role"], "content": content})
        msgs.append({"role": "user",
                     "content": build_user_prompt(query, blocks)})
        return msgs

    def _build_context(self, hits: List[RetrievedChunk]) -> Tuple[List[dict], List[Source]]:
        """子块定位 + 父块补全上下文，并去重（同一父块只出现一次）。

        预算按**去重后的实际块数**分配，不按命中数。
        原实现用 len(hits)（如 6）去除预算，但 6 条命中常来自 3 个父块，
        去重后每块只拿到 1000 字，而父块上限 2200 字——等于把父块拦腰截断，
        父块上下文白补了。改为先收集去重结果，再按真实块数分配。
        """
        r = self.cfg.retrieval
        parent_ids = [h.chunk.parent_id for h in hits
                      if r.use_parent_context and h.chunk.parent_id]
        parents = self.registry.get_chunks(list(set(parent_ids)))

        # 步骤一：先算出「哪些块要进上下文」，同一父块只保留排名最靠前的那条命中
        picked: List[Tuple[RetrievedChunk, str]] = []
        seen: set[str] = set()
        for h in hits:
            key = h.chunk.parent_id or h.chunk.chunk_id
            if key in seen:
                continue
            seen.add(key)
            text = h.chunk.text
            if r.use_parent_context and h.chunk.parent_id in parents:
                text = parents[h.chunk.parent_id].text
            picked.append((h, text))

        # 步骤二：预算按去重后的块数分配；总需求不超预算时完全不截断
        n = max(1, len(picked))
        per_block = max(400, r.max_context_chars // n)
        need = sum(len(t) for _, t in picked)
        if need <= r.max_context_chars:
            per_block = max(len(t) for _, t in picked) if picked else per_block

        blocks: List[dict] = []
        sources: List[Source] = []
        idx = 0
        for h, text in picked:
            if len(text) > per_block:
                cut = text[:per_block].rsplit("。", 1)[0]
                text = (cut + "……") if len(cut) > per_block * 0.5 else text[:per_block] + "……"

            idx += 1
            blocks.append({
                "index": idx,
                "title": h.chunk.title,
                "section": h.chunk.section,
                "path": h.chunk.path,
                "page": h.chunk.meta.get("page"),
                "updated": h.chunk.meta.get("published_at"),
                "content_type": h.chunk.meta.get("content_type"),
                "source_url": h.chunk.meta.get("source_url"),
                "text": text,
            })
            sources.append(Source(
                index=idx, chunk_id=h.chunk.chunk_id, doc_id=h.chunk.doc_id,
                title=h.chunk.title, section=h.chunk.section, path=h.chunk.path,
                score=round(h.rerank_score if h.rerank_score is not None else h.score, 6),
                dense_score=round(h.dense_score, 6),
                snippet=h.chunk.text[:180].replace("\n", " ") + "…",
                url=str(h.chunk.meta.get("source_url") or ""),
                content_type=str(h.chunk.meta.get("content_type") or "")))
        return blocks, sources

    # ================= 运维 =================
    def stats(self) -> dict:
        s = self.registry.stats()
        s.update({"vectors": self.store.count(), "bm25": len(self.bm25),
                  "llm": self.llm.name, "embedder": self.embedder.name,
                  "vector_store": self.store.name, "reranker": self.reranker.name})
        return s

    def close(self) -> None:
        self.llm.close()
        self.store.close()
        self.registry.close()
        if hasattr(self.embedder, "close"):
            self.embedder.close()


def _meta_match(meta: dict, where: dict) -> bool:
    for k, want in where.items():
        got = meta.get(k)
        if isinstance(want, (list, tuple, set)):
            if got not in want:
                return False
        elif got != want:
            return False
    return True


def _cap_per_doc(hits: List[RetrievedChunk], cap: int) -> List[RetrievedChunk]:
    """同一篇文档最多留 cap 条候选，其余顺次后移。

    一篇长文档切成几十个碎片时，它的碎片往往集体会排在一起（向量相似度接近），
    把 top_k 全占掉。上下文组装虽然按父块去过重，但那时已经截断过候选了，
    去重后只剩两三篇文档——答案就会漏掉其他同样相关的资料。
    这里在截断前按文档限流，保证 top_k 覆盖到更多文档；保持原有融合顺序。
    """
    if cap <= 0:
        return hits
    seen: dict[str, int] = {}
    kept: List[RetrievedChunk] = []
    overflow: List[RetrievedChunk] = []
    for h in hits:
        key = h.chunk.doc_id or h.chunk.chunk_id
        n = seen.get(key, 0)
        if n < cap:
            seen[key] = n + 1
            kept.append(h)
        else:
            overflow.append(h)
    return kept + overflow
