"""BM25 稀疏检索：中文用 jieba 分词，兜底用字符二元组。

为什么混合模式一定要有稀疏检索：
向量检索对「专有名词、编号、缩写」不敏感（比如「员工手册 3.1 条」「报销时限」「VPN 申请」），
而知识库问答里这类精确匹配恰恰是高频刚需。Dense + Sparse 融合是性价比最高的一步优化。
"""
from __future__ import annotations

import hashlib
import logging
import pickle
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

log = logging.getLogger(__name__)

STOPWORDS = set("""
的 了 和 是 在 我 有 就 不 人 都 一 一个 上 也 很 到 说 要 去 你 会 着 没有 看 好 自己 这
那 什么 怎么 如何 为什么 可以 需要 请问 一下 那些 这个 那个 我们 你们 他们 吗 呢 吧 啊
the a an of to in for on and or is are be as at by it this that with from
""".split())

_JIEBA = None


def _tokenize(text: str) -> List[str]:
    global _JIEBA
    text = text.lower()
    try:
        if _JIEBA is None:
            import jieba
            jieba.setLogLevel(logging.WARNING)
            _JIEBA = jieba
        toks = [t.strip() for t in _JIEBA.lcut(text)]
    except Exception:  # noqa: BLE001  jieba 不可用时退化为字符二元组
        toks = [text[i:i + 2] for i in range(max(0, len(text) - 1))]
    out: List[str] = []
    for t in toks:
        if not t or t in STOPWORDS:
            continue
        if len(t) > 1 or "\u4e00" <= t <= "\u9fff":
            out.append(t)
    return out


def _text_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


class BM25Index:
    """全量内存索引。10 万 chunk 量级内存占用约 200~400MB，单机可承受。

    增量分词：按 chunk_id + 文本哈希复用上次的分词结果。
    上千份文档时，改一份文档只会给那一份重新分词，其余直接命中缓存——
    真正贵的是 jieba 分词，BM25Okapi 构造（统计词频）本身很快。
    """

    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path else None
        self._ids: List[str] = []
        self._tokens: List[List[str]] = []
        self._hashes: List[str] = []
        self._bm25 = None
        self._id_pos: Dict[str, int] = {}
        self.last_reused = 0          # 最近一次 build 复用了多少条分词结果
        self.last_tokenized = 0       # 最近一次 build 真正分了多少条

    def build(self, items: Iterable[Tuple[str, str]]) -> int:
        cached_tok: Dict[str, List[str]] = {}
        cached_hash: Dict[str, str] = {}
        for cid, tok, h in zip(self._ids, self._tokens, self._hashes):
            cached_tok[cid] = tok
            cached_hash[cid] = h

        ids: List[str] = []
        corpus: List[List[str]] = []
        hashes: List[str] = []
        reused = tokenized = 0
        for chunk_id, text in items:
            h = _text_hash(text)
            ids.append(chunk_id)
            hashes.append(h)
            if cached_hash.get(chunk_id) == h:
                corpus.append(cached_tok[chunk_id])
                reused += 1
            else:
                corpus.append(_tokenize(text))
                tokenized += 1

        self._ids = ids
        self._tokens = corpus
        self._hashes = hashes
        self._id_pos = {cid: i for i, cid in enumerate(ids)}
        self.last_reused = reused
        self.last_tokenized = tokenized
        if not corpus:
            self._bm25 = None
            return 0
        from rank_bm25 import BM25Okapi

        self._bm25 = BM25Okapi(corpus, k1=1.5, b=0.75)
        return len(ids)

    def search(self, query: str, top_k: int = 40) -> List[Tuple[str, float]]:
        if self._bm25 is None or not self._ids:
            return []
        qtok = _tokenize(query)
        if not qtok:
            return []
        import numpy as np

        scores = np.asarray(self._bm25.get_scores(qtok), dtype=np.float32)
        idx = np.nonzero(scores > 0)[0]
        if idx.size == 0:
            return []
        if idx.size > top_k:
            part = idx[np.argpartition(-scores[idx], top_k - 1)[:top_k]]
        else:
            part = idx
        part = part[np.argsort(-scores[part])]
        return [(self._ids[int(i)], float(scores[int(i)])) for i in part]

    # ---------- 持久化 ----------
    def save(self) -> None:
        if not self.path:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "wb") as f:
            pickle.dump({"ids": self._ids, "tokens": self._tokens,
                         "hashes": self._hashes}, f, protocol=4)

    def load(self) -> bool:
        if not self.path or not self.path.exists():
            return False
        try:
            with open(self.path, "rb") as f:
                data = pickle.load(f)
            ids = data.get("ids") or []
            tokens = data.get("tokens") or []
            hashes = data.get("hashes") or []
            if len(ids) != len(tokens) or not ids:
                return False
            if len(hashes) != len(ids):
                # 旧格式（没有哈希）：无法判断文本是否变化，宁可重建一次
                log.info("BM25 索引为旧格式，将重建一次以启用增量分词")
                return False
            self._ids = ids
            self._tokens = tokens
            self._hashes = hashes
            self._id_pos = {c: i for i, c in enumerate(ids)}
            from rank_bm25 import BM25Okapi
            self._bm25 = BM25Okapi(tokens, k1=1.5, b=0.75)
            return True
        except Exception as exc:  # noqa: BLE001
            log.warning("BM25 索引加载失败，将重建: %s", exc)
            return False

    def __len__(self) -> int:
        return len(self._ids)

