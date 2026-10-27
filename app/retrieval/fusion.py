"""Dense + Sparse 融合：RRF（Reciprocal Rank Fusion）。

为什么不直接加权求和分数：余弦相似度（0~1）和 BM25（0~30+）量纲完全不同，
归一化后加权也随语料变化而漂移。RRF 只看排名，天然免调参，是工业界最稳的默认选择。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple


@dataclass
class FusedHit:
    chunk_id: str
    score: float = 0.0
    dense_score: float = 0.0
    sparse_score: float = 0.0
    rank_dense: int = -1
    rank_sparse: int = -1


def rrf_fuse(dense: Sequence[Tuple[str, float]], sparse: Sequence[Tuple[str, float]],
             rrf_k: int = 60, dense_weight: float = 1.0,
             sparse_weight: float = 1.0, top_k: int = 30) -> List[FusedHit]:
    hits: Dict[str, FusedHit] = {}

    for rank, (cid, score) in enumerate(dense):
        h = hits.setdefault(cid, FusedHit(chunk_id=cid))
        h.dense_score = float(score)
        h.rank_dense = rank
        h.score += dense_weight / (rrf_k + rank + 1)

    for rank, (cid, score) in enumerate(sparse):
        h = hits.setdefault(cid, FusedHit(chunk_id=cid))
        h.sparse_score = float(score)
        h.rank_sparse = rank
        h.score += sparse_weight / (rrf_k + rank + 1)

    ordered = sorted(hits.values(), key=lambda h: -h.score)
    return ordered[:top_k]
