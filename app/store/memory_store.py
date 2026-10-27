"""内存向量库（穷举余弦）。用于单测、离线演示，以及 Chroma 不可用时的降级。"""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np

from app.store.base import VectorStore, Where


class MemoryStore(VectorStore):
    name = "memory"

    def __init__(self) -> None:
        self._ids: List[str] = []
        self._vecs: List[np.ndarray] = []
        self._meta: List[dict] = []
        self._index: Dict[str, int] = {}

    def add(self, ids, vectors, metadatas=None) -> None:
        for i, (cid, vec) in enumerate(zip(ids, vectors)):
            v = np.asarray(vec, dtype=np.float32)
            n = float(np.linalg.norm(v))
            v = v / n if n > 0 else v
            meta = dict(metadatas[i]) if metadatas else {}
            if cid in self._index:
                self._vecs[self._index[cid]] = v
                self._meta[self._index[cid]] = meta
                continue
            self._index[cid] = len(self._ids)
            self._ids.append(cid)
            self._vecs.append(v)
            self._meta.append(meta)

    def search(self, vector, top_k=40, where=None) -> List[Tuple[str, float]]:
        if not self._ids:
            return []
        q = np.asarray(vector, dtype=np.float32)
        n = float(np.linalg.norm(q))
        q = q / n if n > 0 else q
        mat = np.vstack(self._vecs)
        scores = mat @ q
        order = np.argsort(-scores)
        out: List[Tuple[str, float]] = []
        for idx in order:
            meta = self._meta[int(idx)]
            if where and not _match(meta, where):
                continue
            out.append((self._ids[int(idx)], float(scores[int(idx)])))
            if len(out) >= top_k:
                break
        return out

    def delete_doc(self, doc_id: str) -> None:
        keep = [i for i, m in enumerate(self._meta) if m.get("doc_id") != doc_id]
        self._ids = [self._ids[i] for i in keep]
        self._vecs = [self._vecs[i] for i in keep]
        self._meta = [self._meta[i] for i in keep]
        self._index = {cid: i for i, cid in enumerate(self._ids)}

    def count(self) -> int:
        return len(self._ids)

    def reset(self) -> None:
        self._ids, self._vecs, self._meta, self._index = [], [], [], {}


def _match(meta: dict, where: Where) -> bool:
    for k, want in where.items():
        got = meta.get(k)
        if isinstance(want, (list, tuple, set)):
            if got not in want:
                return False
        elif got != want:
            return False
    return True
