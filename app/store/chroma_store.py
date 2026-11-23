"""Chroma 持久化向量库。

选它的理由：单机嵌入式、零运维、支持 metadata 过滤，上千份文档（10 万级 chunk）够用。
真要上千万级或需要原生稀疏向量/多租户隔离，换 Qdrant，只需实现 app/store/base.py。
"""
from __future__ import annotations

import logging
from typing import List, Sequence, Tuple

from app.store.base import VectorStore, Where

log = logging.getLogger(__name__)


class ChromaStore(VectorStore):
    name = "chroma"

    def __init__(self, path: str, collection: str = "kb") -> None:
        import chromadb
        from chromadb.config import Settings

        self._path = path
        self._client = chromadb.PersistentClient(
            path=str(path), settings=Settings(anonymized_telemetry=False, allow_reset=True))
        self._name = collection
        # embedding_function 必须显式关掉：不传的话 chroma 会拉它自己的默认模型
        # （all-MiniLM-L6-v2），首次使用要联网下载，离线机器上就是无限等待。
        # 向量本来就是我们自己算好传进来的，根本不需要它的 embedding。
        self._col = self._client.get_or_create_collection(
            name=collection, metadata={"hnsw:space": "cosine"},
            embedding_function=None)
        log.info("Chroma 就绪: %s / %s (已有 %d 条)", path, collection, self._col.count())

    def _where(self, where: Where | None):
        if not where:
            return None
        conds = []
        for k, v in where.items():
            if isinstance(v, (list, tuple, set)):
                conds.append({k: {"$in": list(v)}})
            else:
                conds.append({k: v})
        return conds[0] if len(conds) == 1 else {"$and": conds}

    def add(self, ids: Sequence[str], vectors: Sequence[Sequence[float]],
            metadatas: Sequence[dict] | None = None) -> None:
        metas = []
        for m in (metadatas or [{} for _ in ids]):
            clean = {}
            for k, v in (m or {}).items():
                if isinstance(v, (str, int, float, bool)):
                    clean[k] = v
            metas.append(clean)
        self._col.upsert(ids=list(ids), embeddings=[list(v) for v in vectors],
                         metadatas=metas)

    def search(self, vector, top_k=40, where=None) -> List[Tuple[str, float]]:
        n = self._col.count()
        if n == 0:
            return []
        res = self._col.query(query_embeddings=[list(vector)],
                              n_results=min(top_k, n),
                              where=self._where(where),
                              include=["distances"])
        ids = (res.get("ids") or [[]])[0]
        dists = (res.get("distances") or [[]])[0]
        return [(cid, 1.0 - float(d)) for cid, d in zip(ids, dists)]

    def delete_doc(self, doc_id: str) -> None:
        self._col.delete(where={"doc_id": doc_id})

    def count(self) -> int:
        return self._col.count()

    def reset(self) -> None:
        try:
            self._client.delete_collection(self._name)
        except Exception:  # noqa: BLE001
            pass
        self._col = self._client.get_or_create_collection(
            name=self._name, metadata={"hnsw:space": "cosine"})
        # 删完必须真的是空的。旧版本这里 delete 失败会被 except 吞掉，
        # 然后 get_or_create 又拿回原 collection，于是「重置成功」的日志下面
        # 藏着一个带旧维度（1024）的库，直到查询才报维度不匹配。
        left = self._col.count()
        if left:
            raise RuntimeError(
                f"向量库重置失败：collection '{self._name}' 仍残留 {left} 条记录。"
                f"请停止服务后手动删除向量库目录（{self._path}）再重新入库。")
