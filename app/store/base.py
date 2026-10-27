"""向量库抽象。换库只需实现 add/search/delete_doc/count/reset 五个方法。"""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

Where = Dict[str, object]


class VectorStore:
    name = "base"

    def add(self, ids: Sequence[str], vectors: Sequence[Sequence[float]],
            metadatas: Sequence[dict] | None = None) -> None:
        raise NotImplementedError

    def search(self, vector: Sequence[float], top_k: int = 40,
               where: Where | None = None) -> List[Tuple[str, float]]:
        """返回 [(chunk_id, score)]，score 越大越相关。"""
        raise NotImplementedError

    def delete_doc(self, doc_id: str) -> None:
        raise NotImplementedError

    def count(self) -> int:
        raise NotImplementedError

    def reset(self) -> None:
        raise NotImplementedError

    def close(self) -> None:
        pass


def build_store(cfg) -> VectorStore:
    if cfg.vector_store.provider == "chroma":
        from app.store.chroma_store import ChromaStore
        return ChromaStore(cfg.vector_store.path, cfg.vector_store.collection)
    from app.store.memory_store import MemoryStore
    return MemoryStore()
