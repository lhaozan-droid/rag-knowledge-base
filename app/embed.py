"""Embedding 层：可插拔 + 向量缓存 + 离线兜底。

混合模式要点：这里所有实现都是**本地执行**的（ollama / sentence-transformers），
只有 openai_compat 会把文本发出去，默认不用。
"""
from __future__ import annotations

import hashlib
import logging
import re
import sqlite3
import struct
import threading
from pathlib import Path
from typing import List, Sequence

import numpy as np

log = logging.getLogger(__name__)

_WORD = re.compile(r"[a-zA-Z0-9_]+")
_CJK_RUN = re.compile(r"[\u4e00-\u9fff]+")


class Embedder:
    name: str = "base"
    dim: int = 0

    @property
    def cache_key(self) -> str:
        """向量缓存的命名空间。

        必须带模型标识：缓存表是按文本 sha1 索引的，如果两种模型共用同一个键，
        换模型后老缓存会被直接命中，于是拿到上一代模型的向量（维度甚至都对不上），
        写入时不报错，直到查询才抛「expecting embedding with dimension of N」。
        """
        model = getattr(self, "model_name", None) or getattr(self, "model", None)
        return f"{self.name}:{model}:{self.dim}" if model else f"{self.name}:{self.dim}"

    def encode(self, texts: Sequence[str]) -> List[List[float]]:
        raise NotImplementedError

    def encode_one(self, text: str) -> List[float]:
        return self.encode([text])[0]

    def describe(self) -> str:
        """一句人能读懂的身份：哪个 provider、哪个模型、多少维。

        入库报告要打这一行：成功时 build_embedder 是静默的，只有降级才 warning。
        不主动报出来，首跑用户就无法从报告判断向量是真模型算的还是 hash 兜底算的——
        而 hash 兜底时召回基本不可用，报告上必须能一眼看出来。
        """
        impl = getattr(self, "inner", self)     # CachedEmbedder 透传到真实实现
        name = getattr(impl, "name", "?")
        model = getattr(impl, "model_name", None) or getattr(impl, "model", None)
        model = model if isinstance(model, str) else ""
        dim = getattr(impl, "dim", 0) or 0
        if name == "hash":
            return f"hash（{dim} 维，离线哈希，非语义模型）"
        return f"{model or name}（{name} / {dim} 维）"


def _ngram_tokens(text: str):
    """离线哈希向量用的 token 流：英文按词，中文按字+二元组。"""
    t = text.lower()
    out = []
    for w in _WORD.findall(t):
        out.append((w, 1.0))
    for run in _CJK_RUN.findall(t):
        for ch in run:
            out.append((ch, 0.5))
        for i in range(len(run) - 1):
            out.append((run[i:i + 2], 1.0))
    return out


class HashEmbedder(Embedder):
    """无模型依赖的确定性向量。仅用于离线开发、单测和线上降级，效果远弱于 bge。"""
    name = "hash"

    def __init__(self, dim: int = 1024):
        self.dim = dim

    def encode(self, texts: Sequence[str]) -> List[List[float]]:
        vecs = []
        for t in texts:
            v = np.zeros(self.dim, dtype=np.float32)
            for tok, w in _ngram_tokens(t):
                h = hashlib.md5(tok.encode("utf-8")).digest()
                idx = int.from_bytes(h[:4], "little") % self.dim
                v[idx] += w if h[4] & 1 else -w
            n = float(np.linalg.norm(v))
            vecs.append((v / n if n > 0 else v).tolist())
        return vecs


class OllamaEmbedder(Embedder):
    """调用本机 Ollama。推荐模型：bge-m3（1024 维，中英+长文本）。"""

    name = "ollama"

    def __init__(self, base_url: str = "http://127.0.0.1:11434", model: str = "bge-m3",
                 timeout: float = 120.0):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    def encode(self, texts: Sequence[str]) -> List[List[float]]:
        import httpx

        items = [t if t.strip() else " " for t in texts]
        with httpx.Client(timeout=self.timeout) as client:
            try:
                r = client.post(f"{self.base_url}/api/embed",
                                json={"model": self.model, "input": items})
                if r.status_code == 404:
                    raise _LegacyOllama()
                r.raise_for_status()
                data = r.json()
                embs = data.get("embeddings") or []
            except _LegacyOllama:
                embs = []
                for t in items:
                    r = client.post(f"{self.base_url}/api/embeddings",
                                    json={"model": self.model, "prompt": t})
                    r.raise_for_status()
                    embs.append(r.json()["embedding"])
        if len(embs) != len(items):
            raise RuntimeError(f"Ollama 返回向量数量不匹配: {len(embs)} != {len(items)}")
        if embs:
            self.dim = len(embs[0])
        return embs


class _LegacyOllama(Exception):
    pass


class OpenAICompatEmbedder(Embedder):
    """OpenAI 兼容的 /embeddings（vLLM、Xinference、One-API 等自建网关都兼容）。"""

    name = "openai_compat"

    def __init__(self, base_url: str, api_key: str, model: str, timeout: float = 60.0):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout

    def encode(self, texts: Sequence[str]) -> List[List[float]]:
        import httpx

        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        with httpx.Client(timeout=self.timeout) as client:
            r = client.post(f"{self.base_url}/embeddings", headers=headers,
                            json={"model": self.model, "input": list(texts)})
            r.raise_for_status()
            data = r.json()["data"]
        data = sorted(data, key=lambda d: d.get("index", 0))
        embs = [d["embedding"] for d in data]
        if embs:
            self.dim = len(embs[0])
        return embs


class FastEmbedEmbedder(Embedder):
    """fastembed + onnxruntime：不装 torch、不装 ollama，模型只在本地跑。

    选它的原因：目标机器既没有 ollama 也没有 torch，但从 chromadb 那套依赖里
    已经带了 onnxruntime，复用它只需要再下一个模型文件（几百 MB）。
    推荐 `jinaai/jina-embeddings-v2-base-zh`：768 维、8192 token 上下文，
    700 字的制度段落不会被截断；`bge-small-zh-v1.5` 只有 512 token，会丢后半段。
    """

    name = "fastembed"

    def __init__(self, model: str = "jinaai/jina-embeddings-v2-base-zh",
                 cache_dir: str | None = None, batch_size: int = 16):
        import os

        # 镜像与符号链接必须在 huggingface_hub 被 import 之前设好。
        # Windows 默认没有创建 symlink 的权限，不关掉会下载完却加载不到模型文件。
        os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
        os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"
        from fastembed import TextEmbedding

        self.model_name = model
        self.batch_size = max(1, batch_size)
        self._model = TextEmbedding(model_name=model, cache_dir=cache_dir)
        probe = next(iter(self._model.embed(["初始化"])))
        self.dim = len(probe)

    def encode(self, texts: Sequence[str]) -> List[List[float]]:
        items = [t if t.strip() else " " for t in texts]
        arr = list(self._model.embed(items, batch_size=self.batch_size))
        return np.asarray(arr, dtype=np.float32).tolist()


class SentenceTransformerEmbedder(Embedder):
    name = "sentence_transformers"

    def __init__(self, model: str = "BAAI/bge-small-zh-v1.5", device: str = "cpu",
                 batch_size: int = 32):
        from sentence_transformers import SentenceTransformer  # 可选依赖

        self.model = SentenceTransformer(model, device=device)
        self.batch_size = batch_size
        self.dim = int(self.model.get_sentence_embedding_dimension())

    def encode(self, texts: Sequence[str]) -> List[List[float]]:
        arr = self.model.encode(list(texts), batch_size=self.batch_size,
                                normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(arr, dtype=np.float32).tolist()


class CachedEmbedder(Embedder):
    """按文本 sha1 缓存向量。增量入库时，未改动的段落完全不需要重算。"""

    def __init__(self, inner: Embedder, cache_path: str | Path, batch_size: int = 16):
        self.inner = inner
        self.name = inner.name
        self.dim = inner.dim
        self.batch_size = max(1, batch_size)
        self._lock = threading.Lock()
        self._ns = inner.cache_key
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(cache_path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS vec(hash TEXT PRIMARY KEY, dim INTEGER, vec BLOB)")
        self._conn.commit()

    def _key(self, text: str) -> str:
        return hashlib.sha1(f"{self._ns}\x00{text}".encode("utf-8")).hexdigest()

    def _lookup(self, keys: List[str]) -> dict:
        found = {}
        for i in range(0, len(keys), 400):
            part = keys[i:i + 400]
            q = f"SELECT hash, vec FROM vec WHERE hash IN ({','.join('?' * len(part))})"
            for h, blob in self._conn.execute(q, part):
                found[h] = list(struct.unpack(f"<{len(blob) // 4}f", blob))
        return found

    def _store(self, pairs: List[tuple]) -> None:
        rows = [(h, len(v), struct.pack(f"<{len(v)}f", *v)) for h, v in pairs]
        with self._lock:
            self._conn.executemany(
                "INSERT OR REPLACE INTO vec(hash, dim, vec) VALUES (?,?,?)", rows)
            self._conn.commit()

    def encode(self, texts: Sequence[str]) -> List[List[float]]:
        if not texts:
            return []
        keys = [self._key(t) for t in texts]
        found = self._lookup(keys)
        missing = [(k, t) for k, t in zip(keys, texts) if k not in found]
        if missing:
            uniq = {}
            for k, t in missing:
                uniq.setdefault(k, t)
            ks = list(uniq.keys())
            new = []
            for i in range(0, len(ks), self.batch_size):
                batch_keys = ks[i:i + self.batch_size]
                batch_texts = [uniq[k] for k in batch_keys]
                vecs = self.inner.encode(batch_texts)
                if vecs:
                    self.dim = len(vecs[0])
                new.extend(zip(batch_keys, vecs))
            self._store(new)
            for k, v in new:
                found[k] = v
        return [found[k] for k in keys]

    def close(self) -> None:
        self._conn.close()


def build_embedder(cfg) -> Embedder:
    """按配置构建。任何异常都降级到 HashEmbedder，保证服务不因模型问题整体挂掉。"""
    e = cfg.embedding
    impl: Embedder
    try:
        if e.provider == "ollama":
            impl = OllamaEmbedder(e.ollama.base_url, e.model)
            impl.encode(["健康检查"])
        elif e.provider == "fastembed":
            # 模型文件放项目 data/models 下，避免落在系统 Temp 里被清理工具扫掉
            impl = FastEmbedEmbedder(e.model, cache_dir=str(cfg.data_path / "models"),
                                     batch_size=e.batch_size)
        elif e.provider == "sentence_transformers":
            impl = SentenceTransformerEmbedder(e.model, device=e.device,
                                               batch_size=e.batch_size)
        elif e.provider == "openai_compat":
            impl = OpenAICompatEmbedder(e.openai_compat.base_url,
                                        e.openai_compat.api_key,
                                        e.openai_compat.model or e.model)
        elif e.provider == "hash":
            impl = HashEmbedder(e.dim)
        else:
            raise ValueError(f"未知 embedding.provider: {e.provider}")
    except Exception as exc:  # noqa: BLE001
        log.warning("Embedding provider '%s' 初始化失败（%s），降级为 hash 向量。"
                    "生产环境请修好这一项，否则召回质量不可用。", e.provider, exc)
        impl = HashEmbedder(e.dim)
    if cfg.embedding.cache and not isinstance(impl, CachedEmbedder):
        impl = CachedEmbedder(impl, cfg.embed_cache_path, cfg.embedding.batch_size)
    return impl
