"""对比不同 Embedding 方案的语义召回能力。

为什么要单独写这个：项目自带的 eval_set 用的是制度原词（"年假""住宿标准"），
BM25 单路就能全中，Recall@6 = 100% 看不出向量模型好坏。
这个脚本专挑「口语问法」——员工真实会这么问，但字面和制度原文对不上，
才能看出「按意思找」这条路到底有没有用。

用法：
    python -m eval.compare_embedder
"""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import load_config  # noqa: E402
from app.embed import FastEmbedEmbedder, HashEmbedder  # noqa: E402

# 口语问法：刻意避开制度原词，模拟员工真实提问习惯。
# 允许一条问题有多个可接受答案——制度里不同章节都可能承载同一主题，
# 只认一个标准答案会让指标失真。
QUERIES: List[Tuple[str, Tuple[str, ...]]] = [
    ("出去办事住哪能报销多少", ("住宿标准",)),
    ("领证结婚能休几天假", ("婚假",)),
    ("孩子出生男方有假吗", ("产假与陪产假",)),
    ("每个月几号发工资", ("发薪日",)),
    ("打卡忘了怎么补救", ("补卡",)),
    ("电脑中毒了该找谁", ("终端安全", "电脑与网络", "安全事件")),
]


def load_chunks(cfg) -> Tuple[List[str], List[str], List[str]]:
    conn = sqlite3.connect(str(cfg.registry_path))
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT text, title, section FROM chunks WHERE is_parent=0 ORDER BY chunk_id"
    ).fetchall()
    conn.close()
    return ([r["text"] for r in rows], [r["title"] for r in rows],
            [r["section"] or "" for r in rows])


def rank(query: str, qvec, docvecs: np.ndarray) -> np.ndarray:
    q = qvec / (np.linalg.norm(qvec) + 1e-9)
    d = docvecs / (np.linalg.norm(docvecs, axis=1, keepdims=True) + 1e-9)
    return np.argsort(-(d @ q))


def main() -> None:
    cfg = load_config()
    texts, titles, sections = load_chunks(cfg)
    print(f"语料：{len(texts)} 个检索单元\n")

    variants = {
        "hash（原方案，等价随机）": HashEmbedder(cfg.embedding.dim),
        f"fastembed · {cfg.embedding.model.split('/')[-1]}": FastEmbedEmbedder(
            cfg.embedding.model, cache_dir=str(cfg.data_path / "models")),
    }

    results = {}
    for label, emb in variants.items():
        vecs = np.asarray(emb.encode(texts), dtype=np.float32)
        rows = []
        for q, expect in QUERIES:
            qv = np.asarray(emb.encode_one(q), dtype=np.float32)
            order = rank(q, qv, vecs)
            top = [(int(i), sections[i]) for i in order[:3]]
            hit = next((pos + 1 for pos, (i, _) in enumerate(top)
                        if any(e in sections[i] for e in expect)), None)
            rows.append((q, expect, top, hit))
        results[label] = rows

    for label, rows in results.items():
        hits = [r[3] for r in rows if r[3]]
        mrr = sum(1 / h for h in hits) / len(rows)
        print(f"=== {label} ===")
        print(f"    向量路命中率 top3 = {len(hits)}/{len(rows)}   MRR = {mrr:.3f}")
        for q, expect, top, hit in rows:
            mark = f"命中#{hit}" if hit else "未命中"
            got = " | ".join(s or "(无)" for _, s in top)
            print(f"    [{mark:6s}] {q:16s} → {got}")
        print()


if __name__ == "__main__":
    main()
