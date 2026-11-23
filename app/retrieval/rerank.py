"""重排（Rerank）。

RRF 融合解决的是"召回排序"，重排解决的是"精度"：把 30 个候选交给交叉编码器
（query 和 doc 一起过模型）打分，通常能把 Top-6 的准确率再提 10~20 个点。
代价是延迟。建议顺序：
  先 none → 用 eval/ 里的黄金测试集跑 recall@k；如果 recall@20 高但 Top-5 差，再开重排。
"""
from __future__ import annotations

import json
import logging
import re
from typing import List, Optional

from app.models import RetrievedChunk

log = logging.getLogger(__name__)


class BaseReranker:
    name = "none"

    def rerank(self, query: str, candidates: List[RetrievedChunk],
               top_k: int) -> List[RetrievedChunk]:
        return candidates[:top_k]


class CrossEncoderReranker(BaseReranker):
    name = "cross_encoder"

    def __init__(self, model: str = "BAAI/bge-reranker-base", device: str = "cpu"):
        from sentence_transformers import CrossEncoder  # 可选依赖

        self.model = CrossEncoder(model, device=device)

    def rerank(self, query, candidates, top_k):
        if not candidates:
            return []
        pairs = [(query, c.chunk.text) for c in candidates]
        scores = self.model.predict(pairs)
        for c, s in zip(candidates, scores):
            c.rerank_score = float(s)
        return sorted(candidates, key=lambda c: -(c.rerank_score or 0.0))[:top_k]


class LLMReranker(BaseReranker):
    """用大模型做相关性打分。零额外模型部署，但每轮多一次 LLM 调用，
    务必把 llm_candidates 控制在 12 以内，否则延迟和成本不划算。"""

    name = "llm"

    def __init__(self, llm, candidates: int = 12):
        self.llm = llm
        self.candidates = candidates

    def rerank(self, query, candidates, top_k):
        if not candidates:
            return []
        cands = candidates[: self.candidates]
        listing = "\n".join(
            f"{i + 1}. {c.chunk.title} / {c.chunk.section} :: {c.chunk.text[:220]}"
            for i, c in enumerate(cands))
        prompt = (f"问题：{query}\n\n候选资料：\n{listing}\n\n"
                  "请为每条资料与问题的相关性打分（0-10，整数）。"
                  "只输出 JSON 数组，如 [{\"id\":1,\"score\":8}]，不要解释。")
        try:
            # 思考 token 与正文共用这个预算：本处要输出 12 条候选的 JSON 数组
            # （约 150~200 token），推理模型还要先花约 100 思考 token，给 400 时
            # 预算会被吃掉大半，JSON 被截断解析失败 → 静默退回融合顺序，
            # 界面上和「没开精排」一模一样。按「思考 + 完整 JSON」留足余量。
            res = self.llm.chat([{"role": "user", "content": prompt}],
                                temperature=0.0, max_tokens=1024)
            raw = re.search(r"\[.*\]", res.text, re.S)
            if not raw:
                raise ValueError("未返回 JSON 数组")
            scores = {int(d["id"]): float(d["score"]) for d in json.loads(raw.group(0))}
            for i, c in enumerate(cands, 1):
                c.rerank_score = scores.get(i, 0.0)
            return sorted(cands, key=lambda c: -(c.rerank_score or 0.0))[:top_k]
        except Exception as exc:  # noqa: BLE001
            log.warning("[rerank] 重排未生效: %s，回退到融合顺序", exc)
            return candidates[:top_k]


def build_reranker(cfg, llm=None) -> BaseReranker:
    provider = cfg.rerank.provider
    if provider == "cross_encoder":
        try:
            return CrossEncoderReranker(cfg.rerank.model, device=cfg.embedding.device)
        except Exception as exc:  # noqa: BLE001
            log.warning("交叉编码器加载失败（%s），重排降级为 none", exc)
            return BaseReranker()
    if provider == "llm" and llm is not None:
        return LLMReranker(llm, cfg.rerank.llm_candidates)
    return BaseReranker()
