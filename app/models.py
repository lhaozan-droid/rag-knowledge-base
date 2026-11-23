"""核心数据模型。

设计原则：所有在检索链路上流动的数据都是有类型的 dataclass，
避免用裸 dict 传递导致字段含义漂移（大规模入库时这类 bug 极难排查）。
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


@dataclass
class Document:
    """一份知识库文档。doc_id 由路径 + 内容哈希决定，保证同内容不同路径不互相覆盖。"""
    doc_id: str
    path: str                 # 相对知识库根目录的路径
    title: str
    source_type: str          # md / txt / pdf / docx / pptx / xlsx / html
    content_hash: str
    mtime: float
    char_count: int = 0
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Chunk:
    """检索单元。children 参与向量检索，parent 用于喂给 LLM 时补全上下文。"""
    chunk_id: str
    doc_id: str
    text: str
    ordinal: int
    title: str
    section: str = ""         # 标题层级路径，如 "第3章 考勤管理 > 3.2 请假流程"
    path: str = ""
    parent_id: str = ""
    is_parent: int = 0
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievedChunk:
    """一次检索命中的片段，保留各阶段分数便于排查召回问题。"""
    chunk: Chunk
    score: float = 0.0
    dense_score: float = 0.0
    sparse_score: float = 0.0
    rerank_score: Optional[float] = None
    rank_dense: int = -1
    rank_sparse: int = -1


@dataclass
class Source:
    """对外暴露的引用出处，前端/企微卡片直接渲染这个结构。"""
    index: int
    chunk_id: str
    doc_id: str
    title: str
    section: str
    path: str
    score: float                # 融合分：rrf_k=60、两路权重 1.0 时上界 ≈ 2/61 ≈ 0.0328；
                                # rerank.provider 开成 cross_encoder 后这里变成 rerank 分值，
                                # 量纲完全不同。只供排序排查，别当「相似度」展示。
    snippet: str
    url: str = ""
    content_type: str = ""
    dense_score: float = 0.0    # 向量余弦相似度（0~1），语义不随 rerank 配置漂移，
                                # 展示「相关度」用它。0.0 表示该片段只从 BM25 路命中、
                                # 没有 dense 分，此时不要显示成「相关度 0.00」。

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Turn:
    role: str                 # user / assistant
    content: str
    ts: float = 0.0


@dataclass
class Answer:
    text: str
    sources: List[Source] = field(default_factory=list)
    rewritten_query: str = ""
    answered: bool = True
    conversation_id: str = ""
    latency_ms: int = 0
    usage: Dict[str, int] = field(default_factory=dict)
    flags: Dict[str, Any] = field(default_factory=dict)
    # 结构化视图：`text` 仍是权威完整回答，下面三个字段只是它的稳定拆分，
    # 供前端答案态稳定拿到「结论句 + 分条规则 + 补充说明」，缺失时前端降级渲染 text。
    conclusion: str = ""
    rules: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "conclusion": self.conclusion,
            "rules": self.rules,
            "notes": self.notes,
            "sources": [s.to_dict() for s in self.sources],
            "rewritten_query": self.rewritten_query,
            "answered": self.answered,
            "conversation_id": self.conversation_id,
            "latency_ms": self.latency_ms,
            "usage": self.usage,
            "flags": self.flags,
        }


@dataclass
class IngestReport:
    added: int = 0
    updated: int = 0
    skipped: int = 0
    failed: int = 0
    chunks: int = 0
    docs_total: int = 0
    chunks_total: int = 0
    elapsed_s: float = 0.0
    errors: List[str] = field(default_factory=list)
    # 以下两类必须单列：它们不是"成功"，也不是"失败"，但都代表知识库少了一块内容。
    unsupported: List[str] = field(default_factory=list)   # 格式不支持，压根没读
    empty_text: List[str] = field(default_factory=list)    # 读了但没有可用文本

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
