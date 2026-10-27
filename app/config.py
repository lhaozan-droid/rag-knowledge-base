"""配置加载：YAML + 环境变量插值 + 类型化 dataclass。

支持 ${VAR} 与 ${VAR:-默认值} 两种写法，避免把 API Key 写进仓库。
"""
from __future__ import annotations

import dataclasses
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Type, get_type_hints

import yaml

# .env 必须由配置层自己加载，不能只依赖 cli.py。
# tools/ 下的脚本、以及将来任何非 CLI 入口都不会经过 cli.py，
# 那样 ${LLM_API_KEY} 会展开成空字符串，请求带着空 key 出去直接被 401，
# 表面上"检索正常"、实际回答全退化成离线摘录——很难发现。
try:  # python-dotenv 是可选依赖，缺了也不该炸
    from dotenv import load_dotenv

    # 指定项目根目录的 .env，而不是靠 cwd 猜——从任何目录运行脚本都能拿到
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
except Exception:  # noqa: BLE001
    pass

_ENV_PAT = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


def _expand(value: Any) -> Any:
    if isinstance(value, str):
        def repl(m: re.Match) -> str:
            name, default = m.group(1), m.group(2)
            env = os.environ.get(name)
            if env is not None:
                return env
            return default if default is not None else ""
        return _ENV_PAT.sub(repl, value)
    if isinstance(value, dict):
        return {k: _expand(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand(v) for v in value]
    return value


def _build(cls: Type, data: Any):
    """按 dataclass 声明递归构造，忽略未知字段（方便用户加注释/临时字段）。"""
    if not isinstance(data, dict):
        return cls()
    hints = get_type_hints(cls)
    kwargs: Dict[str, Any] = {}
    for f in dataclasses.fields(cls):
        if f.name not in data:
            continue
        tp = hints.get(f.name, f.type)
        raw = data[f.name]
        if dataclasses.is_dataclass(tp):
            kwargs[f.name] = _build(tp, raw)
        else:
            kwargs[f.name] = raw
    return cls(**kwargs)


@dataclass
class AppConfig:
    name: str = "企业知识库"
    data_dir: str = "./data"
    log_level: str = "INFO"


@dataclass
class OllamaConfig:
    base_url: str = "http://127.0.0.1:11434"


@dataclass
class OpenAICompatConfig:
    base_url: str = ""
    api_key: str = ""
    model: str = ""


@dataclass
class EmbeddingConfig:
    """混合模式的核心开关：本地 Embedding，文档原文不出内网。"""
    provider: str = "fastembed"     # fastembed | ollama | sentence_transformers | openai_compat | hash
    model: str = "jinaai/jina-embeddings-v2-base-zh"
    dim: int = 768
    batch_size: int = 64
    normalize: bool = True
    cache: bool = True              # 文本级向量缓存，增量入库时省 90% 算力
    device: str = "cpu"
    ollama: OllamaConfig = field(default_factory=OllamaConfig)
    openai_compat: OpenAICompatConfig = field(default_factory=OpenAICompatConfig)


@dataclass
class VectorStoreConfig:
    provider: str = "memory"        # chroma | memory
    path: str = "./data/chroma"
    collection: str = "kb_main"


@dataclass
class LLMConfig:
    provider: str = "extractive"    # openai_compat | extractive
    base_url: str = ""
    api_key: str = ""
    model: str = ""
    temperature: float = 0.1
    max_tokens: int = 1200
    timeout_s: float = 60.0
    max_retries: int = 2


@dataclass
class RetrievalConfig:
    dense_top_k: int = 40
    sparse_top_k: int = 40
    fusion_k: int = 30
    rrf_k: int = 60
    dense_weight: float = 1.0
    sparse_weight: float = 1.0
    final_top_k: int = 6
    max_per_doc: int = 2            # 同一篇文档最多占几条候选，防止单篇碎片占满 top_k
    max_context_chars: int = 6000
    min_top_score: float = 0.0      # dense 余弦相似度阈值，0 表示不启用（建议先跑评估再定）
    # 检索分数低于此值时，先做「错别字纠错」重检再决定是否拒答；0 表示关闭。
    # 应高于 min_top_score，专门拦「勉强命中但跑偏」的查询（如同音错别字）。
    correct_trigger_score: float = 0.0
    use_parent_context: bool = True
    # 领域同义词扩展，只作用于 BM25 稀疏路。补「休假/请假」这类词面鸿沟，
    # 关掉即可退回纯字面匹配，方便做 A/B 对比。
    expand_glossary: bool = True


@dataclass
class RerankConfig:
    provider: str = "none"          # none | cross_encoder | llm
    model: str = "BAAI/bge-reranker-base"
    top_k: int = 6
    llm_candidates: int = 12


@dataclass
class IngestConfig:
    # 一级子目录当部门名，用于权限过滤。
    # 知识库按「部门/子目录」组织时该开；目录名只是「来源/分类」时必须关，
    # 否则会把这类目录名写成 dept，污染按部门过滤。
    dept_from_dir: bool = True
    # 知识库根目录（管理后台「全部重新入库」用）。相对路径以服务进程 cwd（项目根）为基准。
    root: str = ""


@dataclass
class ChunkingConfig:
    max_chars: int = 700            # 子块上限：太大召回噪声多，太小语义不完整
    overlap_chars: int = 120
    min_chars: int = 80
    parent_max_chars: int = 2200    # 父块上限：喂给 LLM 的上下文窗口预算


@dataclass
class ConversationConfig:
    max_turns: int = 6
    history_turns_for_llm: int = 3
    rewrite: bool = True


@dataclass
class GuardConfig:
    redact_pii: bool = True
    require_citation: bool = True
    no_answer_text: str = "根据现有资料，我没能找到能支撑这个问题的内容。建议咨询相关业务负责人，我可以帮你把这个问题记录下来。"


@dataclass
class Config:
    app: AppConfig = field(default_factory=AppConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    vector_store: VectorStoreConfig = field(default_factory=VectorStoreConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    rerank: RerankConfig = field(default_factory=RerankConfig)
    ingest: IngestConfig = field(default_factory=IngestConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    conversation: ConversationConfig = field(default_factory=ConversationConfig)
    guard: GuardConfig = field(default_factory=GuardConfig)

    # ---------- 派生路径 ----------
    @property
    def data_path(self) -> Path:
        p = Path(self.app.data_dir)
        return p if p.is_absolute() else (Path.cwd() / p)

    @property
    def registry_path(self) -> Path:
        return self.data_path / "registry.db"

    @property
    def bm25_path(self) -> Path:
        return self.data_path / f"bm25_{self.vector_store.collection}.pkl"

    @property
    def embed_cache_path(self) -> Path:
        return self.data_path / "embed_cache.db"

    def ensure_dirs(self) -> None:
        self.data_path.mkdir(parents=True, exist_ok=True)
        if self.vector_store.provider == "chroma":
            Path(self.vector_store.path).mkdir(parents=True, exist_ok=True)

    # ---------- 序列化 ----------
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Config":
        return _build(cls, _expand(data or {}))

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    def summary(self) -> Dict[str, Any]:
        """脱敏后的配置摘要，供 /health 与日志使用。"""
        d = self.to_dict()
        if d["llm"].get("api_key"):
            d["llm"]["api_key"] = "***"
        if d["embedding"].get("openai_compat", {}).get("api_key"):
            d["embedding"]["openai_compat"]["api_key"] = "***"
        return d


def load_config(path: str | Path | None = None) -> Config:
    """加载配置。找不到文件时使用内置默认值（全离线可跑）。

    默认按「项目根目录 → 当前目录」的顺序找 config.yaml。只认 cwd 的话，
    从别的目录启动（服务、定时任务）会静默退化成默认配置：
    api_key 变空、min_top_score 变 0，表面上能跑，实际答不对也不拒答。
    """
    if path is None:
        env_path = os.environ.get("RAG_CONFIG")
        if env_path:
            path = env_path
        else:
            root = Path(__file__).resolve().parents[1]
            for cand in (root / "config.yaml", root / "config.yml",
                         Path("config.yaml"), Path("config.yml")):
                if cand.exists():
                    path = cand
                    break
    if path and Path(path).exists():
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        return Config.from_dict(raw)
    return Config()
