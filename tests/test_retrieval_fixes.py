"""D4 / D5 / D6 的回归测试。

这三个缺陷的共同点是「不报错、只是效果变差」，没有测试就会悄悄复发：
- D4 上下文预算按命中数均分，父块被拦腰截断
- D5 稀疏路带 where 时先取 top_k 再过滤，过滤后可能所剩无几
- D6 每次入库全量重新分词，改一份文档要等整库
"""
from app.models import Chunk, RetrievedChunk
from app.rag.engine import RagEngine
from app.retrieval.bm25 import BM25Index
from tests.conftest import DEMO, make_cfg

LONG = "年假相关规定原文。" * 280  # 约 2800 字，超过父块上限的典型长章节


def _hit(chunk_id: str, parent_id: str, text: str = "子块内容") -> RetrievedChunk:
    return RetrievedChunk(chunk=Chunk(
        chunk_id=chunk_id, doc_id="d1", text=text, ordinal=0,
        title="员工手册", section="第三章 年假", parent_id=parent_id))


# ---------------- D4：预算按去重后的块数分配 ----------------

def test_context_budget_uses_unique_blocks(tmp_path, monkeypatch):
    """6 条命中来自 2 个父块 → 每块应分到 3000 字，而不是 6000//6=1000。"""
    cfg = make_cfg(tmp_path, retrieval={"max_context_chars": 6000})
    eng = RagEngine(cfg)
    hits = [_hit(f"c{i}", f"p{i % 2}") for i in range(6)]

    parents = {f"p{i}": Chunk(chunk_id=f"p{i}", doc_id="d1", text=LONG,
                              ordinal=0, title="员工手册", is_parent=1)
               for i in range(2)}
    monkeypatch.setattr(eng.registry, "get_chunks", lambda ids: dict(parents))

    blocks, sources = eng._build_context(hits)

    assert len(blocks) == 2, "同一父块只能出现一次"
    assert len(sources) == 2
    for b in blocks:
        assert len(b["text"]) >= len(LONG) - 2, "预算足够时父块不应被截断"
    total = sum(len(b["text"]) for b in blocks)
    assert total <= 6000
    eng.close()


def test_context_budget_caps_at_budget_when_overflow(tmp_path, monkeypatch):
    """父块总长超预算时按块数均分，且总量不超预算。"""
    cfg = make_cfg(tmp_path, retrieval={"max_context_chars": 2000})
    eng = RagEngine(cfg)
    hits = [_hit(f"c{i}", f"p{i % 4}") for i in range(8)]

    parents = {f"p{i}": Chunk(chunk_id=f"p{i}", doc_id="d1", text=LONG,
                              ordinal=0, title="员工手册", is_parent=1)
               for i in range(4)}
    monkeypatch.setattr(eng.registry, "get_chunks", lambda ids: dict(parents))

    blocks, _ = eng._build_context(hits)
    assert len(blocks) == 4
    assert sum(len(b["text"]) for b in blocks) <= 2000 + 40
    eng.close()


# ---------------- D5：带 where 时扩大稀疏候选池 ----------------

def test_sparse_pool_expands_when_filtering(tmp_path):
    """带 where 时，稀疏路必须先取更大候选池再过滤，否则过滤后可能一条不剩。"""
    eng = RagEngine(make_cfg(tmp_path, retrieval={"sparse_top_k": 10}))
    eng.ingest(DEMO, reset=True)
    calls = []

    def spy(query, top_k=40):
        calls.append(top_k)
        return []

    eng.bm25.search = spy  # 只关心传下去的候选池大小
    eng.retrieve("年假有多少天")
    assert calls[-1] == 10, "不带 where 时按配置取"

    eng.retrieve("年假有多少天", where={"dept": "技术部"})
    assert calls[-1] > 10, "带 where 时必须放大候选池，否则过滤后可能所剩无几"
    eng.close()


# ---------------- D6：BM25 增量分词 ----------------

def test_bm25_reuses_tokens_when_unchanged():
    idx = BM25Index()
    items = [(f"c{i}", f"第 {i} 段制度原文，涉及考勤与请假流程。") for i in range(20)]
    idx.build(items)
    assert idx.last_tokenized == 20 and idx.last_reused == 0

    idx.build(items)  # 文本没变
    assert idx.last_reused == 20 and idx.last_tokenized == 0

    changed = list(items)
    changed[3] = ("c3", "改过的第三段，内容不同了")
    idx.build(changed)
    assert idx.last_tokenized == 1, "只有改动的块需要重新分词"
    assert idx.last_reused == 19


def test_bm25_drops_removed_chunks():
    idx = BM25Index()
    idx.build([("a", "甲方"), ("b", "乙方")])
    idx.build([("a", "甲方")])
    assert len(idx) == 1
    assert idx.search("乙方", 10) == []


def test_bm25_save_load_keeps_token_cache(tmp_path):
    path = tmp_path / "bm25.pkl"
    idx = BM25Index(path)
    idx.build([("a", "年假天数"), ("b", "住宿标准")])
    idx.save()

    same = BM25Index(path)
    assert same.load() is True
    same.build([("a", "年假天数"), ("b", "住宿标准")])
    assert same.last_reused == 2, "落盘后再重建应能复用分词"


# ---------------- 传入空索引时的 falsy 陷阱 ----------------

def test_pipeline_uses_passed_empty_index(tmp_path):
    """BM25Index 定义了 __len__，空索引是 falsy。

    构造 IngestPipeline 时若写成 `bm25 or BM25Index(...)`，
    会静默替换成新对象，调用方手里的索引永远填不上数据（新库首次入库后稀疏路失效）。
    """
    eng = RagEngine(make_cfg(tmp_path))
    assert len(eng.bm25) == 0
    eng.ingest(DEMO, reset=True)
    assert len(eng.bm25) == eng.registry.count_chunks() > 0
    eng.close()


def test_engine_rebuilds_missing_bm25_on_start(tmp_path):
    """索引文件被删/格式过期时，启动应自愈，而不是静默退化成单路召回。"""
    eng = RagEngine(make_cfg(tmp_path))
    eng.ingest(DEMO, reset=True)
    n = len(eng.bm25)
    eng.close()

    eng2 = RagEngine(make_cfg(tmp_path))
    assert len(eng2.bm25) == n
    # 删掉索引文件后重新构造，应自动重建
    eng2.close()
    (tmp_path / "bm25_kb_main.pkl").unlink(missing_ok=True)
    eng3 = RagEngine(make_cfg(tmp_path))
    assert len(eng3.bm25) == n, "启动时应自动重建 BM25 索引"
    eng3.close()


# ---------------- LLM 精排的预算（静默降级类） ----------------

def test_llm_rerank_budget_leaves_room_for_json():
    """精排要输出 12 条候选的 JSON，预算必须涵盖「思考 + 完整 JSON」。

    回归点：旧值 400 被推理模型的思考吃掉大半，JSON 被截断 → 解析失败 →
    静默回退融合顺序，界面与「没开精排」完全一样，没人能发现精排没生效。
    """
    from app.retrieval.rerank import LLMReranker

    kwargs = {}

    class _L:
        name = "openai_compat"

        def chat(self, messages, **k):
            kwargs.update(k)
            return type("R", (), {"text": '[{"id":1,"score":9},{"id":2,"score":3}]',
                                  "finish_reason": "stop"})()

    cands = [_hit("c1", "p1", "报销时限 30 天"), _hit("c2", "p2", "年假 15 天")]
    out = LLMReranker(_L(), candidates=12).rerank("报销多久到账", cands, top_k=1)
    assert kwargs.get("max_tokens", 0) >= 1024, "精排预算给太小会截断 JSON"
    assert out and out[0].chunk.chunk_id == "c1"
