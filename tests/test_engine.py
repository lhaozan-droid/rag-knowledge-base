"""端到端：入库 → 检索 → 拼 Prompt → 引用校验 → 兜底。全部离线可跑。"""
from app.llm.client import ExtractiveLLM
from app.rag.engine import RagEngine
from tests.conftest import DEMO, RecordingLLM, make_cfg


def test_ingest_report(engine):
    report = engine.ingest(DEMO)  # 第二次入库应全部跳过（幂等）
    assert report.skipped == 5
    assert report.added == 0
    stats = engine.stats()
    assert stats["docs"] == 5
    assert stats["chunks"] > 30
    assert stats["vectors"] == stats["chunks"]


def test_answer_carries_sources_and_citations(engine):
    ans = engine.answer("年假有多少天")
    assert ans.sources, "必须给出引用出处"
    assert "[1]" in ans.text
    assert ans.latency_ms >= 0
    assert ans.conversation_id

    # 结构化字段必须与旧字段共存：/chat 返回 conclusion + rules + notes，且 text 仍保留
    payload = ans.to_dict()
    assert "conclusion" in payload
    assert "rules" in payload
    assert "notes" in payload
    assert payload["text"] == ans.text
    assert "sources" in payload and "answered" in payload
    assert "conversation_id" in payload and "latency_ms" in payload


def test_retrieved_context_reaches_prompt(tmp_path):
    llm = RecordingLLM("年假天数为 15 天 [1]")
    eng = RagEngine(make_cfg(tmp_path), llm=llm)
    eng.ingest(DEMO, reset=True)
    ans = eng.answer("年假有多少天")

    assert llm.blocks, "上下文块不能为空"
    joined = "\n".join(b["text"] for b in llm.blocks)
    assert "15 天" in joined, "检索应命中员工手册的年假条款"
    assert ans.answered is True
    assert ans.flags["citations"] == [1]
    eng.close()


def test_citation_validation_flags_missing(tmp_path):
    llm = RecordingLLM("年假有 15 天。")  # 故意不写引用
    eng = RagEngine(make_cfg(tmp_path), llm=llm)
    eng.ingest(DEMO, reset=True)
    ans = eng.answer("年假有多少天")
    assert ans.flags["citation_missing"] is True
    assert ans.answered is False
    assert "未标注出处" in ans.text
    eng.close()


def test_refusal_when_no_confident_hit(tmp_path):
    cfg = make_cfg(tmp_path, retrieval={"min_top_score": 0.99})
    eng = RagEngine(cfg)
    eng.ingest(DEMO, reset=True)
    ans = eng.answer("今天中午食堂吃什么")
    assert ans.answered is False
    assert ans.sources == []
    assert ans.flags.get("no_hit") is True
    rows = eng.registry.list_unanswered()
    assert rows and rows[0]["query"] == "今天中午食堂吃什么"
    eng.close()


def test_multi_turn_uses_history(tmp_path):
    llm = RecordingLLM("按制度执行 [1]")
    eng = RagEngine(make_cfg(tmp_path), llm=llm)
    eng.ingest(DEMO, reset=True)
    a1 = eng.answer("年假有多少天")
    a2 = eng.answer("那病假呢", conversation_id=a1.conversation_id)

    assert a2.conversation_id == a1.conversation_id
    assert a2.rewritten_query != "那病假呢", "追问应被改写成独立查询"
    # 写死的期望值带 [1]：改写结果里的引用角标不是"事实"，护栏必须原样接受它。
    # 若这里变回原问题，说明数字护栏又把角标当成"抄自上一答的数字"了。
    assert a2.rewritten_query == "年假申请流程怎么走 [1]", \
        f"带角标的改写应被接受，实际 {a2.rewritten_query!r}"
    assert llm.rewrite_calls == 1, "有历史且命中指代时，应恰好触发一次改写"
    # system + 历史2轮 + 当前问题
    assert len(llm.messages) >= 4
    roles = [m["role"] for m in llm.messages]
    assert roles[0] == "system" and roles[-1] == "user"
    eng.close()


def test_extractive_fallback_without_llm(engine):
    """没有 API Key 时，至少要给出原文句子而不是空响应。"""
    assert engine.llm.name == "extractive"
    ans = engine.answer("密码多久必须更换一次")
    assert ans.text.strip()
    assert ans.sources
    assert "离线摘录" in ans.text
    # 不配 Key 是默认状态，这份文案是读者的第一印象：不许出现配置键名，
    # 否则等于告诉读者「你得先懂配置才能用」。
    assert "openai_compat" not in ans.text
    assert "llm.provider" not in ans.text


def test_stream_events(tmp_path):
    llm = RecordingLLM("年假 15 天 [1]")
    eng = RagEngine(make_cfg(tmp_path), llm=llm)
    eng.ingest(DEMO, reset=True)
    events = list(eng.answer_stream("年假"))
    types = [e["type"] for e in events]
    assert types[0] == "meta"
    assert "sources" in types
    assert "delta" in types
    assert types[-1] == "done"
    assert "15 天" in events[-1]["text"]
    eng.close()
