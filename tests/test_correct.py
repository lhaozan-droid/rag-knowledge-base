"""查询纠错单元测试。"""
from app.rag.correct import correct_query, llm_correct_query


class _FakeLLM:
    def __init__(self, name="openai", out=""):
        self.name = name
        self.out = out
        self.calls = []

    def chat(self, messages, **kwargs):
        self.calls.append(messages)
        return type("R", (), {"text": self.out, "usage": {}})()


def test_correct_known_typo():
    assert correct_query("年佳怎么申请") == "年假怎么申请"


def test_correct_no_typo_unchanged():
    assert correct_query("年假怎么申请") == "年假怎么申请"
    assert correct_query("报销单怎么填") == "报销单怎么填"


def test_correct_empty():
    assert correct_query("") == ""
    assert correct_query(None) == ""


def test_llm_correct_extractive_returns_original():
    llm = _FakeLLM(name="extractive", out="年假")
    assert llm_correct_query("年佳怎么申请", llm) == "年佳怎么申请"
    assert llm.calls == []


def test_llm_correct_normal():
    llm = _FakeLLM(name="openai", out="年假怎么申请")
    assert llm_correct_query("年佳怎么申请", llm) == "年假怎么申请"


def test_llm_correct_identical_result_returns_original():
    # 纠错结果与原文相同 → 不采纳，原样返回
    llm = _FakeLLM(name="openai", out="年佳怎么申请")
    assert llm_correct_query("年佳怎么申请", llm) == "年佳怎么申请"


def test_llm_correct_garbage_falls_back():
    # 输出超长（异常）→ 原样返回
    llm = _FakeLLM(name="openai", out="这是纠正后的查询" * 20)
    assert llm_correct_query("年佳怎么申请", llm) == "年佳怎么申请"


def test_llm_correct_raises_falls_back():
    llm = _FakeLLM(name="openai", out="年假")

    def boom(messages, **kwargs):
        raise RuntimeError("boom")

    llm.chat = boom
    assert llm_correct_query("年佳怎么申请", llm) == "年佳怎么申请"


def test_llm_correct_budget_leaves_room_for_reasoning():
    """纠错调用的 max_tokens 必须给推理模型留够思考预算。

    回归点：旧值 120 被思考 token 吃光、正文返回空，纠错静默不生效，
    界面上和「本来就没有错别字」完全一样，无法区分。
    """
    kwargs = {}

    class _L:
        name = "openai"

        def chat(self, messages, **k):
            kwargs.update(k)
            return type("R", (), {"text": "年假怎么申请", "finish_reason": "stop"})()

    llm_correct_query("年佳怎么申请", _L())
    assert kwargs.get("max_tokens", 0) >= 512, "纠错预算给太小会被思考吃光"
