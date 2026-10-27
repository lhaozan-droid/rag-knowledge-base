"""答案判定的单测。

这套判据是「拒答会不会被算成已回答」的总闸门，逻辑错了不会报错，
只会让统计悄悄失真——所以必须逐条钉住。
"""
import pytest

from app.rag.verdict import (REFUSAL_MARK, judge, looks_like_refusal_head,
                             strip_refusal_mark)


ALLOWED = [1, 2, 3]


def test_normal_answer_with_citations():
    v = judge("年假 5 天 [1]，满 10 年 10 天 [2]。", ALLOWED)
    assert v.answered and not v.refused
    assert v.citations == [1, 2]
    assert v.reason is None
    assert not v.should_log_unanswered


def test_marked_refusal_ignores_citations():
    """模型拒答时即使顺手带了引用，也不能算已回答。"""
    text = "NO_ANSWER\n资料中没有提到加班餐补标准，建议咨询人力资源部 [1][2]。"
    v = judge(text, ALLOWED)
    assert v.refused and not v.answered
    assert v.reason == "model_refused"
    assert v.citations == []
    assert "加班餐补" in v.text
    assert "[" not in v.text


def test_refusal_mark_variants():
    for raw in ["NO_ANSWER\n没找到", "[NO_ANSWER] 没找到", "no_answer：没找到"]:
        assert looks_like_refusal_head(raw), raw
        assert judge(raw, ALLOWED).refused


def test_heuristic_refusal_without_mark():
    """模型没按格式输出标记时，用启发式兜住，但只在没有引用时启用。"""
    v = judge("资料中未提及此项内容。", ALLOWED)
    assert v.refused and v.reason == "model_refused"
    assert v.flags["refusal"] == "heuristic"


def test_heuristic_does_not_fire_when_cited():
    v = judge("资料中未提及试用期延长，但制度写明最长 6 个月 [1]。", ALLOWED)
    assert not v.refused
    assert v.citations == [1]


def test_long_answer_without_citation_is_engineering_issue():
    text = "关于这个问题，" + "说明" * 120  # 远超启发式长度上限
    v = judge(text, ALLOWED)
    assert not v.refused
    assert v.reason == "citation_missing"
    assert not v.answered
    # 未标引用是工程问题，不是知识缺口：进了未答清单会让人误以为库里没这份资料，
    # 照着清单去补文档就补错方向了。
    assert not v.should_log_unanswered


def test_citation_outside_allowed_is_dropped():
    """引用编号必须来自本轮资料，跨轮残留的 [9] 不能算数。"""
    v = judge("答案见 [9]。", ALLOWED)
    assert v.citations == []
    assert v.reason == "citation_missing"


def test_require_citation_off():
    v = judge("无法确定。", ALLOWED, require_citation=False)
    assert v.answered is True or v.refused is True


def test_strip_mark_keeps_body():
    body = strip_refusal_mark("NO_ANSWER：资料中没有该条款，建议查《员工手册》。")
    assert body.startswith("资料中没有该条款")
    assert REFUSAL_MARK not in body


def test_empty_text_is_not_answered():
    v = judge("", ALLOWED)
    assert not v.answered
    # 模型吐空也是工程问题（engine 里会转成 LLMError 走摘录兜底），
    # 不该当成「知识库里没有这个问题的答案」。
    assert not v.should_log_unanswered
