"""同义词扩展的单元测试。

重点不是「扩展得越多越好」，而是：
1. 该扩的必须扩到（休假/请假这类真实词面鸿沟）；
2. 不该扩的不能乱扩（无关问题不能被塞进制度词，否则 BM25 会召回噪声）。
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.rag.glossary import describe, expand_for_sparse, expand_terms  # noqa: E402


def test_leave_query_gets_expanded():
    """用户点名的场景：问「休假怎么申请」，必须扩出「请假申请/年休假」。"""
    terms = expand_terms("休假怎么申请")
    assert "请假申请" in terms
    assert "年休假" in terms
    assert "直属主管" in terms


def test_expanded_query_keeps_original():
    q = "休假怎么申请"
    out = expand_for_sparse(q)
    assert out.startswith(q)
    assert len(out) > len(q)


def test_no_duplicate_terms():
    terms = expand_terms("休假请假年假调休")
    assert len(terms) == len(set(terms))
    # 原 query 里已有的词不应该再塞一遍
    for w in terms:
        assert w not in "休假请假年假调休" or len(w) > 4


def test_out_of_scope_query_not_expanded():
    """库外问题不能被强行塞制度词，否则会把无关文档召回来。"""
    for q in ("公司有多少员工", "怎么写竞品分析报告", "今天天气怎么样"):
        assert expand_terms(q) == [], f"{q} 不应命中任何同义词组"


def test_empty_query_safe():
    assert expand_terms("") == []
    assert expand_for_sparse("") == ""


def test_max_terms_cap():
    q = "休假 报账 出差 打卡 密码 入职"
    out = expand_for_sparse(q, max_terms=5)
    # 原 query + 最多 5 个扩展词
    assert len(out.split()) <= len(q.split()) + 5


def test_same_group_expanded_once():
    """同一组里的多个触发词同时出现（「休假」+「请假」），扩展结果不应翻倍。"""
    a = expand_terms("休假")
    b = expand_terms("休假请假")
    assert b == a


def test_describe_runs():
    assert "→" in describe("休假怎么申请")
    assert "无扩展" in describe("公司有多少员工")


def test_payment_arrival_query_expands_to_dawu():
    """「到账」在语料里 0 次，制度写的是「随工资打款」，必须扩出「打款」。

    不扩的话「报销多久能到账」的稀疏路一个词都命不中写着答案的那一节。
    """
    terms = expand_terms("报销多久能到账")
    assert "打款" in terms
    assert "打款" in expand_for_sparse("报销多久能到账")


def test_salary_date_not_expanded():
    """「工资什么时候发」的答案在员工手册的发薪日，不能被拽到费用报销的「打款」节。"""
    assert expand_terms("工资什么时候发") == []


def test_travel_expansion_excludes_sibling_sections():
    """出差组不能收兄弟小节的标题词。

    「差旅补贴」「交通工具」是同一篇文档里并列的主题，收进来会把补贴节顶到
    第一，把真正写住宿标准的那节挤出 max_per_doc 配额。
    """
    terms = expand_terms("出差住宿费能报多少")
    assert "差旅补贴" not in terms
    assert "交通工具" not in terms
    # 指向住宿标准那一节的说法要在
    assert "住宿标准" in terms or "住宿费" in terms


def test_vehicle_query_expands_to_section_body_words():
    """「坐飞机 / 打车」必须能扩到 §2.2 正文里的说法。

    扩展词只能取正文原词：「交通工具」是这一节的标题、不进 chunk 正文，
    写进扩展词等于零命中，所以它必须在触发词里出现、绝不能出现在扩展词里。
    """
    terms = expand_terms("出差能坐飞机吗")
    assert "二等座" in terms
    assert "经济舱" in terms
    assert "交通工具" not in terms

    taxi = expand_terms("出差可以打车吗")
    assert "二等座" in taxi and "经济舱" in taxi


def test_vehicle_group_does_not_drag_subsidy_section():
    """「市内交通」在 §2.2 和 §2.3 都有，不能当扩展词，否则会把补贴节一起拉上来。"""
    assert "市内交通" not in expand_terms("出差可以打车吗")
    assert "差旅补贴" not in expand_terms("出差可以打车吗")
