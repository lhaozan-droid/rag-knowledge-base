"""query 归一的单元测试。

只钉两件事：标点/空白被剥干净，词本身一个不动。
真实检索效果（dense 排名是否稳定）不在这里验，靠 A/B 实测。
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.rag.engine import normalize_query  # noqa: E402


def test_strip_trailing_question_mark():
    """全角问号必须剥掉——就是它把 dense 排名搞翻的。"""
    assert normalize_query("报销多久能到账？") == "报销多久能到账"


def test_strip_repeated_punctuation():
    assert normalize_query("报销多久能到账！！！") == "报销多久能到账"
    assert normalize_query("报销多久能到账？！。") == "报销多久能到账"


def test_strip_whitespace_and_wrap():
    assert normalize_query("  年假多少天  ") == "年假多少天"
    assert normalize_query("年假\t多少\n天") == "年假 多少 天"


def test_strip_book_title_marks():
    """《员工手册》？→ 员工手册：书名号对 BM25 是纯噪声。"""
    assert normalize_query("《员工手册》？") == "员工手册"


def test_fullwidth_letters_normalized():
    """NFKC 顺手把全角字母转半角，ＶＰＮ 要能当 VPN 命中。"""
    assert normalize_query("ＶＰＮ怎么申请？") == "VPN怎么申请"


def test_keep_words_and_order():
    """不删词、不改词序：只动首尾标点与空白。"""
    out = normalize_query("公司有班车吗，几点发车？")
    assert out == "公司有班车吗,几点发车"
    assert "班车" in out and out.index("班车") < out.index("发车")


def test_punctuation_only_query_falls_back():
    """只发一个「？」时回退原串，不能把空串送进 embedding。"""
    assert normalize_query("？") == "？"
    assert normalize_query("。。。") == "。。。"
    assert normalize_query("   ") == "   "


def test_empty_query_safe():
    assert normalize_query("") == ""


def test_plain_query_untouched():
    assert normalize_query("报销多久能到账") == "报销多久能到账"
