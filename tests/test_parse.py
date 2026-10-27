"""结构化拆分 parse_answer 的纯函数单测：结论句 + 分条规则 + 补充说明。"""
from app.rag.parse import parse_answer


def test_markdown_list_splits_conclusion_and_rules():
    text = (
        "年假为 15 天，入职满 1 年即可享受 [1]。\n\n"
        "1. 累计工作满 1 年不满 10 年的，年休假 5 天\n"
        "2. 满 10 年不满 20 年的，年休假 10 天\n"
        "- 国家法定节假日不计入年休假\n"
    )
    conclusion, rules, notes = parse_answer(text)
    assert conclusion == "年假为 15 天，入职满 1 年即可享受 [1]。"
    assert rules == [
        "累计工作满 1 年不满 10 年的，年休假 5 天",
        "满 10 年不满 20 年的，年休假 10 天",
        "国家法定节假日不计入年休假",
    ]
    assert notes == []


def test_dash_and_star_bullets_strip_markers():
    text = "结论句。\n- 第一条\n* 第二条\n• 第三条"
    conclusion, rules, notes = parse_answer(text)
    assert conclusion == "结论句。"
    assert rules == ["第一条", "第二条", "第三条"]
    assert notes == []


def test_bold_line_is_not_a_bullet():
    text = "**加粗结论**\n1. 规则项"
    conclusion, rules, notes = parse_answer(text)
    assert conclusion == "**加粗结论**"
    assert rules == ["规则项"]
    assert notes == []


def test_no_list_returns_conclusion_only():
    text = "仅退款需在售后入口发起，审核通过后 3-5 个工作日到账。"
    conclusion, rules, notes = parse_answer(text)
    assert conclusion == text
    assert rules == []
    assert notes == []


def test_empty_text():
    assert parse_answer("") == ("", [], [])
    assert parse_answer("   \n  ") == ("", [], [])


def test_list_first_leaves_conclusion_empty():
    text = "- 第一步\n- 第二步"
    conclusion, rules, notes = parse_answer(text)
    assert conclusion == ""
    assert rules == ["第一步", "第二步"]
    assert notes == []


def test_heading_marker_stripped_from_conclusion():
    text = "## 结论\n年假 15 天。\n1. 规则一"
    conclusion, rules, notes = parse_answer(text)
    assert conclusion == "结论"
    assert rules == ["规则一"]
    # 结论句之后的正文（无论列表前后）都不丢，进入 notes
    assert notes == ["年假 15 天。"]


def test_trailing_prose_goes_to_notes():
    text = (
        "违规处罚自公示之日起生效 [1]。\n\n"
        "1. 首次违规：警告\n"
        "2. 再次违规：限流 7 天\n\n"
        "如果不认可处罚，可在 7 天内发起申诉 [2]。"
    )
    conclusion, rules, notes = parse_answer(text)
    assert conclusion == "违规处罚自公示之日起生效 [1]。"
    assert rules == ["首次违规：警告", "再次违规：限流 7 天"]
    assert notes == ["如果不认可处罚，可在 7 天内发起申诉 [2]。"]


def test_subheading_and_trailing_note_preserved():
    text = (
        "退款流程：在售后入口发起，审核通过后 3-5 个工作日到账 [1]。\n\n"
        "操作步骤：\n"
        "1. 打开订单详情页\n"
        "2. 选择\"仅退款\"\n\n"
        "补充说明：金额以实际到账为准，如有疑问联系平台客服。"
    )
    conclusion, rules, notes = parse_answer(text)
    assert conclusion == "退款流程：在售后入口发起，审核通过后 3-5 个工作日到账 [1]。"
    assert rules == ["打开订单详情页", "选择\"仅退款\""]
    assert notes == ["操作步骤：", "补充说明：金额以实际到账为准，如有疑问联系平台客服。"]


def test_extractive_fallback_shape():
    text = (
        "（当前为离线摘录模式：未接入生成模型，仅陈列最相关的原文句子）\n\n"
        "- 年假天数为 15 天 —— 《员工手册》 / 第3章 休假 [1]\n"
        "- 病假需提供医院证明 —— 《员工手册》 [2]\n\n"
        "接入大模型后，同样的检索结果会被整合成一段完整的、带引用的回答。"
    )
    conclusion, rules, notes = parse_answer(text)
    assert conclusion.startswith("（当前为离线摘录模式")
    assert len(rules) == 2
    assert "[1]" in rules[0] and "[2]" in rules[1]
    # 列表之后的落款/提示进入 notes，不再丢失，也不混进结论
    assert "接入大模型" not in conclusion
    assert any("接入大模型后" in n for n in notes)
