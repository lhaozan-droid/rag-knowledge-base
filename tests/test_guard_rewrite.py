from app.rag.guard import redact, redact_blocks
from app.rag.rewrite import needs_rewrite, rewrite_query


def test_redact_pii():
    raw = "值班电话 13812345678，邮箱 hr@example.com，身份证 110101199001011234"
    out, n = redact(raw)
    assert "13812345678" not in out
    assert "hr@example.com" not in out
    assert "110101199001011234" not in out
    assert n == 3
    assert out.count("【") == 3


def test_redact_keeps_normal_numbers():
    out, n = redact("年假 15 天，报销上限 600 元/晚，分机 8000")
    assert n == 0
    assert "15" in out and "600" in out


def test_redact_blocks():
    blocks = [{"index": 1, "text": "联系人 13900001111"}]
    out, n = redact_blocks(blocks)
    assert n == 1 and "13900001111" not in out[0]["text"]


HIST = [{"role": "user", "content": "年假有多少天？"},
        {"role": "assistant", "content": "年假 15 天 [1]"}]


def test_no_history_never_rewrites():
    assert needs_rewrite("那病假呢", []) is False


def test_pronoun_triggers_rewrite():
    assert needs_rewrite("那病假呢", HIST) is True
    assert needs_rewrite("这个流程需要谁审批", HIST) is True


def test_self_contained_question_skips_rewrite():
    assert needs_rewrite("出差住宿标准在一线城市是多少钱", HIST) is False


def test_short_but_complete_questions_skip_rewrite():
    """短不等于承接上文。

    中文问句天然就短，早期用 `len(q) <= 10` 判追问，控制台 19 个入口
    （6 条热门问题 + 13 个分类叶子）有 18 个被误判，会话里从第二问起检索串
    被拼成「上一问 / 本问」，走拼接回退的部署上实测多条名次变差。
    """
    for q in ("报销多久能到账", "年假多少天", "出差住宿费能报多少", "密码忘了"):
        assert needs_rewrite(q, HIST) is False, f"{q} 是独立问句，不该触发改写"


def test_elision_form_triggers_rewrite():
    """「那…呢」是中文最常见的省略追问，没有显式指代词也要认出来。"""
    for q in ("那病假呢", "那婚假呢", "那出差呢"):
        assert needs_rewrite(q, HIST) is True, f"{q} 是省略追问"


def test_all_console_entries_skip_rewrite():
    """控制台上每一个会发 query 的入口都必须是「独立问句」。

    早期 `len(q) <= 10` 把所有短问句都当追问，19 个入口里 18 个被误判；
    这里直接从 console.html 取入口（6 条热门问题 + 13 个分类叶子），
    并**先归一**再判——判据看到的是 `normalize_query` 之后的串，不是页面原文
    （「试用期多久、怎么转正？」原文 11 字，归一后 10 字，正好卡在旧判据线上）。
    前端增删入口时这个测试自动覆盖，不用改测试。
    """
    import pathlib
    import re

    import pytest

    from app.rag.engine import normalize_query

    html_path = pathlib.Path(__file__).resolve().parents[1] / "app" / "api" / "console.html"
    if not html_path.exists():  # pragma: no cover - 副本里没有前端资源时跳过
        pytest.skip("console.html 不存在，跳过入口回归")
    html = html_path.read_text(encoding="utf-8")
    hot = re.findall(r"\{q:'([^']+)'", html)
    tree = html.split("const KB_TREE")[1].split("];")[0]
    leaves = [m for m in re.findall(r"\{name:'([^']+)'\}", tree) if m not in _TREE_GROUPS]
    entries = hot + leaves
    if len(entries) < 10:  # pragma: no cover - 前端结构变了就别误报
        pytest.skip("未能从 console.html 解析出入口，跳过")

    bad = [q for q in entries if needs_rewrite(normalize_query(q), HIST)]
    assert not bad, f"这些入口是独立问句，不该被当成追问：{bad}"


_TREE_GROUPS = {"人力资源", "行政与后勤", "财务制度", "IT 与信息安全", "公司制度"}


def test_extractive_never_changes_query():
    """触发改写 ≠ 一定改变 query。

    离线摘录模式没有生成能力，拼接实测是净负收益，所以这里必须原样返回：
    真正要检索的就是用户问的那句。
    """
    class Extractive:
        name = "extractive"

    for q in ("那病假呢", "这个流程要多久", "报销多久能到账"):
        assert rewrite_query(q, HIST, Extractive()) == q


class BadLLM:
    name = "openai_compat"

    def chat(self, *a, **k):
        raise RuntimeError("upstream down")


def test_rewrite_falls_back_on_error():
    """模型改写失败时退回原问题，而不是拼上一问。

    旧行为是拼「上一问 / 本问」，实测拼接对检索是纯负收益（连真追问都变差），
    已经不做了。
    """
    out = rewrite_query("那病假呢", HIST, BadLLM())
    assert out == "那病假呢"
    assert "年假有多少天" not in out


class GoodLLM:
    name = "openai_compat"

    def chat(self, *a, **k):
        class R:
            text = "病假的申请材料和工资发放比例是什么？"
        return R()


def test_rewrite_uses_llm_output():
    out = rewrite_query("那病假呢", HIST, GoodLLM())
    assert out == "病假的申请材料和工资发放比例是什么？"


class _BudgetLLM:
    name = "openai_compat"

    def __init__(self, text="病假多少天"):
        self.kwargs = {}
        self.text = text

    def chat(self, messages, **kwargs):
        self.kwargs = kwargs
        return type("R", (), {"text": self.text, "finish_reason": "stop"})()


def test_rewrite_budget_leaves_room_for_reasoning():
    """改写调用的 max_tokens 必须给推理模型留够思考预算。

    回归点：旧值 120 只是「大多数时候失败、偶尔侥幸通过」——实测思考量在
    90~171 token 之间波动，120 下 4 次里 3 次 `finish_reason='length'`、正文空、
    静默回退原问题。这个数是「思考 + 一句正文」，不是「一句改写最多多少字」。
    """
    llm = _BudgetLLM()
    rewrite_query("那病假呢", HIST, llm)
    assert llm.kwargs.get("max_tokens", 0) >= 512, "改写预算给太小会被思考吃光"


# ---------------- 改写结果护栏：不许把上一答的内容抄进检索串 ----------------
# 实测缺陷：Q1「年假多少天」→ Q2「这个流程怎么走」，三次改写里两次把上一轮
# 回答里的「提前 3 个工作日」「连续休假超过 5 天」搬进了检索串，其中一次还被
# 改写成「…吗？」的是非问，最终答案以「结论：是的。」开头——在确认上一答的
# 前提。上一答若是错的，错的前提会先进检索串、再被检索结果"确认"一遍。

def test_rewrite_rejects_numbers_copied_from_previous_answer():
    llm = _BudgetLLM("年假申请流程需要提前 3 个工作日提交吗？")
    assert rewrite_query("这个流程怎么走", HIST, llm) == "这个流程怎么走"


def test_rewrite_rejects_chinese_numeral_leak():
    llm = _BudgetLLM("连续休假超过五天的年假申请流程怎么走")
    assert rewrite_query("这个流程怎么走", HIST, llm) == "这个流程怎么走"


def test_rewrite_keeps_numbers_that_came_from_the_question():
    """数字本来就在用户问题里，就不能当泄漏拦掉。"""
    hist = [{"role": "user", "content": "报销超过 5000 元要谁审批？"},
            {"role": "assistant", "content": "5000 元以上由部门负责人审批 [1]"}]
    llm = _BudgetLLM("报销超过 5000 元的审批流程怎么走")
    assert rewrite_query("这个流程怎么走", hist, llm) == "报销超过 5000 元的审批流程怎么走"


def test_rewrite_rejects_turning_into_yesno():
    llm = _BudgetLLM("年假申请流程是不是要提前提交？")
    assert rewrite_query("这个流程怎么走", HIST, llm) == "这个流程怎么走"


def test_rewrite_allows_yesno_when_question_already_yesno():
    hist = [{"role": "user", "content": "年假没休完能留到明年吗？"},
            {"role": "assistant", "content": "可以，但要在次年 3 月 31 日前用完 [1]"}]
    llm = _BudgetLLM("年假没休完能留到明年三月底吗？")
    assert rewrite_query("这个规定还生效吗", hist, llm) == "年假没休完能留到明年三月底吗？"


def test_rewrite_rejects_overlong_candidate():
    """超过 40 字的"改写"通常是在复述答案，不是补主语。"""
    llm = _BudgetLLM("年假申请的完整流程是先由员工在系统提交后经直属主管审批" * 3)
    assert rewrite_query("这个流程怎么走", HIST, llm) == "这个流程怎么走"


# ---------------- 引用角标不能被当成"抄来的数字" ----------------
# 回归现场：桩回「按制度执行 [1]」，角标里的 1 被判成"引入了用户没说过的数字"，
# 改写被拒、`test_multi_turn_uses_history` 直接红。真实场景里上一答几乎总带
# `[1] [2]`，这条误杀会把改写本身拦死——比它要防的缺陷更糟。

def test_citation_markers_are_not_numbers():
    r"""引用角标 `[1]` `【2】` `（3）` `(4)` 不是"事实"，剥掉再判。

    这张表就是验收表，四条都要真跑：
        '年假申请流程怎么走 [1]'            → 接受
        '年假申请流程怎么走 [1][2]'          → 接受
        '年假申请要提前 3 个工作日吗'        → 拒绝（裸数字 + 是非问）
        '年假申请流程怎么走'                 → 接受
    """
    history_with_cites = [
        {"role": "user", "content": "年假多少天"},
        {"role": "assistant", "content": "年假为 15 天 [1][2]"}]

    accepted = ("年假申请流程怎么走 [1]", "年假申请流程怎么走 [1][2]",
                "年假申请流程怎么走【1】", "年假申请流程怎么走（1）",
                "年假申请流程怎么走 (1)", "年假申请流程怎么走")
    for cand in accepted:
        assert rewrite_query("那病假呢", history_with_cites, _BudgetLLM(cand)) == cand, \
            f"{cand!r} 的角标不是事实，不该被拒"

    rejected = ("年假申请要提前 3 个工作日吗", "年假申请要提前 3 个工作日",
                "年假申请是不是要提前提交？")
    for cand in rejected:
        assert rewrite_query("那病假呢", history_with_cites, _BudgetLLM(cand)) == "那病假呢", \
            f"{cand!r} 引入了原问没有的事实，应该被拒"


def test_introduced_numbers_ignores_citation_brackets():
    """直接打函数：两种括号形式都剥干净。"""
    from app.rag.rewrite import _introduced_numbers

    hist = [{"role": "user", "content": "年假多少天"},
            {"role": "assistant", "content": "年假为 15 天 [1][2]"}]
    assert _introduced_numbers("那病假呢", "按制度执行 [1]", hist) == []
    assert _introduced_numbers("那病假呢", "年假申请流程怎么走 [2]", hist) == []
    assert _introduced_numbers("那病假呢", "按制度执行 【3】", hist) == []
    assert _introduced_numbers("那病假呢", "按制度执行（4）", hist) == []
    assert _introduced_numbers("那病假呢", "年假申请要提前 3 个工作日吗", hist) == ["3"]
    assert _introduced_numbers("那病假呢", "年假申请流程怎么走", hist) == []


def test_number_guard_still_fires_without_brackets():
    llm = _BudgetLLM("年假申请流程需要提前 3 个工作日提交")
    assert rewrite_query("这个流程怎么走", HIST, llm) == "这个流程怎么走"


def test_rewrite_prompt_does_not_carry_assistant_turns():
    """改写输入里不该出现 assistant 的回答——那是泄漏源，从源头不给。"""
    from app.llm.prompts import build_rewrite_prompt

    prompt = build_rewrite_prompt(HIST, "这个流程怎么走")
    assert "年假 15 天 [1]" not in prompt, "助手回答不该进改写提示词"
    assert "年假有多少天？" in prompt, "用户自己的上一问要在"



