"""多轮追问的查询改写。

不做改写的后果：员工问「年假怎么算」→「那病假呢」，
检索器只拿到「那病假呢」，向量和 BM25 都会跑偏，命中一堆无关片段。

判据只有两条（成本敏感）：
- 没有历史 → 不改写，零成本；
- 命中指代词（这个 / 它 / 上述 / 第X步 …）或中文省略追问的固定形态「那…呢」
  → 认为承接上文，值得改写一次；其余**一律不改写**。

同一批 19 个入口回算：旧判据命中 18 个，现判据命中 0 个；而真追问
（「那病假呢」「这个流程怎么走」「它要多久」「第2步是什么」）仍全部触发。

为什么**不用长度**当判据：中文问句天然就短。「报销多久能到账」7 个字、
「年假多少天」5 个字，但它们自带主体和疑问焦点，是完全独立的问句。
早期版本用 `len(q) <= 10` 判「极短追问」，控制台里 19 个入口（6 条热门问题
+ 13 个分类叶子）有 18 个落在这条判据里被误判成追问——只剩
「年假没休完能留到明年吗？」（去掉句末问号仍 11 字）不受影响。
会话里从第二问起检索串被拼成「上一问 / 本问」，在调用拼接回退的部署上
（离线摘录模式，或改写调用返回空）实测多条名次变差。
长度不是"是否承接上文"的信号。

注意判据看到的**不是**页面上的原文：`engine.answer()` 先跑 `normalize_query`
（去句末标点、全半角归一），再调这里。所以「试用期多久、怎么转正？」
原文 11 字、进判据是 10 字，正好卡在旧判据的 10 字线上——复算入口数时
必须先归一，否则会数成 17/19。

改写是要花钱的：只有命中上面两条判据时才会多一次模型调用。
另外注意 **`needs_rewrite()` 返回 True 不等于一定会改变检索输入**——
离线摘录模式（extractive）拿不到生成能力，做不了真正的指代消解，直接按原问题
检索（理由见 `rewrite_query`）。
"""
from __future__ import annotations

import logging
import re
from typing import Dict, List

from app.llm.prompts import REWRITE_SYSTEM, build_rewrite_prompt

log = logging.getLogger(__name__)

PRONOUN_RE = re.compile(
    r"(这个|那个|它|它们|上面|上述|刚才|前面|前面说的|该制度|该规定|这条|那条|"
    r"其中|这里的|那里的|这个流程|这个规定|这个标准|同样的|还有呢|然后呢|"
    r"第[一二三四五六七八九十\d]+步|继续|再详细|具体点|举例)")
# 「那…呢」是中文里最短的省略追问（「那病假呢」「那婚假呢」）：
# 没有显式指代词，但「那」+ 句尾「呢」这个框式结构几乎只用于承接上文。
# 长度上限 8 是防误伤——「那么远的地方呢」这类长句不该按追问处理。
ELISION_RE = re.compile(r"^那.{0,8}呢$")

# ---- 改写结果的确定性护栏 -------------------------------------------------
# 改写最大的风险不是"改得不全"，是"改多了"：模型会把上一轮**回答**里的事实和
# 数字抄进检索串。实测（Q1「年假多少天」→ Q2「这个流程怎么走」）三次改写里
# 有两次把「提前 3 个工作日」「连续休假超过 5 天」搬了进来，后果有两个：
#   1. 锚定附和：第 1 次被改写成「…需要提前 3 个工作日…吗？」的是非问，
#      最终答案开头就是「结论：是的。」——模型在确认上一答的前提。上一答若是
#      错的，错的前提会先进检索串、再被检索结果"确认"一遍，比不消解更危险；
#   2. 指标失真：带答案内容的改写 best_dense 比干净问法高 0.16~0.18，
#      任何用改写后串跑的阈值标定/召回评测都不可比。
# 下面三条护栏都是纯确定性判断，不额外调模型：
#   a. 不许出现**用户从没说过的**数字（本轮问题 + 历史里 user 轮为白名单；
#      含「三天」这类中文数字 + 单位）。用户自己说过的数字放行——那是消解，
#      不是抄袭；
#   b. 不许把非是非问改成「…吗？」；
#   c. 长度上限收到 40 字——一句"只补主语"的改写通常 10~25 字。
_NUM_RE = re.compile(
    r"\d+(?:\.\d+)?"
    r"|[一二三四五六七八九十百千万两]+(?=天|个工作日|工作日|个月|周|元|次|小时|分钟)")

# 引用角标不是"数字"，是答案的标注：`[1] [2]`（半角方括号）、`【1】`（全角方括号）、
# `（1）`（全角括号）、`(1)`（半角括号）。必须先剥掉再查数字，否则 `按制度执行 [1]`
# 会被判成"引入了用户没说过的数字 1"——而真实场景里上一答**几乎总带角标**，模型也常
# 把它抄进改写结果，这条护栏就会在线上频繁误杀、把改写本身拦死（回归现场：
# `test_multi_turn_uses_history` 直接红）。
_CITE_RE = re.compile(r"\[\s*\d+\s*\]|【\s*\d+\s*】|（\s*\d+\s*）|\(\s*\d+\s*\)")

_YESNO_MARKS = ("吗", "是不是", "对不对", "行不行", "要不要")


def _strip_citations(text: str) -> str:
    """剥掉引用角标，只留正文。改写结果不该带角标，但带也不该被判成"抄数字"。"""
    return _CITE_RE.sub("", text or "")


def _numbers(text: str) -> set:
    return set(_NUM_RE.findall(_strip_citations(text)))


def _user_said_numbers(query: str, history: List[Dict[str, str]]) -> set:
    """用户自己说过的数字（本轮问题 + 历史里 user 轮）。

    这些数字出现在改写结果里是合理的——用户上一问提过「5000 元」，这一问
    「那这个流程怎么走」补全成「5000 元以上的报销流程怎么走」不算抄答案。
    """
    known = _numbers(query)
    for t in history or []:
        if t.get("role") == "user":
            known |= _numbers(t.get("content", ""))
    return known


def _introduced_numbers(query: str, out: str, history: List[Dict[str, str]]) -> list:
    """改写结果里出现了、但用户从没说过的数字——只可能是从上一答抄来的。"""
    known = _user_said_numbers(query, history)
    return [n for n in _numbers(out) if n not in known]


def _turned_into_yesno(query: str, out: str) -> bool:
    """原问不是是非问，改写却变成了「…吗？」。"""
    if any(m in (query or "") for m in _YESNO_MARKS):
        return False
    return any(m in (out or "") for m in _YESNO_MARKS)


def needs_rewrite(query: str, history: List[Dict[str, str]]) -> bool:
    """这一问是否值得做一次改写。

    返回 True 只表示「配了生成模型时值得改写」，不表示检索输入一定会变：
    extractive 模式下改写不了，仍然用原问题检索。
    """
    if not history:
        return False
    q = (query or "").strip()
    if not q:
        return False
    # 指代词优先：即使同时含疑问焦点也必须改写（「这个流程怎么走」）
    if PRONOUN_RE.search(q) or ELISION_RE.match(q):
        return True
    return False


def rewrite_query(query: str, history: List[Dict[str, str]], llm) -> str:
    if not needs_rewrite(query, history):
        return query
    if getattr(llm, "name", "") == "extractive":
        # 离线摘录模式没有生成能力，做不了真正的指代消解。
        # 早期这里用「上一问 / 本问」拼接来"近似"消解，实测是纯负收益：
        # 拼出来的串既不是用户的问题、也不是一个语义单元，上一问的主题强行
        # 占了一半向量权重。此前的 14 条对照（独立问句 + 真追问）里没有一例
        # 拼接优于单发，连「那病假呢」的名次都往后退（如「报销」那一轮的 top1
        # 从正确条款跑到 §3.1 报销时限）。
        # 所以宁可不解消，直接按原问题检索——这是诚实的降级，不是缩水。
        #
        # 注意这个判断靠 `name` 字符串，不是能力探测。将来若新增别的离线 provider
        # 而 name 不叫 extractive，会走到下面的 llm.chat()——离线对象没有这个方法，
        # 抛 AttributeError 被 except 捕获、返回原问题，行为仍然安全，只是日志里
        # 会多一条「改写未生效」，所以不影响正确性，改 provider 时看一眼日志即可。
        return query
    try:
        # 思考 token 与正文共用这个预算：推理模型（deepseek-v4-flash 等）先花思考
        # token，再吐正文。同一条 prompt 实测思考量在 90~171 token 之间波动，给 120
        # 时 4 次里 3 次 finish_reason='length'、text=''（剩下 1 次以 98/120 侥幸通过），
        # 而本函数只会静默回退原问题——功能看着像"上线了"，其实大部分时候没生效，
        # 日志也只是一行 warning。所以别按「一句改写最多多少字」来配这个数，
        # 要按「思考 + 一句正文」留出几倍余量。
        res = llm.chat(
            [{"role": "system", "content": REWRITE_SYSTEM},
             {"role": "user", "content": build_rewrite_prompt(history, query)}],
            temperature=0.0, max_tokens=512)
        out = (res.text or "").strip().strip("「」\"'`”’ ")
        out = out.split("\n")[0].strip()
        # 护栏顺序：先看"改得对不对"（抄答案 / 变是非问），再看"像不像一句话"。
        # 命中任何一条都退回原问题——宁可不解消，也不把上一答的前提带进检索。
        reason = ""
        if not out or not (2 <= len(out) <= 40):
            reason = f"长度不合（{len(out)} 字）"
        elif leaked := _introduced_numbers(query, out, history):
            reason = f"引入了用户没说过的数字 {leaked}（可能抄自上一轮回答）"
        elif _turned_into_yesno(query, out):
            reason = "把原问改成了是非问"
        if not reason:
            if out == query:
                # 模型判定原问题已完整、原样返回：不是失败，也别打成"改写生效"
                log.debug("[rewrite] 无需改写: %r", query)
                return query
            log.debug("[rewrite] 改写生效: %s -> %s", query, out)
            return out
        log.warning("[rewrite] 改写未生效: %s，退回原问题 %r（候选 %r，finish_reason=%r）",
                    reason, query, out, getattr(res, "finish_reason", ""))
    except Exception as exc:  # noqa: BLE001
        log.warning("[rewrite] 改写未生效: 调用失败 %s，退回原问题 %r", exc, query)
    return query
