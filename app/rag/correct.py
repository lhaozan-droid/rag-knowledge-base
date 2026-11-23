"""查询纠错：解决「同音/形近错别字」导致的检索落空或跑偏。

要解决的问题（真实案例）：
    员工问「年佳怎么申请」，其实想问的是「年假」。
    系统按字面检索「年佳」，知识库里没有这个词（制度里写的是「年假 / 请假申请」），
    向量检索被这个生僻词带偏，top 命中全是「假期」「休息」这类泛内容，
    top1 完全跑偏，用户以为系统不懂。

为什么确定性词表 + LLM 纠错要一起用：
- 确定性词表：已知错字零成本、零风险、100% 命中，但覆盖不全（枚举不完所有错法）；
- LLM 纠错：能识别任意同音/形近错字，通用，但有「过度纠正」的风险，
  所以只在「检索分数可疑」时触发，且「纠错后显著变好才采纳」，否则退回原结果。

触发时机（由 engine 控制，本模块不管）：
    检索后 best_dense 低于 correct_trigger_score（默认 0.55，高于拒答阈值 0.48），
    说明「勉强命中但质量可疑」或「直接落空」，此时先纠错重检，再决定要不要拒答。
"""
from __future__ import annotations

import logging

log = logging.getLogger(__name__)

# 确定性纠错表：错写 → 正确。加一条立即生效，不用重训、不用重建索引。
# 只放「有十足把握」的错字，拿不准的别放——误改一个正确的专有名词比不改更糟。
# 每条映射的「正确词」都必须是 glossary.GROUPS 里的真实词条（且能在语料里检索到），
# 否则纠错只是把错字换成了另一个检索不到的写法，等于白纠。
CORRECTIONS = {
    "年佳": "年假",   # 同音：年佳 → 年假
    "报消": "报销",   # 同音：报消 → 报销
    "打咔": "打卡",   # 同音：打咔 → 打卡
    "转证": "转正",   # 同音：转证 → 转正
}

# 按长度倒序替换，避免长词被短词抢先切走
_ORDERED = sorted(CORRECTIONS.items(), key=lambda kv: -len(kv[0]))


def correct_query(query: str) -> str:
    """确定性纠错：把已知错字替换为正确写法。零成本，任何情况都先跑。"""
    q = (query or "").strip()
    for wrong, right in _ORDERED:
        if wrong in q:
            q = q.replace(wrong, right)
    return q


CORRECT_SYSTEM = """你是检索查询纠错器。任务：纠正用户查询中明显的错别字，输出纠正后的查询。

规则：
1. 只纠正「明显是错别字」的字词，尤其是同音字、形近字（如「年佳」应为「年假」）；
2. 专有名词、品牌、产品名原则上不改，除非你有十足把握它是错字；
3. 拿不准的地方保持原样，宁可少改，不要乱改；
4. 不要回答问题，不要解释，不要加引号；
5. 如果查询本身没有错别字，原样输出；
6. 只输出纠正后的这一句话。"""


def llm_correct_query(query: str, llm) -> str:
    """LLM 纠错（通用兜底）。失败或无法纠正时原样返回。"""
    if getattr(llm, "name", "") == "extractive":
        return query
    try:
        # 思考 token 与正文共用这个预算（与 rewrite.py 同一坑）：推理模型会先花
        # 思考 token，给 120 时长查询会全被思考吃光、正文返回空，本函数只会静默
        # 改回原查询——纠错功能等于没上线，界面上和「没写错字」完全一样。
        # 别按「一句纠错最多多少字」来配，要按「思考 + 一句正文」来配。
        res = llm.chat(
            [{"role": "system", "content": CORRECT_SYSTEM},
             {"role": "user", "content": f"请纠正下面查询中的错别字：\n{query}"}],
            temperature=0.0, max_tokens=512)
        out = (res.text or "").strip().strip("「」\"'`”’ ")
        out = out.split("\n")[0].strip()
        if out and out != query and 2 <= len(out) <= 120:
            log.debug("[correct] 纠错生效: %s -> %s", query, out)
            return out
        if not out:
            # 空正文几乎一定是预算被思考吃光，别让它和「本来就没写错字」混在一起
            log.warning("[correct] 纠错未生效: 返回空正文（finish_reason=%r），保持原查询 %r",
                        getattr(res, "finish_reason", ""), query)
        return query
    except Exception as exc:  # noqa: BLE001
        log.warning("[correct] 纠错未生效: 调用失败 %s，保持原查询 %r", exc, query)
        return query
