"""答案判定：同步与流式共用同一套判据。

为什么单独抽一层：原来 answer() 和 answer_stream() 各写一套判断，
流式那套漏了阈值、漏了引用校验、漏了未答记录，导致「调试台看到的」和
「接口返回的」不是同一件事。这类不一致不报错，只会悄悄骗人。

判定的三类结果，必须能分辨：
- no_hit           检索没命中（最外层闸门）
- model_refused    模型自述资料不足 / 越界拒答
- citation_missing 答了但没标引用（说明分块或 Prompt 有问题）
- ok               正常回答

前两种进未答清单（知识缺口），第三种只做标记不进清单：
no_hit 说明知识缺口，model_refused 说明文档表述和员工问法对不上，
citation_missing 说明工程环节有毛病——它不该被当成"知识库里没有"。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

# 让模型用一个固定标记表达「拒绝作答」，而不是靠字符串匹配去猜。
# 猜法很脆：模型换个说法（"本制度未涉及" / "无法确认"）就漏判。
REFUSAL_MARK = "NO_ANSWER"

_REFUSAL_HEAD = re.compile(r"^\s*[\[【(\s]*NO[_ ]?ANSWER[\]】)\s:：]*", re.IGNORECASE)
_CITE_RE = re.compile(r"\[(\d{1,2})\]")

# 兜底启发式：模型没按格式输出标记时，用它兜住明显拒答的短回答。
# 只在「没有引用」时启用，避免把正常回答误判成拒答。
_REFUSAL_HINTS = (
    "资料中未提及", "资料未提及", "资料中没有", "资料中不包含", "未提及",
    "无法回答", "无法确定", "无法确认", "没有找到", "未找到相关",
    "不在我的职责范围", "仅能回答", "超出我的",
)
_REFUSAL_MAX_LEN = 160


@dataclass
class Verdict:
    """一次回答的判定结果。answer() 和 answer_stream() 都产出这个。"""

    refused: bool = False
    citations: List[int] = field(default_factory=list)
    answered: bool = False
    reason: Optional[str] = None          # no_hit / model_refused / citation_missing
    text: str = ""                        # 去掉标记后的正文
    flags: Dict[str, Any] = field(default_factory=dict)

    @property
    def should_log_unanswered(self) -> bool:
        """是否进「未答清单」。

        只收 no_hit 和 model_refused：这两类才对应知识缺口，是补文档的依据。
        citation_missing 是工程问题（该标引用没标），混进来会让盲区清单失真——
        实际有答案的问题也出现在「知识库没有」的列表里，照着补文档就补错方向了。
        """
        return self.reason in ("no_hit", "model_refused")


def _valid_citations(text: str, allowed: Optional[Sequence[int]]) -> List[int]:
    found = {int(m) for m in _CITE_RE.findall(text)}
    if allowed is not None:
        found &= {int(i) for i in allowed}
    return sorted(found)


def looks_like_refusal_head(head: str) -> bool:
    """流式专用：只拿到开头一小段就能判断是不是拒答。

    之所以能这么早判断：我们要求模型把标记放在第一行。
    """
    return bool(_REFUSAL_HEAD.match(head)) or head.strip().upper().startswith("NO")


def strip_refusal_mark(text: str) -> str:
    body = _REFUSAL_HEAD.sub("", text, count=1).strip()
    body = body.lstrip("，,。.：: \n\t")
    return body.strip()


def judge(text: str, allowed_indices: Optional[Sequence[int]],
          require_citation: bool = True,
          min_chars_for_refusal_check: int = _REFUSAL_MAX_LEN) -> Verdict:
    """判定一次生成结果。纯函数，无副作用，好测。

    allowed_indices 传当前 sources 的编号集合：模型引用了不存在的编号会被过滤掉。
    """
    raw = (text or "").strip()
    citations = _valid_citations(raw, allowed_indices)

    marked = looks_like_refusal_head(raw)
    heuristic = (not marked and not citations and len(raw) <= min_chars_for_refusal_check
                 and any(h in raw for h in _REFUSAL_HINTS))

    if marked or heuristic:
        body = strip_refusal_mark(raw) if marked else raw
        # 拒答文本里不该出现引用编号：否则员工看到的是
        # 「资料中没有提到 XX [1][2]」——一边说没有，一边给出来源，自相矛盾。
        body = _CITE_RE.sub("", body).strip()
        body = re.sub(r"[，,]\s*([。.])", r"\1", body)
        body = re.sub(r"\s{2,}", " ", body).strip()
        return Verdict(refused=True, citations=[], answered=False,
                       reason="model_refused", text=body,
                       flags={"refusal": "marked" if marked else "heuristic"})

    if require_citation and not citations:
        return Verdict(refused=False, citations=[], answered=False,
                       reason="citation_missing", text=raw,
                       flags={"refusal": None})

    return Verdict(refused=False, citations=citations, answered=True,
                   reason=None, text=raw, flags={"refusal": None})
