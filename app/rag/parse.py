"""把自由 markdown 回答拆成「结论句 + 分条规则 + 补充说明」，供 /chat 结构化返回。

取舍说明：这里选择「生成后确定性解析」，而不是「让 LLM 直接输出 JSON」，
原因有三：
1. 离线摘录兜底（ExtractiveLLM）没有生成能力，只能从原文抽句子，产不出 JSON；
2. 判定层（judge）依赖 raw text 里的 [n] 引用编号和 NO_ANSWER 标记，
   改成 JSON 输出会让引用校验和拒答判定整条链路重写；
3. `text` 仍是权威完整回答，企微和任何读 `text` 的客户端零回归；
   `conclusion` / `rules` / `notes` 只是 `text` 的稳定视图，不增加 token、不引入幻觉。

因此本模块是纯函数、确定性、可单测，不触碰生成链路。
"""
from __future__ import annotations

import re
from typing import List, Tuple

# 列表行标记：- / *（排除 ** 加粗）/ • / ·，或 1. / 1) / 1、
_LIST_ITEM_RE = re.compile(r"^\s*(?:[-•·]|\*(?!\*)|(?:\d{1,2})[.)、])\s+(.+?)\s*$")
_HEADING_RE = re.compile(r"^\s*#{1,6}\s*")


def _strip_heading(line: str) -> str:
    return _HEADING_RE.sub("", line).strip()


def parse_answer(text: str) -> Tuple[str, List[str], List[str]]:
    """返回 (conclusion, rules, notes)。

    - conclusion：第一条列表之前的正文（结论句），去掉 # 标题符号；
    - rules：全部列表行，去掉列表标记，保留加粗与 [n] 引用编号；
    - notes：除结论句外的其它正文行（列表前的子标题 + 列表后的补充说明/时效提示），
      按原文顺序保留、去掉 # 标题符号——避免结构化渲染时丢掉二级指引。
    - 没有任何列表行时：rules/notes 为空、conclusion 取首行正文，
      调用方（前端）应据此降级为渲染完整 `text`。
    """
    text = (text or "").strip()
    if not text:
        return "", [], []

    lines = [ln.strip() for ln in text.splitlines()]
    conclusion = ""
    rules: List[str] = []
    notes: List[str] = []
    in_items = False

    for ln in lines:
        if not ln:
            continue
        m = _LIST_ITEM_RE.match(ln)
        if m:
            in_items = True
            rules.append(m.group(1).strip())
        elif not in_items and not conclusion:
            # 只有列表之前的第一条正文才算结论
            conclusion = _strip_heading(ln)
        else:
            # 结论句之后的正文（列表前子标题、列表后补充说明）一律进 notes，不丢
            notes.append(_strip_heading(ln))

    if not rules:
        return conclusion or (_strip_heading(lines[0]) if lines else ""), [], []
    return conclusion, rules, notes
