"""入模前的护栏：PII 脱敏。

混合模式最容易出事的地方：文档在本地，但检索片段要发给云端大模型。
员工手册里常有手机号、身份证号、银行账号，直接外发就是合规事故。
这里在拼上下文之前做一次正则脱敏，宁可让模型看到【手机号】也不让原文出网。
"""
from __future__ import annotations

import re
from typing import List, Tuple

PATTERNS: List[Tuple[re.Pattern, str]] = [
    (re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)"), "【手机号】"),
    (re.compile(r"(?<!\d)\d{17}[\dXx](?!\d)"), "【身份证号】"),
    (re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"), "【邮箱】"),
    (re.compile(r"(?<!\d)\d{16,19}(?!\d)"), "【银行卡号】"),
]


def redact(text: str) -> Tuple[str, int]:
    total = 0
    for pat, repl in PATTERNS:
        text, n = pat.subn(repl, text)
        total += n
    return text, total


def redact_blocks(blocks: List[dict]) -> Tuple[List[dict], int]:
    out, total = [], 0
    for b in blocks:
        text, n = redact(b.get("text", ""))
        total += n
        out.append({**b, "text": text})
    return out, total
