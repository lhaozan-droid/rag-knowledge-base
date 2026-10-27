"""大模型调用层：OpenAI 兼容协议 + 重试 + 流式 + 离线摘录兜底。

混合模式下，这里只接收「已经脱敏的检索片段」，不接收原始文档。
"""
from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional, Sequence

import httpx

log = logging.getLogger(__name__)

Message = Dict[str, str]


@dataclass
class ChatResult:
    text: str
    model: str = ""
    usage: Dict[str, int] = field(default_factory=dict)
    finish_reason: str = ""


class LLMError(RuntimeError):
    """上游模型不可用。引擎捕获后走兜底回答，不把 500 抛给用户。"""


class ChatLLM:
    name = "base"

    def chat(self, messages: Sequence[Message], context_blocks: Optional[List[dict]] = None,
             query: str = "", temperature: Optional[float] = None,
             max_tokens: Optional[int] = None) -> ChatResult:
        raise NotImplementedError

    def stream(self, messages: Sequence[Message], context_blocks: Optional[List[dict]] = None,
               query: str = "", temperature: Optional[float] = None,
               max_tokens: Optional[int] = None) -> Iterator[str]:
        yield self.chat(messages, context_blocks, query, temperature, max_tokens).text

    def close(self) -> None:
        pass


class OpenAICompatLLM(ChatLLM):
    """兼容 DeepSeek / 通义 / 混元 / vLLM / One-API / Ollama(OpenAI 模式)。"""

    name = "openai_compat"

    def __init__(self, base_url: str, api_key: str, model: str,
                 temperature: float = 0.1, max_tokens: int = 1200,
                 timeout_s: float = 60.0, max_retries: int = 2):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max(0, max_retries)
        self._client = httpx.Client(timeout=timeout_s)

    @property
    def _url(self) -> str:
        return f"{self.base_url}/chat/completions"

    @property
    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"} if self.api_key else \
               {"Content-Type": "application/json"}

    def _payload(self, messages, temperature, max_tokens, stream) -> dict:
        return {
            "model": self.model,
            "messages": list(messages),
            "temperature": self.temperature if temperature is None else temperature,
            "max_tokens": self.max_tokens if max_tokens is None else max_tokens,
            "stream": stream,
        }

    def chat(self, messages, context_blocks=None, query="", temperature=None,
             max_tokens=None) -> ChatResult:
        payload = self._payload(messages, temperature, max_tokens, False)
        last: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                r = self._client.post(self._url, json=payload, headers=self._headers)
                if r.status_code == 429 or r.status_code >= 500:
                    raise httpx.HTTPStatusError(f"HTTP {r.status_code}: {r.text[:200]}",
                                                request=r.request, response=r)
                if r.status_code >= 400:
                    # 4xx（鉴权/参数错误）重试没意义，直接暴露
                    raise LLMError(f"模型接口返回 {r.status_code}: {r.text[:300]}")
                data = r.json()
                choice = (data.get("choices") or [{}])[0]
                return ChatResult(
                    text=(choice.get("message") or {}).get("content", "") or "",
                    model=data.get("model", self.model),
                    usage=data.get("usage") or {},
                    finish_reason=choice.get("finish_reason", "") or "")
            except (httpx.TimeoutException, httpx.TransportError, httpx.HTTPStatusError) as exc:
                last = exc
                if attempt < self.max_retries:
                    wait = min(8.0, 0.8 * (2 ** attempt))
                    log.warning("模型调用失败（第 %d 次），%.1fs 后重试: %s",
                                attempt + 1, wait, exc)
                    time.sleep(wait)
        raise LLMError(f"调用大模型失败: {last}")

    def stream(self, messages, context_blocks=None, query="", temperature=None,
               max_tokens=None) -> Iterator[str]:
        payload = self._payload(messages, temperature, max_tokens, True)
        try:
            with self._client.stream("POST", self._url, json=payload,
                                     headers=self._headers) as r:
                if r.status_code >= 400:
                    body = r.read().decode("utf-8", "ignore")[:300]
                    raise LLMError(f"模型流式接口返回 {r.status_code}: {body}")
                for line in r.iter_lines():
                    if not line or not line.startswith("data:"):
                        continue
                    chunk = line[5:].strip()
                    if chunk == "[DONE]":
                        break
                    try:
                        data = json.loads(chunk)
                    except json.JSONDecodeError:
                        continue
                    delta = ((data.get("choices") or [{}])[0].get("delta") or {})
                    piece = delta.get("content")
                    if piece:
                        yield piece
        except httpx.HTTPError as exc:
            raise LLMError(f"流式调用失败: {exc}") from exc

    def close(self) -> None:
        self._client.close()


class ExtractiveLLM(ChatLLM):
    """零依赖兜底：不生成，只从检索片段里抽最相关的句子并标注出处。

    价值有两个：
    1. 没有 API Key 也能把整条链路（入库→检索→引用）跑通做演示；
    2. 云端模型超时/限流时，用户至少还能拿到原文依据，而不是一个 500。
    """

    name = "extractive"

    def chat(self, messages, context_blocks=None, query="", temperature=None,
             max_tokens=None) -> ChatResult:
        blocks = context_blocks or []
        q = query or (messages[-1]["content"][-200:] if messages else "")
        if not blocks:
            return ChatResult(text="（离线模式）没有检索到可用资料。", finish_reason="extractive")

        qgrams = _bigrams(q)
        # IDF 加权：像"上海""600"这种只在一两个片段里出现的词，才是真正的区分信号；
        # "公司""制度"这种到处都有的词，权重应该压低。
        df: Dict[str, int] = {}
        for b in blocks:
            for g in qgrams & _bigrams(b.get("text", "")):
                df[g] = df.get(g, 0) + 1

        def weight(g: str) -> float:
            return 1.0 / df.get(g, 1)

        picked = []
        for b in blocks:
            for sent in _split_sentences(b.get("text", "")):
                if len(sent) < 6:
                    continue
                sent_grams = _bigrams(sent)
                matched = qgrams & sent_grams
                if not matched:
                    continue
                score = sum(weight(g) for g in matched) / (len(qgrams) ** 0.5 + 1e-6)
                picked.append((score, b.get("index"), b.get("title", ""),
                               b.get("section", ""), sent))
        picked.sort(key=lambda x: -x[0])
        top = picked[:4] or picked[:1]

        if not top:
            return ChatResult(text="（离线模式）检索到的内容过短，无法摘录。",
                              finish_reason="extractive")

        lines = ["（当前为离线摘录模式：未接入生成模型，仅陈列最相关的原文句子）", ""]
        for _, idx, title, section, sent in top:
            loc = f"《{title}》" + (f" / {section}" if section and section != title else "")
            lines.append(f"- {sent.strip()} —— {loc} [{idx}]")
        # 落款说「多出来的是什么」，不说「该改哪个配置项」：
        # 不配 Key 是默认状态，每个读者第一次跑起来都会看到这一行，
        # 出现 llm.provider / openai_compat 这种内部术语等于先把门槛立在第一屏。
        lines += ["", "接入大模型后，同样的检索结果会被整合成一段完整的、带引用的回答。"]
        return ChatResult(text="\n".join(lines), finish_reason="extractive",
                          usage={"note": 0})


def _bigrams(s: str) -> set:
    s = re.sub(r"\s+", "", s)
    return {s[i:i + 2] for i in range(max(0, len(s) - 1))}


def _split_sentences(text: str) -> List[str]:
    return [x.strip() for x in re.split(r"[。！？；\n]", text) if x.strip()]


def build_llm(cfg) -> ChatLLM:
    c = cfg.llm
    # 措辞明确「只影响问答」：这行在 `ingest` 时也会打（引擎构造就会建 LLM），
    # 而入库全程不调生成模型。写成 "LLM: xxx" 会让首跑用户以为必须先配 Key。
    if c.provider == "openai_compat" and c.base_url and c.model:
        log.info("问答模型: %s @ %s（仅用于生成回答，入库不需要它）",
                 c.model, c.base_url)
        return OpenAICompatLLM(c.base_url, c.api_key, c.model, c.temperature,
                               c.max_tokens, c.timeout_s, c.max_retries)
    if c.provider not in ("extractive", ""):
        log.warning("llm.provider=%s 缺少 base_url/model，降级为离线摘录模式", c.provider)
    else:
        log.info("问答模型: 离线摘录模式（未配置生成模型；只影响问答，入库不受影响）")
    return ExtractiveLLM()
