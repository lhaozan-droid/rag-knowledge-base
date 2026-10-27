import pytest

from app.config import Config
from app.llm.client import ChatLLM, ChatResult
from app.rag.engine import RagEngine

DEMO = __import__("pathlib").Path(__file__).resolve().parents[1] / "demo-docs"


def make_cfg(tmp_path, **overrides):
    data = {
        "app": {"data_dir": str(tmp_path), "name": "测试公司"},
        "embedding": {"provider": "hash", "cache": True, "batch_size": 8},
        "vector_store": {"provider": "memory"},
        "llm": {"provider": "extractive"},
        "chunking": {"max_chars": 400, "overlap_chars": 60,
                     "min_chars": 40, "parent_max_chars": 1200},
        "retrieval": {"final_top_k": 6, "dense_top_k": 20, "sparse_top_k": 20,
                      "fusion_k": 15, "max_context_chars": 4000},
    }
    for k, v in overrides.items():
        data.setdefault(k, {}).update(v)
    return Config.from_dict(data)


class RecordingLLM(ChatLLM):
    """把收到的 messages / 上下文记下来，用于断言检索结果真的进了 Prompt。

    改写调用那一路**故意返回带 `[1]` 角标的内容**：真实模型改写时，历史里
    assistant 的答案本来就带角标，模型也常把它抄进改写结果。桩保持这个形态，
    这条测试才是在守「角标不算泄露」这个规则本身；要是让改写回一句干净的串，
    这个场景就从测试覆盖里消失了（护栏再退回去误杀也不会有人发现）。
    """

    name = "recording"

    def __init__(self, reply: str = "年假天数为 15 天 [1]",
                 rewrite: str = "年假申请流程怎么走 [1]"):
        self.reply = reply
        self.rewrite = rewrite
        self.messages = None
        self.blocks = None
        self.rewrite_calls = 0

    def chat(self, messages, context_blocks=None, query="", temperature=None,
             max_tokens=None):
        self.messages = list(messages)
        self.blocks = context_blocks
        is_rewrite = any(m.get("role") == "user" and "改写后的独立问题" in str(m.get("content", ""))
                         for m in messages)
        if is_rewrite:
            self.rewrite_calls += 1
        return ChatResult(text=self.rewrite if is_rewrite else self.reply,
                          usage={"prompt_tokens": 10, "completion_tokens": 5})


@pytest.fixture
def engine(tmp_path):
    eng = RagEngine(make_cfg(tmp_path))
    eng.ingest(DEMO, reset=True)
    yield eng
    eng.close()
