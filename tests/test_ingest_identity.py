"""文档身份与元数据透传的回归测试。

抓来的语料有三个特点，处理不好会静默出错：
1. 正文带 YAML front-matter —— 不剥离会把 `---` 和 `article_id:` 当正文入库；
2. 文件路径会随分类调整而变 —— 若路径仍是主键，改一次目录就产生一批孤儿记录；
3. 目录名是"来源 / 分类"而不是部门 —— 无脑推 dept 会污染权限过滤。
"""
from __future__ import annotations

from pathlib import Path

from app.ingest.loaders import load_file
from app.ingest.chunker import chunk_document
from app.models import Document
from app.rag.engine import RagEngine
from app.llm.prompts import format_blocks
from tests.conftest import make_cfg

FM_DOC = """---
title: 年假申请流程速览
article_id: demo-art-0001
source_url: https://kb.example.com/articles/demo-art-0001
content_type: image_only
category: 员工制度 / 请假与考勤 / 年假申请
published_at: 2026-08-10
---

# 年假申请流程速览

本条内容以图片形式发布，未提供文字版。
"""


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def test_front_matter_is_stripped_and_parsed(tmp_path):
    p = _write(tmp_path / "a.md", FM_DOC)
    loaded = load_file(p)

    assert "---" not in loaded.text, "front-matter 分隔符不能残留进正文"
    assert "article_id" not in loaded.text
    assert loaded.text.startswith("# 年假申请流程速览"), "正文应从标题开始"

    assert loaded.meta["article_id"] == "demo-art-0001"
    assert loaded.meta["content_type"] == "image_only"
    assert loaded.meta["source_url"].startswith("https://")
    assert loaded.meta["published_at"] == "2026-08-10"
    assert loaded.title == "年假申请流程速览"


def test_doc_id_prefers_article_id_over_path(tmp_path):
    """路径变了但 article_id 不变 → 仍是同一篇文档，不能新增也不能留孤儿。"""
    eng = RagEngine(make_cfg(tmp_path / "db"))
    kb = tmp_path / "kb"
    _write(kb / "员工制度/请假与考勤/demo-art-0001.md", FM_DOC)

    r1 = eng.ingest(kb, reset=True)
    assert r1.docs_total == 1
    doc1 = eng.registry.all_docs()[0]
    assert doc1.doc_id == "demo-art-0001", "身份应取 article_id，而不是路径哈希"

    # 分类调整：文件挪到另一层目录
    moved = kb / "员工制度/考勤管理/demo-art-0001.md"
    moved.parent.mkdir(parents=True, exist_ok=True)
    moved.write_text(FM_DOC, encoding="utf-8")
    (kb / "员工制度/请假与考勤/demo-art-0001.md").unlink()

    r2 = eng.ingest(kb)
    assert r2.docs_total == 1, "移动路径不应产生第二份文档"
    assert eng.registry.count_docs() == 1
    doc2 = eng.registry.all_docs()[0]
    assert doc2.doc_id == "demo-art-0001"
    assert "考勤管理" in doc2.path, "路径应被刷新为新位置"
    eng.close()


def test_dept_from_dir_can_be_disabled(tmp_path):
    eng = RagEngine(make_cfg(tmp_path / "db", ingest={"dept_from_dir": False}))
    kb = tmp_path / "kb"
    _write(kb / "制度/考勤管理/0001.md",
           "# 补卡与异常考勤处理细则\n\n制度正文，足够长以产生分块。" * 6)
    eng.ingest(kb, reset=True)

    ids = [cid for cid, _ in eng.registry.iter_children()]
    chunks = list(eng.registry.get_chunks(ids).values())
    assert chunks, "应产生子块"
    assert all("dept" not in c.meta for c in chunks), "关闭后不应写出 dept"
    eng.close()


def test_content_type_meta_reaches_chunks_and_prompt(tmp_path):
    """文档级 meta 要一路传到 chunk 和 Prompt 块里（content_type / source_url 是数据契约）。"""
    p = _write(tmp_path / "a.md", FM_DOC)
    loaded = load_file(p)
    doc = Document(doc_id=loaded.meta["article_id"], path="a.md", title=loaded.title,
                   source_type=loaded.source_type, content_hash="x",
                   mtime=0.0, char_count=len(loaded.text), meta=dict(loaded.meta))
    chunks = chunk_document(doc, loaded.text, make_cfg(tmp_path).chunking)

    assert chunks, "图片类模板正文必须够长以产生分块，否则索引里搜不到"
    assert all(c.meta.get("content_type") == "image_only" for c in chunks)
    assert all(c.meta.get("source_url") for c in chunks)

    text = format_blocks([{
        "index": 1, "title": doc.title, "section": "", "path": "a.md",
        "content_type": "image_only", "source_url": loaded.meta["source_url"],
        "text": loaded.text,
    }])
    assert "原文链接：https://kb.example.com" in text
    # Prompt 里不再注入内容形态标签：本地入库不产出这类枚举，
    # 塞进去只会让模型误判资料完整程度；但 content_type 仍留在块数据里供上层使用。
    assert "image_only" not in text
