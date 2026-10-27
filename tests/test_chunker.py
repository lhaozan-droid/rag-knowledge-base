from app.config import ChunkingConfig
from app.ingest.chunker import chunk_document, embedding_text
from app.models import Document

CFG = ChunkingConfig(max_chars=100, overlap_chars=20, min_chars=30,
                     parent_max_chars=300)


def doc(title="测试文档"):
    return Document(doc_id="d1", path="a.md", title=title,
                    source_type="markdown", content_hash="h", mtime=0.0)


def test_heading_path_is_preserved():
    text = "# 员工手册\n\n## 第三章 年假\n\n### 3.2 申请流程\n年假需提前 3 个工作日提交。\n"
    chunks = chunk_document(doc(), text, CFG)
    sections = {c.section for c in chunks}
    assert "员工手册 > 第三章 年假 > 3.2 申请流程" in sections


def test_parent_child_relation():
    text = "# A\n\n第一段内容比较短。\n\n第二段内容也比较短。\n"
    chunks = chunk_document(doc(), text, CFG)
    parents = [c for c in chunks if c.is_parent == 1]
    children = [c for c in chunks if c.is_parent == 0]
    assert parents and children
    parent_ids = {p.chunk_id for p in parents}
    assert all(c.parent_id in parent_ids for c in children)


def test_long_section_splits_with_overlap():
    body = "这是一句用来测试分块的中文句子，长度适中。" * 40
    text = "# 长章节\n\n" + body
    chunks = chunk_document(doc(), text, CFG)
    children = [c for c in chunks if c.is_parent == 0]
    assert len(children) > 3
    assert all(len(c.text) <= CFG.max_chars + CFG.overlap_chars for c in children)
    # 有重叠会导致子块文本总量大于原文
    assert sum(len(c.text) for c in children) > len(body)


def test_no_heading_document_still_chunks():
    text = "一开始就没有标题的纯文本内容。" * 30
    chunks = chunk_document(doc("无标题"), text, CFG)
    assert chunks
    assert all(c.section == "无标题" for c in chunks)


def test_empty_document():
    assert chunk_document(doc(), "   \n\n ", CFG) == []


def test_embedding_text_contains_title_and_section():
    text = "# 员工手册\n\n## 年假\n\n年假 15 天。\n"
    chunks = chunk_document(doc("员工手册"), text, CFG)
    child = [c for c in chunks if c.is_parent == 0][0]
    et = embedding_text(child)
    assert "员工手册" in et
    assert "年假 15 天" in et
