"""管理后台：文档清单 / 单篇删除 / 入库报告。全部离线可跑。"""
import pytest


def test_list_docs_metadata_and_filter(engine):
    docs = engine.list_docs()
    assert len(docs) == 5
    keys = {"doc_id", "title", "path", "source_type", "char_count",
            "chunk_count", "status", "updated_at", "mtime", "meta"}
    for d in docs:
        assert keys <= set(d.keys())
        assert d["chunk_count"] > 0
        assert d["status"] == "active"
    # source_type 精确筛选：demo 库全是 markdown
    assert len(engine.list_docs(source_type="markdown")) == 5
    assert engine.list_docs(source_type="pdf") == []
    # q 关键字：命中标题
    by_title = engine.list_docs(q=docs[0]["title"][:4])
    assert any(d["doc_id"] == docs[0]["doc_id"] for d in by_title)
    # q 关键字：命中路径（"05-新员工" 只在 path 里，不在 title 里）
    by_path = engine.list_docs(q="05-新员工")
    assert len(by_path) == 1
    assert by_path[0]["path"].startswith("05-新员工")


def test_delete_doc_removes_chunks_and_vectors(engine):
    doc = engine.list_docs()[0]
    doc_id = doc["doc_id"]
    total_chunks = engine.registry.count_chunks_of_doc(doc_id)  # 含父块

    res = engine.delete_doc(doc_id)
    assert res["doc_id"] == doc_id
    assert res["chunks_deleted"] == total_chunks

    assert engine.registry.get_doc(doc_id) is None
    assert engine.registry.count_chunks_of_doc(doc_id) == 0
    # 向量库 / BM25 / 注册表三者数量重新一致（都只剩子块）
    assert engine.store.count() == engine.registry.count_chunks()
    assert len(engine.bm25) == engine.registry.count_chunks()
    assert len(engine.list_docs()) == 4

    # 重复删除 → 幂等失败而非静默成功
    with pytest.raises(KeyError):
        engine.delete_doc(doc_id)


def test_last_ingest_report(engine):
    rpt = engine.last_ingest_report()
    assert rpt is not None
    assert rpt["docs_total"] == 5
    assert rpt["chunks_total"] > 0
    assert engine.last_ingest_at() is not None
    assert engine.last_ingest_at() > 0
