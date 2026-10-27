"""知识库分类：多级分类树、归类、按分类筛选与分页。全部离线可跑。"""
import pytest


def _tree(engine):
    """返回 {分类名: id} 便于断言，只取一级/二级。"""
    return {c["name"]: c["id"] for c in engine.list_categories()}


def test_create_category_tree_and_reject_duplicate(engine):
    root = engine.create_category("员工制度")
    sub = engine.create_category("请假与考勤", root)
    sub2 = engine.create_category("年假申请", sub)

    cats = {c["id"]: c for c in engine.list_categories()}
    assert cats[sub]["parent_id"] == root
    assert cats[sub2]["parent_id"] == sub

    # 同级重名必须拒绝，否则树上会出现两个同名节点，用户分不清该点哪个
    with pytest.raises(ValueError):
        engine.create_category("请假与考勤", root)
    # 不同父级下允许同名（「员工制度/规则」与「财务制度/规则」是两回事）
    other = engine.create_category("财务制度")
    assert engine.create_category("规则", other)


def test_delete_category_blocked_by_children_then_ok(engine):
    root = engine.create_category("分组")
    kid = engine.create_category("子项", root)

    # 有子分类一律拒绝：否则子树变孤儿，再也找不回来
    with pytest.raises(ValueError):
        engine.delete_category(root)

    res = engine.delete_category(kid)
    assert res["id"] == kid
    assert res["docs_unassigned"] == 0
    assert engine.registry.get_category(kid) is None

    # 子项清掉后父级可删
    assert engine.delete_category(root)["id"] == root


def test_delete_category_with_docs_requires_force(engine):
    root = engine.create_category("有文档的分组")
    doc = engine.list_docs()[0]
    engine.set_doc_category(doc["doc_id"], root)

    # 默认拒绝：误删分类不该连带毁掉文档
    with pytest.raises(ValueError):
        engine.delete_category(root)

    res = engine.delete_category(root, force=True)
    assert res["docs_unassigned"] == 1
    # 文档还在，只是回到未分类
    assert engine.registry.get_doc(doc["doc_id"]) is not None
    assert engine.count_docs(unclassified=True) == 5


def test_move_category_rejects_cycle(engine):
    a = engine.create_category("A")
    b = engine.create_category("B", a)
    c = engine.create_category("C", b)

    with pytest.raises(ValueError):
        engine.move_category(a, a)          # 移到自己下面
    with pytest.raises(ValueError):
        engine.move_category(a, b)          # 移到自己的子孙下面
    with pytest.raises(ValueError):
        engine.move_category(a, c)

    # 合法移动
    engine.move_category(c, a)
    assert engine.registry.get_category(c)["parent_id"] == a
    with pytest.raises(KeyError):
        engine.move_category(c, 99999)      # 目标不存在


def test_assign_category_and_filter_includes_descendants(engine):
    root = engine.create_category("员工制度")
    sub = engine.create_category("请假与考勤", root)
    deep = engine.create_category("年假申请", sub)

    docs = engine.list_docs()
    engine.set_doc_category(docs[0]["doc_id"], deep)
    engine.set_doc_category(docs[1]["doc_id"], sub)

    # 父节点筛选含子孙：点「员工制度」应看到挂在任意子层的文档
    assert engine.count_docs(category_id=root) == 2
    assert engine.count_docs(category_id=sub) == 2
    assert engine.count_docs(category_id=deep) == 1

    # 只看直接挂靠时，父节点自身为 0
    assert engine.count_docs(category_id=root, include_sub=False) == 0

    # 列表里带出分类全路径，前端不必再回查
    row = [d for d in engine.list_docs(category_id=deep) if d["doc_id"] == docs[0]["doc_id"]][0]
    assert row["category_path"] == "员工制度 / 请假与考勤 / 年假申请"

    # 批量归类
    ids = [d["doc_id"] for d in docs[2:4]]
    assert engine.set_docs_category(ids, root) == 2
    assert engine.count_docs(category_id=root) == 4
    # 空列表不应报错
    assert engine.set_docs_category([], root) == 0


def test_unclassified_filter_is_independent(engine):
    """「未分类」必须用独立开关：category_id=0 同时表示「不筛选」。"""
    assert engine.count_docs() == 5
    assert engine.count_docs(unclassified=True) == 5
    assert engine.count_docs(category_id=None) == 5

    root = engine.create_category("归类用")
    engine.set_doc_category(engine.list_docs()[0]["doc_id"], root)
    assert engine.count_docs(unclassified=True) == 4
    assert engine.count_docs() == 5                       # 不筛选仍是全部
    # 置回未分类
    engine.set_doc_category(engine.list_docs()[0]["doc_id"], 0)
    assert engine.count_docs(unclassified=True) == 5


def test_list_docs_pagination_matches_total(engine):
    """分页总数必须与筛选条件一致，否则前端翻页会算出错误页数。"""
    total = engine.count_docs()
    assert total == 5

    page1 = engine.list_docs(limit=2, offset=0)
    page2 = engine.list_docs(limit=2, offset=2)
    page3 = engine.list_docs(limit=2, offset=4)
    assert len(page1) == 2 and len(page2) == 2 and len(page3) == 1
    ids = [d["doc_id"] for d in page1 + page2 + page3]
    assert len(set(ids)) == total          # 不重不漏

    # 超出范围的页返回空，而不是报错
    assert engine.list_docs(limit=2, offset=100) == []
    # limit=0 保持旧行为：一次返回全部
    assert len(engine.list_docs()) == total


def test_migrate_categories_from_meta_is_idempotent(engine):
    """历史 meta.category 字符串 → 分类树；重复执行不产生重复分类。"""
    # 造两条带历史分类的文档：分隔符是「 / 」，名称里可含裸斜杠
    reg = engine.registry
    docs = engine.list_docs()
    for d, cat in zip(docs[:2], ["人力/行政 / 假期管理", "人力/行政 / 假期管理"]):
        reg._conn.execute("UPDATE docs SET category_id=0, meta=? WHERE doc_id=?",
                          (__import__("json").dumps({"category": cat}), d["doc_id"]))
    reg._conn.commit()

    first = engine.migrate_categories_from_meta()
    assert first["docs"] == 2
    cats = engine.list_categories()
    names = [c["name"] for c in cats]
    # 「人力/行政」作为一个整体节点，不能被裸斜杠劈成「人力」「行政」
    assert "人力/行政" in names
    assert "假期管理" in names
    assert not {"人力", "行政"} <= set(names)

    # 归类到了叶子节点
    row = [d for d in engine.list_docs() if d["doc_id"] == docs[0]["doc_id"]][0]
    assert row["category_path"] == "人力/行政 / 假期管理"

    # 幂等：再跑一次不新增分类、不重复归类
    second = engine.migrate_categories_from_meta()
    assert second["docs"] == 0
    assert len(engine.list_categories()) == len(cats)


def test_category_doc_counts_include_descendants(engine):
    root = engine.create_category("根")
    sub = engine.create_category("子", root)
    docs = engine.list_docs()
    engine.set_doc_category(docs[0]["doc_id"], sub)
    engine.set_doc_category(docs[1]["doc_id"], root)

    cats = {c["id"]: c for c in engine.list_categories()}
    assert cats[root]["doc_count"] == 1        # 直接挂靠
    assert cats[root]["total_count"] == 2      # 含子孙
    assert cats[sub]["doc_count"] == 1
    assert cats[sub]["total_count"] == 1


def test_delete_docs_batch(engine):
    docs = engine.list_docs()
    ids = [d["doc_id"] for d in docs[:2]]
    res = engine.delete_docs(ids)
    assert res["deleted"] == 2
    assert res["errors"] == []
    assert engine.count_docs() == 3
    # 三处存储一致（向量库 / BM25 / 注册表）
    assert engine.store.count() == engine.registry.count_chunks()
    assert len(engine.bm25) == engine.registry.count_chunks()

    # 不存在的 id 单独报错，不影响其它
    res2 = engine.delete_docs(["__not_exist__"] + [docs[2]["doc_id"]])
    assert res2["deleted"] == 1
    assert len(res2["errors"]) == 1
