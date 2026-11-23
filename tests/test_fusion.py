from app.retrieval.fusion import rrf_fuse


def test_both_channels_hit_wins():
    dense = [("a", 0.9), ("b", 0.8), ("c", 0.7)]
    sparse = [("c", 12.0), ("d", 9.0)]
    fused = rrf_fuse(dense, sparse, rrf_k=60, top_k=4)
    # c 同时出现在两路，RRF 分数最高
    assert fused[0].chunk_id == "c"
    assert {f.chunk_id for f in fused} == {"a", "b", "c", "d"}


def test_weights_shift_order():
    dense = [("a", 0.9)]
    sparse = [("b", 5.0)]
    assert rrf_fuse(dense, sparse, dense_weight=3.0, sparse_weight=1.0)[0].chunk_id == "a"
    assert rrf_fuse(dense, sparse, dense_weight=1.0, sparse_weight=3.0)[0].chunk_id == "b"


def test_scores_and_ranks_recorded():
    fused = rrf_fuse([("a", 0.5)], [("a", 3.0)], top_k=1)
    hit = fused[0]
    assert hit.dense_score == 0.5
    assert hit.sparse_score == 3.0
    assert hit.rank_dense == 0 and hit.rank_sparse == 0
    assert hit.score > 0.03  # 2/(60+1)


def test_empty_input():
    assert rrf_fuse([], []) == []
