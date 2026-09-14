import numpy as np

from twotower.metrics import hit_at_k, ndcg_at_k, rank_of_positive


def test_hit_at_k():
    assert hit_at_k(0, 1) == 1.0
    assert hit_at_k(3, 3) == 0.0  # rank 3 is position 4, not in top-3
    assert hit_at_k(4, 5) == 1.0


def test_ndcg_at_k():
    assert ndcg_at_k(0, 10) == 1.0  # top position → ideal
    assert abs(ndcg_at_k(1, 10) - 1.0 / np.log2(3)) < 1e-9
    assert ndcg_at_k(10, 10) == 0.0  # outside top-k


def test_rank_of_positive():
    # index 0 is the positive; two negatives, one higher one lower.
    assert rank_of_positive(np.array([0.5, 0.9, 0.1])) == 1
    assert rank_of_positive(np.array([2.0, 0.9, 0.1])) == 0  # positive ranked first
    assert rank_of_positive(np.array([0.0, 1.0, 2.0, 3.0])) == 3  # ranked last
