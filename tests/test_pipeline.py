"""End-to-end: training on synthetic data must beat the popularity baseline.

This is the "did it actually learn" test - the whole point of a personalized
recommender is to outperform recommending globally-popular items.
"""

import numpy as np

from twotower.data import make_synthetic
from twotower.evaluate import evaluate_model, evaluate_popularity
from twotower.train import train


def test_two_tower_beats_popularity():
    ds = make_synthetic(n_users=200, n_items=150, pos_per_user=15, seed=0)
    model, test_pos, user_pos = train(ds, dim=32, epochs=40, lr=3.0, seed=0)

    m = evaluate_model(model, test_pos, user_pos, ds.n_items, ds.item_features, seed=0)
    p = evaluate_popularity(ds.popularity(), test_pos, user_pos, ds.n_items, seed=0)

    # Comfortable margin so the test is robust to seed/platform variance.
    assert m["ndcg@10"] > p["ndcg@10"] * 1.5
    assert m["hit@10"] > 0.30


def test_leave_one_out_split_is_disjoint():
    ds = make_synthetic(n_users=50, n_items=40, pos_per_user=10, seed=1)
    rng = np.random.default_rng(1)
    train_pairs, test_pos, user_pos = ds.leave_one_out(rng)
    # A user's held-out test item must not also appear in their train pairs.
    train_by_user: dict[int, set[int]] = {}
    for u, i in train_pairs:
        train_by_user.setdefault(u, set()).add(i)
    for u, held in test_pos.items():
        assert held not in train_by_user.get(u, set())
