"""Leave-one-out evaluation against a popularity baseline.

For each user we rank their held-out positive against a fixed set of sampled
negatives, and report HitRate@K and NDCG@K. The popularity baseline is scored on
the *same* candidate sets (same seed), so the comparison is apples-to-apples —
the model has to beat "just recommend popular items" using personalization.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable

import numpy as np

from .metrics import hit_at_k, ndcg_at_k, rank_of_positive
from .model import TwoTowerModel
from .negatives import sample_negatives


def _evaluate(
    score_fn: Callable[[int, np.ndarray], np.ndarray],
    test_pos: dict[int, int],
    user_pos_sets: dict[int, set[int]],
    n_items: int,
    k_list: Iterable[int] = (5, 10),
    n_negatives: int = 100,
    seed: int = 0,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    k_list = list(k_list)
    hits = {k: 0.0 for k in k_list}
    ndcgs = {k: 0.0 for k in k_list}
    count = 0
    for u, pos_item in test_pos.items():
        negs = sample_negatives(user_pos_sets[u], n_items, n_negatives, rng)
        candidates = np.array([pos_item, *negs])
        scores = score_fn(u, candidates)
        rank = rank_of_positive(scores)
        for k in k_list:
            hits[k] += hit_at_k(rank, k)
            ndcgs[k] += ndcg_at_k(rank, k)
        count += 1
    if count == 0:
        return {}
    out = {}
    for k in k_list:
        out[f"hit@{k}"] = round(hits[k] / count, 4)
        out[f"ndcg@{k}"] = round(ndcgs[k] / count, 4)
    out["n_users_evaluated"] = count
    return out


def evaluate_model(
    model: TwoTowerModel,
    test_pos: dict[int, int],
    user_pos_sets: dict[int, set[int]],
    n_items: int,
    item_features: np.ndarray | None = None,
    k_list: Iterable[int] = (5, 10),
    n_negatives: int = 100,
    seed: int = 0,
) -> dict[str, float]:
    def score_fn(u: int, candidates: np.ndarray) -> np.ndarray:
        return model.score_user_items(u, candidates, item_features)

    return _evaluate(score_fn, test_pos, user_pos_sets, n_items, k_list, n_negatives, seed)


def evaluate_popularity(
    popularity: np.ndarray,
    test_pos: dict[int, int],
    user_pos_sets: dict[int, set[int]],
    n_items: int,
    k_list: Iterable[int] = (5, 10),
    n_negatives: int = 100,
    seed: int = 0,
) -> dict[str, float]:
    def score_fn(_u: int, candidates: np.ndarray) -> np.ndarray:
        return popularity[candidates]

    return _evaluate(score_fn, test_pos, user_pos_sets, n_items, k_list, n_negatives, seed)
