"""Ranking metrics for leave-one-out evaluation.

In the leave-one-out protocol each user has exactly one held-out positive item,
ranked against a set of sampled negatives. With a single relevant item,
Recall@K == HitRate@K, so we report HitRate@K and NDCG@K. Both are computed
from the 0-indexed ``rank`` of the positive item (0 = ranked first).
"""

from __future__ import annotations

import math


def hit_at_k(rank: int, k: int) -> float:
    return 1.0 if rank < k else 0.0


def ndcg_at_k(rank: int, k: int) -> float:
    """DCG of a single relevant item at position ``rank`` (ideal DCG == 1)."""
    if rank < k:
        return 1.0 / math.log2(rank + 2)
    return 0.0


def rank_of_positive(scores) -> int:
    """Given a candidate score array whose index 0 is the positive item, return
    the 0-indexed rank of the positive (number of candidates scoring strictly
    higher). Ties do not advantage the positive."""
    positive = scores[0]
    return int((scores[1:] > positive).sum())
