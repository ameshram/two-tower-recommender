"""Uniform negative sampling.

Both training (BPR triples) and evaluation (candidate sets) need items the user
has *not* interacted with. We sample uniformly and reject known positives.
"""

from __future__ import annotations

import numpy as np


def sample_negatives(
    positives: set[int], n_items: int, count: int, rng: np.random.Generator
) -> list[int]:
    """Return ``count`` distinct item ids not in ``positives``."""
    if n_items - len(positives) < count:
        raise ValueError("not enough non-positive items to sample the requested negatives")
    negs: set[int] = set()
    while len(negs) < count:
        cand = int(rng.integers(n_items))
        if cand not in positives:
            negs.add(cand)
    return list(negs)
