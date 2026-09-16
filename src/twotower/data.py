"""Datasets: a synthetic latent-factor generator (offline default) and a
MovieLens 100k loader (opt-in).

Both produce a `Dataset` of implicit positive interactions per user. The
synthetic generator draws interactions from a known latent-factor model, so a
correctly-implemented two-tower can recover that structure and beat a popularity
baseline - which is exactly what the eval asserts, with no network or download.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Dataset:
    n_users: int
    n_items: int
    all_pos: dict[int, list[int]]  # user -> item ids, in interaction order
    item_features: np.ndarray | None = None  # (n_items, F) multi-hot, optional

    def popularity(self) -> np.ndarray:
        """Item interaction counts over ALL positives (train + held-out)."""
        counts = np.zeros(self.n_items)
        for items in self.all_pos.values():
            for i in items:
                counts[i] += 1
        return counts

    def train_popularity(self) -> np.ndarray:
        """Item counts over TRAINING interactions only - excludes each user's
        held-out last item (see ``leave_one_out``). The leave-one-out popularity
        baseline must be scored with this, not ``popularity()``: counting the
        test positives would leak the held-out item into the baseline it is
        being compared against.
        """
        counts = np.zeros(self.n_items)
        for items in self.all_pos.values():
            train_items = items[:-1] if len(items) >= 2 else items
            for i in train_items:
                counts[i] += 1
        return counts

    @property
    def n_item_features(self) -> int:
        return 0 if self.item_features is None else self.item_features.shape[1]

    def leave_one_out(
        self, rng: np.random.Generator
    ) -> tuple[list[tuple[int, int]], dict[int, int], dict[int, set[int]]]:
        """Hold out each user's last interaction for test.

        Returns (train_pairs, test_pos, user_pos_sets). Users with fewer than 2
        interactions contribute to training and to the exclusion sets but are
        not evaluated. ``user_pos_sets`` holds every positive (train+test) so
        negative sampling never collides with a known interaction.
        """
        train_pairs: list[tuple[int, int]] = []
        test_pos: dict[int, int] = {}
        user_pos_sets: dict[int, set[int]] = {}
        for u, items in self.all_pos.items():
            user_pos_sets[u] = set(items)
            if len(items) >= 2:
                *train_items, held_out = items
                test_pos[u] = held_out
            else:
                train_items = items
            for i in train_items:
                train_pairs.append((u, i))
        return train_pairs, test_pos, user_pos_sets


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


def make_synthetic(
    n_users: int = 300,
    n_items: int = 200,
    latent_dim: int = 8,
    pos_per_user: int = 20,
    seed: int = 0,
) -> Dataset:
    """Draw implicit interactions from a latent-factor model.

    Each user u and item i get true latent vectors; interaction logits are
    U_true @ V_true.T plus a per-item popularity bias. Positives are sampled
    per user from the softmax over items - so the data has both a popularity
    signal (which the baseline exploits) and user-specific structure (which
    only a personalized model can exploit).
    """
    rng = np.random.default_rng(seed)
    u_true = rng.normal(0, 1, (n_users, latent_dim))
    v_true = rng.normal(0, 1, (n_items, latent_dim))
    item_bias = rng.normal(0, 0.8, n_items)  # popularity structure

    all_pos: dict[int, list[int]] = {}
    for u in range(n_users):
        logits = u_true[u] @ v_true.T + item_bias
        probs = _softmax(logits)
        k = min(pos_per_user, n_items)
        items = rng.choice(n_items, size=k, replace=False, p=probs)
        all_pos[u] = list(map(int, items))  # order = interaction order for LOO
    return Dataset(n_users=n_users, n_items=n_items, all_pos=all_pos)


def load_movielens(path: str | Path, min_rating: float = 4.0) -> Dataset:
    """Load MovieLens 100k from an extracted ml-100k directory.

    Positives are ratings >= ``min_rating``, ordered by timestamp. Item genre
    columns become the item-tower content features. Download with
    scripts/download_movielens.py.
    """
    root = Path(path)
    data_file = root / "u.data"
    item_file = root / "u.item"
    if not data_file.exists():
        raise FileNotFoundError(f"{data_file} not found - run scripts/download_movielens.py first")

    # u.data: user \t item \t rating \t timestamp  (ids are 1-indexed)
    rows = []
    for line in data_file.read_text().splitlines():
        u, i, r, ts = line.split("\t")
        rows.append((int(u) - 1, int(i) - 1, float(r), int(ts)))
    n_users = max(r[0] for r in rows) + 1
    n_items = max(r[1] for r in rows) + 1

    rows.sort(key=lambda x: x[3])  # by timestamp → interaction order
    all_pos: dict[int, list[int]] = {}
    for u, i, r, _ in rows:
        if r >= min_rating:
            all_pos.setdefault(u, []).append(i)

    # u.item: 24 fields; last 19 are genre flags.
    item_features = None
    if item_file.exists():
        feats = np.zeros((n_items, 19))
        for line in item_file.read_text(encoding="latin-1").splitlines():
            parts = line.split("|")
            iid = int(parts[0]) - 1
            feats[iid] = np.array([float(x) for x in parts[-19:]])
        item_features = feats

    all_pos = {u: items for u, items in all_pos.items() if items}
    return Dataset(n_users=n_users, n_items=n_items, all_pos=all_pos, item_features=item_features)
