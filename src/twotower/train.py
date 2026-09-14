"""Training loop (BPR over leave-one-out train pairs) and a CLI.

    python -m twotower.train                       # synthetic data
    python -m twotower.train --movielens data/ml-100k --epochs 20
"""

from __future__ import annotations

import argparse

import numpy as np

from .data import Dataset, load_movielens, make_synthetic
from .evaluate import evaluate_model, evaluate_popularity
from .model import TwoTowerModel


def _sample_batch_negatives(
    users: np.ndarray, user_pos_sets: dict[int, set[int]], n_items: int, rng: np.random.Generator
) -> np.ndarray:
    """One negative per row, rejecting the user's known positives."""
    negs = rng.integers(n_items, size=len(users))
    for idx, u in enumerate(users):
        while int(negs[idx]) in user_pos_sets[u]:
            negs[idx] = rng.integers(n_items)
    return negs


def train(
    dataset: Dataset,
    dim: int = 32,
    epochs: int = 50,
    lr: float = 3.0,  # tuned for mean-reduced BPR at 0.1-scale init; see docs/model.md
    batch_size: int = 256,
    reg: float = 1e-5,
    seed: int = 0,
    verbose: bool = False,
) -> tuple[TwoTowerModel, dict[int, int], dict[int, set[int]]]:
    rng = np.random.default_rng(seed)
    train_pairs, test_pos, user_pos_sets = dataset.leave_one_out(rng)
    pairs = np.array(train_pairs)
    feats = dataset.item_features

    model = TwoTowerModel(
        dataset.n_users, dataset.n_items, dim=dim, n_item_features=dataset.n_item_features, reg=reg, seed=seed
    )
    for epoch in range(epochs):
        rng.shuffle(pairs)
        total, n_batches = 0.0, 0
        for start in range(0, len(pairs), batch_size):
            batch = pairs[start : start + batch_size]
            users, pos = batch[:, 0], batch[:, 1]
            negs = _sample_batch_negatives(users, user_pos_sets, dataset.n_items, rng)
            total += model.bpr_step(users, pos, negs, lr, feats)
            n_batches += 1
        if verbose:
            print(f"epoch {epoch + 1:3d}/{epochs}  bpr_loss={total / max(1, n_batches):.4f}")
    return model, test_pos, user_pos_sets


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--movielens", default=None, help="path to an extracted ml-100k dir")
    ap.add_argument("--dim", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr", type=float, default=3.0)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="save trained model to this .npz")
    args = ap.parse_args()

    dataset = load_movielens(args.movielens) if args.movielens else make_synthetic(seed=args.seed)
    print(f"users={dataset.n_users} items={dataset.n_items} item_features={dataset.n_item_features}")

    model, test_pos, user_pos = train(
        dataset, dim=args.dim, epochs=args.epochs, lr=args.lr, batch_size=args.batch, seed=args.seed, verbose=True
    )

    model_metrics = evaluate_model(
        model, test_pos, user_pos, dataset.n_items, dataset.item_features, seed=args.seed
    )
    pop_metrics = evaluate_popularity(
        dataset.train_popularity(), test_pos, user_pos, dataset.n_items, seed=args.seed
    )
    print(f"\nmodel:      {model_metrics}")
    print(f"popularity: {pop_metrics}")

    if args.out:
        model.save(args.out)
        print(f"saved model to {args.out}")


if __name__ == "__main__":
    main()
