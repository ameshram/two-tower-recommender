"""Two-tower retrieval model, implemented from scratch in NumPy.

Two towers produce embeddings that are scored by dot product:

    user_vec(u) = U[u]                                   # user tower (id embedding)
    item_vec(i) = V[i] + item_features[i] @ Wg           # item tower (id + content)
    score(u, i) = user_vec(u) · item_vec(i)

Trained with the Bayesian Personalized Ranking (BPR) pairwise loss on triples
(user u, positive i, negative j):

    L = mean_b  softplus( -(score(u,i) - score(u,j)) )

The gradients are derived by hand (see docs/model.md) and checked numerically in
tests/test_model.py. Implementing this in NumPy - rather than calling a
framework - is deliberate: it demonstrates the mechanics. A PyTorch port is the
production path (README → "What I'd change in v2").
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


class TwoTowerModel:
    def __init__(
        self,
        n_users: int,
        n_items: int,
        dim: int = 32,
        n_item_features: int = 0,
        reg: float = 1e-5,
        seed: int = 0,
    ) -> None:
        rng = np.random.default_rng(seed)
        self.n_users = n_users
        self.n_items = n_items
        self.dim = dim
        self.reg = reg
        self.U = rng.normal(0, 0.1, (n_users, dim))
        self.V = rng.normal(0, 0.1, (n_items, dim))
        self.Wg = rng.normal(0, 0.1, (n_item_features, dim)) if n_item_features else None

    # ---- forward -----------------------------------------------------
    def _item_vec(self, items: np.ndarray, item_features: np.ndarray | None) -> np.ndarray:
        base = self.V[items]
        if self.Wg is not None and item_features is not None:
            base = base + item_features[items] @ self.Wg
        return base

    def score_user_items(
        self, user: int, item_ids, item_features: np.ndarray | None = None
    ) -> np.ndarray:
        """Score a single user against a list/array of candidate items."""
        items = np.asarray(item_ids)
        return self._item_vec(items, item_features) @ self.U[user]

    # ---- loss + analytic gradients (no regularization) ---------------
    def data_loss_and_grads(
        self,
        users: np.ndarray,
        pos: np.ndarray,
        neg: np.ndarray,
        item_features: np.ndarray | None = None,
    ) -> tuple[float, np.ndarray, np.ndarray, np.ndarray | None]:
        """Mean BPR loss over the batch and dense gradients wrt U, V, Wg."""
        b = len(users)
        uv = self.U[users]  # (B, d)
        piv = self._item_vec(pos, item_features)  # (B, d)
        niv = self._item_vec(neg, item_features)  # (B, d)

        x = np.sum(uv * (piv - niv), axis=1)  # (B,)
        loss = float(np.mean(np.logaddexp(0.0, -x)))  # softplus(-x)

        # dL/dx = (sigmoid(x) - 1) / B
        c = (1.0 / (1.0 + np.exp(-x)) - 1.0) / b  # (B,)
        cw = c[:, None]  # (B, 1)

        g_u = np.zeros_like(self.U)
        np.add.at(g_u, users, cw * (piv - niv))

        g_v = np.zeros_like(self.V)
        np.add.at(g_v, pos, cw * uv)
        np.add.at(g_v, neg, -cw * uv)

        g_wg = None
        if self.Wg is not None and item_features is not None:
            w = cw * uv  # (B, d)
            g_wg = (item_features[pos] - item_features[neg]).T @ w  # (F, d)
        return loss, g_u, g_v, g_wg

    # ---- one SGD step (data grad + L2 weight decay) ------------------
    def bpr_step(
        self,
        users: np.ndarray,
        pos: np.ndarray,
        neg: np.ndarray,
        lr: float,
        item_features: np.ndarray | None = None,
    ) -> float:
        loss, g_u, g_v, g_wg = self.data_loss_and_grads(users, pos, neg, item_features)
        self.U -= lr * (g_u + self.reg * self.U)
        self.V -= lr * (g_v + self.reg * self.V)
        if self.Wg is not None and g_wg is not None:
            self.Wg -= lr * (g_wg + self.reg * self.Wg)
        return loss

    # ---- persistence -------------------------------------------------
    def save(self, path: str | Path) -> None:
        arrays = {"U": self.U, "V": self.V, "reg": np.array(self.reg)}
        if self.Wg is not None:
            arrays["Wg"] = self.Wg
        np.savez(path, **arrays)

    @classmethod
    def load(cls, path: str | Path) -> TwoTowerModel:
        d = np.load(path)
        wg = d["Wg"] if "Wg" in d.files else None
        m = cls(
            n_users=d["U"].shape[0],
            n_items=d["V"].shape[0],
            dim=d["U"].shape[1],
            n_item_features=0 if wg is None else wg.shape[0],
            reg=float(d["reg"]),
        )
        m.U, m.V = d["U"], d["V"]
        if wg is not None:
            m.Wg = wg
        return m
