"""Model correctness: analytic gradients, learning, and persistence.

The gradient check is the important one - it proves the hand-derived BPR
gradients match a numerical finite-difference estimate, so the "from scratch"
claim is actually correct, not just plausible.
"""

import numpy as np

from twotower.model import TwoTowerModel


def _tiny_batch(seed=0):
    rng = np.random.default_rng(seed)
    n_users, n_items, dim, n_feat = 5, 6, 3, 4
    model = TwoTowerModel(n_users, n_items, dim=dim, n_item_features=n_feat, reg=0.0, seed=seed)
    item_features = rng.normal(0, 1, (n_items, n_feat))
    users = rng.integers(n_users, size=8)
    pos = rng.integers(n_items, size=8)
    neg = rng.integers(n_items, size=8)
    return model, item_features, users, pos, neg


def _numerical_grad(model, attr, item_features, users, pos, neg, eps=1e-5):
    arr = getattr(model, attr)
    grad = np.zeros_like(arr)
    it = np.nditer(arr, flags=["multi_index"])
    while not it.finished:
        idx = it.multi_index
        orig = arr[idx]
        arr[idx] = orig + eps
        lp, *_ = model.data_loss_and_grads(users, pos, neg, item_features)
        arr[idx] = orig - eps
        lm, *_ = model.data_loss_and_grads(users, pos, neg, item_features)
        arr[idx] = orig
        grad[idx] = (lp - lm) / (2 * eps)
        it.iternext()
    return grad


def test_gradient_check():
    model, feats, users, pos, neg = _tiny_batch()
    _, g_u, g_v, g_wg = model.data_loss_and_grads(users, pos, neg, feats)

    for attr, analytic in [("U", g_u), ("V", g_v), ("Wg", g_wg)]:
        numeric = _numerical_grad(model, attr, feats, users, pos, neg)
        max_diff = np.max(np.abs(numeric - analytic))
        assert max_diff < 1e-6, f"{attr} gradient mismatch: {max_diff}"


def test_model_overfits_tiny_data():
    """After training, the model ranks a trained positive above its negative."""
    model = TwoTowerModel(4, 8, dim=8, reg=0.0, seed=0)
    users = np.array([0, 1, 2, 3])
    pos = np.array([0, 1, 2, 3])
    neg = np.array([4, 5, 6, 7])
    for _ in range(500):
        model.bpr_step(users, pos, neg, lr=1.0)
    for u, p, n in zip(users, pos, neg, strict=True):
        s = model.score_user_items(u, np.array([p, n]))
        assert s[0] > s[1]


def test_save_load_roundtrip(tmp_path):
    model = TwoTowerModel(4, 6, dim=5, n_item_features=3, seed=1)
    feats = np.random.default_rng(1).normal(0, 1, (6, 3))
    before = model.score_user_items(2, np.array([0, 1, 2]), feats)
    path = tmp_path / "m.npz"
    model.save(path)
    loaded = TwoTowerModel.load(path)
    after = loaded.score_user_items(2, np.array([0, 1, 2]), feats)
    assert np.allclose(before, after)
