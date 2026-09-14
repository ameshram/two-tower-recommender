# The model & the math

## Two towers

Each user and item is mapped to a vector in the same space; relevance is their
dot product:

```
user_vec(u) = U[u]                              # user tower: id embedding
item_vec(i) = V[i] + item_features[i] · Wg      # item tower: id embedding + content
score(u, i) = user_vec(u) · item_vec(i)
```

`U ∈ ℝ^{users×d}`, `V ∈ ℝ^{items×d}`, and `Wg ∈ ℝ^{F×d}` projects item content
features (MovieLens genres) into the embedding space. With no features the item
tower is a pure id embedding (equivalent to matrix factorization). The "two
tower" structure matters at serving time: item vectors are **precomputable and
indexable** (ANN), so retrieval over millions of items is a single user-tower
forward pass plus a nearest-neighbor lookup.

## Loss: Bayesian Personalized Ranking (BPR)

Implicit feedback has no negatives, only observed positives. BPR learns from
*pairwise* preferences: for a user `u`, an observed positive `i` should score
higher than an unobserved item `j`. With `x = score(u,i) − score(u,j)`:

```
L = mean over triples of  softplus(−x) = mean −log σ(x)
```

## Gradients (derived by hand, checked numerically)

Let `σ = σ(x)`. Then `dL/dx = (σ − 1)` per triple (÷ batch size for the mean).
Since `x = u·(i_vec − j_vec)`:

```
∂L/∂U[u]  = (σ−1) · (i_vec − j_vec)
∂L/∂V[i]  = (σ−1) · u_vec
∂L/∂V[j]  = −(σ−1) · u_vec
∂L/∂Wg    = (σ−1) · outer(feat[i] − feat[j], u_vec)
```

These are implemented in `model.data_loss_and_grads` and verified against a
central-difference numerical gradient in `tests/test_model.py::test_gradient_check`
(max abs error < 1e-6). L2 weight decay is added in the SGD step, not the data
loss, so the gradient check stays clean.

## A note on the learning rate

The loss uses a **mean** reduction and embeddings init at scale 0.1, so the
raw gradient magnitude is small; the tuned default `lr=3.0` is what that scale
requires to converge in ~50 epochs. It is a hyperparameter, not a magic number —
lower it if you increase the init scale or switch to a sum reduction. The eval
(`make eval`) is what certifies the choice: it asserts the trained model beats
the popularity baseline by a wide margin.

## Complexity & serving

- Training: `O(triples · d)` per epoch.
- Serving: item vectors precomputed once; per-request cost is one user-tower
  pass + ANN over item vectors. This is the property that makes two-tower models
  the standard *retrieval* stage in large-scale recommenders (a heavier ranking
  model then re-scores the shortlist).

## What I'd change in v2

- **PyTorch port** with the same interface — mini-batch autograd, GPU, and
  larger embeddings; this NumPy version exists to show the mechanics.
- **In-batch / popularity-corrected negative sampling** (sampled softmax) instead
  of uniform negatives.
- **A real ANN index** (FAISS/hnswlib) for the retrieval step + latency numbers.
- **Sequence-aware user tower** (recent interactions) rather than a static id.
