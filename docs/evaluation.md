# Evaluation methodology

A recommender is only as trustworthy as its offline eval. This one is built to
answer one question honestly: **does personalization beat just recommending
popular items?**

## Protocol: leave-one-out with sampled negatives

For each user, hold out their **last** interaction (by time) as the test
positive. At eval time, build a candidate set of `{held-out positive} ∪ {N
uniformly-sampled negatives the user has not interacted with}` (default N=100),
score all candidates, and record the rank of the positive. This is the standard
LOO protocol used in the implicit-feedback recommender literature (e.g. NCF).

## Metrics

- **HitRate@K** — fraction of users whose held-out positive is in the top-K.
- **NDCG@K** — position-discounted version (rewards ranking the positive higher).

With a single relevant item, Recall@K ≡ HitRate@K, so we report HitRate + NDCG at
K ∈ {5, 10}.

## The baseline is the point

Every metric is reported for the trained model **and** for a **popularity
baseline** scored on the *identical* candidate sets (same seed). A model that
can't beat "recommend the globally most popular items" has learned nothing
useful. We therefore also report **NDCG@10 uplift over popularity** as the
headline, and gate on it.

## Latest offline run (synthetic, seed 0)

| Metric | Two-tower | Popularity |
|---|---|---|
| HitRate@10 | **0.47** | 0.16 |
| NDCG@10 | **0.257** | 0.077 |

**NDCG@10 uplift: 3.3×.** Reproduce with `make eval`.

## Regression gates (`eval/thresholds.yaml`)

```
min_hit_at_10: 0.30
min_ndcg_at_10: 0.13
min_ndcg10_uplift_over_popularity: 1.8
```

Set below the achieved values with headroom for seed/platform variance;
`run_eval.py --gate` exits non-zero on violation, so CI blocks a regression.

## Synthetic vs. MovieLens

- **Synthetic (default, offline).** Interactions are drawn from a known
  latent-factor model with a popularity bias, so there is genuine personalized
  structure to recover *and* a strong popularity signal to beat. Deterministic,
  fast, no download — this is what CI runs.
- **MovieLens 100k (`--movielens`).** Real ratings (≥4 = positive), ordered by
  timestamp, with genre features feeding the item tower. Download with
  `make data`. Not run in CI (network + slower), but the same eval applies.

## Honest caveats

- LOO with sampled negatives is a *proxy* for production retrieval quality; it
  can be optimistic (a full-catalog ranking is harder) and is sensitive to N.
- The synthetic result confirms the model and pipeline are correct; it is not a
  claim about any real recommendation workload. MovieLens numbers are the more
  realistic reference.
