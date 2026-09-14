"""Train a two-tower model and evaluate it against a popularity baseline, with
CI regression gates.

Offline by default (synthetic data). `--movielens PATH` uses real MovieLens.
`--gate` exits non-zero if any threshold in thresholds.yaml is violated, so CI
blocks a regression. Writes eval/report.md and eval/report.json.

    python -m eval.run_eval --gate
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from twotower.data import load_movielens, make_synthetic  # noqa: E402
from twotower.evaluate import evaluate_model, evaluate_popularity  # noqa: E402
from twotower.train import train  # noqa: E402


def evaluate(movielens: str | None, dim: int, epochs: int, lr: float, seed: int) -> dict:
    dataset = load_movielens(movielens) if movielens else make_synthetic(seed=seed)
    model, test_pos, user_pos = train(
        dataset, dim=dim, epochs=epochs, lr=lr, seed=seed, verbose=False
    )
    model_m = evaluate_model(
        model, test_pos, user_pos, dataset.n_items, dataset.item_features, seed=seed
    )
    pop_m = evaluate_popularity(dataset.popularity(), test_pos, user_pos, dataset.n_items, seed=seed)
    uplift = round(model_m["ndcg@10"] / pop_m["ndcg@10"], 3) if pop_m["ndcg@10"] else float("inf")
    return {
        "dataset": "movielens" if movielens else "synthetic",
        "n_users": dataset.n_users,
        "n_items": dataset.n_items,
        "config": {"dim": dim, "epochs": epochs, "lr": lr, "seed": seed},
        "model": model_m,
        "popularity": pop_m,
        "ndcg10_uplift_over_popularity": uplift,
    }


def check_gates(report: dict, thresholds: dict) -> list[str]:
    failures = []
    checks = [
        ("hit@10", report["model"]["hit@10"], "min", thresholds.get("min_hit_at_10")),
        ("ndcg@10", report["model"]["ndcg@10"], "min", thresholds.get("min_ndcg_at_10")),
        (
            "ndcg10_uplift",
            report["ndcg10_uplift_over_popularity"],
            "min",
            thresholds.get("min_ndcg10_uplift_over_popularity"),
        ),
    ]
    for name, val, _kind, lim in checks:
        if lim is not None and val < lim:
            failures.append(f"{name}={val} < min {lim}")
    return failures


def render_markdown(report: dict, failures: list[str]) -> str:
    m, p = report["model"], report["popularity"]
    lines = [
        "# Evaluation report",
        "",
        f"- dataset: **{report['dataset']}** ({report['n_users']} users, {report['n_items']} items)",
        f"- config: `{report['config']}`",
        "",
        "| Metric | Two-tower | Popularity baseline |",
        "|---|---|---|",
        f"| HitRate@5 | {m['hit@5']} | {p['hit@5']} |",
        f"| NDCG@5 | {m['ndcg@5']} | {p['ndcg@5']} |",
        f"| HitRate@10 | {m['hit@10']} | {p['hit@10']} |",
        f"| NDCG@10 | {m['ndcg@10']} | {p['ndcg@10']} |",
        "",
        f"**NDCG@10 uplift over popularity: {report['ndcg10_uplift_over_popularity']}×**",
        "",
        "## Gates",
        "",
        "✅ all gates passed" if not failures else "❌ " + "; ".join(failures),
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--movielens", default=None)
    ap.add_argument("--dim", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr", type=float, default=3.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gate", action="store_true")
    ap.add_argument("--thresholds", default=str(Path(__file__).parent / "thresholds.yaml"))
    ap.add_argument("--report", default=str(Path(__file__).parent / "report.md"))
    args = ap.parse_args()

    report = evaluate(args.movielens, args.dim, args.epochs, args.lr, args.seed)
    thresholds = yaml.safe_load(Path(args.thresholds).read_text())
    failures = check_gates(report, thresholds)

    md = render_markdown(report, failures)
    Path(args.report).write_text(md)
    Path(args.report).with_suffix(".json").write_text(json.dumps(report, indent=2))
    print(md)

    if args.gate and failures:
        print(f"\nGATE FAILED: {len(failures)} threshold(s) violated.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
