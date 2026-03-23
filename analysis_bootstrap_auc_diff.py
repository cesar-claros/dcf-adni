"""
Paired Bootstrap Test for AUC Difference
==========================================

Tests whether the AUC gain of the stacked model over BMCA alone is
statistically supported using a paired bootstrap procedure.

Method
------
Both models are evaluated on the same bootstrap resample of matched pairs
(sampled with replacement at the pair level). The AUC difference
(stacked − BMCA) is computed for each resample, producing an empirical
distribution of the paired difference. The 95% CI of this distribution is
the primary inferential quantity.

Pairing is essential here: stacked and BMCA scores are computed on the same
subjects in each resample, so their AUC estimates share the same sampling
noise. The paired difference has lower variance than an unpaired comparison
would, making it the correct approach when both models are evaluated on the
same test set.

The proportion of bootstrap resamples where Δ ≤ 0 (i.e., stacked is no
better than BMCA) is reported as a one-sided p-value under H0: Δ ≤ 0.

Loads
-----
results/stacking_model.joblib — produced by model_stacking_evaluation.py

Outputs
-------
results/bootstrap_auc_diff.csv — CI and p-value for each model pair
plots/bootstrap_auc_diff.pdf   — histogram of bootstrap Δ distribution

Usage::

    python analysis_bootstrap_auc_diff.py
    python analysis_bootstrap_auc_diff.py --n_boot 10000 --seed 42
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

logging.basicConfig(level=logging.INFO, format="%(name)s — %(message)s")
logger = logging.getLogger(__name__)


# =============================================================================
# Bootstrap test
# =============================================================================


def paired_bootstrap_auc_diff(
    y_true: np.ndarray,
    score_a: np.ndarray,
    score_b: np.ndarray,
    groups: np.ndarray,
    n_boot: int = 10000,
    seed: int = 0,
) -> dict:
    """
    Paired bootstrap distribution of AUC(B) − AUC(A).

    Both models are scored on the same bootstrap resample of matched pairs,
    preserving the within-sample correlation between their AUC estimates.
    Pairs are resampled with replacement at the group level.

    Args:
        y_true: True binary labels.
        score_a: Predicted probabilities for model A.
        score_b: Predicted probabilities for model B.
        groups: Matched pair IDs (one value per subject; each pair has 2 rows).
        n_boot: Number of bootstrap resamples.
        seed: Random seed.

    Returns:
        dict with keys:
            observed_diff   — AUC(B) − AUC(A) on the original test set
            mean_diff       — mean of bootstrap distribution
            ci_low, ci_high — 2.5th and 97.5th percentiles of bootstrap Δ
            p_value         — proportion of resamples where Δ ≤ 0
            diffs           — full array of bootstrap Δ values
    """
    rng = np.random.default_rng(seed)
    unique_groups = np.unique(groups)
    diffs = []

    for _ in range(n_boot):
        sampled = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        idx = np.concatenate([np.where(groups == g)[0] for g in sampled])
        y_b = y_true[idx]
        if len(np.unique(y_b)) < 2:
            continue
        auc_a = roc_auc_score(y_b, score_a[idx])
        auc_b = roc_auc_score(y_b, score_b[idx])
        diffs.append(auc_b - auc_a)

    diffs = np.array(diffs)
    observed_diff = roc_auc_score(y_true, score_b) - roc_auc_score(y_true, score_a)

    return {
        "observed_diff": observed_diff,
        "mean_diff": float(np.mean(diffs)),
        "ci_low": float(np.percentile(diffs, 2.5)),
        "ci_high": float(np.percentile(diffs, 97.5)),
        "p_value": float(np.mean(diffs <= 0)),
        "diffs": diffs,
    }


# =============================================================================
# Plotting
# =============================================================================


def plot_bootstrap_distributions(comparisons: list[dict], output_path: str) -> None:
    """
    Plot bootstrap Δ distributions for each model comparison.

    Each panel shows the histogram of AUC(B) − AUC(A) bootstrap resamples,
    with the 95% CI shaded and a vertical line at Δ = 0.
    """
    n = len(comparisons)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5), sharey=False)
    if n == 1:
        axes = [axes]

    for ax, comp in zip(axes, comparisons):
        diffs = comp["result"]["diffs"]
        ci_low = comp["result"]["ci_low"]
        ci_high = comp["result"]["ci_high"]
        observed = comp["result"]["observed_diff"]
        p = comp["result"]["p_value"]

        ax.hist(diffs, bins=60, color="steelblue", alpha=0.75, edgecolor="white", linewidth=0.4)

        # Shade 95% CI
        ci_mask = (diffs >= ci_low) & (diffs <= ci_high)
        ax.hist(diffs[ci_mask], bins=60, color="steelblue", alpha=0.35,
                edgecolor="white", linewidth=0.4)

        # Reference lines
        ax.axvline(0, color="black", lw=1.2, linestyle="--", label="Δ = 0")
        ax.axvline(observed, color="firebrick", lw=1.5, linestyle="-",
                   label=f"Observed Δ = {observed:+.3f}")
        ax.axvline(ci_low, color="steelblue", lw=1.0, linestyle=":",
                   label=f"95% CI [{ci_low:+.3f}, {ci_high:+.3f}]")
        ax.axvline(ci_high, color="steelblue", lw=1.0, linestyle=":")

        ax.set_title(f"{comp['label']}\np (Δ ≤ 0) = {p:.3f}", fontsize=10)
        ax.set_xlabel("AUC difference (B − A)")
        ax.set_ylabel("Bootstrap frequency")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Paired Bootstrap Distribution of AUC Differences", fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Plot saved to {output_path}")


# =============================================================================
# Entry point
# =============================================================================


def run(
    artifact_path: str = "results/stacking_model.joblib",
    output_dir: str = "results",
    plots_dir: str = "plots",
    n_boot: int = 10000,
    seed: int = 0,
) -> pd.DataFrame:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    Path(plots_dir).mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading stacking artifacts from {artifact_path}...")
    artifacts = joblib.load(artifact_path)

    result = artifacts["result"]
    y_true = result["y_true"]
    groups = result["groups"]
    bmca_scores = result["bmca"]["scores"]
    mrf_scores = result["mrf"]["scores"]
    stacked_scores = result["stacked"]["scores"]

    logger.info(
        f"Test set: {len(y_true)} subjects, {int(y_true.sum())} transitions, "
        f"{len(np.unique(groups))} matched pairs."
    )
    logger.info(
        f"Observed AUCs — BMCA: {result['bmca']['auc']:.3f}, "
        f"MRF: {result['mrf']['auc']:.3f}, "
        f"Stacked: {result['stacked']['auc']:.3f}"
    )

    # ── Primary comparison: Stacked vs BMCA ──────────────────────────────────
    logger.info(f"\nRunning paired bootstrap (n={n_boot}) — Stacked vs BMCA...")
    stacked_vs_bmca = paired_bootstrap_auc_diff(
        y_true, bmca_scores, stacked_scores, groups, n_boot=n_boot, seed=seed
    )
    logger.info(
        f"  Observed Δ = {stacked_vs_bmca['observed_diff']:+.3f}\n"
        f"  95% CI:    [{stacked_vs_bmca['ci_low']:+.3f}, {stacked_vs_bmca['ci_high']:+.3f}]\n"
        f"  p (Δ ≤ 0): {stacked_vs_bmca['p_value']:.3f}"
    )

    # ── Reference comparison: BMCA vs MRF ────────────────────────────────────
    logger.info(f"\nRunning paired bootstrap (n={n_boot}) — BMCA vs MRF (reference)...")
    bmca_vs_mrf = paired_bootstrap_auc_diff(
        y_true, mrf_scores, bmca_scores, groups, n_boot=n_boot, seed=seed
    )
    logger.info(
        f"  Observed Δ = {bmca_vs_mrf['observed_diff']:+.3f}\n"
        f"  95% CI:    [{bmca_vs_mrf['ci_low']:+.3f}, {bmca_vs_mrf['ci_high']:+.3f}]\n"
        f"  p (Δ ≤ 0): {bmca_vs_mrf['p_value']:.3f}"
    )

    # ── Plot ──────────────────────────────────────────────────────────────────
    comparisons = [
        {
            "label": "Stacked vs BMCA\n(A = BMCA, B = Stacked)",
            "result": stacked_vs_bmca,
        },
        {
            "label": "BMCA vs MRF (reference)\n(A = MRF, B = BMCA)",
            "result": bmca_vs_mrf,
        },
    ]
    plot_bootstrap_distributions(
        comparisons, output_path=f"{plots_dir}/bootstrap_auc_diff.pdf"
    )

    # ── Save CSV ──────────────────────────────────────────────────────────────
    rows = [
        {
            "comparison": "stacked_vs_bmca",
            "model_a": "bmca",
            "model_b": "stacked",
            "auc_a": round(result["bmca"]["auc"], 4),
            "auc_b": round(result["stacked"]["auc"], 4),
            "observed_diff": round(stacked_vs_bmca["observed_diff"], 4),
            "bootstrap_mean_diff": round(stacked_vs_bmca["mean_diff"], 4),
            "ci_low_95": round(stacked_vs_bmca["ci_low"], 4),
            "ci_high_95": round(stacked_vs_bmca["ci_high"], 4),
            "p_value_one_sided": round(stacked_vs_bmca["p_value"], 4),
            "n_boot": n_boot,
        },
        {
            "comparison": "bmca_vs_mrf",
            "model_a": "mrf",
            "model_b": "bmca",
            "auc_a": round(result["mrf"]["auc"], 4),
            "auc_b": round(result["bmca"]["auc"], 4),
            "observed_diff": round(bmca_vs_mrf["observed_diff"], 4),
            "bootstrap_mean_diff": round(bmca_vs_mrf["mean_diff"], 4),
            "ci_low_95": round(bmca_vs_mrf["ci_low"], 4),
            "ci_high_95": round(bmca_vs_mrf["ci_high"], 4),
            "p_value_one_sided": round(bmca_vs_mrf["p_value"], 4),
            "n_boot": n_boot,
        },
    ]
    out_df = pd.DataFrame(rows)
    out_df.to_csv(f"{output_dir}/bootstrap_auc_diff.csv", index=False)
    logger.info(f"Results saved to {output_dir}/bootstrap_auc_diff.csv")

    return out_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Paired bootstrap test for AUC difference between stacked and BMCA models"
    )
    parser.add_argument(
        "--artifact", default="results/stacking_model.joblib",
        help="Path to stacking_model.joblib produced by model_stacking_evaluation.py",
    )
    parser.add_argument("--output_dir", default="results")
    parser.add_argument("--plots_dir", default="plots")
    parser.add_argument(
        "--n_boot", type=int, default=10000,
        help="Number of bootstrap resamples (default: 10000)",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    run(
        artifact_path=args.artifact,
        output_dir=args.output_dir,
        plots_dir=args.plots_dir,
        n_boot=args.n_boot,
        seed=args.seed,
    )
