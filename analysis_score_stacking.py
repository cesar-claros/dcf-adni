"""
Post-hoc score stacking: fit logistic regression on BMCA and MRF OOF scores.

This tests whether MRF adds value at the prediction level (not feature level).
Score stacking avoids the "feature dilution" problem of BMCA+MRF feature union
by combining model outputs rather than inputs.

Reads oof_predictions.csv from a completed CV run. No new model training needed.

Usage::

    python analysis_score_stacking.py \
        --oof_dir results_paper/training/combined_seed0 \
        --output_dir results_paper/stacking/combined_seed0

    # Run on multiple seeds
    python analysis_score_stacking.py \
        --oof_dir results_paper/training/combined_seed0 \
                  results_paper/training/combined_seed1 \
        --output_dir results_paper/stacking
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from model_strate_cv_evaluation import _bootstrap_auc, _bootstrap_paired_auc_diff

logging.basicConfig(level=logging.INFO, format="%(name)s — %(message)s")
logger = logging.getLogger(__name__)


def _stack_oof(oof_df: pd.DataFrame, seed: int = 0, n_folds: int = 5) -> dict:
    """Fit logistic regression stacker on BMCA + MRF OOF scores via nested CV.

    Uses leave-one-group-out CV to avoid overfitting the stacker on the same
    OOF predictions it will be evaluated on.
    """
    y = oof_df["y_true"].values
    groups = oof_df["group"].values
    bmca_scores = oof_df["oof_bmca"].values
    mrf_scores = oof_df["oof_mrf"].values

    # Stack: [BMCA_score, MRF_score] -> logistic regression
    X_stack = np.column_stack([bmca_scores, mrf_scores])

    # Nested CV stacking to avoid information leakage
    cv = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    stacked_scores = np.full(len(y), np.nan)
    coefficients = []

    for train_idx, test_idx in cv.split(X_stack, y=y, groups=groups):
        lr = LogisticRegression(penalty="l2", C=1.0, solver="lbfgs", max_iter=1000)
        lr.fit(X_stack[train_idx], y[train_idx])
        stacked_scores[test_idx] = lr.predict_proba(X_stack[test_idx])[:, 1]
        coefficients.append(lr.coef_[0].copy())

    avg_coef = np.mean(coefficients, axis=0)
    stacked_auc = roc_auc_score(y, stacked_scores)
    ci_low, ci_high = _bootstrap_auc(y, stacked_scores, groups, n_boot=2000, seed=seed)

    # MRF weight relative to BMCA
    bmca_weight = avg_coef[0]
    mrf_weight = avg_coef[1]
    mrf_relative = abs(mrf_weight) / (abs(bmca_weight) + abs(mrf_weight)) if (abs(bmca_weight) + abs(mrf_weight)) > 0 else 0.0

    return {
        "stacked_auc": stacked_auc,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "bmca_coefficient": bmca_weight,
        "mrf_coefficient": mrf_weight,
        "mrf_relative_weight": mrf_relative,
        "stacked_scores": stacked_scores,
    }


def run_single(oof_path: str, output_dir: str, seed: int = 0) -> dict:
    """Run score stacking on a single OOF predictions file."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    oof_df = pd.read_csv(oof_path)

    if "oof_bmca" not in oof_df.columns or "oof_mrf" not in oof_df.columns:
        logger.warning(f"Missing BMCA or MRF OOF scores in {oof_path}")
        return {}

    result = _stack_oof(oof_df, seed=seed)

    y = oof_df["y_true"].values
    groups = oof_df["group"].values
    bmca_scores = oof_df["oof_bmca"].values

    # BMCA-only AUC for comparison
    bmca_auc = roc_auc_score(y, bmca_scores)
    bmca_ci_low, bmca_ci_high = _bootstrap_auc(y, bmca_scores, groups, n_boot=2000, seed=seed)

    # Paired bootstrap: Stacked vs BMCA
    bt = _bootstrap_paired_auc_diff(
        y, result["stacked_scores"], bmca_scores, groups,
        n_boot=10000, seed=seed,
    )

    # Summary
    summary = pd.DataFrame([
        {
            "model": "BMCA",
            "auc": round(bmca_auc, 4),
            "ci_low_95": round(bmca_ci_low, 4),
            "ci_high_95": round(bmca_ci_high, 4),
        },
        {
            "model": "Stacked(BMCA+MRF)",
            "auc": round(result["stacked_auc"], 4),
            "ci_low_95": round(result["ci_low"], 4),
            "ci_high_95": round(result["ci_high"], 4),
        },
    ])
    summary.to_csv(f"{output_dir}/stacking_summary.csv", index=False)

    # Stacker details
    details = pd.DataFrame([{
        "bmca_coefficient": round(result["bmca_coefficient"], 4),
        "mrf_coefficient": round(result["mrf_coefficient"], 4),
        "mrf_relative_weight": round(result["mrf_relative_weight"], 4),
        "delta_vs_bmca": round(bt["observed_diff"], 4),
        "delta_ci_low": round(bt["ci_low"], 4),
        "delta_ci_high": round(bt["ci_high"], 4),
        "delta_p_value": round(bt["p_value"], 4),
    }])
    details.to_csv(f"{output_dir}/stacking_details.csv", index=False)

    logger.info(
        f"  Stacked AUC = {result['stacked_auc']:.3f} [{result['ci_low']:.3f}, {result['ci_high']:.3f}]  "
        f"BMCA coef = {result['bmca_coefficient']:.3f}, MRF coef = {result['mrf_coefficient']:.3f} "
        f"(MRF relative weight = {result['mrf_relative_weight']:.1%})"
    )
    logger.info(
        f"  Stacked vs BMCA: Δ = {bt['observed_diff']:+.4f}  "
        f"95% CI [{bt['ci_low']:+.4f}, {bt['ci_high']:+.4f}]  "
        f"p(Δ≤0) = {bt['p_value']:.4f}"
    )

    return result


def run(
    oof_dirs: list[str],
    output_dir: str = "results_paper/stacking",
) -> None:
    """Run score stacking on OOF predictions from one or more directories."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    all_results = []
    for oof_dir in oof_dirs:
        oof_path = Path(oof_dir) / "oof_predictions.csv"
        if not oof_path.exists():
            logger.warning(f"Skipping {oof_dir}: oof_predictions.csv not found")
            continue

        # Extract seed from directory name (e.g., combined_seed0 -> 0)
        dir_name = Path(oof_dir).name
        seed = 0
        if "_seed" in dir_name:
            try:
                seed = int(dir_name.rsplit("_seed", 1)[1])
            except ValueError:
                pass

        sub_output = f"{output_dir}/{dir_name}"
        logger.info(f"\n{'='*60}\nScore stacking: {dir_name} (seed={seed})\n{'='*60}")
        result = run_single(str(oof_path), sub_output, seed=seed)
        if result:
            all_results.append({"source": dir_name, "seed": seed, **{
                k: v for k, v in result.items() if k != "stacked_scores"
            }})

    if all_results:
        agg_df = pd.DataFrame(all_results)
        agg_df.to_csv(f"{output_dir}/stacking_aggregated.csv", index=False)
        logger.info(f"\nAggregated stacking results:\n{agg_df.to_string(index=False)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Score stacking: BMCA + MRF OOF scores")
    parser.add_argument(
        "--oof_dir", nargs="+", required=True,
        help="Directories containing oof_predictions.csv",
    )
    parser.add_argument("--output_dir", default="results_paper/stacking")
    args = parser.parse_args()

    run(oof_dirs=args.oof_dir, output_dir=args.output_dir)
