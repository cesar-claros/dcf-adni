"""
Strategy E: Full CV evaluation over all primary pairs.

Instead of a single train/test split (16 test pairs), this script runs
StratifiedGroupKFold over all 47 primary CN->dementia pairs. Augmentation
pairs are always included in training. Each primary pair rotates through
exactly one test fold, yielding out-of-fold (OOF) predictions for all 47
pairs — tripling the effective test size.

For each feature set (BMCA, MRF, BMCA+MRF), the script:
1. Runs k-fold outer CV (groups = matched pair IDs).
2. Within each fold, runs Optuna hyperparameter tuning with an inner CV.
3. Collects OOF predictions for primary pairs.
4. Reports OOF AUC with bootstrap 95% CI over all 47 pairs.

Usage::

    python scripts/model_strate_cv_evaluation.py --n_iter 50 --n_outer 5 --seed 0 --n_jobs 1
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dcf_adni.modeling.bootstrap import bootstrap_auc_ci, paired_bootstrap_auc_diff
from dcf_adni.modeling.protocols import run_nested_cv
from dcf_adni.modeling.schema import (
    GROUP_COL,
    METADATA_COLS,
    TRANSITION_COL,
    feature_cols,
    load_combined,
)
from dcf_adni.paths import RESULTS_DIR

logging.basicConfig(level=logging.INFO, format="%(name)s — %(message)s")
logger = logging.getLogger(__name__)

def run(
    bmca_path: str = "data/adni_bmca_features_strate_combined_matched.csv",
    mrf_path: str = "data/adni_mrf_features_strate_combined_matched.csv",
    output_dir: str = str(RESULTS_DIR / "strate_cv"),
    n_outer: int = 5,
    n_inner: int = 5,
    n_iter: int = 50,
    seed: int = 0,
    n_jobs: int = 1,
    bmca_audit: str | None = None,
    mrf_audit: str | None = None,
    training_mode: str = "combined",
) -> dict:
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    bmca_df = load_combined(bmca_path)
    mrf_df = load_combined(mrf_path)

    bmca_features = feature_cols(bmca_df, bmca_audit)
    mrf_features = feature_cols(mrf_df, mrf_audit)

    # Build BMCA+MRF by merging on metadata
    meta_cols = [c for c in bmca_df.columns if c in METADATA_COLS]
    bmca_mrf_df = bmca_df.merge(
        mrf_df.drop(columns=[c for c in meta_cols if c != "subject_id"], errors="ignore"),
        on="subject_id",
        how="inner",
        suffixes=("", "_mrf_dup"),
    )
    # Drop any duplicate columns from merge
    bmca_mrf_df = bmca_mrf_df[[c for c in bmca_mrf_df.columns if not c.endswith("_mrf_dup")]]
    bmca_mrf_features = sorted(set(bmca_features) | set(mrf_features))

    if training_mode == "augmentation_only":
        n_cv_pairs = bmca_df[bmca_df["analysis_set"] == "augmentation"][GROUP_COL].nunique()
        pop_label = "augmentation"
    else:
        n_cv_pairs = bmca_df[bmca_df["analysis_set"] == "primary"][GROUP_COL].nunique()
        pop_label = "primary"
    logger.info(
        f"Full CV ({training_mode}): {n_cv_pairs} {pop_label} pairs, "
        f"{n_outer}-fold outer CV, {n_iter} Optuna trials per fold."
    )

    results = {}
    for name, df, feats in [
        ("BMCA", bmca_df, bmca_features),
        ("MRF", mrf_df, mrf_features),
        ("BMCA+MRF", bmca_mrf_df, bmca_mrf_features),
    ]:
        logger.info(f"\n{'='*60}\n{name} ({len(feats)} features)\n{'='*60}")
        r = run_nested_cv(
            df, feats, name,
            n_outer=n_outer, n_inner=n_inner, n_iter=n_iter,
            seed=seed, n_jobs=n_jobs, training_mode=training_mode,
        )
        results[name] = r

        r["importance_df"].to_csv(f"{output_dir}/{name.lower().replace('+','_')}_importance.csv", index=False)

    # LIBRA raw-score baseline (no model — just AUC of pre-computed score)
    _LIBRA_COL = "libra_supported_rescaled_0_100"
    y_cv = results["BMCA"]["y_true"]
    groups_cv = results["BMCA"]["groups"]
    if _LIBRA_COL in mrf_df.columns:
        if training_mode == "augmentation_only":
            libra_pop = mrf_df[mrf_df["analysis_set"] == "augmentation"]
        else:
            libra_pop = mrf_df[mrf_df["analysis_set"] == "primary"]
        libra_scores = libra_pop[_LIBRA_COL].values
        libra_valid = ~np.isnan(libra_scores) & ~np.isnan(y_cv)
        if libra_valid.sum() > 0 and len(np.unique(y_cv[libra_valid])) >= 2:
            libra_auc = roc_auc_score(y_cv[libra_valid], libra_scores[libra_valid])
            libra_ci_low, libra_ci_high = bootstrap_auc_ci(
                y_cv[libra_valid], libra_scores[libra_valid],
                groups_cv[libra_valid], n_boot=2000, seed=seed,
            )
            results["LIBRA"] = {
                "name": "LIBRA",
                "oof_auc": libra_auc,
                "ci_low": libra_ci_low,
                "ci_high": libra_ci_high,
                "fold_aucs": [],
                "importance_df": pd.DataFrame(
                    {"feature": [_LIBRA_COL], "importance": [100.0]}
                ),
                "oof_scores": libra_scores,
                "y_true": y_cv,
                "groups": groups_cv,
            }
            logger.info(
                f"  [LIBRA] Raw-score AUC = {libra_auc:.3f}  "
                f"95% CI [{libra_ci_low:.3f}, {libra_ci_high:.3f}]"
            )

    # Save OOF predictions
    oof_df = pd.DataFrame({"group": results["BMCA"]["groups"], "y_true": results["BMCA"]["y_true"]})
    for name, r in results.items():
        oof_df[f"oof_{name.lower().replace('+','_')}"] = r["oof_scores"]
    oof_df.to_csv(f"{output_dir}/oof_predictions.csv", index=False)

    # Summary table
    summary_rows = []
    for name, r in results.items():
        summary_rows.append({
            "model": name,
            "oof_auc": round(r["oof_auc"], 4),
            "ci_low_95": round(r["ci_low"], 4),
            "ci_high_95": round(r["ci_high"], 4),
            "fold_aucs": str([round(a, 3) for a in r["fold_aucs"]]),
        })
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(f"{output_dir}/strate_cv_summary.csv", index=False)

    logger.info(f"\n{'='*60}\nSummary\n{'='*60}")
    logger.info(f"\n{summary_df.to_string(index=False)}")

    # Paired bootstrap AUC difference tests
    y_true = results["BMCA"]["y_true"]
    groups = results["BMCA"]["groups"]
    comparisons = [
        ("BMCA+MRF", "BMCA"),
        ("MRF", "BMCA"),
        ("LIBRA", "BMCA"),
    ]
    bootstrap_rows = []
    for name_a, name_b in comparisons:
        if name_a in results and name_b in results:
            bt = paired_bootstrap_auc_diff(
                y_true, results[name_a]["oof_scores"], results[name_b]["oof_scores"],
                groups, n_boot=10000, seed=seed,
            )
            bootstrap_rows.append({
                "comparison": f"{name_a} vs {name_b}",
                "observed_diff": round(bt["observed_diff"], 4),
                "ci_low_95": round(bt["ci_low"], 4),
                "ci_high_95": round(bt["ci_high"], 4),
                "p_value": round(bt["p_value"], 4),
                "n_boot": bt["n_boot"],
            })
            logger.info(
                f"  {name_a} vs {name_b}: Δ = {bt['observed_diff']:+.4f}  "
                f"95% CI [{bt['ci_low']:+.4f}, {bt['ci_high']:+.4f}]  "
                f"p(Δ≤0) = {bt['p_value']:.4f}"
            )
    if bootstrap_rows:
        bt_df = pd.DataFrame(bootstrap_rows)
        bt_df.to_csv(f"{output_dir}/bootstrap_paired_diff.csv", index=False)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Strategy E full CV evaluation")
    parser.add_argument("--bmca", default="data/adni_bmca_features_strate_combined_matched.csv")
    parser.add_argument("--mrf", default="data/adni_mrf_features_strate_combined_matched.csv")
    parser.add_argument("--output_dir", default=str(RESULTS_DIR / "strate_cv"))
    parser.add_argument("--n_outer", type=int, default=5)
    parser.add_argument("--n_inner", type=int, default=5)
    parser.add_argument("--n_iter", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n_jobs", type=int, default=1)
    parser.add_argument("--bmca_audit", default=None, help="BMCA column audit CSV; only keep_for_modeling=1 features used")
    parser.add_argument("--mrf_audit", default=None, help="MRF column audit CSV; only keep_for_modeling=1 features used")
    parser.add_argument(
        "--training_mode", default="combined",
        choices=["combined", "primary_only", "augmentation_only"],
        help="Which subjects to use: combined (primary+aug), primary_only, or augmentation_only",
    )
    args = parser.parse_args()

    run(
        bmca_path=args.bmca,
        mrf_path=args.mrf,
        output_dir=args.output_dir,
        n_outer=args.n_outer,
        n_inner=args.n_inner,
        n_iter=args.n_iter,
        seed=args.seed,
        n_jobs=args.n_jobs,
        bmca_audit=args.bmca_audit,
        mrf_audit=args.mrf_audit,
        training_mode=args.training_mode,
    )
