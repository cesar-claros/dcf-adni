"""
Compare BMCA feature importances between the two clinical pathways:
  - Primary: CN -> MCI/Dementia
  - Augmentation: MCI -> Dementia

Trains BMCA models via nested CV on each population separately, extracts
feature importances, and computes Spearman rank correlation.

Usage::

    python analysis_bmca_feature_comparison.py \
        --bmca data/adni_bmca_features_L4_combined_matched.csv \
        --bmca_audit data/adni_bmca_features_L4_column_audit.csv \
        --output_dir results_paper/bmca_comparison
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr

from model_strate_cv_evaluation import (
    _feature_cols,
    _load_combined,
    run_cv_for_feature_set,
)

logging.basicConfig(level=logging.INFO, format="%(name)s — %(message)s")
logger = logging.getLogger(__name__)


def run(
    bmca_path: str,
    output_dir: str = "results_paper/bmca_comparison",
    bmca_audit: str | None = None,
    n_outer: int = 5,
    n_inner: int = 5,
    n_iter: int = 50,
    seed: int = 0,
    n_jobs: int = 1,
) -> dict:
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    bmca_df = _load_combined(bmca_path)
    bmca_features = _feature_cols(bmca_df, bmca_audit)

    logger.info(f"BMCA feature comparison: {len(bmca_features)} features")

    # Train on primary pairs only (CN -> MCI/Dementia)
    logger.info(f"\n{'='*60}\nBMCA on PRIMARY pairs (CN -> MCI/Dementia)\n{'='*60}")
    r_primary = run_cv_for_feature_set(
        bmca_df, bmca_features, "BMCA_primary",
        n_outer=n_outer, n_inner=n_inner, n_iter=n_iter,
        seed=seed, n_jobs=n_jobs, training_mode="primary_only",
    )

    # Train on augmentation pairs only (MCI -> Dementia)
    logger.info(f"\n{'='*60}\nBMCA on AUGMENTATION pairs (MCI -> Dementia)\n{'='*60}")
    r_aug = run_cv_for_feature_set(
        bmca_df, bmca_features, "BMCA_augmentation",
        n_outer=n_outer, n_inner=n_inner, n_iter=n_iter,
        seed=seed, n_jobs=n_jobs, training_mode="augmentation_only",
    )

    # Save individual importance tables
    primary_imp = r_primary["importance_df"].copy()
    aug_imp = r_aug["importance_df"].copy()
    primary_imp.to_csv(f"{output_dir}/primary_importance.csv", index=False)
    aug_imp.to_csv(f"{output_dir}/augmentation_importance.csv", index=False)

    # Build comparison table
    primary_imp = primary_imp.rename(columns={"importance": "importance_primary"})
    aug_imp = aug_imp.rename(columns={"importance": "importance_augmentation"})

    comparison = primary_imp.merge(aug_imp, on="feature", how="outer")
    comparison["rank_primary"] = comparison["importance_primary"].rank(ascending=False).astype(int)
    comparison["rank_augmentation"] = (
        comparison["importance_augmentation"].rank(ascending=False).astype(int)
    )
    comparison["rank_diff"] = comparison["rank_primary"] - comparison["rank_augmentation"]
    comparison = comparison.sort_values("rank_primary").reset_index(drop=True)

    # Spearman rank correlation
    rho, p_val = spearmanr(comparison["rank_primary"], comparison["rank_augmentation"])
    logger.info(f"\nSpearman rank correlation: rho = {rho:.3f}, p = {p_val:.4f}")

    # Add summary row
    comparison.to_csv(f"{output_dir}/comparison_table.csv", index=False)

    # Save summary statistics
    summary = pd.DataFrame([{
        "spearman_rho": round(rho, 4),
        "spearman_p": round(p_val, 6),
        "n_features": len(comparison),
        "primary_oof_auc": round(r_primary["oof_auc"], 4),
        "augmentation_oof_auc": round(r_aug["oof_auc"], 4),
    }])
    summary.to_csv(f"{output_dir}/comparison_summary.csv", index=False)

    logger.info(f"\nPrimary OOF AUC: {r_primary['oof_auc']:.3f}")
    logger.info(f"Augmentation OOF AUC: {r_aug['oof_auc']:.3f}")
    logger.info(f"\nTop 10 features by pathway:")
    logger.info(f"\n{'Primary (CN→MCI/Dem)':<40} {'Augmentation (MCI→Dem)':<40}")
    logger.info(f"{'-'*40} {'-'*40}")
    for i in range(min(10, len(comparison))):
        p_row = comparison[comparison["rank_primary"] == i + 1].iloc[0]
        a_row = comparison[comparison["rank_augmentation"] == i + 1].iloc[0]
        logger.info(
            f"{i+1:2d}. {p_row['feature']:<36} {i+1:2d}. {a_row['feature']:<36}"
        )

    return {
        "primary": r_primary,
        "augmentation": r_aug,
        "comparison": comparison,
        "spearman_rho": rho,
        "spearman_p": p_val,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BMCA feature importance comparison")
    parser.add_argument("--bmca", required=True)
    parser.add_argument("--output_dir", default="results_paper/bmca_comparison")
    parser.add_argument("--bmca_audit", default=None)
    parser.add_argument("--n_outer", type=int, default=5)
    parser.add_argument("--n_inner", type=int, default=5)
    parser.add_argument("--n_iter", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n_jobs", type=int, default=1)
    args = parser.parse_args()

    run(
        bmca_path=args.bmca,
        output_dir=args.output_dir,
        bmca_audit=args.bmca_audit,
        n_outer=args.n_outer,
        n_inner=args.n_inner,
        n_iter=args.n_iter,
        seed=args.seed,
        n_jobs=args.n_jobs,
    )
