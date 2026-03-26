"""
Within-pair contrast diagnostics: mechanistic analysis of MRF signal.

For each matched pair, computes the delta vector (transition - control) for
all MRF features. Then measures:
  1. Per-feature signal-to-noise ratio (SNR = mean(delta) / std(delta))
  2. L1-regularized logistic regression on delta vectors (contrast AUC)
  3. Number of features selected by L1 regularization

Runs on both primary (CN->MCI/Dementia) and augmentation (MCI->Dementia)
populations to demonstrate pathway-specificity: MRF deltas carry signal for
augmentation pairs but not primary pairs.

Usage::

    python analysis_within_pair_contrast.py \
        --mrf data/adni_mrf_features_L4_combined_matched.csv \
        --mrf_audit data/adni_mrf_features_L4_column_audit.csv \
        --output_dir results_paper/contrast
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from model_strate_cv_evaluation import (
    GROUP_COL,
    LABEL_COL,
    _feature_cols,
    _load_combined,
)

logging.basicConfig(level=logging.INFO, format="%(name)s — %(message)s")
logger = logging.getLogger(__name__)


def _build_pair_deltas(
    df: pd.DataFrame,
    feature_cols: list[str],
    population: str,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Build delta vectors (transition - control) for each matched pair.

    Returns (delta_df, labels) where labels are always 1 for transition pairs
    and 0 for stable pairs (for the contrast model, all deltas are positive-class
    since we subtract control from transition).
    """
    pop_df = df[df["analysis_set"] == population].copy()

    deltas = []
    pair_ids = []

    for group_id, pair in pop_df.groupby(GROUP_COL):
        if len(pair) != 2:
            continue

        transition_row = pair[pair[LABEL_COL] == 1]
        control_row = pair[pair[LABEL_COL] == 0]

        if len(transition_row) != 1 or len(control_row) != 1:
            continue

        delta = transition_row[feature_cols].values[0] - control_row[feature_cols].values[0]
        deltas.append(delta)
        pair_ids.append(group_id)

    delta_df = pd.DataFrame(deltas, columns=feature_cols, index=pair_ids)
    delta_df.index.name = "group"

    return delta_df, np.ones(len(delta_df))


def _compute_snr(delta_df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-feature signal-to-noise ratio from delta vectors."""
    rows = []
    for col in delta_df.columns:
        vals = delta_df[col].dropna()
        if len(vals) < 5:
            continue
        mean_delta = vals.mean()
        std_delta = vals.std()
        snr = mean_delta / std_delta if std_delta > 0 else 0.0
        rows.append({
            "feature": col,
            "mean_delta": mean_delta,
            "std_delta": std_delta,
            "snr": snr,
            "abs_snr": abs(snr),
            "n_valid": len(vals),
        })
    return pd.DataFrame(rows).sort_values("abs_snr", ascending=False).reset_index(drop=True)


def _contrast_model(
    delta_df: pd.DataFrame,
    seed: int = 0,
    n_folds: int = 5,
) -> dict:
    """Fit L1-regularized logistic regression on delta vectors.

    Uses a binary classification setup: for each pair, the contrast vector
    (transition - control) should be distinguishable from shuffled contrasts.
    We create negative examples by randomly flipping the sign of delta vectors.
    """
    X = delta_df.values.copy()
    n_pairs = X.shape[0]

    # Handle NaN: replace with 0 (neutral delta)
    X = np.nan_to_num(X, nan=0.0)

    # Create positive (real deltas) and negative (sign-flipped) examples
    rng = np.random.RandomState(seed)
    X_neg = -X[rng.permutation(n_pairs)]
    X_all = np.vstack([X, X_neg])
    y_all = np.concatenate([np.ones(n_pairs), np.zeros(n_pairs)])

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_all)

    # L1-regularized logistic regression with CV
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    try:
        lr = LogisticRegressionCV(
            penalty="l1",
            solver="saga",
            Cs=20,
            cv=cv,
            scoring="roc_auc",
            max_iter=5000,
            random_state=seed,
        )
        lr.fit(X_scaled, y_all)

        # OOF predictions via cross-validation
        oof_scores = np.full(len(y_all), np.nan)
        for train_idx, test_idx in cv.split(X_scaled, y_all):
            lr_fold = LogisticRegressionCV(
                penalty="l1", solver="saga", Cs=20,
                cv=3, scoring="roc_auc", max_iter=5000, random_state=seed,
            )
            lr_fold.fit(X_scaled[train_idx], y_all[train_idx])
            oof_scores[test_idx] = lr_fold.predict_proba(X_scaled[test_idx])[:, 1]

        oof_auc = roc_auc_score(y_all, oof_scores)
        best_C = float(lr.C_[0])
        n_selected = int(np.sum(np.abs(lr.coef_[0]) > 1e-6))

        # Feature coefficients
        coef_df = pd.DataFrame({
            "feature": delta_df.columns,
            "coefficient": lr.coef_[0],
            "abs_coefficient": np.abs(lr.coef_[0]),
        }).sort_values("abs_coefficient", ascending=False).reset_index(drop=True)

    except Exception as e:
        logger.warning(f"Contrast model failed: {e}")
        oof_auc = 0.5
        best_C = None
        n_selected = 0
        coef_df = pd.DataFrame(columns=["feature", "coefficient", "abs_coefficient"])

    return {
        "contrast_auc": oof_auc,
        "best_C": best_C,
        "n_features_selected": n_selected,
        "n_features_total": X.shape[1],
        "n_pairs": n_pairs,
        "coefficients_df": coef_df,
    }


def run(
    mrf_path: str,
    output_dir: str = "results_paper/contrast",
    mrf_audit: str | None = None,
    seed: int = 0,
) -> dict:
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    mrf_df = _load_combined(mrf_path)
    mrf_features = _feature_cols(mrf_df, mrf_audit)

    results = {}
    for population, label in [("primary", "CN->MCI/Dementia"), ("augmentation", "MCI->Dementia")]:
        logger.info(f"\n{'='*60}\n{population.upper()} pairs ({label})\n{'='*60}")

        delta_df, _ = _build_pair_deltas(mrf_df, mrf_features, population)
        logger.info(f"  {len(delta_df)} pairs, {len(mrf_features)} MRF features")

        # SNR analysis
        snr_df = _compute_snr(delta_df)
        snr_df.to_csv(f"{output_dir}/{population}_snr.csv", index=False)

        logger.info(f"\n  Top 10 features by |SNR|:")
        for _, row in snr_df.head(10).iterrows():
            logger.info(
                f"    {row['feature']:<35} SNR = {row['snr']:+.3f}  "
                f"(mean = {row['mean_delta']:+.4f}, std = {row['std_delta']:.4f})"
            )

        # Contrast model
        contrast = _contrast_model(delta_df, seed=seed)
        logger.info(
            f"\n  Contrast model: OOF AUC = {contrast['contrast_auc']:.3f}, "
            f"C = {contrast['best_C']}, "
            f"{contrast['n_features_selected']}/{contrast['n_features_total']} features selected"
        )

        contrast["coefficients_df"].to_csv(
            f"{output_dir}/{population}_contrast_coefficients.csv", index=False
        )

        results[population] = {
            "n_pairs": len(delta_df),
            "contrast_auc": contrast["contrast_auc"],
            "best_C": contrast["best_C"],
            "n_features_selected": contrast["n_features_selected"],
            "n_features_total": contrast["n_features_total"],
            "median_abs_snr": float(snr_df["abs_snr"].median()),
            "max_abs_snr": float(snr_df["abs_snr"].max()),
            "top_snr_feature": snr_df.iloc[0]["feature"] if len(snr_df) > 0 else "",
        }

    # Comparison table
    comparison = pd.DataFrame([
        {"population": pop, **info}
        for pop, info in results.items()
    ])
    comparison.to_csv(f"{output_dir}/contrast_comparison.csv", index=False)

    logger.info(f"\n{'='*60}\nComparison\n{'='*60}")
    logger.info(f"\n{comparison.to_string(index=False)}")

    # Interpretation
    if len(results) == 2:
        p_auc = results["primary"]["contrast_auc"]
        a_auc = results["augmentation"]["contrast_auc"]
        p_snr = results["primary"]["median_abs_snr"]
        a_snr = results["augmentation"]["median_abs_snr"]
        logger.info(f"\n  Contrast AUC ratio (aug/primary): {a_auc/p_auc:.2f}")
        logger.info(f"  Median |SNR| ratio (aug/primary): {a_snr/p_snr:.1f}x")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Within-pair contrast diagnostics")
    parser.add_argument("--mrf", required=True)
    parser.add_argument("--mrf_audit", default=None)
    parser.add_argument("--output_dir", default="results_paper/contrast")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    run(
        mrf_path=args.mrf,
        output_dir=args.output_dir,
        mrf_audit=args.mrf_audit,
        seed=args.seed,
    )
