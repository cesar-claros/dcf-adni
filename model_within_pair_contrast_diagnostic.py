"""
Diagnostic: Within-Pair Contrast Signal — Augmentation vs Primary Pairs
========================================================================

Experiment 13 showed the within-pair MRF contrast model achieves OOF AUC = 0.440
on primary pairs (CN→MCI), with max within-pair SNR = 0.185. This diagnostic runs
the identical contrast model on augmentation pairs (CI→dementia, 267 pairs) and
compares the two datasets head-to-head.

Key question
------------
Is the within-pair contrast failure in Experiment 13 pathway-specific (CN→MCI
has no within-pair MRF signal) or general (the contrast approach itself is
uninformative regardless of the transition type)?

Context
-------
- Exp. 12 Stage 1: MRF trained as independent subjects on 534 augmentation subjects
  achieved OOF AUC = 0.733 for CI→dementia. MRF has real population-level signal
  for this task.
- Exp. 13: Within-pair MRF contrast on 119 primary pairs achieves OOF AUC = 0.440
  (below chance). Within-pair MRF signal is absent for CN→MCI.

If augmentation pairs show a contrast OOF AUC >> 0.5, it confirms:
  1. Within-pair MRF variation is informative for CI→dementia (later-stage task).
  2. The CN→MCI failure is pathway-specific, not a methodological artefact.
  3. The matched design eliminates more within-pair MRF variance for CN→MCI than
     for CI→dementia.

If augmentation pairs also fail (OOF AUC ≈ 0.5), it confirms:
  1. Within-pair MRF differences are noise regardless of transition type.
  2. The independent-subject AUC = 0.733 (Exp. 12) reflects between-pair signal
     that matching eliminates, not within-pair signal.

Design
------
For both analysis sets independently:
  1. Compute within-pair SNR = |mean_delta| / std_delta per MRF feature.
  2. Build sign-flipped contrast dataset (delta = transition − stable, 50% flipped).
  3. Train L1 logistic regression (k=5 OOF KFold on pairs, C by inner 3-fold CV).
  4. Report OOF AUC, selected features, top coefficients.

No BMCA model is trained — this is a pure MRF within-pair signal diagnostic.

Usage::

    python model_within_pair_contrast_diagnostic.py
    python model_within_pair_contrast_diagnostic.py --seed 42 --cv_folds 5
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format="%(name)s — %(message)s")
logger = logging.getLogger(__name__)

_METADATA_COLS = {
    "subject_id", "pair_id", "group", "transition", "transition_label",
    "matched_cohort", "analysis_set", "evaluation_eligible",
    "abs_age_gap", "split", "split_group_source",
    "first_conversion_month", "baseline_diagnosis", "n_followup_visits_ge12_with_diag",
}

LABEL_COL = "transition_label"
GROUP_COL = "group"


def _feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in _METADATA_COLS]


# =============================================================================
# SNR diagnostic
# =============================================================================


def compute_snr(mrf_df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """
    Within-pair SNR = |mean(delta)| / std(delta) per feature.
    Delta is always transition − stable (before sign flipping).
    """
    raw_deltas = []
    for _, sub_df in mrf_df.groupby(GROUP_COL):
        trans_rows = sub_df[sub_df[LABEL_COL] == 1]
        stable_rows = sub_df[sub_df[LABEL_COL] == 0]
        if len(trans_rows) != 1 or len(stable_rows) != 1:
            continue
        delta = (
            trans_rows.iloc[0][feature_cols].values.astype(float)
            - stable_rows.iloc[0][feature_cols].values.astype(float)
        )
        raw_deltas.append(np.where(np.isnan(delta), 0.0, delta))

    raw_deltas = np.array(raw_deltas)
    means = np.nanmean(raw_deltas, axis=0)
    stds = np.nanstd(raw_deltas, axis=0)
    snr = np.abs(means) / (stds + 1e-9)

    return (
        pd.DataFrame({"feature": feature_cols, "mean_delta": means, "std_delta": stds, "snr": snr})
        .sort_values("snr", ascending=False)
        .reset_index(drop=True)
    )


# =============================================================================
# Contrast model
# =============================================================================


def run_contrast_model(
    mrf_df: pd.DataFrame,
    feature_cols: list[str],
    label: str,
    seed: int = 0,
    cv_folds: int = 5,
) -> dict:
    """
    Build the sign-flipped contrast dataset and train L1 logistic regression.

    Returns OOF AUC, selected features, coefficients, and summary stats.
    """
    rng = np.random.default_rng(seed)
    delta_rows, direction_labels = [], []

    skipped = 0
    for _, sub_df in mrf_df.groupby(GROUP_COL):
        trans_rows = sub_df[sub_df[LABEL_COL] == 1]
        stable_rows = sub_df[sub_df[LABEL_COL] == 0]
        if len(trans_rows) != 1 or len(stable_rows) != 1:
            skipped += 1
            continue

        delta = (
            trans_rows.iloc[0][feature_cols].values.astype(float)
            - stable_rows.iloc[0][feature_cols].values.astype(float)
        )
        delta = np.where(np.isnan(delta), 0.0, delta)

        flipped = bool(rng.random() < 0.5)
        if flipped:
            delta = -delta
            direction_label = 0
        else:
            direction_label = 1

        delta_rows.append(delta)
        direction_labels.append(direction_label)

    if skipped > 0:
        logger.warning(f"{label}: skipped {skipped} malformed pairs")

    X = np.array(delta_rows)
    y = np.array(direction_labels)
    n_pairs = len(X)
    logger.info(
        f"{label}: {n_pairs} pairs | "
        f"direction=1: {y.sum()}, direction=0: {n_pairs - y.sum()}"
    )

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # OOF CV on pairs
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    oof_proba = np.full(n_pairs, np.nan)

    for tr_idx, val_idx in kf.split(X_scaled):
        m = LogisticRegressionCV(
            Cs=10, cv=3, penalty="l1", solver="saga",
            scoring="roc_auc", random_state=seed, max_iter=5000,
        )
        m.fit(X_scaled[tr_idx], y[tr_idx])
        oof_proba[val_idx] = m.predict_proba(X_scaled[val_idx])[:, 1]

    oof_auc = roc_auc_score(y, oof_proba)
    logger.info(f"{label}: OOF AUC = {oof_auc:.4f}")

    # Final model on all pairs
    final_model = LogisticRegressionCV(
        Cs=10, cv=5, penalty="l1", solver="saga",
        scoring="roc_auc", random_state=seed, max_iter=5000,
    )
    final_model.fit(X_scaled, y)

    n_nonzero = int((final_model.coef_[0] != 0).sum())
    logger.info(
        f"{label}: selected C = {final_model.C_[0]:.4f}, "
        f"{n_nonzero}/{len(feature_cols)} features non-zero (L1)"
    )

    coef_df = (
        pd.DataFrame({
            "feature": feature_cols,
            "coefficient": final_model.coef_[0],
            "abs_coefficient": np.abs(final_model.coef_[0]),
        })
        .sort_values("abs_coefficient", ascending=False)
        .reset_index(drop=True)
    )

    return {
        "label": label,
        "oof_auc": oof_auc,
        "n_pairs": n_pairs,
        "n_features_selected": n_nonzero,
        "n_features_total": len(feature_cols),
        "selected_C": float(final_model.C_[0]),
        "coef_df": coef_df,
    }


# =============================================================================
# Entry point
# =============================================================================


def run(
    mrf_train_path: str = "data/adni_mrf_features_train.csv",
    output_dir: str = "results",
    seed: int = 0,
    cv_folds: int = 5,
) -> dict:
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    mrf_train = pd.read_csv(mrf_train_path)
    feature_cols = _feature_cols(mrf_train)

    mrf_primary = mrf_train[mrf_train["analysis_set"] == "primary"].copy()
    mrf_aug = mrf_train[mrf_train["analysis_set"] == "augmentation"].copy()

    logger.info(
        f"\n{'='*60}\n"
        f"Within-Pair Contrast Diagnostic: Augmentation vs Primary\n"
        f"{'='*60}\n"
        f"Primary pairs:     {mrf_primary[GROUP_COL].nunique()} "
        f"({len(mrf_primary)} subjects, CN→MCI)\n"
        f"Augmentation pairs:{mrf_aug[GROUP_COL].nunique()} "
        f"({len(mrf_aug)} subjects, CI→dementia)\n"
        f"MRF features:      {len(feature_cols)}"
    )

    # ------------------------------------------------------------------
    # SNR diagnostic for both sets
    # ------------------------------------------------------------------
    logger.info("\n--- Within-pair SNR: primary (CN→MCI) ---")
    snr_primary = compute_snr(mrf_primary, feature_cols)
    snr_primary.columns = ["feature", "mean_delta_primary", "std_delta_primary", "snr_primary"]

    logger.info("\n--- Within-pair SNR: augmentation (CI→dementia) ---")
    snr_aug = compute_snr(mrf_aug, feature_cols)
    snr_aug.columns = ["feature", "mean_delta_aug", "std_delta_aug", "snr_aug"]

    snr_comparison = snr_primary.merge(snr_aug, on="feature")
    snr_comparison["snr_ratio_aug_vs_primary"] = (
        snr_comparison["snr_aug"] / (snr_comparison["snr_primary"] + 1e-9)
    )
    snr_comparison = snr_comparison.sort_values("snr_aug", ascending=False).reset_index(drop=True)

    logger.info(
        f"\nSNR comparison (sorted by augmentation SNR):\n"
        f"{snr_comparison[['feature', 'snr_primary', 'snr_aug', 'snr_ratio_aug_vs_primary']].head(15).to_string(index=False)}"
    )
    logger.info(
        f"\nSummary:\n"
        f"  Primary   — max SNR: {snr_primary['snr_primary'].max():.4f}, "
        f"median: {snr_primary['snr_primary'].median():.4f}\n"
        f"  Augmentation — max SNR: {snr_aug['snr_aug'].max():.4f}, "
        f"median: {snr_aug['snr_aug'].median():.4f}"
    )

    snr_comparison.to_csv(f"{output_dir}/contrast_diagnostic_snr_comparison.csv", index=False)

    # ------------------------------------------------------------------
    # Contrast model: primary pairs
    # ------------------------------------------------------------------
    logger.info("\n--- Contrast model: primary pairs (CN→MCI) ---")
    result_primary = run_contrast_model(
        mrf_primary, feature_cols, label="Primary (CN→MCI)",
        seed=seed, cv_folds=cv_folds,
    )
    result_primary["coef_df"].to_csv(
        f"{output_dir}/contrast_diagnostic_primary_coefficients.csv", index=False
    )

    # ------------------------------------------------------------------
    # Contrast model: augmentation pairs
    # ------------------------------------------------------------------
    logger.info("\n--- Contrast model: augmentation pairs (CI→dementia) ---")
    result_aug = run_contrast_model(
        mrf_aug, feature_cols, label="Augmentation (CI→dementia)",
        seed=seed, cv_folds=cv_folds,
    )
    result_aug["coef_df"].to_csv(
        f"{output_dir}/contrast_diagnostic_augmentation_coefficients.csv", index=False
    )

    # ------------------------------------------------------------------
    # Comparison summary
    # ------------------------------------------------------------------
    summary = pd.DataFrame([
        {
            "analysis_set": r["label"],
            "n_pairs": r["n_pairs"],
            "contrast_oof_auc": round(r["oof_auc"], 4),
            "n_features_selected_l1": r["n_features_selected"],
            "n_features_total": r["n_features_total"],
            "selected_C": round(r["selected_C"], 4),
            "max_within_pair_snr": round(
                snr_primary["snr_primary"].max() if "Primary" in r["label"]
                else snr_aug["snr_aug"].max(), 4
            ),
            "median_within_pair_snr": round(
                snr_primary["snr_primary"].median() if "Primary" in r["label"]
                else snr_aug["snr_aug"].median(), 4
            ),
        }
        for r in [result_primary, result_aug]
    ])
    summary.to_csv(f"{output_dir}/contrast_diagnostic_summary.csv", index=False)

    # ------------------------------------------------------------------
    # Final log
    # ------------------------------------------------------------------
    logger.info(
        f"\n{'='*60}\n"
        f"SUMMARY: Within-Pair Contrast Diagnostic\n"
        f"{'='*60}\n"
        f"  {'Set':<30} {'Pairs':>6} {'OOF AUC':>9} {'L1 selected':>12} {'Max SNR':>9} {'Median SNR':>11}\n"
        f"  {'-'*79}\n"
        f"  {'Primary (CN→MCI)':<30} {result_primary['n_pairs']:>6} "
        f"{result_primary['oof_auc']:>9.4f} "
        f"{result_primary['n_features_selected']:>12}/{result_primary['n_features_total']} "
        f"{snr_primary['snr_primary'].max():>9.4f} "
        f"{snr_primary['snr_primary'].median():>11.4f}\n"
        f"  {'Augmentation (CI→dementia)':<30} {result_aug['n_pairs']:>6} "
        f"{result_aug['oof_auc']:>9.4f} "
        f"{result_aug['n_features_selected']:>12}/{result_aug['n_features_total']} "
        f"{snr_aug['snr_aug'].max():>9.4f} "
        f"{snr_aug['snr_aug'].median():>11.4f}\n"
        f"\nInterpretation:\n"
        f"  Expected if pathway-specific: aug OOF AUC >> 0.5, primary ≈ 0.5\n"
        f"  Expected if general failure:  both OOF AUC ≈ 0.5\n"
        f"{'='*60}"
    )

    return {
        "result_primary": result_primary,
        "result_aug": result_aug,
        "snr_comparison": snr_comparison,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Diagnostic: Within-pair contrast signal for augmentation vs primary pairs"
    )
    parser.add_argument("--mrf_train", default="data/adni_mrf_features_train.csv")
    parser.add_argument("--output_dir", default="results")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cv_folds", type=int, default=5)
    args = parser.parse_args()

    run(
        mrf_train_path=args.mrf_train,
        output_dir=args.output_dir,
        seed=args.seed,
        cv_folds=args.cv_folds,
    )
