"""
Strategy 4: Within-Pair Contrast Model
=======================================

Tests whether within-pair MRF differences (transition vs stable subject) carry
discriminative signal when modelled directly as signed contrast vectors.

All prior experiments treated subjects as independent rows. The matched
case-control design makes a different comparison natural: "Given two subjects
matched on age, sex, and APOE genotype, which MRF profile is associated with
transition?" This is the direct analogue of conditional logistic regression —
the gold standard for matched case-control studies.

Design
------
1. **Construct contrast dataset (primary training pairs only):**
   - For each of the 119 primary training pairs, compute:
     delta = transition_features − stable_features.
   - With probability 0.5 (seed-controlled RNG), flip the sign and set
     direction_label = 0; otherwise direction_label = 1.
   - Result: 119 balanced samples (direction_label ∈ {0, 1}).
   - NaN within-pair differences are set to 0 (no detectable difference).

2. **Diagnostic:** Compute within-pair SNR = |mean_delta| / std_delta per MRF
   feature. If SNR is near zero for all features, within-pair MRF variation is
   pure noise and the contrast model is expected to fail.

3. **Train L1 logistic regression with OOF scoring (pair-level k=5 CV):**
   - Each row is a pair (not a subject). KFold on pairs is correct here.
   - OOF direction probabilities (P(delta is trans−stable direction)) computed
     for each training pair without seeing its own label.
   - Final model trained on all 119 primary pairs.

4. **Assign contrast scores to all subjects:**
   - Primary training: OOF direction proba → subject-level scores (leakage-free).
     transition score = P(trans is transition) computed from flip-corrected OOF proba.
     stable score = 1 − transition score.
   - Augmentation training: 0.5 (neutral — contrast model not trained on CI→dementia).
   - Test: out-of-sample. For each test pair (A, B):
     P(A is transition) = avg(model(A−B), 1 − model(B−A)) for robustness.
     P(B is transition) = 1 − P(A is transition).

5. **Train BMCA baseline and BMCA + mrf_contrast_score** head-to-head with Optuna
   (50 trials, StratifiedGroupKFold k=5, primary-only val_mask).

6. **Evaluate** on primary test set; run paired bootstrap AUC difference test.

Usage::

    python model_within_pair_contrast_evaluation.py
    python model_within_pair_contrast_evaluation.py --n_iter 100 --seed 42
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import RocCurveDisplay, roc_auc_score
from sklearn.model_selection import KFold, StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
from src.utils_model import train_model

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
SUBJECT_ID_COL = "subject_id"
CONTRAST_SCORE_COL = "mrf_contrast_score"


# =============================================================================
# Data helpers
# =============================================================================


def _feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in _METADATA_COLS]


def _evaluation_eligible(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["evaluation_eligible"] == 1].copy()


def _bootstrap_auc(
    y_true: np.ndarray,
    y_score: np.ndarray,
    groups: np.ndarray,
    n_boot: int = 1000,
    seed: int = 0,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    unique_groups = np.unique(groups)
    boot_aucs = []
    for _ in range(n_boot):
        sampled = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        idx = np.concatenate([np.where(groups == g)[0] for g in sampled])
        y_b, s_b = y_true[idx], y_score[idx]
        if len(np.unique(y_b)) < 2:
            continue
        boot_aucs.append(roc_auc_score(y_b, s_b))
    return float(np.percentile(boot_aucs, 2.5)), float(np.percentile(boot_aucs, 97.5))


def _bootstrap_auc_diff(
    y_true: np.ndarray,
    y_score_a: np.ndarray,
    y_score_b: np.ndarray,
    groups: np.ndarray,
    n_boot: int = 10_000,
    seed: int = 0,
) -> dict:
    rng = np.random.default_rng(seed)
    unique_groups = np.unique(groups)
    obs_diff = roc_auc_score(y_true, y_score_a) - roc_auc_score(y_true, y_score_b)
    boot_diffs = []
    for _ in range(n_boot):
        sampled = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        idx = np.concatenate([np.where(groups == g)[0] for g in sampled])
        y_b = y_true[idx]
        if len(np.unique(y_b)) < 2:
            continue
        boot_diffs.append(
            roc_auc_score(y_b, y_score_a[idx]) - roc_auc_score(y_b, y_score_b[idx])
        )
    boot_diffs = np.array(boot_diffs)
    return {
        "observed_diff": obs_diff,
        "ci_low": float(np.percentile(boot_diffs, 2.5)),
        "ci_high": float(np.percentile(boot_diffs, 97.5)),
        "p_value": float(np.mean(boot_diffs <= 0)),
        "n_boot": len(boot_diffs),
    }


# =============================================================================
# SNR diagnostic
# =============================================================================


def compute_snr_table(
    mrf_primary: pd.DataFrame,
    feature_cols: list[str],
) -> pd.DataFrame:
    """
    Compute mean within-pair delta and SNR per MRF feature.

    Delta is always transition − stable (before sign flipping).
    SNR = |mean_delta| / std_delta. High SNR means the feature
    consistently differs between transition and stable subjects within pairs.
    """
    raw_deltas = []
    for _, sub_df in mrf_primary.groupby(GROUP_COL):
        trans_rows = sub_df[sub_df[LABEL_COL] == 1]
        stable_rows = sub_df[sub_df[LABEL_COL] == 0]
        if len(trans_rows) != 1 or len(stable_rows) != 1:
            continue
        delta = (
            trans_rows.iloc[0][feature_cols].values.astype(float)
            - stable_rows.iloc[0][feature_cols].values.astype(float)
        )
        raw_deltas.append(np.where(np.isnan(delta), 0.0, delta))

    raw_deltas = np.array(raw_deltas)  # (n_pairs, n_features)
    means = np.nanmean(raw_deltas, axis=0)
    stds = np.nanstd(raw_deltas, axis=0)
    snr = np.abs(means) / (stds + 1e-9)

    return (
        pd.DataFrame({"feature": feature_cols, "mean_delta": means, "std_delta": stds, "snr": snr})
        .sort_values("snr", ascending=False)
        .reset_index(drop=True)
    )


# =============================================================================
# Contrast dataset construction
# =============================================================================


def build_contrast_dataset(
    mrf_primary: pd.DataFrame,
    feature_cols: list[str],
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    """
    Build a balanced contrast dataset from primary training pairs.

    For each pair:
    - Compute delta = transition_features − stable_features.
    - With P=0.5, flip sign → delta = stable − transition, direction_label = 0.
    - Otherwise direction_label = 1.
    - NaN differences are set to 0 (no detectable within-pair difference).

    Returns
    -------
    delta_df         : (n_pairs, n_features) signed delta vectors
    direction_labels : (n_pairs,) — 1 if trans−stable direction, 0 if flipped
    pair_info        : DataFrame [group, transition_id, stable_id, flipped]
    """
    delta_rows, info_rows, labels = [], [], []

    for grp, sub_df in mrf_primary.groupby(GROUP_COL):
        trans_rows = sub_df[sub_df[LABEL_COL] == 1]
        stable_rows = sub_df[sub_df[LABEL_COL] == 0]
        if len(trans_rows) != 1 or len(stable_rows) != 1:
            logger.warning(
                f"Skipping malformed pair {grp} "
                f"(n_trans={len(trans_rows)}, n_stable={len(stable_rows)})"
            )
            continue

        trans = trans_rows.iloc[0]
        stable = stable_rows.iloc[0]

        delta = (
            trans[feature_cols].values.astype(float)
            - stable[feature_cols].values.astype(float)
        )
        delta = np.where(np.isnan(delta), 0.0, delta)

        flipped = bool(rng.random() < 0.5)
        if flipped:
            delta = -delta
            direction_label = 0
        else:
            direction_label = 1

        delta_rows.append(delta)
        labels.append(direction_label)
        info_rows.append({
            GROUP_COL: grp,
            "transition_id": trans[SUBJECT_ID_COL],
            "stable_id": stable[SUBJECT_ID_COL],
            "flipped": flipped,
        })

    delta_df = pd.DataFrame(delta_rows, columns=feature_cols)
    pair_info = pd.DataFrame(info_rows).reset_index(drop=True)
    return delta_df, np.array(labels), pair_info


# =============================================================================
# Contrast model training with OOF
# =============================================================================


def train_contrast_model(
    delta_df: pd.DataFrame,
    direction_labels: np.ndarray,
    seed: int = 0,
    cv_folds: int = 5,
) -> tuple[StandardScaler, LogisticRegressionCV, np.ndarray, float]:
    """
    Train L1 logistic regression on contrast vectors at the pair level.

    OOF direction probabilities (P(direction_label=1 | delta)) are computed
    via k-fold CV. The final model is refitted on all pairs.

    Returns
    -------
    scaler       : fitted StandardScaler applied to delta vectors
    model        : LogisticRegressionCV fitted on all pairs
    oof_proba    : (n_pairs,) OOF P(direction_label=1 | delta_i)
    oof_auc      : AUC of OOF predictions vs direction_labels (near 0.5 = no signal)
    """
    X = delta_df.values.astype(float)
    y = direction_labels

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Each row is a pair — KFold is correct (no group structure within contrast dataset)
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    oof_proba = np.full(len(X), np.nan)

    for tr_idx, val_idx in kf.split(X_scaled):
        m = LogisticRegressionCV(
            Cs=10, cv=3, penalty="l1", solver="saga",
            scoring="roc_auc", random_state=seed, max_iter=5000,
        )
        m.fit(X_scaled[tr_idx], y[tr_idx])
        oof_proba[val_idx] = m.predict_proba(X_scaled[val_idx])[:, 1]

    oof_auc = roc_auc_score(y, oof_proba)
    logger.info(f"Contrast model OOF AUC (direction labels): {oof_auc:.4f}")

    # Final model on all pairs
    final_model = LogisticRegressionCV(
        Cs=10, cv=5, penalty="l1", solver="saga",
        scoring="roc_auc", random_state=seed, max_iter=5000,
    )
    final_model.fit(X_scaled, y)
    logger.info(
        f"Contrast model: selected C = {final_model.C_[0]:.4f}, "
        f"n_nonzero coefs = {(final_model.coef_[0] != 0).sum()}/{X.shape[1]}"
    )

    return scaler, final_model, oof_proba, oof_auc


# =============================================================================
# Contrast score assignment
# =============================================================================


def assign_contrast_scores(
    scaler: StandardScaler,
    contrast_model: LogisticRegressionCV,
    mrf_primary_train: pd.DataFrame,
    mrf_aug_train: pd.DataFrame,
    mrf_test: pd.DataFrame,
    feature_cols: list[str],
    pair_info: pd.DataFrame,
    oof_proba: np.ndarray,
) -> tuple[pd.Series, pd.Series]:
    """
    Map contrast model output to subject-level P(subject is transition member).

    Training (primary): OOF direction proba → subject-level scores (leakage-free).
      - oof_proba[i] = P(delta_i is trans−stable direction)
      - If NOT flipped: delta_i = trans−stable → P(trans is transition) = oof_proba[i]
      - If flipped:     delta_i = stable−trans → P(trans is transition) = 1 − oof_proba[i]
    Training (augmentation): 0.5 (neutral).
    Test: out-of-sample from final model, averaged over both delta directions.
    """
    train_scores: dict = {}

    # --- Primary training subjects: OOF → subject level ---
    for i, row in pair_info.iterrows():
        trans_id = row["transition_id"]
        stable_id = row["stable_id"]
        flipped = row["flipped"]
        p_trans = oof_proba[i] if not flipped else 1.0 - oof_proba[i]
        train_scores[trans_id] = float(p_trans)
        train_scores[stable_id] = 1.0 - float(p_trans)

    # --- Augmentation subjects: neutral ---
    for sid in mrf_aug_train[SUBJECT_ID_COL].values:
        train_scores[sid] = 0.5

    train_score_series = pd.Series(train_scores, name=CONTRAST_SCORE_COL)

    # --- Test subjects: out-of-sample, both delta directions averaged ---
    test_scores: dict = {}
    for _, sub_df in mrf_test.groupby(GROUP_COL):
        subjects = sub_df.reset_index(drop=True)
        if len(subjects) < 2:
            # Singleton (unmatched control from remaining_test) — assign neutral
            for sid in subjects[SUBJECT_ID_COL]:
                test_scores[sid] = 0.5
            continue

        subj_a = subjects.iloc[0]
        subj_b = subjects.iloc[1]

        def _delta(x, y_subj):
            d = (
                x[feature_cols].values.astype(float)
                - y_subj[feature_cols].values.astype(float)
            )
            return np.where(np.isnan(d), 0.0, d)

        delta_ab = _delta(subj_a, subj_b)
        delta_ba = _delta(subj_b, subj_a)

        p_ab = contrast_model.predict_proba(
            scaler.transform(delta_ab.reshape(1, -1))
        )[0, 1]
        p_ba = contrast_model.predict_proba(
            scaler.transform(delta_ba.reshape(1, -1))
        )[0, 1]

        # Average both directions for robustness (logistic regression is approximately
        # antisymmetric: P(y=1|-delta) ≈ 1 - P(y=1|delta) when intercept ≈ 0)
        p_a_is_trans = (p_ab + (1.0 - p_ba)) / 2.0
        p_b_is_trans = 1.0 - p_a_is_trans

        test_scores[subj_a[SUBJECT_ID_COL]] = float(p_a_is_trans)
        test_scores[subj_b[SUBJECT_ID_COL]] = float(p_b_is_trans)

    test_score_series = pd.Series(test_scores, name=CONTRAST_SCORE_COL)

    trans_ids = pair_info["transition_id"].tolist()
    stable_ids = pair_info["stable_id"].tolist()
    logger.info(
        f"Contrast score (primary train) — "
        f"transition mean: {train_score_series.loc[trans_ids].mean():.3f}, "
        f"stable mean: {train_score_series.loc[stable_ids].mean():.3f}"
    )
    logger.info(
        f"Contrast score (test) — "
        f"mean: {test_score_series.mean():.3f}, std: {test_score_series.std():.3f}"
    )

    return train_score_series, test_score_series


def attach_contrast_score(
    bmca_df: pd.DataFrame,
    scores: pd.Series,
) -> pd.DataFrame:
    """Append mrf_contrast_score column to a BMCA feature table."""
    out = bmca_df.copy()
    out[CONTRAST_SCORE_COL] = scores.reindex(out[SUBJECT_ID_COL].values).values
    return out


# =============================================================================
# BMCA training and evaluation
# =============================================================================


def _train_and_evaluate(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    model_name: str,
    n_iter: int = 50,
    n_splits: int = 5,
    seed: int = 0,
    n_jobs: int = -1,
    gpu: bool = False,
    n_boot: int = 1000,
) -> dict:
    feature_cols = _feature_cols(train_df)

    X_train = train_df[feature_cols]
    y_train = train_df[LABEL_COL].astype(float)
    groups_train = train_df[GROUP_COL]

    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    val_mask = (train_df["analysis_set"] == "primary").values

    eligible = _evaluation_eligible(test_df)
    X_test = eligible[feature_cols]
    y_test = eligible[LABEL_COL].astype(float)

    logger.info(
        f"Training {model_name}: {len(feature_cols)} features, {len(X_train)} train rows."
    )

    study, best_model, _ = train_model(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        model="catboost",
        seed_rf=seed,
        seed_bayes=seed,
        cv=cv,
        n_iter=n_iter,
        groups=groups_train,
        cat_vars=None,
        n_jobs=n_jobs,
        gpu=gpu,
        val_mask=val_mask,
    )

    groups_test = eligible[GROUP_COL].values
    y_test_arr = y_test.values
    y_score = best_model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test_arr, y_score)
    ci_low, ci_high = _bootstrap_auc(
        y_test_arr, y_score, groups_test, n_boot=n_boot, seed=seed
    )

    imp_df = (
        pd.DataFrame({
            "feature": feature_cols,
            "importance": best_model.get_feature_importance(),
        })
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

    logger.info(
        f"{model_name}  AUC = {auc:.3f}  95% CI [{ci_low:.3f}, {ci_high:.3f}]  "
        f"CV = {study.best_value:.4f}"
    )

    return {
        "model_name": model_name,
        "study": study,
        "model": best_model,
        "feature_cols": feature_cols,
        "auc": auc,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "inner_cv_auc": study.best_value,
        "y_true": y_test_arr,
        "y_score": y_score,
        "groups": groups_test,
        "importance": imp_df,
        "best_params": study.best_params,
    }


# =============================================================================
# Output
# =============================================================================


def _plot_roc(results: list[dict], output_path: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    colors = ["steelblue", "forestgreen"]
    n_pairs = int(results[0]["y_true"].sum())
    for r, color in zip(results, colors):
        RocCurveDisplay.from_predictions(
            r["y_true"], r["y_score"], ax=ax,
            name=f"{r['model_name']} (AUC = {r['auc']:.3f})",
            color=color,
        )
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="Chance")
    ax.set_title(
        f"Within-Pair Contrast Model — ROC Comparison\n"
        f"(primary test set, n = {n_pairs} pairs)"
    )
    ax.set_xlabel("1 − Specificity (FPR)")
    ax.set_ylabel("Sensitivity (TPR)")
    ax.grid(True, which="both", alpha=0.3)
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"ROC plot saved to {output_path}")


# =============================================================================
# Entry point
# =============================================================================


def run(
    bmca_train_path: str = "data/adni_bmca_features_train.csv",
    bmca_test_path: str = "data/adni_bmca_features_test.csv",
    mrf_train_path: str = "data/adni_mrf_features_train.csv",
    mrf_test_path: str = "data/adni_mrf_features_test.csv",
    output_dir: str = "results",
    plots_dir: str = "plots",
    n_iter: int = 50,
    n_splits: int = 5,
    n_boot: int = 1000,
    seed: int = 0,
    n_jobs: int = -1,
    gpu: bool = False,
    contrast_cv_folds: int = 5,
) -> dict:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    Path(plots_dir).mkdir(parents=True, exist_ok=True)

    bmca_train = pd.read_csv(bmca_train_path)
    bmca_test = pd.read_csv(bmca_test_path)
    mrf_train = pd.read_csv(mrf_train_path)
    mrf_test = pd.read_csv(mrf_test_path)

    mrf_primary_train = mrf_train[mrf_train["analysis_set"] == "primary"].copy()
    mrf_aug_train = mrf_train[mrf_train["analysis_set"] == "augmentation"].copy()
    mrf_feature_cols = _feature_cols(mrf_train)

    n_primary_pairs = mrf_primary_train[GROUP_COL].nunique()
    n_aug_pairs = mrf_aug_train[GROUP_COL].nunique()
    n_test_pairs = mrf_test[GROUP_COL].nunique()

    logger.info(
        f"\n{'='*60}\n"
        f"Strategy 4: Within-Pair Contrast Model\n"
        f"{'='*60}\n"
        f"Primary training pairs:     {n_primary_pairs}\n"
        f"Augmentation training pairs:{n_aug_pairs}\n"
        f"Test pairs:                 {n_test_pairs}\n"
        f"MRF features:               {len(mrf_feature_cols)}"
    )

    # ------------------------------------------------------------------
    # SNR diagnostic
    # ------------------------------------------------------------------
    logger.info("\n--- Computing within-pair SNR diagnostic ---")
    snr_df = compute_snr_table(mrf_primary_train, mrf_feature_cols)
    logger.info(
        f"\nTop 10 MRF features by within-pair SNR:\n"
        f"{snr_df.head(10).to_string(index=False)}"
    )
    snr_df.to_csv(f"{output_dir}/within_pair_contrast_snr.csv", index=False)

    max_snr = snr_df["snr"].max()
    if max_snr < 0.05:
        logger.warning(
            f"Maximum within-pair SNR = {max_snr:.4f}. "
            "MRF differences within matched pairs are near zero — "
            "the contrast model is expected to perform at chance."
        )

    # ------------------------------------------------------------------
    # Build contrast dataset
    # ------------------------------------------------------------------
    logger.info("\n--- Building contrast dataset ---")
    rng = np.random.default_rng(seed)
    delta_df, direction_labels, pair_info = build_contrast_dataset(
        mrf_primary_train, mrf_feature_cols, rng
    )
    n_pairs = len(delta_df)
    n_pos = int(direction_labels.sum())
    logger.info(
        f"Contrast dataset: {n_pairs} pairs | "
        f"direction_label=1: {n_pos}, direction_label=0: {n_pairs - n_pos}"
    )

    # ------------------------------------------------------------------
    # Train contrast model
    # ------------------------------------------------------------------
    logger.info("\n--- Training contrast model (L1 logistic regression on pairs) ---")
    scaler, contrast_model, oof_proba, oof_auc = train_contrast_model(
        delta_df, direction_labels, seed=seed, cv_folds=contrast_cv_folds
    )

    if oof_auc < 0.55:
        logger.warning(
            f"Contrast model OOF AUC = {oof_auc:.4f} is near chance (0.50). "
            "Within-pair MRF differences may not carry reliable directional signal."
        )

    # L1 feature coefficients
    coef_df = (
        pd.DataFrame({
            "feature": mrf_feature_cols,
            "coefficient": contrast_model.coef_[0],
            "abs_coefficient": np.abs(contrast_model.coef_[0]),
        })
        .sort_values("abs_coefficient", ascending=False)
        .reset_index(drop=True)
    )
    n_nonzero = int((coef_df["coefficient"] != 0).sum())
    logger.info(
        f"\nL1 selected {n_nonzero}/{len(mrf_feature_cols)} features "
        f"(C = {contrast_model.C_[0]:.4f})\n"
        f"Top 10 by |coefficient|:\n{coef_df.head(10).to_string(index=False)}"
    )
    coef_df.to_csv(f"{output_dir}/within_pair_contrast_coefficients.csv", index=False)

    # ------------------------------------------------------------------
    # Assign contrast scores to all subjects
    # ------------------------------------------------------------------
    logger.info("\n--- Assigning contrast scores to all subjects ---")
    train_scores, test_scores = assign_contrast_scores(
        scaler, contrast_model,
        mrf_primary_train, mrf_aug_train, mrf_test,
        mrf_feature_cols, pair_info, oof_proba,
    )

    bmca_train_with_score = attach_contrast_score(bmca_train, train_scores)
    bmca_test_with_score = attach_contrast_score(bmca_test, test_scores)

    n_missing_train = bmca_train_with_score[CONTRAST_SCORE_COL].isna().sum()
    n_missing_test = bmca_test_with_score[CONTRAST_SCORE_COL].isna().sum()
    if n_missing_train > 0 or n_missing_test > 0:
        logger.warning(
            f"Contrast score missing for {n_missing_train} train / "
            f"{n_missing_test} test subjects."
        )

    # ------------------------------------------------------------------
    # Train BMCA baseline and BMCA + contrast score
    # ------------------------------------------------------------------
    logger.info("\n--- Training BMCA baseline ---")
    r_base = _train_and_evaluate(
        bmca_train, bmca_test, "BMCA baseline",
        n_iter=n_iter, n_splits=n_splits, seed=seed,
        n_jobs=n_jobs, gpu=gpu, n_boot=n_boot,
    )

    logger.info("\n--- Training BMCA + mrf_contrast_score ---")
    r_contrast = _train_and_evaluate(
        bmca_train_with_score, bmca_test_with_score, "BMCA + contrast score",
        n_iter=n_iter, n_splits=n_splits, seed=seed,
        n_jobs=n_jobs, gpu=gpu, n_boot=n_boot,
    )

    # ------------------------------------------------------------------
    # Paired bootstrap comparison
    # ------------------------------------------------------------------
    diff = _bootstrap_auc_diff(
        r_base["y_true"], r_contrast["y_score"], r_base["y_score"],
        r_base["groups"], n_boot=10_000, seed=seed,
    )
    logger.info(
        f"\nBMCA+contrast vs BMCA baseline: "
        f"Δ = {diff['observed_diff']:+.4f}  "
        f"95% CI [{diff['ci_low']:+.4f}, {diff['ci_high']:+.4f}]  "
        f"p(Δ ≤ 0) = {diff['p_value']:.3f}"
    )

    cs_row = r_contrast["importance"][
        r_contrast["importance"]["feature"] == CONTRAST_SCORE_COL
    ]
    if not cs_row.empty:
        rank_pos = cs_row.index[0] + 1
        importance_val = cs_row["importance"].values[0]
        logger.info(
            f"mrf_contrast_score: rank {rank_pos}/{len(r_contrast['feature_cols'])}, "
            f"importance = {importance_val:.2f}%"
        )

    # ------------------------------------------------------------------
    # Save outputs
    # ------------------------------------------------------------------
    _plot_roc([r_base, r_contrast], f"{plots_dir}/within_pair_contrast_roc.pdf")

    metrics = pd.DataFrame([
        {
            "model": r["model_name"],
            "auc": round(r["auc"], 4),
            "auc_ci_low_95": round(r["ci_low"], 4),
            "auc_ci_high_95": round(r["ci_high"], 4),
            "best_inner_cv_auc": round(r["inner_cv_auc"], 4),
            "n_features": len(r["feature_cols"]),
            **{f"param_{k}": v for k, v in r["best_params"].items()},
        }
        for r in [r_base, r_contrast]
    ])
    metrics.to_csv(f"{output_dir}/within_pair_contrast_evaluation.csv", index=False)

    pd.DataFrame([{
        "comparison": "BMCA+contrast vs BMCA baseline",
        **{k: round(v, 4) if isinstance(v, float) else v for k, v in diff.items()},
    }]).to_csv(f"{output_dir}/within_pair_contrast_bootstrap_diff.csv", index=False)

    pd.DataFrame([{
        "contrast_model_oof_auc": round(oof_auc, 4),
        "n_training_pairs": n_pairs,
        "n_features_selected_l1": n_nonzero,
        "n_mrf_features_total": len(mrf_feature_cols),
        "selected_C": float(contrast_model.C_[0]),
        "max_within_pair_snr": round(float(snr_df["snr"].max()), 4),
        "median_within_pair_snr": round(float(snr_df["snr"].median()), 4),
    }]).to_csv(f"{output_dir}/within_pair_contrast_model_summary.csv", index=False)

    r_base["importance"].to_csv(
        f"{output_dir}/within_pair_contrast_baseline_importance.csv", index=False
    )
    r_contrast["importance"].to_csv(
        f"{output_dir}/within_pair_contrast_combined_importance.csv", index=False
    )

    joblib.dump({
        "scaler": scaler,
        "contrast_model": contrast_model,
        "oof_proba": oof_proba,
        "pair_info": pair_info,
        "snr_df": snr_df,
        "coef_df": coef_df,
        "model_base": r_base["model"],
        "model_contrast": r_contrast["model"],
        "study_base": r_base["study"],
        "study_contrast": r_contrast["study"],
        "result_base": {k: v for k, v in r_base.items() if k not in ("model", "study", "importance")},
        "result_contrast": {k: v for k, v in r_contrast.items() if k not in ("model", "study", "importance")},
        "diff": diff,
    }, f"{output_dir}/within_pair_contrast_models.joblib")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    logger.info(
        f"\n{'='*60}\n"
        f"SUMMARY: Strategy 4 — Within-Pair Contrast Model\n"
        f"{'='*60}\n"
        f"Contrast model (L1 logistic regression on {n_pairs} pairs):\n"
        f"  OOF AUC (direction labels):  {oof_auc:.4f}\n"
        f"  Features selected by L1:     {n_nonzero}/{len(mrf_feature_cols)}\n"
        f"  Selected C:                  {contrast_model.C_[0]:.4f}\n"
        f"  Max within-pair SNR:         {snr_df['snr'].max():.4f} "
        f"({snr_df.iloc[0]['feature']})\n"
        f"\nBMCA evaluation (primary test set):\n"
        f"  {'Model':<30} {'AUC':>8} {'95% CI':>22} {'CV AUC':>8}\n"
        f"  {'-'*68}\n"
        f"  {'BMCA baseline':<30} {r_base['auc']:>8.4f} "
        f"[{r_base['ci_low']:.3f}, {r_base['ci_high']:.3f}] "
        f"{r_base['inner_cv_auc']:>8.4f}\n"
        f"  {'BMCA + contrast score':<30} {r_contrast['auc']:>8.4f} "
        f"[{r_contrast['ci_low']:.3f}, {r_contrast['ci_high']:.3f}] "
        f"{r_contrast['inner_cv_auc']:>8.4f}\n"
        f"\n  Δ AUC:    {diff['observed_diff']:+.4f}\n"
        f"  95% CI:   [{diff['ci_low']:+.4f}, {diff['ci_high']:+.4f}]\n"
        f"  p(Δ≤0):   {diff['p_value']:.3f}\n"
        f"{'='*60}"
    )

    return {
        "scaler": scaler,
        "contrast_model": contrast_model,
        "oof_auc": oof_auc,
        "snr_df": snr_df,
        "result_base": r_base,
        "result_contrast": r_contrast,
        "diff": diff,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Strategy 4: Within-Pair Contrast Model for BMCA"
    )
    parser.add_argument("--bmca_train", default="data/adni_bmca_features_train.csv")
    parser.add_argument("--bmca_test", default="data/adni_bmca_features_test.csv")
    parser.add_argument("--mrf_train", default="data/adni_mrf_features_train.csv")
    parser.add_argument("--mrf_test", default="data/adni_mrf_features_test.csv")
    parser.add_argument("--output_dir", default="results")
    parser.add_argument("--plots_dir", default="plots")
    parser.add_argument("--n_iter", type=int, default=50)
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--n_boot", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n_jobs", type=int, default=-1)
    parser.add_argument("--gpu", action="store_true", default=False)
    parser.add_argument("--contrast_cv_folds", type=int, default=5,
                        help="K-fold CV folds for contrast model OOF scoring")
    args = parser.parse_args()

    run(
        bmca_train_path=args.bmca_train,
        bmca_test_path=args.bmca_test,
        mrf_train_path=args.mrf_train,
        mrf_test_path=args.mrf_test,
        output_dir=args.output_dir,
        plots_dir=args.plots_dir,
        n_iter=args.n_iter,
        n_splits=args.n_splits,
        n_boot=args.n_boot,
        seed=args.seed,
        n_jobs=args.n_jobs,
        gpu=args.gpu,
        contrast_cv_folds=args.contrast_cv_folds,
    )
