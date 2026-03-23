"""
Strategy 2: Augmentation-Distilled MRF Score
=============================================

Tests whether training an MRF model exclusively on augmentation pairs
(stable CN vs CI→dementia) and using its predictions as a single summary
feature can transfer the CI→dementia vascular signal to the CN→MCI task.

Motivation
----------
MRF features are near-chance for CN→MCI (AUC ~0.47–0.62) but may carry a
stronger signal for CI→dementia, where vascular burden is a well-established
driver. The 267 augmentation pairs provide a clean, large dataset for learning
this task. Rather than adding 30 noisy raw MRF features to the BMCA model, a
single distilled "vascular risk profile" score — trained where MRF should work
— is added as one column. If the CI→dementia vascular pathway overlaps with
CN→MCI, the score will carry transferable signal.

Design
------
**Stage 1 — MRF model on augmentation pairs only:**
  1. Filter training data to augmentation pairs (534 subjects, 267 pairs).
  2. Train CatBoost on MRF features with Optuna (50 trials, k=5 CV).
  3. Compute OOF predictions for all augmentation subjects via cross_val_predict.
  4. Apply the fitted model to primary train and all test subjects (out-of-sample).

**Stage 2 — BMCA + mrf_aug_score:**
  5. Assemble mrf_aug_score for all subjects in the training set:
     - Augmentation subjects: OOF predictions from Stage 1 (leakage-free)
     - Primary train subjects: Stage 1 model predictions (out-of-sample)
  6. Add mrf_aug_score as one column to the BMCA feature matrix.
  7. Train BMCA + mrf_aug_score on the full training pool (primary + augmentation)
     with val_mask (primary-only CV scoring), same seed as Stage 1 baseline.

**Comparison:**
  8. Train BMCA-only baseline with the same seed for a fair head-to-head.
  9. Evaluate both on primary test set; run paired bootstrap AUC difference test.

Leakage handling
----------------
Augmentation subjects in Stage 2 training receive their OOF predictions from
Stage 1 (computed without seeing their own labels). Primary train and test
subjects are fully out-of-sample for Stage 1. No label leakage.

Usage::

    python model_aug_distilled_mrf_evaluation.py
    python model_aug_distilled_mrf_evaluation.py --n_iter 100 --seed 42
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
from catboost import CatBoostClassifier
from sklearn.metrics import RocCurveDisplay, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

import optuna
from optuna.samplers import TPESampler

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
MRF_SCORE_COL = "mrf_aug_score"


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
# Stage 1: MRF model trained on augmentation pairs
# =============================================================================


def train_mrf_on_augmentation(
    mrf_train: pd.DataFrame,
    n_iter: int = 50,
    n_splits: int = 5,
    seed: int = 0,
) -> tuple[optuna.Study, CatBoostClassifier, np.ndarray, pd.Series]:
    """
    Train MRF CatBoost on augmentation pairs and compute OOF predictions.

    Returns
    -------
    study : Optuna study
    model : CatBoostClassifier refitted on all augmentation subjects
    oof_proba : np.ndarray of shape (n_augmentation,) — OOF predictions
    aug_index : pd.Series — subject_id index aligning oof_proba rows
    """
    aug = mrf_train[mrf_train["analysis_set"] == "augmentation"].copy()
    mrf_feats = _feature_cols(aug)

    X_aug = aug[mrf_feats]
    y_aug = aug[LABEL_COL].values.astype(float)
    groups_aug = aug[GROUP_COL].values

    n_pairs = aug[GROUP_COL].nunique()
    logger.info(
        f"Stage 1 — MRF on augmentation: "
        f"{len(aug)} subjects, {n_pairs} pairs, {len(mrf_feats)} features"
    )

    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    splits = list(cv.split(X_aug, y_aug, groups_aug))

    # ---- Optuna search ----
    def objective(trial: optuna.Trial) -> float:
        params = {
            "iterations": trial.suggest_int("iterations", 100, 1000),
            "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-1, log=True),
            "depth": trial.suggest_int("depth", 2, 8),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-3, 1e3, log=True),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
            "border_count": trial.suggest_int("border_count", 32, 255),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 50),
            "random_strength": trial.suggest_float("random_strength", 0.0, 10.0),
        }
        fold_aucs = []
        for tr_idx, val_idx in splits:
            m = CatBoostClassifier(
                **params, random_seed=seed, verbose=0, allow_writing_files=False
            )
            m.fit(X_aug.iloc[tr_idx], y_aug[tr_idx])
            y_proba = m.predict_proba(X_aug.iloc[val_idx])[:, 1]
            if len(np.unique(y_aug[val_idx])) == 2:
                fold_aucs.append(roc_auc_score(y_aug[val_idx], y_proba))
        return float(np.mean(fold_aucs)) if fold_aucs else 0.0

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="maximize", sampler=TPESampler(seed=seed)
    )
    study.optimize(objective, n_trials=n_iter, show_progress_bar=True)
    logger.info(
        f"Stage 1 best CV AUC: {study.best_value:.4f} | params: {study.best_params}"
    )

    # ---- OOF predictions (leakage-free for augmentation subjects in Stage 2) ----
    oof_proba = np.full(len(aug), np.nan)
    for tr_idx, val_idx in splits:
        m = CatBoostClassifier(
            **study.best_params, random_seed=seed, verbose=0,
            allow_writing_files=False
        )
        m.fit(X_aug.iloc[tr_idx], y_aug[tr_idx])
        oof_proba[val_idx] = m.predict_proba(X_aug.iloc[val_idx])[:, 1]

    oof_auc = roc_auc_score(y_aug, oof_proba)
    logger.info(f"Stage 1 OOF AUC (augmentation): {oof_auc:.4f}")

    # ---- Refit on all augmentation data ----
    best_model = CatBoostClassifier(
        **study.best_params, random_seed=seed, verbose=0, allow_writing_files=False
    )
    best_model.fit(X_aug, y_aug)

    return study, best_model, oof_proba, aug[SUBJECT_ID_COL]


# =============================================================================
# Stage 1: Score assignment
# =============================================================================


def assign_mrf_aug_scores(
    mrf_model: CatBoostClassifier,
    mrf_train: pd.DataFrame,
    mrf_test: pd.DataFrame,
    aug_oof_proba: np.ndarray,
    aug_subject_ids: pd.Series,
) -> tuple[pd.Series, pd.Series]:
    """
    Build the mrf_aug_score column for all training and test subjects.

    Training set:
      - Augmentation subjects → OOF predictions (leakage-free)
      - Primary subjects → Stage 1 model predictions (out-of-sample)

    Test set:
      - All subjects → Stage 1 model predictions (out-of-sample)

    Returns two Series indexed by subject_id.
    """
    mrf_feats = _feature_cols(mrf_train)

    # --- Training scores ---
    train_scores = pd.Series(np.nan, index=mrf_train[SUBJECT_ID_COL], name=MRF_SCORE_COL)

    # Augmentation OOF
    aug_oof_series = pd.Series(aug_oof_proba, index=aug_subject_ids.values)
    train_scores.loc[aug_oof_series.index] = aug_oof_series.values

    # Primary: out-of-sample predictions from Stage 1 model
    primary_mask = mrf_train["analysis_set"] == "primary"
    primary = mrf_train[primary_mask]
    primary_proba = mrf_model.predict_proba(primary[mrf_feats])[:, 1]
    train_scores.loc[primary[SUBJECT_ID_COL].values] = primary_proba

    # --- Test scores ---
    test_scores = pd.Series(
        mrf_model.predict_proba(mrf_test[mrf_feats])[:, 1],
        index=mrf_test[SUBJECT_ID_COL].values,
        name=MRF_SCORE_COL,
    )

    logger.info(
        f"mrf_aug_score stats (train) — "
        f"augmentation: mean={train_scores.loc[aug_oof_series.index].mean():.3f}, "
        f"primary: mean={primary_proba.mean():.3f}"
    )
    logger.info(
        f"mrf_aug_score stats (test) — mean={test_scores.mean():.3f}"
    )

    return train_scores, test_scores


def attach_mrf_score(
    bmca_df: pd.DataFrame,
    scores: pd.Series,
) -> pd.DataFrame:
    """Append mrf_aug_score column to a BMCA feature table."""
    out = bmca_df.copy()
    score_mapped = scores.reindex(out[SUBJECT_ID_COL].values).values
    out[MRF_SCORE_COL] = score_mapped
    return out


# =============================================================================
# Stage 2 training and evaluation
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
        f"Stage 2 training {model_name}: "
        f"{len(feature_cols)} features, {len(X_train)} train rows."
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
        f"Augmentation-Distilled MRF — ROC Comparison\n"
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
) -> dict:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    Path(plots_dir).mkdir(parents=True, exist_ok=True)

    bmca_train = pd.read_csv(bmca_train_path)
    bmca_test = pd.read_csv(bmca_test_path)
    mrf_train = pd.read_csv(mrf_train_path)
    mrf_test = pd.read_csv(mrf_test_path)

    logger.info(
        f"\n{'='*60}\n"
        f"Strategy 2: Augmentation-Distilled MRF Score\n"
        f"{'='*60}"
    )

    # ------------------------------------------------------------------
    # Stage 1: MRF model on augmentation pairs
    # ------------------------------------------------------------------
    logger.info("\n--- Stage 1: Training MRF on augmentation pairs ---")
    s1_study, s1_model, aug_oof, aug_ids = train_mrf_on_augmentation(
        mrf_train, n_iter=n_iter, n_splits=n_splits, seed=seed
    )

    # Stage 1 MRF feature importance
    mrf_feats = _feature_cols(mrf_train)
    s1_imp = (
        pd.DataFrame({
            "feature": mrf_feats,
            "importance": s1_model.get_feature_importance(),
        })
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    logger.info(f"\nStage 1 MRF top 10 features:\n{s1_imp.head(10).to_string(index=False)}")

    # Gate: check Stage 1 AUC is useful
    aug_labels = mrf_train[mrf_train["analysis_set"] == "augmentation"][LABEL_COL].values
    s1_oof_auc = roc_auc_score(aug_labels, aug_oof)
    logger.info(f"\nStage 1 OOF AUC (augmentation): {s1_oof_auc:.4f}")
    if s1_oof_auc < 0.60:
        logger.warning(
            f"Stage 1 OOF AUC = {s1_oof_auc:.4f} < 0.60 threshold. "
            "MRF has limited signal for CI→dementia. "
            "Proceeding but results may be uninformative."
        )

    # ------------------------------------------------------------------
    # Stage 1: Assign scores to all subjects
    # ------------------------------------------------------------------
    logger.info("\n--- Assigning mrf_aug_score to all subjects ---")
    train_scores, test_scores = assign_mrf_aug_scores(
        s1_model, mrf_train, mrf_test, aug_oof, aug_ids
    )

    bmca_train_with_score = attach_mrf_score(bmca_train, train_scores)
    bmca_test_with_score = attach_mrf_score(bmca_test, test_scores)

    # Verify score is present for all subjects
    n_missing_train = bmca_train_with_score[MRF_SCORE_COL].isna().sum()
    n_missing_test = bmca_test_with_score[MRF_SCORE_COL].isna().sum()
    if n_missing_train > 0 or n_missing_test > 0:
        logger.warning(
            f"mrf_aug_score missing for {n_missing_train} train / "
            f"{n_missing_test} test subjects."
        )

    # ------------------------------------------------------------------
    # Stage 2: Train BMCA-only baseline and BMCA + mrf_aug_score
    # ------------------------------------------------------------------
    logger.info("\n--- Stage 2: Training BMCA baseline ---")
    r_base = _train_and_evaluate(
        bmca_train, bmca_test, "BMCA baseline",
        n_iter=n_iter, n_splits=n_splits, seed=seed, n_jobs=n_jobs, gpu=gpu,
        n_boot=n_boot,
    )

    logger.info("\n--- Stage 2: Training BMCA + mrf_aug_score ---")
    r_aug = _train_and_evaluate(
        bmca_train_with_score, bmca_test_with_score, "BMCA + mrf_aug_score",
        n_iter=n_iter, n_splits=n_splits, seed=seed, n_jobs=n_jobs, gpu=gpu,
        n_boot=n_boot,
    )

    # ------------------------------------------------------------------
    # Paired bootstrap comparison
    # ------------------------------------------------------------------
    diff = _bootstrap_auc_diff(
        r_base["y_true"], r_aug["y_score"], r_base["y_score"],
        r_base["groups"], n_boot=10_000, seed=seed,
    )
    logger.info(
        f"\nBMCA+mrf_aug_score vs BMCA baseline: "
        f"Δ = {diff['observed_diff']:+.4f}  "
        f"95% CI [{diff['ci_low']:+.4f}, {diff['ci_high']:+.4f}]  "
        f"p(Δ ≤ 0) = {diff['p_value']:.3f}"
    )

    # mrf_aug_score rank in combined model
    mrf_score_rank = r_aug["importance"][
        r_aug["importance"]["feature"] == MRF_SCORE_COL
    ]
    if not mrf_score_rank.empty:
        rank_pos = mrf_score_rank.index[0] + 1
        importance_val = mrf_score_rank["importance"].values[0]
        logger.info(
            f"mrf_aug_score: rank {rank_pos}/{len(r_aug['feature_cols'])}, "
            f"importance = {importance_val:.2f}%"
        )

    # ------------------------------------------------------------------
    # Save outputs
    # ------------------------------------------------------------------
    _plot_roc([r_base, r_aug], f"{plots_dir}/aug_distilled_mrf_roc.pdf")

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
        for r in [r_base, r_aug]
    ])
    metrics.to_csv(f"{output_dir}/aug_distilled_mrf_evaluation.csv", index=False)

    pd.DataFrame([{
        "comparison": "BMCA+mrf_aug_score vs BMCA baseline",
        **{k: round(v, 4) if isinstance(v, float) else v for k, v in diff.items()},
    }]).to_csv(f"{output_dir}/aug_distilled_mrf_bootstrap_diff.csv", index=False)

    pd.DataFrame({
        "model": ["stage1_mrf_on_augmentation"],
        "oof_auc": [round(s1_oof_auc, 4)],
        "best_cv_auc": [round(s1_study.best_value, 4)],
        **{f"param_{k}": [v] for k, v in s1_study.best_params.items()},
    }).to_csv(f"{output_dir}/aug_distilled_mrf_stage1.csv", index=False)

    s1_imp.to_csv(f"{output_dir}/aug_distilled_mrf_stage1_importance.csv", index=False)
    r_base["importance"].to_csv(
        f"{output_dir}/aug_distilled_mrf_baseline_importance.csv", index=False
    )
    r_aug["importance"].to_csv(
        f"{output_dir}/aug_distilled_mrf_combined_importance.csv", index=False
    )

    joblib.dump({
        "stage1_model": s1_model,
        "stage1_study": s1_study,
        "stage1_oof_proba": aug_oof,
        "stage1_aug_ids": aug_ids,
        "model_base": r_base["model"],
        "model_aug": r_aug["model"],
        "study_base": r_base["study"],
        "study_aug": r_aug["study"],
        "result_base": {k: v for k, v in r_base.items() if k not in ("model", "study", "importance")},
        "result_aug": {k: v for k, v in r_aug.items() if k not in ("model", "study", "importance")},
        "diff": diff,
    }, f"{output_dir}/aug_distilled_mrf_models.joblib")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    logger.info(
        f"\n{'='*60}\n"
        f"SUMMARY: Strategy 2 — Augmentation-Distilled MRF Score\n"
        f"{'='*60}\n"
        f"Stage 1 MRF on augmentation:\n"
        f"  OOF AUC (CI→dementia):  {s1_oof_auc:.4f}\n"
        f"  Best CV AUC:             {s1_study.best_value:.4f}\n"
        f"\nStage 2:\n"
        f"  {'Model':<28} {'AUC':>8} {'95% CI':>20} {'CV AUC':>8}\n"
        f"  {'-'*64}\n"
        f"  {'BMCA baseline':<28} {r_base['auc']:>8.4f} "
        f"[{r_base['ci_low']:.3f}, {r_base['ci_high']:.3f}] "
        f"{r_base['inner_cv_auc']:>8.4f}\n"
        f"  {'BMCA + mrf_aug_score':<28} {r_aug['auc']:>8.4f} "
        f"[{r_aug['ci_low']:.3f}, {r_aug['ci_high']:.3f}] "
        f"{r_aug['inner_cv_auc']:>8.4f}\n"
        f"\n  Δ AUC:    {diff['observed_diff']:+.4f}\n"
        f"  95% CI:   [{diff['ci_low']:+.4f}, {diff['ci_high']:+.4f}]\n"
        f"  p(Δ≤0):   {diff['p_value']:.3f}\n"
        f"{'='*60}"
    )

    return {
        "stage1_study": s1_study,
        "stage1_model": s1_model,
        "stage1_oof_auc": s1_oof_auc,
        "result_base": r_base,
        "result_aug": r_aug,
        "diff": diff,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Strategy 2: Augmentation-distilled MRF score for BMCA"
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
    )
