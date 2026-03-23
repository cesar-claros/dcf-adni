"""
Strategy 1: Primary-Only Training Experiment
=============================================

Tests the augmentation contamination hypothesis by training BMCA-only and
BMCA+MRF models exclusively on primary matched pairs (CN→MCI/dementia vs
stable CN), excluding all augmentation pairs (stable CN vs CI→dementia).

Motivation
----------
All prior experiments (3–8) trained on the mixed pool of 238 primary + 534
augmentation subjects. The ``val_mask`` restricts inner CV *validation scoring*
to primary pairs, but the model is still *trained* on augmentation data. If
augmentation signal actively interferes with MRF integration — because MRF
features predict CI→dementia through different mechanisms than CN→MCI — then
removing augmentation from training entirely could unmask a real MRF effect.

Design
------
1. Filter training data to primary pairs only (~238 subjects, ~119 pairs).
2. Train BMCA-only CatBoost with Optuna (50 trials, StratifiedGroupKFold k=3).
3. Train BMCA+MRF CatBoost on the same primary-only data (same seed).
4. Evaluate both on the full primary test set (~120 subjects, ~60 pairs).
5. Compare head-to-head with paired bootstrap AUC difference test.

The Optuna search space is tightened relative to the standard pipeline:
``depth`` 2–4, ``iterations`` 100–500, ``min_data_in_leaf`` 5–50, to reduce
overfitting risk with only ~238 training subjects and 29–60 features.

Usage::

    python model_primary_only_evaluation.py
    python model_primary_only_evaluation.py --n_iter 100 --seed 42
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

logging.basicConfig(level=logging.INFO, format="%(name)s — %(message)s")
logger = logging.getLogger(__name__)

# Columns that are not features — metadata, cohort provenance, and outcome-derived variables.
_METADATA_COLS = {
    "subject_id", "pair_id", "group", "transition", "transition_label",
    "matched_cohort", "analysis_set", "evaluation_eligible",
    "abs_age_gap", "split", "split_group_source",
    "first_conversion_month", "baseline_diagnosis", "n_followup_visits_ge12_with_diag",
}

LABEL_COL = "transition_label"
GROUP_COL = "group"
SUBJECT_ID_COL = "subject_id"


# =============================================================================
# Data helpers
# =============================================================================


def _feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in _METADATA_COLS]


def _filter_primary(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only primary analysis-set rows."""
    primary = df[df["analysis_set"] == "primary"].copy()
    logger.info(
        f"Filtered to primary: {len(primary)} subjects, "
        f"{primary[GROUP_COL].nunique()} pairs "
        f"(dropped {len(df) - len(primary)} augmentation rows)."
    )
    return primary


def _merge_feature_tables(
    bmca_df: pd.DataFrame,
    mrf_df: pd.DataFrame,
) -> pd.DataFrame:
    """Merge BMCA and MRF tables on subject_id."""
    bmca_feats = _feature_cols(bmca_df)
    mrf_feats = _feature_cols(mrf_df)

    overlap = set(bmca_feats) & set(mrf_feats)
    if overlap:
        raise ValueError(
            f"Overlapping feature columns between BMCA and MRF: {sorted(overlap)}. "
            "Resolve duplicates before merging."
        )

    merged = bmca_df.merge(
        mrf_df[[SUBJECT_ID_COL] + mrf_feats],
        on=SUBJECT_ID_COL,
        how="inner",
        validate="1:1",
    )
    return merged


def _evaluation_eligible(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["evaluation_eligible"] == 1].copy()


# =============================================================================
# Bootstrap AUC CI (resample at matched-pair level)
# =============================================================================


def _bootstrap_auc(
    y_true: np.ndarray,
    y_score: np.ndarray,
    groups: np.ndarray,
    n_boot: int = 1000,
    seed: int = 0,
) -> tuple[float, float]:
    """Bootstrap 95% CI for AUC by resampling matched pairs with replacement."""
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
    boot_aucs = np.array(boot_aucs)
    return float(np.percentile(boot_aucs, 2.5)), float(np.percentile(boot_aucs, 97.5))


def _bootstrap_auc_diff(
    y_true: np.ndarray,
    y_score_a: np.ndarray,
    y_score_b: np.ndarray,
    groups: np.ndarray,
    n_boot: int = 10_000,
    seed: int = 0,
) -> dict:
    """Paired bootstrap test for AUC(A) - AUC(B), resampling at pair level."""
    rng = np.random.default_rng(seed)
    unique_groups = np.unique(groups)
    obs_auc_a = roc_auc_score(y_true, y_score_a)
    obs_auc_b = roc_auc_score(y_true, y_score_b)
    obs_diff = obs_auc_a - obs_auc_b

    boot_diffs = []
    for _ in range(n_boot):
        sampled = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        idx = np.concatenate([np.where(groups == g)[0] for g in sampled])
        y_b = y_true[idx]
        if len(np.unique(y_b)) < 2:
            continue
        auc_a = roc_auc_score(y_b, y_score_a[idx])
        auc_b = roc_auc_score(y_b, y_score_b[idx])
        boot_diffs.append(auc_a - auc_b)

    boot_diffs = np.array(boot_diffs)
    ci_low = float(np.percentile(boot_diffs, 2.5))
    ci_high = float(np.percentile(boot_diffs, 97.5))
    p_value = float(np.mean(boot_diffs <= 0))

    return {
        "observed_diff": obs_diff,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "p_value": p_value,
        "n_boot": len(boot_diffs),
    }


# =============================================================================
# Training with tightened search space for small samples
# =============================================================================


def _train_catboost_optuna(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    groups: np.ndarray,
    n_iter: int = 50,
    n_splits: int = 3,
    seed: int = 0,
) -> tuple[optuna.Study, CatBoostClassifier]:
    """
    Optuna hyperparameter search for CatBoost with a tightened search space
    suitable for small sample sizes (~238 subjects).

    Returns the study and the best model refitted on the full training set.
    """
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    splits = list(cv.split(X_train, y_train, groups))

    def objective(trial: optuna.Trial) -> float:
        params = {
            "iterations": trial.suggest_int("iterations", 100, 500),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 1e-1, log=True),
            "depth": trial.suggest_int("depth", 2, 4),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-1, 1e3, log=True),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
            "border_count": trial.suggest_int("border_count", 32, 255),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 5, 50),
            "random_strength": trial.suggest_float("random_strength", 0.0, 10.0),
        }

        fold_aucs = []
        for train_idx, val_idx in splits:
            model = CatBoostClassifier(
                **params,
                random_seed=seed,
                verbose=0,
                allow_writing_files=False,
            )
            model.fit(X_train.iloc[train_idx], y_train[train_idx])
            y_proba = model.predict_proba(X_train.iloc[val_idx])[:, 1]
            y_val = y_train[val_idx]
            if len(np.unique(y_val)) == 2:
                fold_aucs.append(roc_auc_score(y_val, y_proba))

        return float(np.mean(fold_aucs)) if fold_aucs else 0.0

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_iter, show_progress_bar=True)

    # Refit best model on full training set
    best_model = CatBoostClassifier(
        **study.best_params,
        random_seed=seed,
        verbose=0,
        allow_writing_files=False,
    )
    best_model.fit(X_train, y_train)

    return study, best_model


# =============================================================================
# Evaluation
# =============================================================================


def _evaluate(
    model: CatBoostClassifier,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    model_name: str,
    n_boot: int = 1000,
    seed: int = 0,
) -> dict:
    """Evaluate on the primary test set (evaluation_eligible == 1)."""
    eligible = _evaluation_eligible(test_df)
    X_test = eligible[feature_cols]
    y_test = eligible[LABEL_COL].values.astype(float)
    groups = eligible[GROUP_COL].values

    y_score = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_score)
    ci_low, ci_high = _bootstrap_auc(y_test, y_score, groups, n_boot=n_boot, seed=seed)

    logger.info(
        f"{model_name}  AUC = {auc:.3f}  95% CI [{ci_low:.3f}, {ci_high:.3f}]"
    )
    return {
        "auc": auc,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "y_true": y_test,
        "y_score": y_score,
        "groups": groups,
    }


def _feature_importance(model: CatBoostClassifier, feature_cols: list[str]) -> pd.DataFrame:
    importances = model.get_feature_importance()
    return (
        pd.DataFrame({"feature": feature_cols, "importance": importances})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


# =============================================================================
# Output
# =============================================================================


def _plot_roc_comparison(
    result_bmca: dict,
    result_bmca_mrf: dict,
    output_path: str,
) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    n_pairs = int(result_bmca["y_true"].sum())

    RocCurveDisplay.from_predictions(
        result_bmca["y_true"],
        result_bmca["y_score"],
        ax=ax,
        name=f"BMCA-only (AUC = {result_bmca['auc']:.3f})",
        color="steelblue",
    )
    RocCurveDisplay.from_predictions(
        result_bmca_mrf["y_true"],
        result_bmca_mrf["y_score"],
        ax=ax,
        name=f"BMCA+MRF (AUC = {result_bmca_mrf['auc']:.3f})",
        color="forestgreen",
    )
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="Chance")
    ax.set_title(
        f"Primary-Only Training — ROC Comparison\n"
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
    logger.info(f"ROC comparison plot saved to {output_path}")


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
    n_splits: int = 3,
    n_boot: int = 1000,
    seed: int = 0,
) -> dict:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    Path(plots_dir).mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    bmca_train = pd.read_csv(bmca_train_path)
    bmca_test = pd.read_csv(bmca_test_path)
    mrf_train = pd.read_csv(mrf_train_path)
    mrf_test = pd.read_csv(mrf_test_path)

    # ------------------------------------------------------------------
    # 2. Filter training data to primary pairs only
    # ------------------------------------------------------------------
    bmca_train_primary = _filter_primary(bmca_train)
    mrf_train_primary = _filter_primary(mrf_train)

    # Merge for BMCA+MRF (train and test)
    combined_train_primary = _merge_feature_tables(bmca_train_primary, mrf_train_primary)
    combined_test = _merge_feature_tables(bmca_test, mrf_test)

    bmca_feature_cols = _feature_cols(bmca_train_primary)
    combined_feature_cols = _feature_cols(combined_train_primary)

    n_primary_pairs = bmca_train_primary[GROUP_COL].nunique()
    logger.info(
        f"\n{'='*60}\n"
        f"Strategy 1: Primary-Only Training\n"
        f"{'='*60}\n"
        f"Training subjects: {len(bmca_train_primary)} ({n_primary_pairs} pairs)\n"
        f"BMCA features: {len(bmca_feature_cols)}\n"
        f"Combined features: {len(combined_feature_cols)}\n"
        f"Inner CV folds: {n_splits}\n"
        f"Optuna trials: {n_iter}\n"
        f"{'='*60}"
    )

    # Sanity check: enough data?
    if n_primary_pairs < 30:
        logger.warning(
            f"Only {n_primary_pairs} training pairs — experiment may be underpowered."
        )

    # ------------------------------------------------------------------
    # 3. Train BMCA-only on primary pairs
    # ------------------------------------------------------------------
    logger.info("\n--- Training BMCA-only (primary-only) ---")
    X_bmca = bmca_train_primary[bmca_feature_cols]
    y_bmca = bmca_train_primary[LABEL_COL].values.astype(float)
    groups_bmca = bmca_train_primary[GROUP_COL].values

    study_bmca, model_bmca = _train_catboost_optuna(
        X_bmca, y_bmca, groups_bmca,
        n_iter=n_iter, n_splits=n_splits, seed=seed,
    )
    logger.info(f"BMCA-only best inner CV AUC: {study_bmca.best_value:.4f}")
    logger.info(f"BMCA-only best params: {study_bmca.best_params}")

    # ------------------------------------------------------------------
    # 4. Train BMCA+MRF on primary pairs (same seed)
    # ------------------------------------------------------------------
    logger.info("\n--- Training BMCA+MRF (primary-only) ---")
    X_combined = combined_train_primary[combined_feature_cols]
    y_combined = combined_train_primary[LABEL_COL].values.astype(float)
    groups_combined = combined_train_primary[GROUP_COL].values

    study_combined, model_combined = _train_catboost_optuna(
        X_combined, y_combined, groups_combined,
        n_iter=n_iter, n_splits=n_splits, seed=seed,
    )
    logger.info(f"BMCA+MRF best inner CV AUC: {study_combined.best_value:.4f}")
    logger.info(f"BMCA+MRF best params: {study_combined.best_params}")

    # ------------------------------------------------------------------
    # 5. Evaluate both on primary test set
    # ------------------------------------------------------------------
    logger.info("\n--- Evaluating on primary test set ---")
    result_bmca = _evaluate(
        model_bmca, bmca_test, bmca_feature_cols,
        "BMCA-only (primary-trained)", n_boot=n_boot, seed=seed,
    )
    result_combined = _evaluate(
        model_combined, combined_test, combined_feature_cols,
        "BMCA+MRF (primary-trained)", n_boot=n_boot, seed=seed,
    )

    # ------------------------------------------------------------------
    # 6. Paired bootstrap AUC difference test
    # ------------------------------------------------------------------
    logger.info("\n--- Paired bootstrap AUC difference ---")
    diff_test = _bootstrap_auc_diff(
        result_bmca["y_true"],
        result_combined["y_score"],
        result_bmca["y_score"],
        result_bmca["groups"],
        n_boot=10_000,
        seed=seed,
    )
    logger.info(
        f"BMCA+MRF vs BMCA-only:  Δ AUC = {diff_test['observed_diff']:+.4f}  "
        f"95% CI [{diff_test['ci_low']:+.4f}, {diff_test['ci_high']:+.4f}]  "
        f"p(Δ ≤ 0) = {diff_test['p_value']:.3f}"
    )

    # ------------------------------------------------------------------
    # 7. Feature importances
    # ------------------------------------------------------------------
    imp_bmca = _feature_importance(model_bmca, bmca_feature_cols)
    imp_combined = _feature_importance(model_combined, combined_feature_cols)

    logger.info(f"\nBMCA-only top 10 features:\n{imp_bmca.head(10).to_string(index=False)}")
    logger.info(f"\nBMCA+MRF top 10 features:\n{imp_combined.head(10).to_string(index=False)}")

    # Separate MRF features in the combined model importance
    mrf_feature_cols = _feature_cols(mrf_train_primary)
    mrf_in_combined = imp_combined[imp_combined["feature"].isin(mrf_feature_cols)]
    logger.info(f"\nMRF features in BMCA+MRF model (top 10):\n{mrf_in_combined.head(10).to_string(index=False)}")

    # ------------------------------------------------------------------
    # 8. Save outputs
    # ------------------------------------------------------------------
    _plot_roc_comparison(result_bmca, result_combined, f"{plots_dir}/primary_only_roc.pdf")

    metrics = pd.DataFrame([
        {
            "model": "bmca_primary_only",
            "auc": round(result_bmca["auc"], 4),
            "auc_ci_low_95": round(result_bmca["ci_low"], 4),
            "auc_ci_high_95": round(result_bmca["ci_high"], 4),
            "best_inner_cv_auc": round(study_bmca.best_value, 4),
            "n_features": len(bmca_feature_cols),
            "n_train_subjects": len(bmca_train_primary),
            "n_train_pairs": n_primary_pairs,
            **{f"param_{k}": v for k, v in study_bmca.best_params.items()},
        },
        {
            "model": "bmca_mrf_primary_only",
            "auc": round(result_combined["auc"], 4),
            "auc_ci_low_95": round(result_combined["ci_low"], 4),
            "auc_ci_high_95": round(result_combined["ci_high"], 4),
            "best_inner_cv_auc": round(study_combined.best_value, 4),
            "n_features": len(combined_feature_cols),
            "n_train_subjects": len(combined_train_primary),
            "n_train_pairs": combined_train_primary[GROUP_COL].nunique(),
            **{f"param_{k}": v for k, v in study_combined.best_params.items()},
        },
    ])
    metrics.to_csv(f"{output_dir}/primary_only_evaluation.csv", index=False)

    bootstrap_df = pd.DataFrame([{
        "comparison": "bmca_mrf_vs_bmca (primary-only trained)",
        "observed_diff": round(diff_test["observed_diff"], 4),
        "ci_low_95": round(diff_test["ci_low"], 4),
        "ci_high_95": round(diff_test["ci_high"], 4),
        "p_value": round(diff_test["p_value"], 4),
        "n_boot": diff_test["n_boot"],
    }])
    bootstrap_df.to_csv(f"{output_dir}/primary_only_bootstrap_diff.csv", index=False)

    imp_bmca.to_csv(f"{output_dir}/primary_only_bmca_importance.csv", index=False)
    imp_combined.to_csv(f"{output_dir}/primary_only_bmca_mrf_importance.csv", index=False)

    joblib.dump(
        {
            "model_bmca": model_bmca,
            "model_bmca_mrf": model_combined,
            "study_bmca": study_bmca,
            "study_bmca_mrf": study_combined,
            "bmca_feature_cols": bmca_feature_cols,
            "combined_feature_cols": combined_feature_cols,
            "result_bmca": result_bmca,
            "result_bmca_mrf": result_combined,
            "diff_test": diff_test,
        },
        f"{output_dir}/primary_only_models.joblib",
    )
    logger.info(f"\nAll results saved to {output_dir}/")

    # ------------------------------------------------------------------
    # 9. Summary
    # ------------------------------------------------------------------
    logger.info(
        f"\n{'='*60}\n"
        f"SUMMARY: Strategy 1 — Primary-Only Training\n"
        f"{'='*60}\n"
        f"                         BMCA-only    BMCA+MRF\n"
        f"  Inner CV AUC:          {study_bmca.best_value:.4f}       {study_combined.best_value:.4f}\n"
        f"  Test AUC:              {result_bmca['auc']:.4f}       {result_combined['auc']:.4f}\n"
        f"  Test 95% CI:           [{result_bmca['ci_low']:.3f}, {result_bmca['ci_high']:.3f}]  "
        f"[{result_combined['ci_low']:.3f}, {result_combined['ci_high']:.3f}]\n"
        f"\n"
        f"  Δ AUC (BMCA+MRF − BMCA): {diff_test['observed_diff']:+.4f}\n"
        f"  Bootstrap 95% CI:        [{diff_test['ci_low']:+.4f}, {diff_test['ci_high']:+.4f}]\n"
        f"  p(Δ ≤ 0):               {diff_test['p_value']:.3f}\n"
        f"{'='*60}"
    )

    return {
        "study_bmca": study_bmca,
        "study_bmca_mrf": study_combined,
        "model_bmca": model_bmca,
        "model_bmca_mrf": model_combined,
        "result_bmca": result_bmca,
        "result_bmca_mrf": result_combined,
        "diff_test": diff_test,
        "importance_bmca": imp_bmca,
        "importance_bmca_mrf": imp_combined,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Strategy 1: Train BMCA and BMCA+MRF on primary pairs only"
    )
    parser.add_argument("--bmca_train", default="data/adni_bmca_features_train.csv")
    parser.add_argument("--bmca_test", default="data/adni_bmca_features_test.csv")
    parser.add_argument("--mrf_train", default="data/adni_mrf_features_train.csv")
    parser.add_argument("--mrf_test", default="data/adni_mrf_features_test.csv")
    parser.add_argument("--output_dir", default="results")
    parser.add_argument("--plots_dir", default="plots")
    parser.add_argument("--n_iter", type=int, default=50,
                        help="Number of Optuna trials (default: 50)")
    parser.add_argument("--n_splits", type=int, default=3,
                        help="Number of inner CV folds (default: 3 for small sample)")
    parser.add_argument("--n_boot", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
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
    )
