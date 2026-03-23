"""
Score-Level Stacking Evaluation
=================================

Tests whether MRF features provide orthogonal predictive signal to BMCA by
training each model independently and combining their probability scores via a
two-input logistic regression stacker.

Design
------
1. Train BMCA and MRF CatBoost models independently with Optuna inner CV,
   using the same primary-only val_mask as model_bmca_evaluation.py and
   model_mrf_evaluation.py. The CV objective is calibrated to the primary
   CN→MCI task; augmentation pairs contribute to training but not to fold
   scoring.
2. Compute out-of-fold (OOF) probability scores for all training subjects
   from each base model using the inner CV splits returned by train_model.
   The OOF scores use the best hyperparameters found by Optuna, re-fit on
   each training fold — no label leakage from the test set.
3. Fit a two-input logistic regression stacker on primary training pairs
   (analysis_set == 'primary') only. Restricting to primary pairs prevents
   the augmentation contrast (stable CN vs CI→dementia) from biasing the
   combination weights toward the easier classification problem.
4. Evaluate BMCA-only, MRF-only, and stacked AUC on the primary held-out
   test set (evaluation_eligible == 1, n=36 pairs) with bootstrap 95% CIs.

Interpretation
--------------
The stacker has exactly two free parameters (one weight per model). A
near-zero or negative MRF coefficient is direct evidence that MRF prediction
adds no orthogonal signal over BMCA — even when each model is allowed to use
its own optimised feature representation and hyperparameters.

Relationship to other scripts
------------------------------
The BMCA and MRF base models are trained with an identical setup to
model_bmca_evaluation.py and model_mrf_evaluation.py (same Optuna budget,
same val_mask, same seed). Running those scripts and this script with the
same seed will produce equivalent base models; the stacking layer is new.

Usage::

    python model_stacking_evaluation.py
    python model_stacking_evaluation.py --n_iter 100 --seed 42 --n_jobs 4
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
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import RocCurveDisplay, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict

sys.path.insert(0, str(Path(__file__).parent))
from src.utils_model import train_model

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


# =============================================================================
# Data helpers
# =============================================================================


def _load_splits(train_path: str, test_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    return train, test


def _feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in _METADATA_COLS]


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


# =============================================================================
# Base model training
# =============================================================================


def _train_base_model(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    label: str,
    n_iter: int,
    n_splits: int,
    seed: int,
    n_jobs: int,
    gpu: bool,
) -> tuple[object, object, list]:
    """
    Train a CatBoost model with Optuna inner CV and primary-only val_mask.

    Identical setup to model_bmca_evaluation.py / model_mrf_evaluation.py.
    Returns (study, best_model, inner_splits).
    """
    feature_cols = _feature_cols(train_df)
    X_train = train_df[feature_cols]
    y_train = train_df[LABEL_COL].astype(float)
    groups_train = train_df[GROUP_COL]
    X_test = test_df[feature_cols]
    y_test = test_df[LABEL_COL].astype(float)

    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    val_mask = (train_df["analysis_set"] == "primary").values

    logger.info(
        f"Training {label} model: {len(feature_cols)} features, "
        f"{len(X_train)} train rows, {n_iter} Optuna trials."
    )

    study, best_model, inner_splits = train_model(
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
    return study, best_model, inner_splits


# =============================================================================
# OOF score computation
# =============================================================================


def _compute_oof_scores(
    model: object,
    train_df: pd.DataFrame,
    inner_splits: list,
) -> np.ndarray:
    """
    Compute out-of-fold probability scores for all training subjects.

    Re-fits the model (same hyperparameters as the best Optuna model) in each
    inner CV fold and predicts on the held-out fold. The pre-computed
    inner_splits are passed directly to cross_val_predict, ensuring that the
    OOF scores use exactly the same fold boundaries used during hyperparameter
    search — no additional data leakage is introduced.

    CatBoost is multi-threaded internally, so cross_val_predict runs folds
    sequentially (n_jobs=1) to avoid resource contention.
    """
    feature_cols = _feature_cols(train_df)
    X_train = train_df[feature_cols]
    y_train = train_df[LABEL_COL].astype(float)

    oof_scores = cross_val_predict(
        clone(model),
        X_train,
        y_train,
        cv=inner_splits,
        method="predict_proba",
        n_jobs=1,
    )[:, 1]

    return oof_scores


# =============================================================================
# Stacker training
# =============================================================================


def _fit_stacker(
    bmca_oof: np.ndarray,
    mrf_oof: np.ndarray,
    y_train: np.ndarray,
    primary_mask: np.ndarray,
    seed: int,
) -> LogisticRegression:
    """
    Fit a two-input logistic regression on primary training OOF scores.

    Stacking weights are learned only on primary CN→MCI pairs so they reflect
    the primary task and are not biased by the augmentation contrast (stable CN
    vs CI→dementia). With two input features, the model cannot overfit on 143
    primary training pairs.
    """
    X_stack = np.column_stack([bmca_oof, mrf_oof])[primary_mask]
    y_stack = y_train[primary_mask]

    stacker = LogisticRegression(C=1.0, max_iter=1000, random_state=seed)
    stacker.fit(X_stack, y_stack)

    bmca_coef, mrf_coef = stacker.coef_[0]
    logger.info(
        f"Stacker coefficients:  BMCA = {bmca_coef:.4f},  MRF = {mrf_coef:.4f}  "
        f"(intercept = {stacker.intercept_[0]:.4f})"
    )
    logger.info(
        "Interpretation: a near-zero or negative MRF coefficient means MRF "
        "adds no orthogonal signal beyond BMCA at the score level."
    )
    return stacker


# =============================================================================
# Evaluation
# =============================================================================


def _evaluate_test(
    bmca_model: object,
    mrf_model: object,
    stacker: LogisticRegression,
    bmca_test_df: pd.DataFrame,
    mrf_test_df: pd.DataFrame,
    n_boot: int,
    seed: int,
) -> dict:
    """
    Evaluate BMCA-only, MRF-only, and stacked AUC on the primary test set.

    Filters both test DataFrames to evaluation_eligible == 1 and verifies
    subject-level alignment before computing scores.
    """
    bmca_elig = _evaluation_eligible(bmca_test_df)
    mrf_elig = _evaluation_eligible(mrf_test_df)

    if not (bmca_elig["subject_id"].values == mrf_elig["subject_id"].values).all():
        raise ValueError(
            "BMCA and MRF test sets are misaligned after filtering to "
            "evaluation_eligible == 1. Both CSVs must be produced from the "
            "same train/test split."
        )

    bmca_feat = _feature_cols(bmca_test_df)
    mrf_feat = _feature_cols(mrf_test_df)

    y_test = bmca_elig[LABEL_COL].values.astype(float)
    groups_test = bmca_elig[GROUP_COL].values

    bmca_scores = bmca_model.predict_proba(bmca_elig[bmca_feat])[:, 1]
    mrf_scores = mrf_model.predict_proba(mrf_elig[mrf_feat])[:, 1]
    stacked_scores = stacker.predict_proba(
        np.column_stack([bmca_scores, mrf_scores])
    )[:, 1]

    result = {}
    for name, scores in [
        ("bmca", bmca_scores),
        ("mrf", mrf_scores),
        ("stacked", stacked_scores),
    ]:
        auc = roc_auc_score(y_test, scores)
        ci_low, ci_high = _bootstrap_auc(y_test, scores, groups_test, n_boot=n_boot, seed=seed)
        result[name] = {"auc": auc, "ci_low": ci_low, "ci_high": ci_high, "scores": scores}
        logger.info(
            f"  {name.upper():>8s}  AUC = {auc:.3f}  95% CI [{ci_low:.3f}, {ci_high:.3f}]"
        )

    result["y_true"] = y_test
    result["groups"] = groups_test
    return result


# =============================================================================
# Output
# =============================================================================


def plot_roc_comparison(result: dict, output_path: str) -> None:
    n_pairs = int(result["y_true"].sum())
    fig, ax = plt.subplots(figsize=(7, 7))

    style = {
        "bmca":    ("steelblue",  "BMCA"),
        "mrf":     ("darkorange", "MRF"),
        "stacked": ("seagreen",   "Stacked (BMCA + MRF)"),
    }
    for name, (color, label) in style.items():
        r = result[name]
        RocCurveDisplay.from_predictions(
            result["y_true"],
            r["scores"],
            ax=ax,
            name=f"{label}  AUC = {r['auc']:.3f}  [{r['ci_low']:.3f}, {r['ci_high']:.3f}]",
            color=color,
        )

    ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="Chance")
    ax.set_title(
        f"Score-Level Stacking — ROC Curves\n"
        f"Primary test set  (n = {n_pairs} pairs)"
    )
    ax.set_xlabel("1 − Specificity (FPR)")
    ax.set_ylabel("Sensitivity (TPR)")
    ax.grid(True, which="both", alpha=0.3)
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"ROC comparison plot saved to {output_path}")


def save_results(
    result: dict,
    stacker: LogisticRegression,
    bmca_study: object,
    mrf_study: object,
    output_dir: str,
) -> None:
    bmca_coef, mrf_coef = stacker.coef_[0]
    rows = []
    for name in ("bmca", "mrf", "stacked"):
        r = result[name]
        rows.append(
            {
                "model": f"stacking_{name}",
                "auc": round(r["auc"], 4),
                "auc_ci_low_95": round(r["ci_low"], 4),
                "auc_ci_high_95": round(r["ci_high"], 4),
                "stacker_bmca_coef": round(float(bmca_coef), 6),
                "stacker_mrf_coef": round(float(mrf_coef), 6),
                "stacker_intercept": round(float(stacker.intercept_[0]), 6),
                "bmca_inner_cv_auc": round(bmca_study.best_value, 4),
                "mrf_inner_cv_auc": round(mrf_study.best_value, 4),
            }
        )
    pd.DataFrame(rows).to_csv(f"{output_dir}/stacking_evaluation.csv", index=False)
    logger.info(f"Results saved to {output_dir}/stacking_evaluation.csv")


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

    # ----- Load data -----
    logger.info("Loading BMCA and MRF feature splits...")
    bmca_train_df, bmca_test_df = _load_splits(bmca_train_path, bmca_test_path)
    mrf_train_df, mrf_test_df = _load_splits(mrf_train_path, mrf_test_path)

    if not (bmca_train_df["subject_id"].values == mrf_train_df["subject_id"].values).all():
        raise ValueError(
            "BMCA and MRF training DataFrames have different subjects or row ordering. "
            "Both must be produced from the same train/test split."
        )

    y_train = bmca_train_df[LABEL_COL].astype(float).values
    primary_mask = (bmca_train_df["analysis_set"] == "primary").values
    logger.info(
        f"Training set: {primary_mask.sum()} primary rows, "
        f"{(~primary_mask).sum()} augmentation rows."
    )

    # ----- Train base models -----
    logger.info("\n=== Training BMCA base model ===")
    bmca_study, bmca_model, bmca_inner_splits = _train_base_model(
        bmca_train_df, bmca_test_df, "BMCA",
        n_iter=n_iter, n_splits=n_splits, seed=seed, n_jobs=n_jobs, gpu=gpu,
    )

    logger.info("\n=== Training MRF base model ===")
    mrf_study, mrf_model, mrf_inner_splits = _train_base_model(
        mrf_train_df, mrf_test_df, "MRF",
        n_iter=n_iter, n_splits=n_splits, seed=seed, n_jobs=n_jobs, gpu=gpu,
    )

    # ----- Compute OOF scores -----
    logger.info("\n=== Computing out-of-fold scores ===")
    bmca_oof = _compute_oof_scores(bmca_model, bmca_train_df, bmca_inner_splits)
    mrf_oof = _compute_oof_scores(mrf_model, mrf_train_df, mrf_inner_splits)

    primary_oof_auc_bmca = roc_auc_score(y_train[primary_mask], bmca_oof[primary_mask])
    primary_oof_auc_mrf = roc_auc_score(y_train[primary_mask], mrf_oof[primary_mask])
    logger.info(
        f"Primary training OOF AUC — BMCA: {primary_oof_auc_bmca:.3f}, "
        f"MRF: {primary_oof_auc_mrf:.3f}"
    )

    # ----- Fit stacker -----
    logger.info("\n=== Fitting stacker on primary training OOF scores ===")
    stacker = _fit_stacker(bmca_oof, mrf_oof, y_train, primary_mask, seed=seed)

    # ----- Evaluate on primary test set -----
    logger.info("\n=== Evaluating on primary test set (n=36 pairs) ===")
    result = _evaluate_test(
        bmca_model, mrf_model, stacker,
        bmca_test_df, mrf_test_df,
        n_boot=n_boot, seed=seed,
    )

    # ----- Save outputs -----
    plot_roc_comparison(result, output_path=f"{plots_dir}/stacking_roc.pdf")
    save_results(result, stacker, bmca_study, mrf_study, output_dir)

    joblib.dump(
        {
            "bmca_model": bmca_model,
            "bmca_study": bmca_study,
            "bmca_oof": bmca_oof,
            "mrf_model": mrf_model,
            "mrf_study": mrf_study,
            "mrf_oof": mrf_oof,
            "stacker": stacker,
            "result": result,
        },
        f"{output_dir}/stacking_model.joblib",
    )
    logger.info(f"Artifacts saved to {output_dir}/stacking_model.joblib")

    return {
        "bmca_model": bmca_model,
        "mrf_model": mrf_model,
        "stacker": stacker,
        "result": result,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Score-level stacking: test whether MRF adds orthogonal signal to BMCA"
    )
    parser.add_argument("--bmca_train", default="data/adni_bmca_features_train.csv")
    parser.add_argument("--bmca_test", default="data/adni_bmca_features_test.csv")
    parser.add_argument("--mrf_train", default="data/adni_mrf_features_train.csv")
    parser.add_argument("--mrf_test", default="data/adni_mrf_features_test.csv")
    parser.add_argument("--output_dir", default="results")
    parser.add_argument("--plots_dir", default="plots")
    parser.add_argument("--n_iter", type=int, default=50,
                        help="Number of Optuna trials per base model (default: 50)")
    parser.add_argument("--n_splits", type=int, default=5,
                        help="Number of inner CV folds (default: 5)")
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
