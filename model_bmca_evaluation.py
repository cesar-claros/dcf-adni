"""
BMCA Model Training and Evaluation
====================================

Trains a CatBoost classifier on the BMCA (Biomarker / Medical / Cognitive
Assessment) feature set produced by data_preprocessing_feature_exports.py,
using Optuna hyperparameter tuning with stratified group k-fold cross-validation
(k=5), and evaluates on the primary held-out test set.

Pipeline
--------
1. Load the BMCA train and test splits from disk.
2. Separate feature columns from metadata (subject_id, group, labels, etc.).
3. Run Optuna TPE search (``n_iter`` trials) with StratifiedGroupKFold(5) as
   the inner CV, optimising ROC-AUC. Groups = matched pair IDs so no pair
   is split across a train/validation boundary.
4. Refit the best hyperparameters on the full training set.
5. Evaluate on the primary test set (evaluation_eligible == 1, n=36 pairs).
6. Compute a bootstrap 95% CI for AUC by resampling matched pairs.
7. Save the trained model, a results CSV, and a ROC plot.

CatBoost is used without WoE pre-transformation: it handles continuous,
binary, and ordinal features natively and tolerates missing values internally
via its default NaN treatment.

One caveat: the column audit retained ``baseline_diagnosis`` despite a large
train/test mode shift (67% → 100%). This variable equals 1 for all primary
subjects (CN at baseline) but varies for augmentation subjects in train.
CatBoost may use it as a proxy for analysis_set membership. Inspect feature
importances after training to assess its influence.

Usage::

    python model_bmca_evaluation.py
    python model_bmca_evaluation.py --n_iter 100 --seed 42 --n_jobs 4
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
from sklearn.metrics import RocCurveDisplay, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, str(Path(__file__).parent))
from src.utils_model import train_model

logging.basicConfig(level=logging.INFO, format="%(name)s — %(message)s")
logger = logging.getLogger(__name__)

# Columns that are not features — metadata, cohort provenance, and outcome-derived variables.
# first_conversion_month directly encodes the label (NaN for stable-CN, non-NaN for
# transition subjects) and must never enter the feature matrix.
# baseline_diagnosis and n_followup_visits_ge12_with_diag are cohort-selection and
# study-participation variables, not baseline risk factors.
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
# Training
# =============================================================================


def train_bmca_model(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    n_iter: int = 50,
    n_splits: int = 5,
    seed: int = 0,
    n_jobs: int = -1,
    gpu: bool = False,
) -> tuple[object, object, dict]:
    """
    Tune and fit a CatBoost model on the BMCA training set.

    Inner CV uses StratifiedGroupKFold(n_splits) with groups = matched pair ID,
    so both members of a matched pair always stay in the same fold.

    Returns the Optuna study, the refitted best model, and the inner CV splits.
    """
    feature_cols = _feature_cols(train_df)

    X_train = train_df[feature_cols]
    y_train = train_df[LABEL_COL].astype(float)
    groups_train = train_df[GROUP_COL]

    X_test = test_df[feature_cols]
    y_test = test_df[LABEL_COL].astype(float)

    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    logger.info(
        f"Training CatBoost on BMCA features: "
        f"{len(feature_cols)} features, "
        f"{len(X_train)} train rows, {len(X_test)} test rows, "
        f"{n_iter} Optuna trials, {n_splits}-fold stratified group CV."
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
    )

    return study, best_model, inner_splits


# =============================================================================
# Evaluation
# =============================================================================


def evaluate_on_test(
    model: object,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    n_boot: int = 1000,
    seed: int = 0,
) -> dict:
    """
    Evaluate the fitted model on the primary test set (evaluation_eligible == 1).
    """
    eligible = _evaluation_eligible(test_df)
    X_test = eligible[feature_cols]
    y_test = eligible[LABEL_COL].values.astype(float)
    groups = eligible[GROUP_COL].values

    y_score = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_score)
    ci_low, ci_high = _bootstrap_auc(y_test, y_score, groups, n_boot=n_boot, seed=seed)

    logger.info(
        f"BMCA CatBoost  AUC = {auc:.3f}  95% CI [{ci_low:.3f}, {ci_high:.3f}]"
    )
    return {
        "auc": auc,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "y_true": y_test,
        "y_score": y_score,
        "groups": groups,
    }


def feature_importance(model: object, feature_cols: list[str]) -> pd.DataFrame:
    """Return CatBoost feature importances sorted descending."""
    importances = model.get_feature_importance()
    return (
        pd.DataFrame({"feature": feature_cols, "importance": importances})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


# =============================================================================
# Output
# =============================================================================


def plot_roc(result: dict, output_path: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    n_pairs = int(result["y_true"].sum())
    RocCurveDisplay.from_predictions(
        result["y_true"],
        result["y_score"],
        ax=ax,
        name=f"BMCA CatBoost (AUC = {result['auc']:.3f})",
        color="steelblue",
        plot_chance_level=True,
    )
    ax.set_title(f"BMCA — ROC Curve  (primary test set, n = {n_pairs} pairs)")
    ax.set_xlabel("1 − Specificity (FPR)")
    ax.set_ylabel("Sensitivity (TPR)")
    ax.grid(True, which="both", alpha=0.3)
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"ROC plot saved to {output_path}")


def save_results(
    result: dict,
    study: object,
    importance_df: pd.DataFrame,
    output_dir: str,
) -> None:
    metrics_df = pd.DataFrame(
        [
            {
                "model": "bmca_catboost",
                "auc": round(result["auc"], 4),
                "auc_ci_low_95": round(result["ci_low"], 4),
                "auc_ci_high_95": round(result["ci_high"], 4),
                "best_inner_cv_auc": round(study.best_value, 4),
                **{f"param_{k}": v for k, v in study.best_params.items()},
            }
        ]
    )
    metrics_df.to_csv(f"{output_dir}/bmca_evaluation.csv", index=False)
    importance_df.to_csv(f"{output_dir}/bmca_feature_importance.csv", index=False)
    logger.info(f"Results saved to {output_dir}/")


# =============================================================================
# Entry point
# =============================================================================


def run(
    train_path: str = "data/adni_bmca_features_train.csv",
    test_path: str = "data/adni_bmca_features_test.csv",
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

    train_df, test_df = _load_splits(train_path, test_path)
    feature_cols = _feature_cols(train_df)

    study, best_model, inner_splits = train_bmca_model(
        train_df, test_df,
        n_iter=n_iter, n_splits=n_splits, seed=seed, n_jobs=n_jobs, gpu=gpu,
    )

    result = evaluate_on_test(best_model, test_df, feature_cols, n_boot=n_boot, seed=seed)
    importance_df = feature_importance(best_model, feature_cols)

    logger.info(f"\nTop 10 features by importance:\n{importance_df.head(10).to_string(index=False)}")

    plot_roc(result, output_path=f"{plots_dir}/bmca_roc.pdf")
    save_results(result, study, importance_df, output_dir)
    joblib.dump(
        {"model": best_model, "study": study, "feature_cols": feature_cols, "result": result},
        f"{output_dir}/bmca_model.joblib",
    )
    logger.info(f"Model saved to {output_dir}/bmca_model.joblib")

    return {"study": study, "model": best_model, "result": result, "importance": importance_df}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train and evaluate a CatBoost model on BMCA features"
    )
    parser.add_argument("--train", default="data/adni_bmca_features_train.csv")
    parser.add_argument("--test", default="data/adni_bmca_features_test.csv")
    parser.add_argument("--output_dir", default="results")
    parser.add_argument("--plots_dir", default="plots")
    parser.add_argument("--n_iter", type=int, default=50,
                        help="Number of Optuna trials (default: 50)")
    parser.add_argument("--n_splits", type=int, default=5,
                        help="Number of inner CV folds (default: 5)")
    parser.add_argument("--n_boot", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n_jobs", type=int, default=-1)
    parser.add_argument("--gpu", action="store_true", default=False)
    args = parser.parse_args()

    run(
        train_path=args.train,
        test_path=args.test,
        output_dir=args.output_dir,
        plots_dir=args.plots_dir,
        n_iter=args.n_iter,
        n_splits=args.n_splits,
        n_boot=args.n_boot,
        seed=args.seed,
        n_jobs=args.n_jobs,
        gpu=args.gpu,
    )
