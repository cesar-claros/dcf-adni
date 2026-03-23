"""
BMCA + Vascular Composite Evaluation
======================================

Tests whether adding a pre-specified vascular burden composite to the BMCA
feature set improves discrimination over BMCA alone.

Composite definition
---------------------
Five MRF features selected for consistent appearance across experiments:

    diastolic_bp   — blood pressure (higher = more vascular burden)
    pulse          — heart rate (higher = more vascular burden)
    BMI            — adiposity (higher = more risk)
    eGFR           — renal function (lower = more risk → negated)
    serum_glucose  — glycaemia (higher = more metabolic dysregulation)

Each feature is standardised using training-set mean and standard deviation
(fit on train, applied to test — no leakage). eGFR is negated before
standardisation so all five components point in the high-values-mean-higher-risk
direction. The composite is the row-wise nanmean across available standardised
values; subjects with fewer than 2 non-missing features receive NaN (handled
natively by CatBoost).

Missing values: diastolic_bp, pulse, and BMI have no missing values in the
primary training set; eGFR and serum_glucose have ~37% missing. Because the
three complete features guarantee at least 3/5 values for every subject, no
subject receives a NaN composite.

Why pre-specified rather than data-adaptive
--------------------------------------------
The residual analysis (Experiment 5) showed no MRF feature reliably predicts
BMCA errors after Bonferroni correction. Data-adaptive selection (forward
selection, LASSO) would exploit noise at n=143 training pairs. A pre-specified
composite avoids selection bias while testing whether the concentrated vascular
signal adds incremental value.

Why these five features
------------------------
- Appear consistently in BMCA+MRF combined importances (diastolic_bp, pulse,
  serum_glucose) and MRF standalone importances (eGFR, BMI, pulse)
- Represent continuous physiological measurements (not binary flags)
- Cover distinct vascular/metabolic channels: blood pressure, heart rate,
  adiposity, renal function, glycaemia

Comparison note
----------------
This script trains BMCA-only and BMCA+composite models with the same seed,
so any AUC difference is attributable to the composite feature rather than
Optuna variance.

Usage::

    python model_bmca_vascular_evaluation.py
    python model_bmca_vascular_evaluation.py --n_iter 100 --seed 42
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

_METADATA_COLS = {
    "subject_id", "pair_id", "group", "transition", "transition_label",
    "matched_cohort", "analysis_set", "evaluation_eligible",
    "abs_age_gap", "split", "split_group_source",
    "first_conversion_month", "baseline_diagnosis", "n_followup_visits_ge12_with_diag",
}

LABEL_COL = "transition_label"
GROUP_COL = "group"

# Features that form the composite and their risk direction.
# True  = keep z-score as-is (higher value → higher risk).
# False = negate z-score    (higher value → lower risk).
_VASCULAR_FEATURES: dict[str, bool] = {
    "diastolic_bp":  True,
    "pulse":         True,
    "BMI":           True,
    "eGFR":          False,   # higher eGFR = better renal function = lower risk
    "serum_glucose": True,
}

_MIN_FEATURES_FOR_COMPOSITE = 2


# =============================================================================
# Data helpers
# =============================================================================


def _load_splits(train_path: str, test_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    return pd.read_csv(train_path), pd.read_csv(test_path)


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
# Vascular composite construction
# =============================================================================


def _fit_composite_params(mrf_train_df: pd.DataFrame) -> dict[str, tuple[float, float]]:
    """
    Compute per-feature mean and std from the training set.

    Returns a dict mapping feature name → (mean, std), computed on all training
    rows (primary + augmentation) so the standardisation reflects the full
    training distribution. Std is set to 1.0 if zero to avoid division by zero.
    """
    params = {}
    for feat in _VASCULAR_FEATURES:
        vals = mrf_train_df[feat].dropna()
        mean = float(vals.mean())
        std = float(vals.std())
        params[feat] = (mean, max(std, 1e-8))
    return params


def _compute_composite(
    mrf_df: pd.DataFrame,
    params: dict[str, tuple[float, float]],
) -> np.ndarray:
    """
    Compute the vascular composite for every row in mrf_df.

    Each feature is z-scored using the training-set params, then direction-
    aligned (eGFR negated). The composite is the nanmean across available
    features; rows with fewer than _MIN_FEATURES_FOR_COMPOSITE non-NaN values
    receive NaN.
    """
    z_scores = {}
    for feat, keep_direction in _VASCULAR_FEATURES.items():
        mean, std = params[feat]
        z = (mrf_df[feat].values.astype(float) - mean) / std
        z_scores[feat] = z if keep_direction else -z

    z_matrix = np.column_stack(list(z_scores.values()))   # (n, 5)
    n_available = np.sum(~np.isnan(z_matrix), axis=1)
    composite = np.where(
        n_available >= _MIN_FEATURES_FOR_COMPOSITE,
        np.nanmean(z_matrix, axis=1),
        np.nan,
    )
    return composite


# =============================================================================
# Model training
# =============================================================================


def _train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    groups_train: pd.Series,
    val_mask: np.ndarray,
    label: str,
    n_iter: int,
    n_splits: int,
    seed: int,
    n_jobs: int,
    gpu: bool,
) -> tuple[object, object, list]:
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    logger.info(
        f"Training {label}: {X_train.shape[1]} features, "
        f"{len(X_train)} train rows, {n_iter} Optuna trials."
    )
    return train_model(
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


# =============================================================================
# Evaluation
# =============================================================================


def _evaluate(
    model: object,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    label: str,
    n_boot: int,
    seed: int,
) -> dict:
    eligible = _evaluation_eligible(test_df)
    X = eligible[feature_cols]
    y = eligible[LABEL_COL].values.astype(float)
    groups = eligible[GROUP_COL].values

    scores = model.predict_proba(X)[:, 1]
    auc = roc_auc_score(y, scores)
    ci_low, ci_high = _bootstrap_auc(y, scores, groups, n_boot=n_boot, seed=seed)
    logger.info(f"{label}  AUC = {auc:.3f}  95% CI [{ci_low:.3f}, {ci_high:.3f}]")
    return {"auc": auc, "ci_low": ci_low, "ci_high": ci_high,
            "y_true": y, "scores": scores, "groups": groups}


def _feature_importance(model: object, feature_cols: list[str]) -> pd.DataFrame:
    return (
        pd.DataFrame({
            "feature": feature_cols,
            "importance": model.get_feature_importance(),
        })
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


# =============================================================================
# Output
# =============================================================================


def _plot_roc_comparison(
    bmca_result: dict,
    vascular_result: dict,
    output_path: str,
) -> None:
    n_pairs = int(bmca_result["y_true"].sum())
    fig, ax = plt.subplots(figsize=(7, 7))

    for result, color, label in [
        (bmca_result,     "steelblue",  "BMCA"),
        (vascular_result, "seagreen",   "BMCA + vascular composite"),
    ]:
        r = result
        RocCurveDisplay.from_predictions(
            r["y_true"], r["scores"], ax=ax, color=color,
            name=f"{label}  AUC = {r['auc']:.3f}  [{r['ci_low']:.3f}, {r['ci_high']:.3f}]",
        )

    ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="Chance")
    ax.set_title(
        f"BMCA vs BMCA + Vascular Composite — ROC Curves\n"
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
    logger.info(f"ROC plot saved to {output_path}")


def _save_results(
    bmca_result: dict,
    vascular_result: dict,
    bmca_study: object,
    vascular_study: object,
    output_dir: str,
) -> None:
    rows = [
        {
            "model": "bmca_catboost",
            "auc": round(bmca_result["auc"], 4),
            "auc_ci_low_95": round(bmca_result["ci_low"], 4),
            "auc_ci_high_95": round(bmca_result["ci_high"], 4),
            "best_inner_cv_auc": round(bmca_study.best_value, 4),
        },
        {
            "model": "bmca_vascular_catboost",
            "auc": round(vascular_result["auc"], 4),
            "auc_ci_low_95": round(vascular_result["ci_low"], 4),
            "auc_ci_high_95": round(vascular_result["ci_high"], 4),
            "best_inner_cv_auc": round(vascular_study.best_value, 4),
        },
    ]
    pd.DataFrame(rows).to_csv(f"{output_dir}/bmca_vascular_evaluation.csv", index=False)
    logger.info(f"Results saved to {output_dir}/bmca_vascular_evaluation.csv")


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
    logger.info("Loading feature splits...")
    bmca_train_df, bmca_test_df = _load_splits(bmca_train_path, bmca_test_path)
    mrf_train_df, mrf_test_df = _load_splits(mrf_train_path, mrf_test_path)

    for split, b_df, m_df in [("train", bmca_train_df, mrf_train_df),
                               ("test",  bmca_test_df,  mrf_test_df)]:
        if not (b_df["subject_id"].values == m_df["subject_id"].values).all():
            raise ValueError(
                f"BMCA and MRF {split} DataFrames have different subjects or row ordering."
            )

    val_mask = (bmca_train_df["analysis_set"] == "primary").values
    y_train = bmca_train_df[LABEL_COL].astype(float)
    y_test = bmca_test_df[LABEL_COL].astype(float)
    groups_train = bmca_train_df[GROUP_COL]

    # ----- Fit composite parameters on training data -----
    logger.info(
        f"Fitting vascular composite from: {list(_VASCULAR_FEATURES.keys())}"
    )
    composite_params = _fit_composite_params(mrf_train_df)

    train_composite = _compute_composite(mrf_train_df, composite_params)
    test_composite  = _compute_composite(mrf_test_df,  composite_params)

    n_nan_train = np.isnan(train_composite).sum()
    n_nan_test  = np.isnan(test_composite).sum()
    logger.info(
        f"Composite computed — train NaN: {n_nan_train}/{len(train_composite)}, "
        f"test NaN: {n_nan_test}/{len(test_composite)}"
    )
    logger.info(
        f"Train composite stats — mean: {np.nanmean(train_composite):.3f}, "
        f"std: {np.nanstd(train_composite):.3f}, "
        f"range: [{np.nanmin(train_composite):.3f}, {np.nanmax(train_composite):.3f}]"
    )

    # ----- Assemble feature matrices -----
    bmca_feat_cols = _feature_cols(bmca_train_df)

    # BMCA-only
    X_bmca_train = bmca_train_df[bmca_feat_cols].copy()
    X_bmca_test  = bmca_test_df[bmca_feat_cols].copy()

    # BMCA + vascular composite
    X_vasc_train = X_bmca_train.copy()
    X_vasc_test  = X_bmca_test.copy()
    X_vasc_train["vascular_composite"] = train_composite
    X_vasc_test["vascular_composite"]  = test_composite
    vasc_feat_cols = bmca_feat_cols + ["vascular_composite"]

    # ----- Train both models with the same seed -----
    logger.info("\n=== Training BMCA-only (baseline) ===")
    bmca_study, bmca_model, _ = _train_model(
        X_bmca_train, y_train, X_bmca_test, y_test,
        groups_train, val_mask, "BMCA-only",
        n_iter=n_iter, n_splits=n_splits, seed=seed, n_jobs=n_jobs, gpu=gpu,
    )

    logger.info("\n=== Training BMCA + vascular composite ===")
    vasc_study, vasc_model, _ = _train_model(
        X_vasc_train, y_train, X_vasc_test, y_test,
        groups_train, val_mask, "BMCA+vascular",
        n_iter=n_iter, n_splits=n_splits, seed=seed, n_jobs=n_jobs, gpu=gpu,
    )

    # ----- Evaluate -----
    logger.info("\n=== Evaluating on primary test set (n=36 pairs) ===")
    bmca_result  = _evaluate(bmca_model,  bmca_test_df,  bmca_feat_cols,  "BMCA-only",     n_boot, seed)
    vasc_result  = _evaluate(vasc_model,  bmca_test_df,  vasc_feat_cols,  "BMCA+vascular", n_boot, seed)

    delta = vasc_result["auc"] - bmca_result["auc"]
    logger.info(f"\nΔ AUC (BMCA+vascular − BMCA-only) = {delta:+.3f}")
    logger.info(
        f"Inner CV AUC — BMCA-only: {bmca_study.best_value:.3f}, "
        f"BMCA+vascular: {vasc_study.best_value:.3f}"
    )

    # ----- Feature importances -----
    bmca_imp  = _feature_importance(bmca_model,  bmca_feat_cols)
    vasc_imp  = _feature_importance(vasc_model,  vasc_feat_cols)

    composite_rank = int(vasc_imp[vasc_imp["feature"] == "vascular_composite"].index[0]) + 1
    composite_importance = float(
        vasc_imp.loc[vasc_imp["feature"] == "vascular_composite", "importance"].iloc[0]
    )
    logger.info(
        f"vascular_composite importance: {composite_importance:.2f}%  "
        f"(rank {composite_rank}/{len(vasc_feat_cols)})"
    )
    logger.info(
        f"\nTop 10 features (BMCA+vascular):\n"
        f"{vasc_imp.head(10).to_string(index=False)}"
    )

    # ----- Save outputs -----
    _plot_roc_comparison(bmca_result, vasc_result, f"{plots_dir}/bmca_vascular_roc.pdf")
    _save_results(bmca_result, vasc_result, bmca_study, vasc_study, output_dir)
    vasc_imp.to_csv(f"{output_dir}/bmca_vascular_feature_importance.csv", index=False)

    joblib.dump(
        {
            "bmca_model": bmca_model,      "bmca_study": bmca_study,
            "vasc_model": vasc_model,      "vasc_study": vasc_study,
            "composite_params": composite_params,
            "bmca_result": bmca_result,    "vasc_result": vasc_result,
            "bmca_importance": bmca_imp,   "vasc_importance": vasc_imp,
        },
        f"{output_dir}/bmca_vascular_model.joblib",
    )
    logger.info(f"Artifacts saved to {output_dir}/bmca_vascular_model.joblib")

    return {
        "bmca_model": bmca_model, "vasc_model": vasc_model,
        "bmca_result": bmca_result, "vasc_result": vasc_result,
        "vasc_importance": vasc_imp,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train and evaluate BMCA with a pre-specified vascular composite feature"
    )
    parser.add_argument("--bmca_train", default="data/adni_bmca_features_train.csv")
    parser.add_argument("--bmca_test",  default="data/adni_bmca_features_test.csv")
    parser.add_argument("--mrf_train",  default="data/adni_mrf_features_train.csv")
    parser.add_argument("--mrf_test",   default="data/adni_mrf_features_test.csv")
    parser.add_argument("--output_dir", default="results")
    parser.add_argument("--plots_dir",  default="plots")
    parser.add_argument("--n_iter",  type=int, default=50,
                        help="Optuna trials per model (default: 50)")
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--n_boot",  type=int, default=1000)
    parser.add_argument("--seed",    type=int, default=0)
    parser.add_argument("--n_jobs",  type=int, default=-1)
    parser.add_argument("--gpu",     action="store_true", default=False)
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
