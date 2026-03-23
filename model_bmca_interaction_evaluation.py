"""
BMCA + Pre-specified Biological Interaction Terms
===================================================

Tests whether MRF features *modify* the effect of BMCA features via
pre-specified cross-domain interaction terms (the "two-hit" hypothesis).

Five interactions, each grounded in a specific biological mechanism:

    1. plasma_ptau217 x hypertension
       Vascular pathology accelerates amyloid-driven neurodegeneration.

    2. plasma_abeta42_abeta40_ratio x serum_glucose
       Insulin resistance impairs amyloid clearance.

    3. csf_tau x eGFR
       Renal dysfunction affects biomarker clearance kinetics.

    4. cdrsb x education_years
       Cognitive reserve masks early decline on clinical assessments.

    5. logical_memory_delayed x heart_disease
       Cardiovascular disease affects cerebral perfusion and memory.

The interaction columns are raw products (CatBoost is scale-invariant
and handles NaN natively). BMCA-only and BMCA+interactions models are
trained head-to-head with the same seed so any AUC difference is
attributable to the interaction features.

Usage::

    python model_bmca_interaction_evaluation.py
    python model_bmca_interaction_evaluation.py --n_iter 100 --seed 42
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

# Pre-specified interaction terms: (bmca_feature, mrf_feature, product_column_name)
_INTERACTIONS = [
    ("plasma_ptau217",              "hypertension",    "ptau217_x_hypertension"),
    ("plasma_abeta42_abeta40_ratio", "serum_glucose",  "abeta_ratio_x_glucose"),
    ("csf_tau",                     "eGFR",            "tau_x_egfr"),
    ("cdrsb",                       "education_years", "cdrsb_x_education"),
    ("logical_memory_delayed",      "heart_disease",   "memory_x_heart_disease"),
]


# =============================================================================
# Data helpers
# =============================================================================


def _load_splits(train_path: str, test_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    return pd.read_csv(train_path), pd.read_csv(test_path)


def _feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in _METADATA_COLS]


def _evaluation_eligible(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["evaluation_eligible"] == 1].copy()


def _verify_alignment(bmca_df: pd.DataFrame, mrf_df: pd.DataFrame, split: str) -> None:
    if not (bmca_df["subject_id"].values == mrf_df["subject_id"].values).all():
        raise ValueError(
            f"BMCA and MRF {split} DataFrames have different subjects or row ordering."
        )


# =============================================================================
# Interaction construction
# =============================================================================


def _compute_interactions(
    bmca_df: pd.DataFrame,
    mrf_df: pd.DataFrame,
) -> pd.DataFrame:
    """Compute raw product interaction columns."""
    interaction_cols = {}
    for bmca_feat, mrf_feat, col_name in _INTERACTIONS:
        bmca_vals = bmca_df[bmca_feat].values.astype(float)
        mrf_vals = mrf_df[mrf_feat].values.astype(float)
        interaction_cols[col_name] = bmca_vals * mrf_vals
    return pd.DataFrame(interaction_cols, index=bmca_df.index)


# =============================================================================
# Bootstrap AUC CI
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
    interaction_result: dict,
    output_path: str,
) -> None:
    n_pairs = int(bmca_result["y_true"].sum())
    fig, ax = plt.subplots(figsize=(7, 7))

    for result, color, label in [
        (bmca_result,        "steelblue", "BMCA"),
        (interaction_result, "darkorange", "BMCA + interactions"),
    ]:
        RocCurveDisplay.from_predictions(
            result["y_true"], result["scores"], ax=ax, color=color,
            name=f"{label}  AUC = {result['auc']:.3f}  [{result['ci_low']:.3f}, {result['ci_high']:.3f}]",
        )

    ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="Chance")
    ax.set_title(
        f"BMCA vs BMCA + Interaction Terms — ROC Curves\n"
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
    interaction_result: dict,
    bmca_study: object,
    interaction_study: object,
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
            "model": "bmca_interaction_catboost",
            "auc": round(interaction_result["auc"], 4),
            "auc_ci_low_95": round(interaction_result["ci_low"], 4),
            "auc_ci_high_95": round(interaction_result["ci_high"], 4),
            "best_inner_cv_auc": round(interaction_study.best_value, 4),
        },
    ]
    pd.DataFrame(rows).to_csv(f"{output_dir}/bmca_interaction_evaluation.csv", index=False)
    logger.info(f"Results saved to {output_dir}/bmca_interaction_evaluation.csv")


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
        _verify_alignment(b_df, m_df, split)

    val_mask = (bmca_train_df["analysis_set"] == "primary").values
    y_train = bmca_train_df[LABEL_COL].astype(float)
    y_test = bmca_test_df[LABEL_COL].astype(float)
    groups_train = bmca_train_df[GROUP_COL]

    # ----- Compute interaction terms -----
    logger.info("Computing interaction terms...")
    train_interactions = _compute_interactions(bmca_train_df, mrf_train_df)
    test_interactions = _compute_interactions(bmca_test_df, mrf_test_df)

    for col_name in train_interactions.columns:
        n_nan = train_interactions[col_name].isna().sum()
        logger.info(f"  {col_name}: {n_nan}/{len(train_interactions)} NaN in train")

    # ----- Assemble feature matrices -----
    bmca_feat_cols = _feature_cols(bmca_train_df)
    interaction_names = [col for _, _, col in _INTERACTIONS]

    X_bmca_train = bmca_train_df[bmca_feat_cols].copy()
    X_bmca_test = bmca_test_df[bmca_feat_cols].copy()

    X_inter_train = pd.concat([X_bmca_train, train_interactions], axis=1)
    X_inter_test = pd.concat([X_bmca_test, test_interactions], axis=1)
    inter_feat_cols = bmca_feat_cols + interaction_names

    logger.info(
        f"Feature dimensions — BMCA: {len(bmca_feat_cols)}, "
        f"BMCA+interactions: {len(inter_feat_cols)} (+{len(interaction_names)} interactions)"
    )

    # ----- Train both models with the same seed -----
    logger.info("\n=== Training BMCA-only (baseline) ===")
    bmca_study, bmca_model, _ = _train_model(
        X_bmca_train, y_train, X_bmca_test, y_test,
        groups_train, val_mask, "BMCA-only",
        n_iter=n_iter, n_splits=n_splits, seed=seed, n_jobs=n_jobs, gpu=gpu,
    )

    logger.info("\n=== Training BMCA + interactions ===")
    inter_study, inter_model, _ = _train_model(
        X_inter_train, y_train, X_inter_test, y_test,
        groups_train, val_mask, "BMCA+interactions",
        n_iter=n_iter, n_splits=n_splits, seed=seed, n_jobs=n_jobs, gpu=gpu,
    )

    # ----- Evaluate -----
    bmca_test_df_inter = bmca_test_df.copy()
    for col in interaction_names:
        bmca_test_df_inter[col] = test_interactions[col].values

    logger.info("\n=== Evaluating on primary test set ===")
    bmca_result = _evaluate(bmca_model, bmca_test_df, bmca_feat_cols, "BMCA-only", n_boot, seed)
    inter_result = _evaluate(inter_model, bmca_test_df_inter, inter_feat_cols, "BMCA+interactions", n_boot, seed)

    delta = inter_result["auc"] - bmca_result["auc"]
    logger.info(f"\nΔ AUC (BMCA+interactions − BMCA-only) = {delta:+.3f}")
    logger.info(
        f"Inner CV AUC — BMCA-only: {bmca_study.best_value:.3f}, "
        f"BMCA+interactions: {inter_study.best_value:.3f}"
    )

    # ----- Feature importances -----
    bmca_imp = _feature_importance(bmca_model, bmca_feat_cols)
    inter_imp = _feature_importance(inter_model, inter_feat_cols)

    logger.info("\nInteraction feature importances:")
    for _, _, col_name in _INTERACTIONS:
        row = inter_imp[inter_imp["feature"] == col_name]
        if not row.empty:
            rank = int(row.index[0]) + 1
            imp = float(row["importance"].iloc[0])
            logger.info(f"  {col_name}: {imp:.2f}%  (rank {rank}/{len(inter_feat_cols)})")

    logger.info(
        f"\nTop 10 features (BMCA+interactions):\n"
        f"{inter_imp.head(10).to_string(index=False)}"
    )

    # ----- Save outputs -----
    _plot_roc_comparison(bmca_result, inter_result, f"{plots_dir}/bmca_interaction_roc.pdf")
    _save_results(bmca_result, inter_result, bmca_study, inter_study, output_dir)
    inter_imp.to_csv(f"{output_dir}/bmca_interaction_feature_importance.csv", index=False)

    joblib.dump(
        {
            "bmca_model": bmca_model,      "bmca_study": bmca_study,
            "inter_model": inter_model,    "inter_study": inter_study,
            "bmca_result": bmca_result,    "inter_result": inter_result,
            "bmca_importance": bmca_imp,   "inter_importance": inter_imp,
        },
        f"{output_dir}/bmca_interaction_model.joblib",
    )
    logger.info(f"Artifacts saved to {output_dir}/bmca_interaction_model.joblib")

    return {
        "bmca_model": bmca_model, "inter_model": inter_model,
        "bmca_result": bmca_result, "inter_result": inter_result,
        "inter_importance": inter_imp,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train and evaluate BMCA with pre-specified biological interaction terms"
    )
    parser.add_argument("--bmca_train", default="data/adni_bmca_features_train.csv")
    parser.add_argument("--bmca_test",  default="data/adni_bmca_features_test.csv")
    parser.add_argument("--mrf_train",  default="data/adni_mrf_features_train.csv")
    parser.add_argument("--mrf_test",   default="data/adni_mrf_features_test.csv")
    parser.add_argument("--output_dir", default="results")
    parser.add_argument("--plots_dir",  default="plots")
    parser.add_argument("--n_iter",  type=int, default=50)
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
