"""
Strategy 2: Education-Adjusted Cognitive Features
==================================================

Tests whether adjusting cognitive assessment features for cognitive reserve
(proxied by education years) improves BMCA discrimination.

Motivation
----------
Experiment 8b revealed that ``cdrsb x education`` was the #1 feature in the
BMCA+interactions model (17.5% importance), exceeding even ``cdrsb`` itself.
The cognitive reserve hypothesis is well-established: higher education provides
compensatory mechanisms that mask early cognitive decline on clinical
assessments. A highly educated subject with CDR-SB = 0.5 may be further along
the disease trajectory than a less educated subject with the same score.

Rather than adding a raw interaction product (which displaces ``cdrsb``'s
importance without improving AUC), this strategy *adjusts* cognitive features
for education, creating more accurate measures of underlying disease state.

Design
------
Three head-to-head comparisons (same seed, same Optuna budget):

1. **BMCA baseline** — standard BMCA features, no adjustment.
2. **BMCA + CDR-SB adjusted** — replace ``cdrsb`` with
   ``cdrsb - f(education_years)`` (linear residual).
3. **BMCA + multi-adjusted** — additionally adjust ``logical_memory_delayed``
   and ``mmse_total`` for education (these have the strongest education
   correlations: rho = 0.278 and 0.239 respectively).

The adjustment function ``f`` is a linear regression fit on training data only.
Education is sourced from the MRF feature table and used solely for the
adjustment — it is NOT included as a separate feature.

Usage::

    python model_bmca_education_adjusted_evaluation.py
    python model_bmca_education_adjusted_evaluation.py --n_iter 100 --seed 42
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
from sklearn.linear_model import LinearRegression
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
# Education adjustment
# =============================================================================


def _fit_education_adjustment(
    train_bmca: pd.DataFrame,
    train_mrf: pd.DataFrame,
    features_to_adjust: list[str],
) -> dict[str, LinearRegression]:
    """
    Fit linear regressions of each cognitive feature on education_years
    using training data only.

    Returns a dict mapping feature_name -> fitted LinearRegression model.
    """
    edu = train_mrf.set_index(SUBJECT_ID_COL)["education_years"]
    merged = train_bmca.set_index(SUBJECT_ID_COL).join(edu)

    adjustments = {}
    for feat in features_to_adjust:
        valid = merged[[feat, "education_years"]].dropna()
        if len(valid) < 10:
            logger.warning(f"Too few valid rows for {feat} adjustment, skipping.")
            continue

        lr = LinearRegression()
        lr.fit(valid[["education_years"]].values, valid[feat].values)
        adjustments[feat] = lr

        from scipy.stats import spearmanr
        rho, p = spearmanr(valid["education_years"], valid[feat])
        logger.info(
            f"  {feat} ~ education_years: "
            f"coef = {lr.coef_[0]:.4f}, intercept = {lr.intercept_:.4f}, "
            f"R² = {lr.score(valid[['education_years']].values, valid[feat].values):.4f}, "
            f"Spearman rho = {rho:.4f}, p = {p:.4f}"
        )

    return adjustments


def _apply_education_adjustment(
    df: pd.DataFrame,
    mrf_df: pd.DataFrame,
    adjustments: dict[str, LinearRegression],
    suffix: str = "_edu_adjusted",
) -> pd.DataFrame:
    """
    Replace each adjusted feature with its education-residualised version.

    adjusted_feature = original - f(education_years)

    Education_years is NOT added as a separate feature.
    """
    out = df.copy()
    edu = mrf_df.set_index(SUBJECT_ID_COL)["education_years"]
    out = out.set_index(SUBJECT_ID_COL)
    out["education_years"] = edu
    out = out.reset_index()

    for feat, lr in adjustments.items():
        valid_mask = out[feat].notna() & out["education_years"].notna()
        predicted = lr.predict(out.loc[valid_mask, ["education_years"]].values)
        new_col = f"{feat}{suffix}"
        out[new_col] = np.nan
        out.loc[valid_mask, new_col] = out.loc[valid_mask, feat].values - predicted

        # Replace original with adjusted
        out = out.drop(columns=[feat]).rename(columns={new_col: feat})

    # Remove the temporary education column — it should not be a feature
    out = out.drop(columns=["education_years"])

    return out


# =============================================================================
# Training wrapper (uses existing train_model from utils)
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
    """Train CatBoost with Optuna and evaluate on primary test set."""
    feature_cols = _feature_cols(train_df)

    X_train = train_df[feature_cols]
    y_train = train_df[LABEL_COL].astype(float)
    groups_train = train_df[GROUP_COL]

    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    val_mask = (train_df["analysis_set"] == "primary").values

    logger.info(
        f"Training {model_name}: "
        f"{len(feature_cols)} features, {len(X_train)} train rows."
    )

    eligible = _evaluation_eligible(test_df)
    X_test = eligible[feature_cols]
    y_test = eligible[LABEL_COL].astype(float)

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

    groups_test = eligible[GROUP_COL].values
    y_test_arr = y_test.values
    y_score = best_model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test_arr, y_score)
    ci_low, ci_high = _bootstrap_auc(y_test_arr, y_score, groups_test, n_boot=n_boot, seed=seed)

    importances = best_model.get_feature_importance()
    imp_df = (
        pd.DataFrame({"feature": feature_cols, "importance": importances})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

    logger.info(f"{model_name}  AUC = {auc:.3f}  95% CI [{ci_low:.3f}, {ci_high:.3f}]")
    logger.info(f"{model_name} best inner CV AUC: {study.best_value:.4f}")

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


def _plot_roc_comparison(results: list[dict], output_path: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    colors = ["steelblue", "darkorange", "forestgreen", "firebrick"]
    n_pairs = int(results[0]["y_true"].sum())

    for r, color in zip(results, colors):
        RocCurveDisplay.from_predictions(
            r["y_true"], r["y_score"], ax=ax,
            name=f"{r['model_name']} (AUC = {r['auc']:.3f})",
            color=color,
        )

    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="Chance")
    ax.set_title(
        f"Education-Adjusted BMCA — ROC Comparison\n"
        f"(primary test set, n = {n_pairs} pairs)"
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

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    bmca_train = pd.read_csv(bmca_train_path)
    bmca_test = pd.read_csv(bmca_test_path)
    mrf_train = pd.read_csv(mrf_train_path)
    mrf_test = pd.read_csv(mrf_test_path)

    logger.info(
        f"\n{'='*60}\n"
        f"Strategy 2: Education-Adjusted Cognitive Features\n"
        f"{'='*60}"
    )

    # ------------------------------------------------------------------
    # 2. Fit education adjustments on training data
    # ------------------------------------------------------------------

    # Variant A: CDR-SB only
    logger.info("\n--- Fitting CDR-SB adjustment ---")
    adj_cdrsb = _fit_education_adjustment(bmca_train, mrf_train, ["cdrsb"])

    # Variant B: CDR-SB + logical_memory_delayed + mmse_total
    logger.info("\n--- Fitting multi-feature adjustment ---")
    adj_multi = _fit_education_adjustment(
        bmca_train, mrf_train,
        ["cdrsb", "logical_memory_delayed", "mmse_total"],
    )

    # ------------------------------------------------------------------
    # 3. Apply adjustments to train and test sets
    # ------------------------------------------------------------------
    logger.info("\n--- Applying adjustments ---")

    # Variant A: CDR-SB adjusted
    bmca_train_cdrsb = _apply_education_adjustment(bmca_train, mrf_train, adj_cdrsb)
    bmca_test_cdrsb = _apply_education_adjustment(bmca_test, mrf_test, adj_cdrsb)

    # Variant B: multi-adjusted
    bmca_train_multi = _apply_education_adjustment(bmca_train, mrf_train, adj_multi)
    bmca_test_multi = _apply_education_adjustment(bmca_test, mrf_test, adj_multi)

    # ------------------------------------------------------------------
    # 4. Train all three models (same seed)
    # ------------------------------------------------------------------
    logger.info("\n--- Training BMCA baseline ---")
    r_baseline = _train_and_evaluate(
        bmca_train, bmca_test, "BMCA baseline",
        n_iter=n_iter, n_splits=n_splits, seed=seed, n_jobs=n_jobs, gpu=gpu, n_boot=n_boot,
    )

    logger.info("\n--- Training BMCA + CDR-SB adjusted ---")
    r_cdrsb = _train_and_evaluate(
        bmca_train_cdrsb, bmca_test_cdrsb, "BMCA cdrsb-adjusted",
        n_iter=n_iter, n_splits=n_splits, seed=seed, n_jobs=n_jobs, gpu=gpu, n_boot=n_boot,
    )

    logger.info("\n--- Training BMCA + multi-adjusted ---")
    r_multi = _train_and_evaluate(
        bmca_train_multi, bmca_test_multi, "BMCA multi-adjusted",
        n_iter=n_iter, n_splits=n_splits, seed=seed, n_jobs=n_jobs, gpu=gpu, n_boot=n_boot,
    )

    # ------------------------------------------------------------------
    # 5. Paired bootstrap comparisons
    # ------------------------------------------------------------------
    logger.info("\n--- Paired bootstrap AUC differences ---")
    all_results = [r_baseline, r_cdrsb, r_multi]

    diff_cdrsb = _bootstrap_auc_diff(
        r_baseline["y_true"], r_cdrsb["y_score"], r_baseline["y_score"],
        r_baseline["groups"], n_boot=10_000, seed=seed,
    )
    diff_multi = _bootstrap_auc_diff(
        r_baseline["y_true"], r_multi["y_score"], r_baseline["y_score"],
        r_baseline["groups"], n_boot=10_000, seed=seed,
    )

    for name, diff in [("cdrsb-adjusted vs baseline", diff_cdrsb),
                       ("multi-adjusted vs baseline", diff_multi)]:
        logger.info(
            f"  {name}: Δ = {diff['observed_diff']:+.4f}  "
            f"95% CI [{diff['ci_low']:+.4f}, {diff['ci_high']:+.4f}]  "
            f"p(Δ ≤ 0) = {diff['p_value']:.3f}"
        )

    # ------------------------------------------------------------------
    # 6. Save outputs
    # ------------------------------------------------------------------
    _plot_roc_comparison(all_results, f"{plots_dir}/education_adjusted_roc.pdf")

    metrics_rows = []
    for r in all_results:
        metrics_rows.append({
            "model": r["model_name"],
            "auc": round(r["auc"], 4),
            "auc_ci_low_95": round(r["ci_low"], 4),
            "auc_ci_high_95": round(r["ci_high"], 4),
            "best_inner_cv_auc": round(r["inner_cv_auc"], 4),
            "n_features": len(r["feature_cols"]),
            **{f"param_{k}": v for k, v in r["best_params"].items()},
        })
    pd.DataFrame(metrics_rows).to_csv(
        f"{output_dir}/education_adjusted_evaluation.csv", index=False
    )

    bootstrap_rows = [
        {
            "comparison": "cdrsb-adjusted vs baseline",
            **{k: round(v, 4) if isinstance(v, float) else v
               for k, v in diff_cdrsb.items()},
        },
        {
            "comparison": "multi-adjusted vs baseline",
            **{k: round(v, 4) if isinstance(v, float) else v
               for k, v in diff_multi.items()},
        },
    ]
    pd.DataFrame(bootstrap_rows).to_csv(
        f"{output_dir}/education_adjusted_bootstrap_diff.csv", index=False
    )

    # Adjustment coefficients
    adj_rows = []
    for feat, lr in adj_multi.items():
        adj_rows.append({
            "feature": feat,
            "education_coef": round(lr.coef_[0], 6),
            "intercept": round(lr.intercept_, 6),
        })
    pd.DataFrame(adj_rows).to_csv(
        f"{output_dir}/education_adjustment_coefficients.csv", index=False
    )

    for r in all_results:
        safe_name = r["model_name"].replace(" ", "_").replace("+", "_")
        r["importance"].to_csv(
            f"{output_dir}/education_adjusted_{safe_name}_importance.csv", index=False
        )

    joblib.dump(
        {r["model_name"]: {
            "model": r["model"], "study": r["study"],
            "feature_cols": r["feature_cols"], "result": {
                "auc": r["auc"], "ci_low": r["ci_low"], "ci_high": r["ci_high"],
                "y_true": r["y_true"], "y_score": r["y_score"], "groups": r["groups"],
            },
        } for r in all_results},
        f"{output_dir}/education_adjusted_models.joblib",
    )

    # ------------------------------------------------------------------
    # 7. Summary
    # ------------------------------------------------------------------
    logger.info(
        f"\n{'='*60}\n"
        f"SUMMARY: Strategy 2 — Education-Adjusted Cognitive Features\n"
        f"{'='*60}\n"
        f"{'Model':<28} {'Test AUC':>10} {'95% CI':>20} {'CV AUC':>8}\n"
        f"{'-'*68}"
    )
    for r in all_results:
        logger.info(
            f"{r['model_name']:<28} {r['auc']:>10.4f} "
            f"[{r['ci_low']:.3f}, {r['ci_high']:.3f}] {r['inner_cv_auc']:>8.4f}"
        )

    logger.info(
        f"\ncdrsb-adjusted vs baseline:  Δ = {diff_cdrsb['observed_diff']:+.4f}  "
        f"CI [{diff_cdrsb['ci_low']:+.4f}, {diff_cdrsb['ci_high']:+.4f}]  "
        f"p = {diff_cdrsb['p_value']:.3f}"
    )
    logger.info(
        f"multi-adjusted vs baseline:  Δ = {diff_multi['observed_diff']:+.4f}  "
        f"CI [{diff_multi['ci_low']:+.4f}, {diff_multi['ci_high']:+.4f}]  "
        f"p = {diff_multi['p_value']:.3f}"
    )
    logger.info(f"{'='*60}")

    return {
        "results": {r["model_name"]: r for r in all_results},
        "diffs": {"cdrsb_adjusted": diff_cdrsb, "multi_adjusted": diff_multi},
        "adjustments": {"cdrsb": adj_cdrsb, "multi": adj_multi},
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Strategy 2: Education-adjusted cognitive features for BMCA"
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
