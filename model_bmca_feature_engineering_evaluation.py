"""
Experiment 11: MRF-Informed Feature Engineering
================================================

Tests whether MRF information can improve BMCA performance when used to
*transform* existing BMCA features rather than adding new ones.

Three biologically motivated transformations are tested individually and
in combination:

A. **Education-adjusted CDR-SB** (validated in Exp. 10):
   Regress CDR-SB on education_years; replace with residual.
   Mechanism: cognitive reserve masks early decline on clinical assessments.

B. **BMI-normalized plasma biomarkers**:
   Divide plasma_ptau217, plasma_ptau217_abeta42_ratio, and
   plasma_abeta42_abeta40_ratio by BMI.
   Mechanism: BMI scales with blood volume, which dilutes plasma biomarker
   concentrations multiplicatively. Division corrects the dilution effect,
   yielding a more accurate estimate of brain-derived biomarker burden.
   Empirical support: BMI normalization improves ptau217-transition
   correlation from rho=0.235 to 0.308 (+31%) in primary pairs.

C. **Vascular-adjusted ADAS-13**:
   Regress ADAS-13 on vascular_risk_count; replace with residual.
   Mechanism: vascular cognitive impairment inflates ADAS-13 through non-AD
   mechanisms (attention, praxis items). Removing vascular variance isolates
   the AD-specific signal.

Models tested (all head-to-head, same seed):
1. Baseline BMCA
2. CDR-SB education-adjusted (A only)
3. BMI-normalized plasma biomarkers (B only)
4. Combined A + B
5. Full A + B + C

Usage::

    python model_bmca_feature_engineering_evaluation.py
    python model_bmca_feature_engineering_evaluation.py --n_iter 100 --seed 42
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
from sklearn.linear_model import LinearRegression
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
# Feature transformations
# =============================================================================


class FeatureTransformSet:
    """
    Encapsulates a set of feature transformations that are fit on training data
    and applied to both train and test.
    """

    def __init__(self, name: str, transforms: list[dict]):
        """
        Parameters
        ----------
        name : str
            Human-readable name for this transform set.
        transforms : list of dict
            Each dict describes one transformation:
            - type: "linear_adjust" or "bmi_normalize"
            - target: BMCA feature column to transform
            - predictor: MRF feature column(s) used for adjustment
        """
        self.name = name
        self.transforms = transforms
        self._fitted_models: dict[str, LinearRegression] = {}
        self._fitted_bmi_medians: dict[str, float] = {}

    def fit(self, bmca_train: pd.DataFrame, mrf_train: pd.DataFrame) -> None:
        """Fit adjustment models on training data."""
        mrf_indexed = mrf_train.set_index(SUBJECT_ID_COL)

        for t in self.transforms:
            if t["type"] == "linear_adjust":
                target = t["target"]
                predictor = t["predictor"]
                merged = bmca_train.set_index(SUBJECT_ID_COL).join(
                    mrf_indexed[[predictor]]
                )
                valid = merged[[target, predictor]].dropna()
                lr = LinearRegression()
                lr.fit(valid[[predictor]].values, valid[target].values)
                self._fitted_models[target] = lr
                logger.info(
                    f"  [{self.name}] {target} ~ {predictor}: "
                    f"coef={lr.coef_[0]:.6f}, R²={lr.score(valid[[predictor]].values, valid[target].values):.4f}"
                )

            elif t["type"] == "bmi_normalize":
                # Store training BMI median for fallback on missing values
                bmi = mrf_indexed["BMI"]
                self._fitted_bmi_medians[t["target"]] = float(bmi.median())

    def transform(self, bmca_df: pd.DataFrame, mrf_df: pd.DataFrame) -> pd.DataFrame:
        """Apply fitted transformations. Returns a copy with modified features."""
        out = bmca_df.copy()
        mrf_indexed = mrf_df.set_index(SUBJECT_ID_COL)

        for t in self.transforms:
            if t["type"] == "linear_adjust":
                target = t["target"]
                predictor = t["predictor"]
                lr = self._fitted_models[target]

                pred_vals = mrf_indexed.reindex(out[SUBJECT_ID_COL])[predictor].values
                valid_mask = ~np.isnan(pred_vals) & ~np.isnan(out[target].values)

                adjusted = out[target].values.copy()
                adjusted[valid_mask] = (
                    out[target].values[valid_mask]
                    - lr.predict(pred_vals[valid_mask].reshape(-1, 1))
                )
                out[target] = adjusted

            elif t["type"] == "bmi_normalize":
                target = t["target"]
                bmi = mrf_indexed.reindex(out[SUBJECT_ID_COL])["BMI"].values

                # Use training median for missing BMI (neutral normalization)
                bmi_filled = np.where(np.isnan(bmi), self._fitted_bmi_medians[target], bmi)
                # Avoid division by zero
                bmi_filled = np.where(bmi_filled == 0, 1.0, bmi_filled)

                original = out[target].values.copy()
                out[target] = np.where(
                    np.isnan(original), np.nan, original / bmi_filled
                )

        return out


def _build_transform_sets() -> dict[str, FeatureTransformSet]:
    """Define the five experimental variants."""

    # A: Education-adjusted CDR-SB
    ts_a = FeatureTransformSet("A: CDR-SB edu-adjusted", [
        {"type": "linear_adjust", "target": "cdrsb", "predictor": "education_years"},
    ])

    # B: BMI-normalized plasma biomarkers
    ts_b = FeatureTransformSet("B: BMI-normalized biomarkers", [
        {"type": "bmi_normalize", "target": "plasma_ptau217"},
        {"type": "bmi_normalize", "target": "plasma_ptau217_abeta42_ratio"},
        {"type": "bmi_normalize", "target": "plasma_abeta42_abeta40_ratio"},
    ])

    # A+B: Combined
    ts_ab = FeatureTransformSet("A+B: Combined", [
        {"type": "linear_adjust", "target": "cdrsb", "predictor": "education_years"},
        {"type": "bmi_normalize", "target": "plasma_ptau217"},
        {"type": "bmi_normalize", "target": "plasma_ptau217_abeta42_ratio"},
        {"type": "bmi_normalize", "target": "plasma_abeta42_abeta40_ratio"},
    ])

    # C: Vascular-adjusted ADAS-13
    ts_c = FeatureTransformSet("C: ADAS-13 vascular-adjusted", [
        {"type": "linear_adjust", "target": "adas13_total",
         "predictor": "vascular_risk_count"},
    ])

    # A+B+C: Full
    ts_full = FeatureTransformSet("A+B+C: Full", [
        {"type": "linear_adjust", "target": "cdrsb", "predictor": "education_years"},
        {"type": "bmi_normalize", "target": "plasma_ptau217"},
        {"type": "bmi_normalize", "target": "plasma_ptau217_abeta42_ratio"},
        {"type": "bmi_normalize", "target": "plasma_abeta42_abeta40_ratio"},
        {"type": "linear_adjust", "target": "adas13_total",
         "predictor": "vascular_risk_count"},
    ])

    return {
        "A_cdrsb_edu": ts_a,
        "B_bmi_biomarkers": ts_b,
        "AB_combined": ts_ab,
        "C_adas_vascular": ts_c,
        "ABC_full": ts_full,
    }


# =============================================================================
# Training wrapper
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

    eligible = _evaluation_eligible(test_df)
    X_test = eligible[feature_cols]
    y_test = eligible[LABEL_COL].astype(float)

    logger.info(
        f"Training {model_name}: "
        f"{len(feature_cols)} features, {len(X_train)} train rows."
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

    groups_test = eligible[GROUP_COL].values
    y_test_arr = y_test.values
    y_score = best_model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test_arr, y_score)
    ci_low, ci_high = _bootstrap_auc(
        y_test_arr, y_score, groups_test, n_boot=n_boot, seed=seed
    )

    importances = best_model.get_feature_importance()
    imp_df = (
        pd.DataFrame({"feature": feature_cols, "importance": importances})
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
# Output helpers
# =============================================================================


def _plot_roc_comparison(results: list[dict], output_path: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 7))
    colors = [
        "steelblue", "darkorange", "forestgreen", "firebrick",
        "mediumpurple", "goldenrod",
    ]
    n_pairs = int(results[0]["y_true"].sum())

    for r, color in zip(results, colors):
        RocCurveDisplay.from_predictions(
            r["y_true"], r["y_score"], ax=ax,
            name=f"{r['model_name']} ({r['auc']:.3f})",
            color=color,
        )

    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="Chance")
    ax.set_title(
        f"MRF-Informed Feature Engineering — ROC Comparison\n"
        f"(primary test set, n = {n_pairs} pairs)"
    )
    ax.set_xlabel("1 − Specificity (FPR)")
    ax.set_ylabel("Sensitivity (TPR)")
    ax.grid(True, which="both", alpha=0.3)
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="lower right", fontsize=7)
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
        f"Experiment 11: MRF-Informed Feature Engineering\n"
        f"{'='*60}"
    )

    # ------------------------------------------------------------------
    # 2. Build and fit transform sets
    # ------------------------------------------------------------------
    transform_sets = _build_transform_sets()

    logger.info("\n--- Fitting transformations on training data ---")
    transformed_data = {}
    for key, ts in transform_sets.items():
        logger.info(f"\n{ts.name}:")
        ts.fit(bmca_train, mrf_train)
        train_t = ts.transform(bmca_train, mrf_train)
        test_t = ts.transform(bmca_test, mrf_test)
        transformed_data[key] = (train_t, test_t)

    # ------------------------------------------------------------------
    # 3. Train all models
    # ------------------------------------------------------------------
    all_results = []

    # Baseline
    logger.info("\n--- Training: Baseline BMCA ---")
    r_base = _train_and_evaluate(
        bmca_train, bmca_test, "Baseline",
        n_iter=n_iter, n_splits=n_splits, seed=seed, n_jobs=n_jobs, gpu=gpu,
        n_boot=n_boot,
    )
    all_results.append(r_base)

    # Each variant
    variant_names = {
        "A_cdrsb_edu": "A: CDR-SB edu-adj",
        "B_bmi_biomarkers": "B: BMI-norm biomarkers",
        "AB_combined": "A+B: Combined",
        "C_adas_vascular": "C: ADAS-13 vasc-adj",
        "ABC_full": "A+B+C: Full",
    }
    for key, display_name in variant_names.items():
        train_t, test_t = transformed_data[key]
        logger.info(f"\n--- Training: {display_name} ---")
        r = _train_and_evaluate(
            train_t, test_t, display_name,
            n_iter=n_iter, n_splits=n_splits, seed=seed, n_jobs=n_jobs, gpu=gpu,
            n_boot=n_boot,
        )
        all_results.append(r)

    # ------------------------------------------------------------------
    # 4. Paired bootstrap comparisons vs baseline
    # ------------------------------------------------------------------
    logger.info("\n--- Paired bootstrap AUC differences vs baseline ---")
    diffs = {}
    for r in all_results[1:]:
        d = _bootstrap_auc_diff(
            r_base["y_true"], r["y_score"], r_base["y_score"],
            r_base["groups"], n_boot=10_000, seed=seed,
        )
        diffs[r["model_name"]] = d
        logger.info(
            f"  {r['model_name']:30s}: Δ = {d['observed_diff']:+.4f}  "
            f"CI [{d['ci_low']:+.4f}, {d['ci_high']:+.4f}]  "
            f"p(Δ ≤ 0) = {d['p_value']:.3f}"
        )

    # ------------------------------------------------------------------
    # 5. Save outputs
    # ------------------------------------------------------------------
    _plot_roc_comparison(all_results, f"{plots_dir}/feature_engineering_roc.pdf")

    # Metrics
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
        f"{output_dir}/feature_engineering_evaluation.csv", index=False
    )

    # Bootstrap diffs
    boot_rows = []
    for name, d in diffs.items():
        boot_rows.append({
            "comparison": f"{name} vs Baseline",
            **{k: round(v, 4) if isinstance(v, float) else v for k, v in d.items()},
        })
    pd.DataFrame(boot_rows).to_csv(
        f"{output_dir}/feature_engineering_bootstrap_diff.csv", index=False
    )

    # Feature importances
    for r in all_results:
        safe = r["model_name"].replace(" ", "_").replace(":", "").replace("+", "_")
        r["importance"].to_csv(
            f"{output_dir}/feature_engineering_{safe}_importance.csv", index=False
        )

    # Save all artifacts
    joblib.dump(
        {
            r["model_name"]: {
                "model": r["model"], "study": r["study"],
                "feature_cols": r["feature_cols"],
                "result": {
                    "auc": r["auc"], "ci_low": r["ci_low"], "ci_high": r["ci_high"],
                    "y_true": r["y_true"], "y_score": r["y_score"],
                    "groups": r["groups"],
                },
            }
            for r in all_results
        },
        f"{output_dir}/feature_engineering_models.joblib",
    )

    # ------------------------------------------------------------------
    # 6. Summary
    # ------------------------------------------------------------------
    logger.info(
        f"\n{'='*70}\n"
        f"SUMMARY: MRF-Informed Feature Engineering\n"
        f"{'='*70}\n"
        f"{'Model':<32} {'AUC':>8} {'95% CI':>20} {'CV':>8} {'Δ':>8}\n"
        f"{'-'*78}"
    )
    for r in all_results:
        delta = r["auc"] - r_base["auc"] if r != r_base else 0.0
        logger.info(
            f"{r['model_name']:<32} {r['auc']:>8.4f} "
            f"[{r['ci_low']:.3f}, {r['ci_high']:.3f}] "
            f"{r['inner_cv_auc']:>8.4f} {delta:>+8.4f}"
        )

    logger.info(f"\n{'Bootstrap Δ vs Baseline':^78}")
    logger.info(f"{'-'*78}")
    for name, d in diffs.items():
        logger.info(
            f"  {name:30s}: Δ = {d['observed_diff']:+.4f}  "
            f"[{d['ci_low']:+.4f}, {d['ci_high']:+.4f}]  "
            f"p = {d['p_value']:.3f}"
        )
    logger.info(f"{'='*70}")

    return {
        "results": {r["model_name"]: r for r in all_results},
        "diffs": diffs,
        "transform_sets": transform_sets,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Experiment 11: MRF-informed feature engineering for BMCA"
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
