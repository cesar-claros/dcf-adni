"""
SHAP and Residual Analysis
===========================

Three complementary analyses to understand where the BMCA model makes errors
and whether MRF features explain those errors.

A. Residual–MRF correlation (primary training OOF)
   ---------------------------------------------------
   For each primary training subject, compute the BMCA out-of-fold residual:
       residual_i = y_true_i − bmca_oof_score_i
   Positive residual: BMCA underestimated a transition subject (missed).
   Negative residual: BMCA overestimated a stable subject (false alarm).
   Each MRF feature is correlated with this residual via Spearman ρ. A large
   positive ρ means the feature tends to be high when BMCA underestimates
   (missed transitions have elevated values); a negative ρ means elevated
   values accompany overestimates (false alarms). These are the features most
   likely to add corrective value.

   Caveat: subjects within a matched pair share age/sex/APOE, so the 286
   primary training rows are not fully independent. Treat p-values as
   approximate guides, not formal tests. The Bonferroni threshold for 31
   MRF features is p < 0.0016.

B. SHAP feature attribution (BMCA, primary test set)
   ----------------------------------------------------
   Computes SHAP values for the BMCA CatBoost model on the 72 primary test
   subjects (36 pairs, evaluation_eligible == 1) using TreeExplainer.
   Produces a beeswarm summary plot showing the direction and magnitude of
   each feature's contribution across subjects.

C. Error subgroup MRF profiles (primary training OOF)
   -----------------------------------------------------
   Classifies each primary training subject as correctly predicted or
   misclassified by BMCA (OOF score threshold = 0.5). Compares MRF feature
   distributions between the two groups using Mann-Whitney U. Features that
   differ between groups are structurally missing from BMCA's representation
   and represent the candidates most worth adding.

   Subjects near the threshold are uncertain, not necessarily wrong; the
   binary split is a diagnostic heuristic, not a definitive classification.

Loads
-----
results/stacking_model.joblib   — bmca_model, bmca_oof, mrf_oof, result
data/adni_bmca_features_train.csv
data/adni_mrf_features_train.csv
data/adni_bmca_features_test.csv

Outputs
-------
results/shap_residual_mrf_correlations.csv — Spearman ρ and p-value per MRF feature
results/shap_error_subgroup.csv            — Mann-Whitney U results per MRF feature
plots/shap_bmca_summary.pdf                — SHAP beeswarm for BMCA on test set
plots/shap_residual_mrf_correlations.pdf   — horizontal bar chart of ρ values
plots/shap_error_subgroup.pdf              — violin plot of top MRF features by subgroup

Usage::

    python analysis_shap_residuals.py
    python analysis_shap_residuals.py --top_k 10
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from scipy.stats import mannwhitneyu, spearmanr
from sklearn.metrics import roc_auc_score

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
# Analysis A — Residual–MRF correlation
# =============================================================================


def residual_mrf_correlation(
    bmca_oof: np.ndarray,
    y_train: np.ndarray,
    primary_mask: np.ndarray,
    mrf_train_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Spearman correlation between BMCA OOF residuals and each MRF feature.

    Restricted to primary training pairs (analysis_set == 'primary').
    """
    mrf_feat_cols = _feature_cols(mrf_train_df)
    residuals = y_train[primary_mask] - bmca_oof[primary_mask]
    mrf_primary = mrf_train_df[primary_mask]

    logger.info(
        f"Residual–MRF correlation: {primary_mask.sum()} primary training rows, "
        f"{len(mrf_feat_cols)} MRF features."
    )
    logger.info(
        f"Residual stats — mean: {residuals.mean():.3f}, "
        f"std: {residuals.std():.3f}, "
        f"range: [{residuals.min():.3f}, {residuals.max():.3f}]"
    )

    rows = []
    for feat in mrf_feat_cols:
        values = mrf_primary[feat].values.astype(float)
        mask = ~np.isnan(values)
        if mask.sum() < 10:
            rows.append({"feature": feat, "spearman_rho": np.nan, "p_value": np.nan, "n": int(mask.sum())})
            continue
        rho, p = spearmanr(residuals[mask], values[mask])
        rows.append({"feature": feat, "spearman_rho": rho, "p_value": p, "n": int(mask.sum())})

    corr_df = (
        pd.DataFrame(rows)
        .assign(abs_rho=lambda d: d["spearman_rho"].abs())
        .sort_values("abs_rho", ascending=False)
        .reset_index(drop=True)
    )
    return corr_df


def plot_mrf_correlations(corr_df: pd.DataFrame, output_path: str, top_k: int = 15) -> None:
    df = corr_df.dropna(subset=["spearman_rho"]).head(top_k).copy()
    df = df.sort_values("spearman_rho")

    bonferroni_threshold = 0.05 / len(corr_df)
    colors = [
        "firebrick" if r > 0 else "steelblue"
        for r in df["spearman_rho"]
    ]
    edge_colors = [
        "black" if p < bonferroni_threshold else "none"
        for p in df["p_value"]
    ]

    fig, ax = plt.subplots(figsize=(8, max(4, len(df) * 0.4)))
    bars = ax.barh(df["feature"], df["spearman_rho"], color=colors, edgecolor=edge_colors,
                   linewidth=1.2, height=0.7)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("Spearman ρ  (residual vs MRF feature)")
    ax.set_title(
        f"Residual–MRF Correlation (top {top_k} by |ρ|)\n"
        f"Positive ρ = feature elevated when BMCA misses a transition  |  "
        f"Bold border = p < {bonferroni_threshold:.4f} (Bonferroni)"
    )
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Correlation plot saved to {output_path}")


# =============================================================================
# Analysis B — SHAP feature attribution
# =============================================================================


def compute_shap_bmca(
    bmca_model: object,
    bmca_test_df: pd.DataFrame,
) -> tuple[object, pd.DataFrame]:
    """
    Compute SHAP values for the BMCA model on the primary test set.

    Accesses the inner CatBoostClassifier via bmca_model.model_ (the
    _CatBoostWrapper stores the fitted CatBoost instance as model_).
    """
    feat_cols = _feature_cols(bmca_test_df)
    eligible = bmca_test_df[bmca_test_df["evaluation_eligible"] == 1].copy()
    X_test = eligible[feat_cols]

    logger.info(
        f"Computing SHAP values for BMCA on {len(X_test)} test subjects "
        f"({len(feat_cols)} features)..."
    )

    explainer = shap.TreeExplainer(bmca_model.model_)
    shap_exp = explainer(X_test)

    return shap_exp, X_test


def plot_shap_summary(shap_exp: object, X_test: pd.DataFrame, output_path: str) -> None:
    shap_vals = shap_exp.values
    # TreeExplainer on a binary CatBoost model may return shape (n, features, 2).
    # Take the positive-class slice in that case.
    if shap_vals.ndim == 3:
        shap_vals = shap_vals[:, :, 1]

    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        shap_vals,
        X_test,
        plot_type="dot",
        show=False,
        max_display=20,
    )
    plt.title("BMCA — SHAP Feature Attribution (primary test set, n=36 pairs)", pad=12)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close("all")
    logger.info(f"SHAP summary plot saved to {output_path}")


# =============================================================================
# Analysis C — Error subgroup MRF profiles
# =============================================================================


def error_subgroup_analysis(
    bmca_oof: np.ndarray,
    y_train: np.ndarray,
    primary_mask: np.ndarray,
    mrf_train_df: pd.DataFrame,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """
    Compare MRF feature distributions between BMCA-correct and BMCA-error subjects.

    A subject is 'error' if the OOF binary prediction (at threshold 0.5) disagrees
    with the true label. Mann-Whitney U tests for each MRF feature.
    """
    mrf_feat_cols = _feature_cols(mrf_train_df)
    y_prim = y_train[primary_mask]
    oof_prim = bmca_oof[primary_mask]
    mrf_prim = mrf_train_df[primary_mask].reset_index(drop=True)

    predicted = (oof_prim >= threshold).astype(int)
    error_mask = predicted != y_prim.astype(int)
    n_error = error_mask.sum()
    n_correct = (~error_mask).sum()

    logger.info(
        f"Error subgroup: {n_error} misclassified, {n_correct} correct "
        f"(OOF threshold = {threshold})."
    )

    rows = []
    for feat in mrf_feat_cols:
        vals = mrf_prim[feat].values.astype(float)
        a = vals[~error_mask & ~np.isnan(vals)]
        b = vals[error_mask & ~np.isnan(vals)]
        if len(a) < 3 or len(b) < 3:
            rows.append({
                "feature": feat, "median_correct": np.nan, "median_error": np.nan,
                "mw_statistic": np.nan, "p_value": np.nan,
                "n_correct": len(a), "n_error": len(b),
            })
            continue
        stat, p = mannwhitneyu(a, b, alternative="two-sided")
        rows.append({
            "feature": feat,
            "median_correct": float(np.nanmedian(a)),
            "median_error": float(np.nanmedian(b)),
            "mw_statistic": stat,
            "p_value": p,
            "n_correct": len(a),
            "n_error": len(b),
        })

    result_df = (
        pd.DataFrame(rows)
        .sort_values("p_value")
        .reset_index(drop=True)
    )
    return result_df, error_mask, mrf_prim


def plot_error_subgroups(
    subgroup_df: pd.DataFrame,
    error_mask: np.ndarray,
    mrf_prim: pd.DataFrame,
    output_path: str,
    top_k: int = 8,
) -> None:
    top_features = subgroup_df.dropna(subset=["p_value"]).head(top_k)["feature"].tolist()
    if not top_features:
        logger.warning("No features to plot in error subgroup analysis.")
        return

    ncols = min(4, len(top_features))
    nrows = int(np.ceil(len(top_features) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    axes = np.array(axes).flatten() if len(top_features) > 1 else [axes]

    for ax, feat in zip(axes, top_features):
        vals = mrf_prim[feat].values.astype(float)
        correct_vals = vals[~error_mask & ~np.isnan(vals)]
        error_vals = vals[error_mask & ~np.isnan(vals)]
        row = subgroup_df[subgroup_df["feature"] == feat].iloc[0]

        ax.violinplot(
            [correct_vals, error_vals],
            positions=[0, 1],
            showmedians=True,
            widths=0.6,
        )
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Correct", "Error"], fontsize=9)
        ax.set_title(
            f"{feat}\np = {row['p_value']:.3f}",
            fontsize=9,
        )
        ax.grid(True, axis="y", alpha=0.3)

    for ax in axes[len(top_features):]:
        ax.set_visible(False)

    fig.suptitle(
        f"MRF Feature Distributions: BMCA-correct vs BMCA-error\n"
        f"(primary training OOF, top {top_k} features by Mann-Whitney p-value)",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Subgroup plot saved to {output_path}")


# =============================================================================
# Entry point
# =============================================================================


def run(
    artifact_path: str = "results/stacking_model.joblib",
    bmca_train_path: str = "data/adni_bmca_features_train.csv",
    mrf_train_path: str = "data/adni_mrf_features_train.csv",
    bmca_test_path: str = "data/adni_bmca_features_test.csv",
    output_dir: str = "results",
    plots_dir: str = "plots",
    top_k: int = 15,
) -> dict:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    Path(plots_dir).mkdir(parents=True, exist_ok=True)

    # ----- Load artifacts -----
    logger.info(f"Loading stacking artifacts from {artifact_path}...")
    artifacts = joblib.load(artifact_path)
    bmca_model = artifacts["bmca_model"]
    bmca_oof = artifacts["bmca_oof"]

    logger.info("Loading feature splits...")
    bmca_train_df = pd.read_csv(bmca_train_path)
    mrf_train_df = pd.read_csv(mrf_train_path)
    bmca_test_df = pd.read_csv(bmca_test_path)

    if not (bmca_train_df["subject_id"].values == mrf_train_df["subject_id"].values).all():
        raise ValueError(
            "BMCA and MRF training DataFrames have different subjects or row ordering."
        )

    y_train = bmca_train_df[LABEL_COL].astype(float).values
    primary_mask = (bmca_train_df["analysis_set"] == "primary").values

    oof_auc_primary = roc_auc_score(y_train[primary_mask], bmca_oof[primary_mask])
    logger.info(f"BMCA OOF AUC on primary training pairs: {oof_auc_primary:.3f}")

    # ── Analysis A: Residual–MRF correlation ─────────────────────────────────
    logger.info("\n=== Analysis A: Residual–MRF correlation ===")
    corr_df = residual_mrf_correlation(bmca_oof, y_train, primary_mask, mrf_train_df)
    corr_df.to_csv(f"{output_dir}/shap_residual_mrf_correlations.csv", index=False)
    logger.info(f"\nTop 10 MRF features by |ρ| with BMCA residual:\n"
                f"{corr_df.head(10)[['feature', 'spearman_rho', 'p_value']].to_string(index=False)}")
    plot_mrf_correlations(corr_df, f"{plots_dir}/shap_residual_mrf_correlations.pdf", top_k=top_k)

    # ── Analysis B: SHAP feature attribution ─────────────────────────────────
    logger.info("\n=== Analysis B: SHAP feature attribution (test set) ===")
    shap_exp, X_test_bmca = compute_shap_bmca(bmca_model, bmca_test_df)
    plot_shap_summary(shap_exp, X_test_bmca, f"{plots_dir}/shap_bmca_summary.pdf")

    shap_vals = shap_exp.values
    if shap_vals.ndim == 3:
        shap_vals = shap_vals[:, :, 1]
    mean_abs_shap = pd.DataFrame({
        "feature": X_test_bmca.columns,
        "mean_abs_shap": np.abs(shap_vals).mean(axis=0),
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    logger.info(f"\nTop 10 BMCA features by mean |SHAP| on test set:\n"
                f"{mean_abs_shap.head(10).to_string(index=False)}")

    # ── Analysis C: Error subgroup MRF profiles ───────────────────────────────
    logger.info("\n=== Analysis C: Error subgroup MRF profiles ===")
    subgroup_df, error_mask, mrf_prim = error_subgroup_analysis(
        bmca_oof, y_train, primary_mask, mrf_train_df
    )
    subgroup_df.to_csv(f"{output_dir}/shap_error_subgroup.csv", index=False)
    logger.info(
        f"\nTop 10 MRF features by Mann-Whitney p-value (correct vs error groups):\n"
        f"{subgroup_df.head(10)[['feature', 'median_correct', 'median_error', 'p_value']].to_string(index=False)}"
    )
    plot_error_subgroups(
        subgroup_df, error_mask, mrf_prim,
        f"{plots_dir}/shap_error_subgroup.pdf",
        top_k=min(top_k, 8),
    )

    logger.info("\nAll outputs saved.")
    return {
        "correlations": corr_df,
        "shap_exp": shap_exp,
        "mean_abs_shap": mean_abs_shap,
        "subgroup": subgroup_df,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SHAP and residual analysis for BMCA errors vs MRF features"
    )
    parser.add_argument("--artifact", default="results/stacking_model.joblib")
    parser.add_argument("--bmca_train", default="data/adni_bmca_features_train.csv")
    parser.add_argument("--mrf_train", default="data/adni_mrf_features_train.csv")
    parser.add_argument("--bmca_test", default="data/adni_bmca_features_test.csv")
    parser.add_argument("--output_dir", default="results")
    parser.add_argument("--plots_dir", default="plots")
    parser.add_argument(
        "--top_k", type=int, default=15,
        help="Number of top features to show in plots (default: 15)",
    )
    args = parser.parse_args()

    run(
        artifact_path=args.artifact,
        bmca_train_path=args.bmca_train,
        mrf_train_path=args.mrf_train,
        bmca_test_path=args.bmca_test,
        output_dir=args.output_dir,
        plots_dir=args.plots_dir,
        top_k=args.top_k,
    )
