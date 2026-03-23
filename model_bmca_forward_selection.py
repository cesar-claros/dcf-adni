"""
BMCA + Conditional Forward Selection of Individual MRF Features
================================================================

Tests which specific individual MRF features carry conditional predictive
value given BMCA. Uses BMCA's best hyperparameters (no re-tuning per
candidate) and OOF predictions to rank all MRF features by their
conditional AUC gain.

Algorithm
---------
1. Train BMCA-only model with Optuna → get best_params and inner_splits.
2. For each MRF feature individually:
   a. Create X_candidate = BMCA features + that MRF feature.
   b. Create a CatBoost model with BMCA's best hyperparameters (no re-tuning).
   c. Compute OOF predictions via cross_val_predict.
   d. Compute OOF AUC on primary training subjects only (val_mask).
3. Rank MRF features by conditional AUC gain over BMCA-only OOF AUC.
4. Greedy selection: add features one at a time while gain >= 0.005 AUC.
5. If any features selected: retrain BMCA + selected with fresh Optuna.
6. Evaluate on held-out test set with bootstrap CI.

Usage::

    python model_bmca_forward_selection.py
    python model_bmca_forward_selection.py --n_iter 100 --seed 42
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
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.metrics import RocCurveDisplay
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from src.utils_model import create_model, train_model, _encode_categoricals

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

_AUC_GAIN_THRESHOLD = 0.005


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
# OOF AUC computation (primary-only)
# =============================================================================


def _oof_auc_primary(
    X: pd.DataFrame,
    y: np.ndarray,
    params: dict,
    inner_splits: list,
    val_mask: np.ndarray,
    seed: int,
) -> float:
    """Compute OOF AUC on primary subjects using fixed hyperparameters."""
    X_enc = _encode_categoricals(X, "catboost")
    oof_scores = np.full(len(y), np.nan)

    for train_idx, val_idx in inner_splits:
        model = create_model("catboost", seed=seed)
        model.set_params(**params)
        model.fit(X_enc.iloc[train_idx], y[train_idx])

        primary_in_val = val_mask[val_idx]
        val_idx_primary = val_idx[primary_in_val]
        if len(val_idx_primary) > 0:
            proba = model.predict_proba(X_enc.iloc[val_idx_primary])[:, 1]
            oof_scores[val_idx_primary] = proba

    valid = ~np.isnan(oof_scores) & val_mask
    if valid.sum() < 2 or len(np.unique(y[valid])) < 2:
        return np.nan
    return roc_auc_score(y[valid], oof_scores[valid])


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


def _plot_ranking(ranking_df: pd.DataFrame, output_path: str) -> None:
    """Bar chart of AUC gains for all MRF features."""
    df = ranking_df.sort_values("delta_auc", ascending=True)
    colors = ["seagreen" if d >= _AUC_GAIN_THRESHOLD else "lightcoral" for d in df["delta_auc"]]

    fig, ax = plt.subplots(figsize=(8, max(6, len(df) * 0.25)))
    ax.barh(range(len(df)), df["delta_auc"].values, color=colors)
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df["mrf_feature"].values, fontsize=7)
    ax.axvline(x=_AUC_GAIN_THRESHOLD, color="black", linestyle="--", lw=0.8,
               label=f"Threshold ({_AUC_GAIN_THRESHOLD})")
    ax.axvline(x=0, color="gray", linestyle="-", lw=0.5)
    ax.set_xlabel("Conditional AUC Gain over BMCA-only")
    ax.set_title("Forward Selection: MRF Feature Ranking by Conditional AUC Gain")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Ranking plot saved to {output_path}")


def _plot_roc_comparison(
    bmca_result: dict,
    selected_result: dict | None,
    output_path: str,
) -> None:
    n_pairs = int(bmca_result["y_true"].sum())
    fig, ax = plt.subplots(figsize=(7, 7))

    entries = [(bmca_result, "steelblue", "BMCA")]
    if selected_result is not None:
        entries.append((selected_result, "seagreen", "BMCA + selected MRF"))

    for result, color, label in entries:
        RocCurveDisplay.from_predictions(
            result["y_true"], result["scores"], ax=ax, color=color,
            name=f"{label}  AUC = {result['auc']:.3f}  [{result['ci_low']:.3f}, {result['ci_high']:.3f}]",
        )

    ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="Chance")
    ax.set_title(
        f"BMCA vs BMCA + Forward-Selected MRF — ROC Curves\n"
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
    y_train_arr = y_train.values
    y_test = bmca_test_df[LABEL_COL].astype(float)
    groups_train = bmca_train_df[GROUP_COL]

    bmca_feat_cols = _feature_cols(bmca_train_df)
    mrf_feat_cols = _feature_cols(mrf_train_df)

    X_bmca_train = bmca_train_df[bmca_feat_cols].copy()
    X_bmca_test = bmca_test_df[bmca_feat_cols].copy()

    logger.info(
        f"BMCA features: {len(bmca_feat_cols)}, "
        f"MRF candidate features: {len(mrf_feat_cols)}"
    )

    # ----- Step 1: Train BMCA-only with Optuna -----
    logger.info("\n=== Step 1: Training BMCA-only (baseline) ===")
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    bmca_study, bmca_model, inner_splits = train_model(
        X_train=X_bmca_train,
        y_train=y_train,
        X_test=X_bmca_test,
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

    best_params = bmca_study.best_params
    logger.info(f"BMCA best params: {best_params}")

    # ----- Step 2: Compute BMCA-only OOF AUC (primary) -----
    bmca_oof_auc = _oof_auc_primary(
        X_bmca_train, y_train_arr, best_params, inner_splits, val_mask, seed
    )
    logger.info(f"BMCA-only OOF AUC (primary): {bmca_oof_auc:.4f}")

    # ----- Step 3: Screen each MRF feature -----
    logger.info(f"\n=== Step 2: Screening {len(mrf_feat_cols)} MRF features ===")
    ranking_rows = []

    for mrf_feat in tqdm(mrf_feat_cols, desc="Screening MRF features"):
        X_candidate = X_bmca_train.copy()
        X_candidate[mrf_feat] = mrf_train_df[mrf_feat].values

        candidate_auc = _oof_auc_primary(
            X_candidate, y_train_arr, best_params, inner_splits, val_mask, seed
        )
        delta = candidate_auc - bmca_oof_auc if not np.isnan(candidate_auc) else np.nan
        ranking_rows.append({
            "mrf_feature": mrf_feat,
            "candidate_oof_auc": round(candidate_auc, 4) if not np.isnan(candidate_auc) else np.nan,
            "bmca_oof_auc": round(bmca_oof_auc, 4),
            "delta_auc": round(delta, 4) if not np.isnan(delta) else np.nan,
        })

    ranking_df = pd.DataFrame(ranking_rows).sort_values("delta_auc", ascending=False)
    ranking_df.to_csv(f"{output_dir}/bmca_forward_selection_ranking.csv", index=False)
    logger.info(f"\nRanking saved to {output_dir}/bmca_forward_selection_ranking.csv")
    logger.info(f"\nTop 10 MRF features by conditional AUC gain:\n{ranking_df.head(10).to_string(index=False)}")

    # ----- Step 4: Greedy forward selection -----
    logger.info(f"\n=== Step 3: Greedy forward selection (threshold = {_AUC_GAIN_THRESHOLD}) ===")
    selected_features = []
    current_auc = bmca_oof_auc

    candidates = ranking_df[ranking_df["delta_auc"] >= _AUC_GAIN_THRESHOLD]["mrf_feature"].tolist()

    for mrf_feat in candidates:
        X_test_candidate = X_bmca_train.copy()
        for sf in selected_features:
            X_test_candidate[sf] = mrf_train_df[sf].values
        X_test_candidate[mrf_feat] = mrf_train_df[mrf_feat].values

        candidate_auc = _oof_auc_primary(
            X_test_candidate, y_train_arr, best_params, inner_splits, val_mask, seed
        )
        gain = candidate_auc - current_auc
        if gain >= _AUC_GAIN_THRESHOLD:
            selected_features.append(mrf_feat)
            current_auc = candidate_auc
            logger.info(f"  Selected: {mrf_feat}  (AUC: {current_auc:.4f}, gain: {gain:+.4f})")
        else:
            logger.info(f"  Skipped: {mrf_feat}  (marginal gain: {gain:+.4f} < {_AUC_GAIN_THRESHOLD})")

    logger.info(f"\nSelected {len(selected_features)} features: {selected_features}")

    # ----- Step 5+6: Retrain and evaluate if features selected -----
    bmca_result = _evaluate(bmca_model, bmca_test_df, bmca_feat_cols, "BMCA-only", n_boot, seed)
    selected_result = None
    selected_study = None
    selected_model = None
    selected_imp = None

    if selected_features:
        logger.info(f"\n=== Step 4: Retraining BMCA + {len(selected_features)} selected MRF features ===")

        combined_feat_cols = bmca_feat_cols + selected_features
        X_combined_train = X_bmca_train.copy()
        X_combined_test = X_bmca_test.copy()
        for sf in selected_features:
            X_combined_train[sf] = mrf_train_df[sf].values
            X_combined_test[sf] = mrf_test_df[sf].values

        cv2 = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        selected_study, selected_model, _ = train_model(
            X_train=X_combined_train,
            y_train=y_train,
            X_test=X_combined_test,
            y_test=y_test,
            model="catboost",
            seed_rf=seed,
            seed_bayes=seed,
            cv=cv2,
            n_iter=n_iter,
            groups=groups_train,
            cat_vars=None,
            n_jobs=n_jobs,
            gpu=gpu,
            val_mask=val_mask,
        )

        test_df_combined = bmca_test_df.copy()
        for sf in selected_features:
            test_df_combined[sf] = mrf_test_df[sf].values

        selected_result = _evaluate(
            selected_model, test_df_combined, combined_feat_cols,
            "BMCA+selected", n_boot, seed
        )
        selected_imp = _feature_importance(selected_model, combined_feat_cols)
        selected_imp.to_csv(f"{output_dir}/bmca_forward_selection_feature_importance.csv", index=False)

        delta = selected_result["auc"] - bmca_result["auc"]
        logger.info(f"\nΔ AUC (BMCA+selected − BMCA-only) = {delta:+.3f}")
    else:
        logger.info("\nNo MRF features passed the selection threshold.")

    # ----- Save evaluation CSV -----
    eval_rows = [{
        "model": "bmca_catboost",
        "auc": round(bmca_result["auc"], 4),
        "auc_ci_low_95": round(bmca_result["ci_low"], 4),
        "auc_ci_high_95": round(bmca_result["ci_high"], 4),
        "best_inner_cv_auc": round(bmca_study.best_value, 4),
        "n_features": len(bmca_feat_cols),
    }]
    if selected_result is not None:
        eval_rows.append({
            "model": "bmca_forward_selected_catboost",
            "auc": round(selected_result["auc"], 4),
            "auc_ci_low_95": round(selected_result["ci_low"], 4),
            "auc_ci_high_95": round(selected_result["ci_high"], 4),
            "best_inner_cv_auc": round(selected_study.best_value, 4),
            "n_features": len(bmca_feat_cols) + len(selected_features),
        })
    pd.DataFrame(eval_rows).to_csv(f"{output_dir}/bmca_forward_selection_evaluation.csv", index=False)

    # ----- Plots -----
    _plot_ranking(ranking_df, f"{plots_dir}/bmca_forward_selection_ranking.pdf")
    _plot_roc_comparison(bmca_result, selected_result, f"{plots_dir}/bmca_forward_selection_roc.pdf")

    # ----- Save artifacts -----
    artifacts = {
        "bmca_model": bmca_model, "bmca_study": bmca_study,
        "bmca_result": bmca_result,
        "ranking_df": ranking_df,
        "selected_features": selected_features,
    }
    if selected_model is not None:
        artifacts.update({
            "selected_model": selected_model,
            "selected_study": selected_study,
            "selected_result": selected_result,
            "selected_importance": selected_imp,
        })
    joblib.dump(artifacts, f"{output_dir}/bmca_forward_selection_model.joblib")
    logger.info(f"Artifacts saved to {output_dir}/bmca_forward_selection_model.joblib")

    return artifacts


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Conditional forward selection of MRF features given BMCA"
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
