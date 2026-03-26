"""
Cross-task transfer evaluation: train on augmentation pairs (MCI->Dementia),
evaluate on primary pairs (CN->MCI/Dementia).

This tests whether MRF signal learned from CI->dementia pairs transfers to
the CN->cognitive decline task. Complete population separation: augmentation
is the training set, primary is the test set.

Usage::

    python analysis_cross_task_transfer.py \
        --bmca data/adni_bmca_features_L4_combined_matched.csv \
        --mrf data/adni_mrf_features_L4_combined_matched.csv \
        --output_dir results_paper/training/aug_to_primary_seed0
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from model_strate_cv_evaluation import (
    GROUP_COL,
    LABEL_COL,
    _METADATA_COLS,
    _bootstrap_auc,
    _bootstrap_paired_auc_diff,
    _feature_cols,
    _load_combined,
)

logging.basicConfig(level=logging.INFO, format="%(name)s — %(message)s")
logger = logging.getLogger(__name__)


def _train_and_predict(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    name: str,
    n_inner: int = 5,
    n_iter: int = 50,
    seed: int = 0,
    n_jobs: int = 1,
) -> dict:
    """Train CatBoost+Optuna on train_df, predict on test_df."""
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    X_train = train_df[feature_cols]
    y_train = train_df[LABEL_COL].astype(float)
    groups_train = train_df[GROUP_COL]

    X_test = test_df[feature_cols]
    y_test = test_df[LABEL_COL].astype(float).values
    groups_test = test_df[GROUP_COL].values

    # Inner CV for hyperparameter tuning
    inner_cv = StratifiedGroupKFold(n_splits=n_inner, shuffle=True, random_state=seed)

    def objective(trial):
        params = {
            "iterations": trial.suggest_int("iterations", 50, 1000),
            "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.3, log=True),
            "depth": trial.suggest_int("depth", 3, 8),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-3, 100, log=True),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
            "border_count": trial.suggest_int("border_count", 32, 255),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 50),
            "random_strength": trial.suggest_float("random_strength", 0.0, 10.0),
            "random_seed": seed,
            "verbose": 0,
            "allow_writing_files": False,
            "nan_mode": "Min",
        }

        inner_aucs = []
        for inner_train_idx, inner_val_idx in inner_cv.split(
            X_train, y=y_train, groups=groups_train
        ):
            X_it, y_it = X_train.iloc[inner_train_idx], y_train.iloc[inner_train_idx]
            X_iv, y_iv = X_train.iloc[inner_val_idx], y_train.iloc[inner_val_idx]

            if len(np.unique(y_iv.values)) < 2:
                continue

            model = CatBoostClassifier(**params)
            model.fit(X_it, y_it, verbose=0)
            preds = model.predict_proba(X_iv)[:, 1]
            inner_aucs.append(roc_auc_score(y_iv.values, preds))

        return np.mean(inner_aucs) if inner_aucs else 0.5

    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed)
    )
    study.optimize(objective, n_trials=n_iter, n_jobs=n_jobs, show_progress_bar=False)

    best_params = study.best_params
    best_params.update(
        {"random_seed": seed, "verbose": 0, "allow_writing_files": False, "nan_mode": "Min"}
    )
    final_model = CatBoostClassifier(**best_params)
    final_model.fit(X_train, y_train, verbose=0)

    test_preds = final_model.predict_proba(X_test)[:, 1]
    test_auc = roc_auc_score(y_test, test_preds)
    ci_low, ci_high = _bootstrap_auc(y_test, test_preds, groups_test, n_boot=2000, seed=seed)

    imp = final_model.get_feature_importance()
    importance_df = (
        pd.DataFrame({"feature": feature_cols, "importance": imp})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

    logger.info(
        f"  [{name}] Transfer AUC = {test_auc:.3f}  "
        f"95% CI [{ci_low:.3f}, {ci_high:.3f}]  "
        f"(train={len(train_df)} aug, test={len(test_df)} primary)"
    )

    return {
        "name": name,
        "test_auc": test_auc,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "test_preds": test_preds,
        "y_test": y_test,
        "groups_test": groups_test,
        "importance_df": importance_df,
    }


def run(
    bmca_path: str,
    mrf_path: str,
    output_dir: str = "results_paper/training/aug_to_primary",
    n_inner: int = 5,
    n_iter: int = 50,
    seed: int = 0,
    n_jobs: int = 1,
    bmca_audit: str | None = None,
    mrf_audit: str | None = None,
) -> dict:
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    bmca_df = _load_combined(bmca_path)
    mrf_df = _load_combined(mrf_path)

    bmca_features = _feature_cols(bmca_df, bmca_audit)
    mrf_features = _feature_cols(mrf_df, mrf_audit)

    # Build BMCA+MRF
    meta_cols = [c for c in bmca_df.columns if c in _METADATA_COLS]
    bmca_mrf_df = bmca_df.merge(
        mrf_df.drop(columns=[c for c in meta_cols if c != "subject_id"], errors="ignore"),
        on="subject_id",
        how="inner",
        suffixes=("", "_mrf_dup"),
    )
    bmca_mrf_df = bmca_mrf_df[[c for c in bmca_mrf_df.columns if not c.endswith("_mrf_dup")]]
    bmca_mrf_features = sorted(set(bmca_features) | set(mrf_features))

    # Split by analysis_set
    bmca_aug = bmca_df[bmca_df["analysis_set"] == "augmentation"]
    bmca_primary = bmca_df[bmca_df["analysis_set"] == "primary"]
    mrf_aug = mrf_df[mrf_df["analysis_set"] == "augmentation"]
    mrf_primary = mrf_df[mrf_df["analysis_set"] == "primary"]
    bmca_mrf_aug = bmca_mrf_df[bmca_mrf_df["analysis_set"] == "augmentation"]
    bmca_mrf_primary = bmca_mrf_df[bmca_mrf_df["analysis_set"] == "primary"]

    n_aug_pairs = bmca_aug[GROUP_COL].nunique()
    n_primary_pairs = bmca_primary[GROUP_COL].nunique()
    logger.info(
        f"Cross-task transfer: train on {n_aug_pairs} augmentation pairs, "
        f"evaluate on {n_primary_pairs} primary pairs."
    )

    results = {}
    for name, train, test, feats in [
        ("BMCA", bmca_aug, bmca_primary, bmca_features),
        ("MRF", mrf_aug, mrf_primary, mrf_features),
        ("BMCA+MRF", bmca_mrf_aug, bmca_mrf_primary, bmca_mrf_features),
    ]:
        logger.info(f"\n{'='*60}\n{name} ({len(feats)} features)\n{'='*60}")
        r = _train_and_predict(
            train, test, feats, name,
            n_inner=n_inner, n_iter=n_iter, seed=seed, n_jobs=n_jobs,
        )
        results[name] = r
        r["importance_df"].to_csv(
            f"{output_dir}/{name.lower().replace('+', '_')}_importance.csv", index=False
        )

    # Summary table
    summary_rows = []
    for name, r in results.items():
        summary_rows.append({
            "model": name,
            "transfer_auc": round(r["test_auc"], 4),
            "ci_low_95": round(r["ci_low"], 4),
            "ci_high_95": round(r["ci_high"], 4),
        })
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(f"{output_dir}/transfer_summary.csv", index=False)
    logger.info(f"\n{summary_df.to_string(index=False)}")

    # Paired bootstrap: BMCA+MRF vs BMCA
    y_test = results["BMCA"]["y_test"]
    groups = results["BMCA"]["groups_test"]
    bt = _bootstrap_paired_auc_diff(
        y_test,
        results["BMCA+MRF"]["test_preds"],
        results["BMCA"]["test_preds"],
        groups,
        n_boot=10000,
        seed=seed,
    )
    bt_row = {
        "comparison": "BMCA+MRF vs BMCA",
        "observed_diff": round(bt["observed_diff"], 4),
        "ci_low_95": round(bt["ci_low"], 4),
        "ci_high_95": round(bt["ci_high"], 4),
        "p_value": round(bt["p_value"], 4),
        "n_boot": bt["n_boot"],
    }
    pd.DataFrame([bt_row]).to_csv(f"{output_dir}/bootstrap_paired_diff.csv", index=False)
    logger.info(
        f"  BMCA+MRF vs BMCA (transfer): Δ = {bt['observed_diff']:+.4f}  "
        f"95% CI [{bt['ci_low']:+.4f}, {bt['ci_high']:+.4f}]  "
        f"p(Δ≤0) = {bt['p_value']:.4f}"
    )

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cross-task transfer: aug -> primary")
    parser.add_argument("--bmca", required=True)
    parser.add_argument("--mrf", required=True)
    parser.add_argument("--output_dir", default="results_paper/training/aug_to_primary_seed0")
    parser.add_argument("--n_inner", type=int, default=5)
    parser.add_argument("--n_iter", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n_jobs", type=int, default=1)
    parser.add_argument("--bmca_audit", default=None)
    parser.add_argument("--mrf_audit", default=None)
    args = parser.parse_args()

    run(
        bmca_path=args.bmca,
        mrf_path=args.mrf,
        output_dir=args.output_dir,
        n_inner=args.n_inner,
        n_iter=args.n_iter,
        seed=args.seed,
        n_jobs=args.n_jobs,
        bmca_audit=args.bmca_audit,
        mrf_audit=args.mrf_audit,
    )
