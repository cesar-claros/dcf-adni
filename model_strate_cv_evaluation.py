"""
Strategy E: Full CV evaluation over all primary pairs.

Instead of a single train/test split (16 test pairs), this script runs
StratifiedGroupKFold over all 47 primary CN->dementia pairs. Augmentation
pairs are always included in training. Each primary pair rotates through
exactly one test fold, yielding out-of-fold (OOF) predictions for all 47
pairs — tripling the effective test size.

For each feature set (BMCA, MRF, BMCA+MRF), the script:
1. Runs k-fold outer CV (groups = matched pair IDs).
2. Within each fold, runs Optuna hyperparameter tuning with an inner CV.
3. Collects OOF predictions for primary pairs.
4. Reports OOF AUC with bootstrap 95% CI over all 47 pairs.

Usage::

    python model_strate_cv_evaluation.py --n_iter 50 --n_outer 5 --seed 0 --n_jobs 1
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

logging.basicConfig(level=logging.INFO, format="%(name)s — %(message)s")
logger = logging.getLogger(__name__)

_METADATA_COLS = {
    "subject_id", "pair_id", "group", "transition", "transition_label",
    "matched_cohort", "analysis_set", "evaluation_eligible",
    "abs_age_gap", "split", "split_group_source",
    "first_conversion_month", "baseline_diagnosis", "n_followup_visits_ge12_with_diag",
}
LABEL_COL = "transition"
GROUP_COL = "group"


def _feature_cols(df: pd.DataFrame, audit_path: str | None = None) -> list[str]:
    all_feats = [c for c in df.columns if c not in _METADATA_COLS]
    if audit_path is None:
        return all_feats
    audit = pd.read_csv(audit_path)
    keep = set(audit.loc[audit["keep_for_modeling"] == 1, "column"])
    return [c for c in all_feats if c in keep]


def _bootstrap_auc(
    y_true: np.ndarray,
    y_score: np.ndarray,
    groups: np.ndarray,
    n_boot: int = 2000,
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


def _bootstrap_paired_auc_diff(
    y_true: np.ndarray,
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    groups: np.ndarray,
    n_boot: int = 10000,
    seed: int = 0,
) -> dict:
    """Paired bootstrap test for AUC(A) - AUC(B), resampling at group level."""
    rng = np.random.default_rng(seed)
    unique_groups = np.unique(groups)
    observed_diff = roc_auc_score(y_true, scores_a) - roc_auc_score(y_true, scores_b)

    boot_diffs = []
    for _ in range(n_boot):
        sampled = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        idx = np.concatenate([np.where(groups == g)[0] for g in sampled])
        y_b = y_true[idx]
        if len(np.unique(y_b)) < 2:
            continue
        auc_a = roc_auc_score(y_b, scores_a[idx])
        auc_b = roc_auc_score(y_b, scores_b[idx])
        boot_diffs.append(auc_a - auc_b)

    boot_diffs = np.array(boot_diffs)
    p_value = float(np.mean(boot_diffs <= 0))

    return {
        "observed_diff": observed_diff,
        "ci_low": float(np.percentile(boot_diffs, 2.5)),
        "ci_high": float(np.percentile(boot_diffs, 97.5)),
        "p_value": p_value,
        "n_boot": len(boot_diffs),
    }


def _load_combined(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df[LABEL_COL] = pd.to_numeric(df[LABEL_COL], errors="coerce")
    df[GROUP_COL] = pd.to_numeric(df[GROUP_COL], errors="coerce")
    return df


def run_cv_for_feature_set(
    df: pd.DataFrame,
    feature_cols: list[str],
    name: str,
    n_outer: int = 5,
    n_inner: int = 5,
    n_iter: int = 50,
    seed: int = 0,
    n_jobs: int = 1,
    training_mode: str = "combined",
) -> dict:
    """Run nested CV: outer folds rotate pairs through test, inner folds tune hyperparams.

    training_mode controls which subjects participate:
      - "combined": primary pairs rotate through test; augmentation always in training.
      - "primary_only": only primary pairs used; no augmentation.
      - "augmentation_only": only augmentation pairs used (CV within augmentation).
    """
    if training_mode == "augmentation_only":
        cv_df = df[df["analysis_set"] == "augmentation"].copy()
        extra_train_df = pd.DataFrame(columns=df.columns)
    else:
        cv_df = df[df["analysis_set"] == "primary"].copy()
        if training_mode == "combined":
            extra_train_df = df[df["analysis_set"] == "augmentation"].copy()
        else:  # primary_only
            extra_train_df = pd.DataFrame(columns=df.columns)

    outer_cv = StratifiedGroupKFold(n_splits=n_outer, shuffle=True, random_state=seed)

    oof_scores = np.full(len(cv_df), np.nan)
    fold_aucs = []
    fold_importances = []

    for fold_idx, (train_idx, test_idx) in enumerate(
        outer_cv.split(cv_df, y=cv_df[LABEL_COL], groups=cv_df[GROUP_COL])
    ):
        cv_train = cv_df.iloc[train_idx]
        cv_test = cv_df.iloc[test_idx]

        # Extra training data (augmentation for combined mode, empty otherwise)
        fold_train = pd.concat([cv_train, extra_train_df], ignore_index=True)

        X_train = fold_train[feature_cols]
        y_train = fold_train[LABEL_COL].astype(float)
        groups_train = fold_train[GROUP_COL]

        X_test = cv_test[feature_cols]
        y_test = cv_test[LABEL_COL].astype(float).values

        # Inner CV for hyperparameter tuning via Optuna
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        inner_cv = StratifiedGroupKFold(n_splits=n_inner, shuffle=True, random_state=seed + fold_idx)
        # In combined mode, only score on primary validation subjects within inner CV.
        # In primary_only/augmentation_only mode, score on all validation subjects.
        if training_mode == "combined":
            inner_scoring_mask = (fold_train["analysis_set"] == "primary").values
        else:
            inner_scoring_mask = np.ones(len(fold_train), dtype=bool)

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

                # Only score on primary validation subjects
                val_scoring = inner_scoring_mask[inner_val_idx]
                if val_scoring.sum() == 0 or len(np.unique(y_iv.values[val_scoring])) < 2:
                    continue

                model = CatBoostClassifier(**params)
                model.fit(X_it, y_it, verbose=0)
                preds = model.predict_proba(X_iv)[:, 1]
                inner_aucs.append(roc_auc_score(y_iv.values[val_scoring], preds[val_scoring]))

            return np.mean(inner_aucs) if inner_aucs else 0.5

        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
        study.optimize(objective, n_trials=n_iter, n_jobs=n_jobs, show_progress_bar=False)

        best_params = study.best_params
        best_params.update({"random_seed": seed, "verbose": 0, "allow_writing_files": False, "nan_mode": "Min"})
        final_model = CatBoostClassifier(**best_params)
        final_model.fit(X_train, y_train, verbose=0)

        fold_preds = final_model.predict_proba(X_test)[:, 1]
        oof_scores[test_idx] = fold_preds

        if len(np.unique(y_test)) >= 2:
            fold_auc = roc_auc_score(y_test, fold_preds)
            fold_aucs.append(fold_auc)

        imp = final_model.get_feature_importance()
        fold_importances.append(pd.Series(imp, index=feature_cols))

        n_test_pairs = cv_test[GROUP_COL].nunique()
        logger.info(
            f"  [{name}] Fold {fold_idx+1}/{n_outer}: "
            f"test pairs={n_test_pairs}, "
            f"fold AUC={fold_auc:.3f}, "
            f"best inner CV={study.best_value:.3f}"
        )

    # OOF AUC over all CV pairs
    y_all = cv_df[LABEL_COL].astype(float).values
    groups_all = cv_df[GROUP_COL].values
    oof_auc = roc_auc_score(y_all, oof_scores)
    ci_low, ci_high = _bootstrap_auc(y_all, oof_scores, groups_all, n_boot=2000, seed=seed)

    # Average feature importance
    avg_importance = pd.concat(fold_importances, axis=1).mean(axis=1)
    importance_df = (
        pd.DataFrame({"feature": feature_cols, "importance": avg_importance})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

    logger.info(
        f"  [{name}] OOF AUC = {oof_auc:.3f}  95% CI [{ci_low:.3f}, {ci_high:.3f}]  "
        f"(n={len(cv_df)//2} pairs, mode={training_mode})"
    )

    return {
        "name": name,
        "oof_auc": oof_auc,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "fold_aucs": fold_aucs,
        "importance_df": importance_df,
        "oof_scores": oof_scores,
        "y_true": y_all,
        "groups": groups_all,
    }


def run(
    bmca_path: str = "data/adni_bmca_features_strate_combined_matched.csv",
    mrf_path: str = "data/adni_mrf_features_strate_combined_matched.csv",
    output_dir: str = "results_strate_cv",
    n_outer: int = 5,
    n_inner: int = 5,
    n_iter: int = 50,
    seed: int = 0,
    n_jobs: int = 1,
    bmca_audit: str | None = None,
    mrf_audit: str | None = None,
    training_mode: str = "combined",
) -> dict:
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    bmca_df = _load_combined(bmca_path)
    mrf_df = _load_combined(mrf_path)

    bmca_features = _feature_cols(bmca_df, bmca_audit)
    mrf_features = _feature_cols(mrf_df, mrf_audit)

    # Build BMCA+MRF by merging on metadata
    meta_cols = [c for c in bmca_df.columns if c in _METADATA_COLS]
    bmca_mrf_df = bmca_df.merge(
        mrf_df.drop(columns=[c for c in meta_cols if c != "subject_id"], errors="ignore"),
        on="subject_id",
        how="inner",
        suffixes=("", "_mrf_dup"),
    )
    # Drop any duplicate columns from merge
    bmca_mrf_df = bmca_mrf_df[[c for c in bmca_mrf_df.columns if not c.endswith("_mrf_dup")]]
    bmca_mrf_features = sorted(set(bmca_features) | set(mrf_features))

    if training_mode == "augmentation_only":
        n_cv_pairs = bmca_df[bmca_df["analysis_set"] == "augmentation"][GROUP_COL].nunique()
        pop_label = "augmentation"
    else:
        n_cv_pairs = bmca_df[bmca_df["analysis_set"] == "primary"][GROUP_COL].nunique()
        pop_label = "primary"
    logger.info(
        f"Full CV ({training_mode}): {n_cv_pairs} {pop_label} pairs, "
        f"{n_outer}-fold outer CV, {n_iter} Optuna trials per fold."
    )

    results = {}
    for name, df, feats in [
        ("BMCA", bmca_df, bmca_features),
        ("MRF", mrf_df, mrf_features),
        ("BMCA+MRF", bmca_mrf_df, bmca_mrf_features),
    ]:
        logger.info(f"\n{'='*60}\n{name} ({len(feats)} features)\n{'='*60}")
        r = run_cv_for_feature_set(
            df, feats, name,
            n_outer=n_outer, n_inner=n_inner, n_iter=n_iter,
            seed=seed, n_jobs=n_jobs, training_mode=training_mode,
        )
        results[name] = r

        r["importance_df"].to_csv(f"{output_dir}/{name.lower().replace('+','_')}_importance.csv", index=False)

    # Save OOF predictions
    oof_df = pd.DataFrame({"group": results["BMCA"]["groups"], "y_true": results["BMCA"]["y_true"]})
    for name, r in results.items():
        oof_df[f"oof_{name.lower().replace('+','_')}"] = r["oof_scores"]
    oof_df.to_csv(f"{output_dir}/oof_predictions.csv", index=False)

    # Summary table
    summary_rows = []
    for name, r in results.items():
        summary_rows.append({
            "model": name,
            "oof_auc": round(r["oof_auc"], 4),
            "ci_low_95": round(r["ci_low"], 4),
            "ci_high_95": round(r["ci_high"], 4),
            "fold_aucs": str([round(a, 3) for a in r["fold_aucs"]]),
        })
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(f"{output_dir}/strate_cv_summary.csv", index=False)

    logger.info(f"\n{'='*60}\nSummary\n{'='*60}")
    logger.info(f"\n{summary_df.to_string(index=False)}")

    # Paired bootstrap AUC difference tests
    y_true = results["BMCA"]["y_true"]
    groups = results["BMCA"]["groups"]
    comparisons = [
        ("BMCA+MRF", "BMCA"),
        ("MRF", "BMCA"),
    ]
    bootstrap_rows = []
    for name_a, name_b in comparisons:
        if name_a in results and name_b in results:
            bt = _bootstrap_paired_auc_diff(
                y_true, results[name_a]["oof_scores"], results[name_b]["oof_scores"],
                groups, n_boot=10000, seed=seed,
            )
            bootstrap_rows.append({
                "comparison": f"{name_a} vs {name_b}",
                "observed_diff": round(bt["observed_diff"], 4),
                "ci_low_95": round(bt["ci_low"], 4),
                "ci_high_95": round(bt["ci_high"], 4),
                "p_value": round(bt["p_value"], 4),
                "n_boot": bt["n_boot"],
            })
            logger.info(
                f"  {name_a} vs {name_b}: Δ = {bt['observed_diff']:+.4f}  "
                f"95% CI [{bt['ci_low']:+.4f}, {bt['ci_high']:+.4f}]  "
                f"p(Δ≤0) = {bt['p_value']:.4f}"
            )
    if bootstrap_rows:
        bt_df = pd.DataFrame(bootstrap_rows)
        bt_df.to_csv(f"{output_dir}/bootstrap_paired_diff.csv", index=False)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Strategy E full CV evaluation")
    parser.add_argument("--bmca", default="data/adni_bmca_features_strate_combined_matched.csv")
    parser.add_argument("--mrf", default="data/adni_mrf_features_strate_combined_matched.csv")
    parser.add_argument("--output_dir", default="results_strate_cv")
    parser.add_argument("--n_outer", type=int, default=5)
    parser.add_argument("--n_inner", type=int, default=5)
    parser.add_argument("--n_iter", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n_jobs", type=int, default=1)
    parser.add_argument("--bmca_audit", default=None, help="BMCA column audit CSV; only keep_for_modeling=1 features used")
    parser.add_argument("--mrf_audit", default=None, help="MRF column audit CSV; only keep_for_modeling=1 features used")
    parser.add_argument(
        "--training_mode", default="combined",
        choices=["combined", "primary_only", "augmentation_only"],
        help="Which subjects to use: combined (primary+aug), primary_only, or augmentation_only",
    )
    args = parser.parse_args()

    run(
        bmca_path=args.bmca,
        mrf_path=args.mrf,
        output_dir=args.output_dir,
        n_outer=args.n_outer,
        n_inner=args.n_inner,
        n_iter=args.n_iter,
        seed=args.seed,
        n_jobs=args.n_jobs,
        bmca_audit=args.bmca_audit,
        mrf_audit=args.mrf_audit,
        training_mode=args.training_mode,
    )
