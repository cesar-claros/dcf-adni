"""
BMCA + Nonlinear MRF Rule Patterns (Adapted H4)
=================================================

Tests whether conjunctive MRF patterns (multi-feature interactions like
"high BP AND low eGFR AND high glucose") add predictive value that
individual features miss.

Algorithm
---------
1.  Train a shallow Random Forest on MRF features (median-imputed).
2.  Extract path rules as binary indicators via extract_rf_rule_matrix().
3.  Deduplicate and filter by support [5%, 50%].
4.  Rank by Cramer's V with the target, take top 50.
5.  Combine BMCA features + top-50 rule indicators.
6.  L1-penalised logistic regression to select surviving rules.
7.  If any survive: train BMCA + surviving rules with CatBoost + Optuna.
8.  Evaluate on held-out test set with bootstrap CI.

Usage::

    python model_bmca_rules_evaluation.py
    python model_bmca_rules_evaluation.py --n_iter 100 --seed 42
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
from scipy.stats import chi2_contingency
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import RocCurveDisplay, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
from src.utils_model import (
    deduplicate_rule_matrix,
    extract_rf_rule_matrix,
    filter_rules_by_support,
    normalize_rule_metadata,
    train_model,
)

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

_RF_N_ESTIMATORS = 300
_RF_MAX_DEPTH = 3
_RF_MIN_SAMPLES_LEAF = 0.05
_TOP_N_RULES = 50
_MIN_SUPPORT = 0.05
_MAX_SUPPORT = 0.50
_COEF_THRESHOLD = 1e-5


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
# Rule extraction pipeline
# =============================================================================


def _cramers_v(x: np.ndarray, y: np.ndarray) -> float:
    """Cramer's V between two binary arrays."""
    table = pd.crosstab(pd.Series(x), pd.Series(y))
    if table.shape[0] < 2 or table.shape[1] < 2:
        return 0.0
    chi2, _, _, _ = chi2_contingency(table)
    n = len(x)
    return float(np.sqrt(chi2 / n))


def _rank_rules_by_association(
    rule_train: pd.DataFrame,
    y_train: np.ndarray,
    top_n: int,
) -> list[str]:
    """Rank rules by Cramer's V with the target and return top_n rule IDs."""
    scores = {}
    for col in rule_train.columns:
        scores[col] = _cramers_v(rule_train[col].values, y_train)
    ranked = sorted(scores, key=scores.get, reverse=True)
    return ranked[:top_n]


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
    rules_result: dict | None,
    output_path: str,
) -> None:
    n_pairs = int(bmca_result["y_true"].sum())
    fig, ax = plt.subplots(figsize=(7, 7))

    entries = [(bmca_result, "steelblue", "BMCA")]
    if rules_result is not None:
        entries.append((rules_result, "darkorange", "BMCA + MRF rules"))

    for result, color, label in entries:
        RocCurveDisplay.from_predictions(
            result["y_true"], result["scores"], ax=ax, color=color,
            name=f"{label}  AUC = {result['auc']:.3f}  [{result['ci_low']:.3f}, {result['ci_high']:.3f}]",
        )

    ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="Chance")
    ax.set_title(
        f"BMCA vs BMCA + MRF Rule Patterns — ROC Curves\n"
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
    X_mrf_train = mrf_train_df[mrf_feat_cols].copy()
    X_mrf_test = mrf_test_df[mrf_feat_cols].copy()

    logger.info(f"BMCA features: {len(bmca_feat_cols)}, MRF features: {len(mrf_feat_cols)}")

    # ----- Step 1: Impute MRF for RF (RF cannot handle NaN) -----
    logger.info("\n=== Step 1: Median-imputing MRF features for RF ===")
    imputer = SimpleImputer(strategy="median")
    X_mrf_train_imp = pd.DataFrame(
        imputer.fit_transform(X_mrf_train),
        columns=mrf_feat_cols,
        index=X_mrf_train.index,
    )
    X_mrf_test_imp = pd.DataFrame(
        imputer.transform(X_mrf_test),
        columns=mrf_feat_cols,
        index=X_mrf_test.index,
    )

    n_imputed = X_mrf_train.isna().sum().sum()
    logger.info(f"Imputed {n_imputed} missing values in MRF train")

    # ----- Step 2: Train shallow RF on MRF -----
    logger.info(
        f"\n=== Step 2: Training RF on MRF features "
        f"(n_estimators={_RF_N_ESTIMATORS}, max_depth={_RF_MAX_DEPTH}) ==="
    )
    rf = RandomForestClassifier(
        n_estimators=_RF_N_ESTIMATORS,
        max_depth=_RF_MAX_DEPTH,
        min_samples_leaf=_RF_MIN_SAMPLES_LEAF,
        random_state=seed,
        n_jobs=n_jobs,
    )
    rf.fit(X_mrf_train_imp, y_train_arr)
    logger.info(f"RF OOB score: {rf.oob_score_:.3f}")

    # ----- Step 3: Extract rules -----
    logger.info("\n=== Step 3: Extracting rule matrix ===")
    rule_train, rule_test, metadata_df = extract_rf_rule_matrix(
        rf, X_mrf_train_imp, X_mrf_test_imp
    )
    n_raw = rule_train.shape[1]
    logger.info(f"Raw rules extracted: {n_raw}")

    # ----- Step 4: Deduplicate and filter by support -----
    rule_train, rule_test, metadata_df = deduplicate_rule_matrix(
        rule_train, rule_test, metadata_df
    )
    n_dedup = rule_train.shape[1]
    logger.info(f"After deduplication: {n_dedup}")

    rule_train, rule_test, metadata_df = filter_rules_by_support(
        rule_train, rule_test, metadata_df,
        min_support=_MIN_SUPPORT, max_support=_MAX_SUPPORT,
    )
    n_filtered = rule_train.shape[1]
    logger.info(f"After support filtering [{_MIN_SUPPORT}, {_MAX_SUPPORT}]: {n_filtered}")

    if n_filtered == 0:
        logger.warning("No rules survived filtering. Exiting.")
        bmca_cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        bmca_study, bmca_model, _ = train_model(
            X_train=X_bmca_train, y_train=y_train,
            X_test=X_bmca_test, y_test=y_test,
            model="catboost", seed_rf=seed, seed_bayes=seed,
            cv=bmca_cv, n_iter=n_iter, groups=groups_train,
            cat_vars=None, n_jobs=n_jobs, gpu=gpu, val_mask=val_mask,
        )
        bmca_result = _evaluate(bmca_model, bmca_test_df, bmca_feat_cols, "BMCA-only", n_boot, seed)
        eval_df = pd.DataFrame([{
            "model": "bmca_catboost", "auc": round(bmca_result["auc"], 4),
            "auc_ci_low_95": round(bmca_result["ci_low"], 4),
            "auc_ci_high_95": round(bmca_result["ci_high"], 4),
            "n_rules_raw": n_raw, "n_rules_dedup": n_dedup,
            "n_rules_filtered": 0, "n_rules_surviving": 0,
        }])
        eval_df.to_csv(f"{output_dir}/bmca_rules_evaluation.csv", index=False)
        return {"bmca_result": bmca_result}

    # ----- Step 5: Rank by Cramer's V, take top N -----
    logger.info(f"\n=== Step 4: Ranking rules by Cramer's V, taking top {_TOP_N_RULES} ===")
    top_rule_ids = _rank_rules_by_association(rule_train, y_train_arr, _TOP_N_RULES)
    rule_train_top = rule_train[top_rule_ids]
    rule_test_top = rule_test[top_rule_ids]
    metadata_top = metadata_df[metadata_df["rule_id"].isin(top_rule_ids)].reset_index(drop=True)
    logger.info(f"Top rules selected: {len(top_rule_ids)}")

    # ----- Step 6: L1 sparse selection -----
    logger.info("\n=== Step 5: L1 logistic regression for rule selection ===")

    # Z-score BMCA features (fit on train); rule indicators stay 0/1
    scaler = StandardScaler()
    X_bmca_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_bmca_train),
        columns=bmca_feat_cols,
        index=X_bmca_train.index,
    )

    X_l1_train = pd.concat([X_bmca_train_scaled, rule_train_top], axis=1)

    # Handle NaN in BMCA features for logistic regression (fill with 0 after scaling)
    X_l1_train = X_l1_train.fillna(0)

    # Pre-compute CV splits with groups (LogisticRegressionCV doesn't pass
    # groups to the splitter, so we provide pre-split indices directly).
    l1_cv_splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    l1_cv_splits = list(l1_cv_splitter.split(X_l1_train, y_train_arr, groups=groups_train.values))
    l1_model = LogisticRegressionCV(
        Cs=100,
        l1_ratios=(1.0,),
        penalty="elasticnet",
        solver="saga",
        scoring="roc_auc",
        cv=l1_cv_splits,
        max_iter=5000,
        random_state=seed,
        n_jobs=n_jobs,
    )
    l1_model.fit(X_l1_train, y_train_arr)

    # Identify surviving rules
    all_l1_cols = list(X_l1_train.columns)
    coefs = l1_model.coef_[0]
    rule_coefs = {
        rule_id: coefs[all_l1_cols.index(rule_id)]
        for rule_id in top_rule_ids
    }
    surviving_rules = [
        rule_id for rule_id, coef in rule_coefs.items()
        if abs(coef) > _COEF_THRESHOLD
    ]
    logger.info(f"Rules surviving L1: {len(surviving_rules)}/{len(top_rule_ids)}")

    # Save surviving rule descriptions
    surviving_df = metadata_top[metadata_top["rule_id"].isin(surviving_rules)].copy()
    surviving_df["l1_coef"] = surviving_df["rule_id"].map(rule_coefs)
    surviving_df = surviving_df.sort_values("l1_coef", key=abs, ascending=False)
    surviving_df.to_csv(f"{output_dir}/bmca_rules_surviving.csv", index=False)
    logger.info(f"Surviving rules saved to {output_dir}/bmca_rules_surviving.csv")

    if surviving_rules:
        logger.info("\nSurviving rules:")
        for _, row in surviving_df.iterrows():
            logger.info(f"  {row['rule_id']}: coef={row['l1_coef']:.4f}  {row['rule']}")

    # ----- Step 7: Train BMCA-only and (if rules survive) BMCA+rules -----
    logger.info("\n=== Step 6: Training BMCA-only (baseline) ===")
    bmca_cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    bmca_study, bmca_model, _ = train_model(
        X_train=X_bmca_train, y_train=y_train,
        X_test=X_bmca_test, y_test=y_test,
        model="catboost", seed_rf=seed, seed_bayes=seed,
        cv=bmca_cv, n_iter=n_iter, groups=groups_train,
        cat_vars=None, n_jobs=n_jobs, gpu=gpu, val_mask=val_mask,
    )

    bmca_result = _evaluate(bmca_model, bmca_test_df, bmca_feat_cols, "BMCA-only", n_boot, seed)

    rules_result = None
    rules_study = None
    rules_model = None
    rules_imp = None

    if surviving_rules:
        logger.info(f"\n=== Step 7: Training BMCA + {len(surviving_rules)} surviving rules ===")
        combined_feat_cols = bmca_feat_cols + surviving_rules
        X_combined_train = pd.concat(
            [X_bmca_train, rule_train_top[surviving_rules]], axis=1
        )
        X_combined_test = pd.concat(
            [X_bmca_test, rule_test_top[surviving_rules]], axis=1
        )

        rules_cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        rules_study, rules_model, _ = train_model(
            X_train=X_combined_train, y_train=y_train,
            X_test=X_combined_test, y_test=y_test,
            model="catboost", seed_rf=seed, seed_bayes=seed,
            cv=rules_cv, n_iter=n_iter, groups=groups_train,
            cat_vars=None, n_jobs=n_jobs, gpu=gpu, val_mask=val_mask,
        )

        test_df_combined = bmca_test_df.copy()
        for rule_id in surviving_rules:
            test_df_combined[rule_id] = rule_test_top[rule_id].values

        rules_result = _evaluate(
            rules_model, test_df_combined, combined_feat_cols,
            "BMCA+rules", n_boot, seed
        )

        rules_imp = _feature_importance(rules_model, combined_feat_cols)
        rules_imp.to_csv(f"{output_dir}/bmca_rules_feature_importance.csv", index=False)

        delta = rules_result["auc"] - bmca_result["auc"]
        logger.info(f"\nΔ AUC (BMCA+rules − BMCA-only) = {delta:+.3f}")
        logger.info(
            f"Inner CV AUC — BMCA-only: {bmca_study.best_value:.3f}, "
            f"BMCA+rules: {rules_study.best_value:.3f}"
        )
    else:
        logger.info("\nNo rules survived L1 selection. BMCA-only result stands.")

    # ----- Save evaluation CSV -----
    eval_rows = [{
        "model": "bmca_catboost",
        "auc": round(bmca_result["auc"], 4),
        "auc_ci_low_95": round(bmca_result["ci_low"], 4),
        "auc_ci_high_95": round(bmca_result["ci_high"], 4),
        "best_inner_cv_auc": round(bmca_study.best_value, 4),
        "n_rules_raw": n_raw,
        "n_rules_dedup": n_dedup,
        "n_rules_filtered": n_filtered,
        "n_rules_surviving": len(surviving_rules),
    }]
    if rules_result is not None:
        eval_rows.append({
            "model": "bmca_rules_catboost",
            "auc": round(rules_result["auc"], 4),
            "auc_ci_low_95": round(rules_result["ci_low"], 4),
            "auc_ci_high_95": round(rules_result["ci_high"], 4),
            "best_inner_cv_auc": round(rules_study.best_value, 4),
            "n_rules_raw": n_raw,
            "n_rules_dedup": n_dedup,
            "n_rules_filtered": n_filtered,
            "n_rules_surviving": len(surviving_rules),
        })
    pd.DataFrame(eval_rows).to_csv(f"{output_dir}/bmca_rules_evaluation.csv", index=False)

    # ----- Plots -----
    _plot_roc_comparison(bmca_result, rules_result, f"{plots_dir}/bmca_rules_roc.pdf")

    # ----- Save artifacts -----
    artifacts = {
        "bmca_model": bmca_model, "bmca_study": bmca_study,
        "bmca_result": bmca_result,
        "rf_model": rf,
        "metadata_top": metadata_top,
        "surviving_rules_df": surviving_df,
        "l1_model": l1_model,
        "imputer": imputer,
        "scaler": scaler,
    }
    if rules_model is not None:
        artifacts.update({
            "rules_model": rules_model,
            "rules_study": rules_study,
            "rules_result": rules_result,
            "rules_importance": rules_imp,
        })
    joblib.dump(artifacts, f"{output_dir}/bmca_rules_model.joblib")
    logger.info(f"Artifacts saved to {output_dir}/bmca_rules_model.joblib")

    return artifacts


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train and evaluate BMCA with nonlinear MRF rule patterns"
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
