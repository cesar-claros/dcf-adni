"""
Strategy 5: MRF-Informed Sample Reweighting (Experiment 14-Adapted)
====================================================================

Uses MRF information to shape *how* the BMCA model is trained — not which
features it uses. Targeting insight from Experiment 14: MRF within-pair signal
is real for augmentation pairs (CI→dementia, contrast OOF AUC = 0.704) and
absent for primary pairs (CN→MCI, contrast OOF AUC = 0.420).

Key adaptation
--------------
The original Strategy 5 applied disagreement weights to all subjects. Experiment
14 shows MRF is near-chance for CN→MCI, so MRF-based weights on primary subjects
would add noise. This script applies MRF-based weights ONLY to augmentation
subjects (where MRF is informative), leaving primary subject weights flat (w=1).

Variants
--------
A (baseline): w = 1.0 for all subjects (standard BMCA retrain)
B (MRF agreement): w_aug = 1 + alpha * P(MRF agrees with label)
                   w_primary = 1.0
                   Upweights augmentation subjects where MRF correctly identifies
                   the transition direction — the MRF-confident cases.
C (disagreement):  w_aug = 1 + alpha * |bmca_oof − mrf_aug_oof|
                   w_primary = 1.0
                   Upweights augmentation subjects where BMCA and MRF disagree —
                   cases where the biomarker/cognitive signal conflicts with the
                   vascular burden signal.

All weights are normalised to mean=1 (preserving effective training scale).

Circularity prevention
----------------------
Weights derive from Stage 1 OOF scores computed with a separate smaller Optuna
run (n_iter_oof trials, seed+100). Stage 2 comparison models use fresh 50-trial
Optuna searches and only see the pre-computed fixed weights.

Two-stage pipeline
------------------
Stage 1A: BMCA OOF   — Optuna (n_iter_oof trials, seed+100, primary-only val_mask)
           → bmca_oof for all training subjects

Stage 1B: MRF aug OOF — Optuna (n_iter_mrf trials, seed+100, augmentation only)
           → mrf_aug_oof for augmentation subjects; gate if OOF AUC < 0.60

Stage 2:  Three models (n_iter trials, seed, primary-only val_mask, with weights)
           A: unweighted baseline
           B: MRF agreement weights on augmentation
           C: Disagreement weights on augmentation

Usage::

    python model_mrf_sample_reweighting.py
    python model_mrf_sample_reweighting.py --alpha 5.0 --n_iter 100
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
from sklearn.metrics import RocCurveDisplay, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

import optuna
from optuna.samplers import TPESampler

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
# Core: Optuna CatBoost with optional sample weights + val_mask scoring
# =============================================================================


def _optuna_catboost(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    groups_train: np.ndarray,
    cv: StratifiedGroupKFold,
    val_mask: np.ndarray,
    weights: np.ndarray | None,
    n_iter: int,
    seed: int,
    label: str = "",
) -> tuple[optuna.Study, CatBoostClassifier, np.ndarray]:
    """
    Optuna hyperparameter search for CatBoost with optional sample weights.

    Validation scoring is restricted to primary subjects via val_mask.
    OOF predictions are computed for ALL training subjects using best params.

    Returns (study, best_model_fit_on_full_train, oof_proba).
    """
    splits = list(cv.split(X_train, y_train, groups_train))
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "iterations": trial.suggest_int("iterations", 100, 1000),
            "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-1, log=True),
            "depth": trial.suggest_int("depth", 2, 8),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-3, 1e3, log=True),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
            "border_count": trial.suggest_int("border_count", 32, 255),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 50),
            "random_strength": trial.suggest_float("random_strength", 0.0, 10.0),
        }
        fold_aucs = []
        for tr_idx, val_idx in splits:
            m = CatBoostClassifier(
                **params, random_seed=seed, verbose=0, allow_writing_files=False
            )
            m.fit(
                X_train.iloc[tr_idx], y_train[tr_idx],
                sample_weight=weights[tr_idx] if weights is not None else None,
            )
            # Score only on primary subjects in this validation fold
            primary_in_val = val_mask[val_idx]
            primary_val_idx = val_idx[primary_in_val]
            if len(primary_val_idx) < 4 or len(np.unique(y_train[primary_val_idx])) < 2:
                continue
            proba = m.predict_proba(X_train.iloc[primary_val_idx])[:, 1]
            fold_aucs.append(roc_auc_score(y_train[primary_val_idx], proba))
        return float(np.mean(fold_aucs)) if fold_aucs else 0.0

    study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_iter, show_progress_bar=True)
    logger.info(
        f"{label} best CV AUC: {study.best_value:.4f} | params: {study.best_params}"
    )

    # OOF predictions for all training subjects (leakage-free per fold)
    oof_proba = np.full(len(X_train), np.nan)
    for tr_idx, val_idx in splits:
        m = CatBoostClassifier(
            **study.best_params, random_seed=seed, verbose=0, allow_writing_files=False
        )
        m.fit(
            X_train.iloc[tr_idx], y_train[tr_idx],
            sample_weight=weights[tr_idx] if weights is not None else None,
        )
        oof_proba[val_idx] = m.predict_proba(X_train.iloc[val_idx])[:, 1]

    # Final model on all training data
    best_model = CatBoostClassifier(
        **study.best_params, random_seed=seed, verbose=0, allow_writing_files=False
    )
    best_model.fit(
        X_train, y_train,
        sample_weight=weights if weights is not None else None,
    )

    return study, best_model, oof_proba


# =============================================================================
# Stage 1B: MRF model on augmentation pairs (same as Exp 12 Stage 1)
# =============================================================================


def _get_mrf_aug_oof(
    mrf_train: pd.DataFrame,
    n_iter: int,
    n_splits: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, float, optuna.Study]:
    """
    Train MRF CatBoost on augmentation subjects and return OOF predictions.

    Returns (oof_proba, aug_labels, oof_auc, study) where oof_proba is
    indexed in the same order as augmentation rows in mrf_train.
    """
    aug = mrf_train[mrf_train["analysis_set"] == "augmentation"].copy()
    mrf_feats = _feature_cols(aug)
    X_aug = aug[mrf_feats]
    y_aug = aug[LABEL_COL].values.astype(float)
    groups_aug = aug[GROUP_COL].values

    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    splits = list(cv.split(X_aug, y_aug, groups_aug))

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "iterations": trial.suggest_int("iterations", 100, 1000),
            "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-1, log=True),
            "depth": trial.suggest_int("depth", 2, 8),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-3, 1e3, log=True),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
            "border_count": trial.suggest_int("border_count", 32, 255),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 50),
            "random_strength": trial.suggest_float("random_strength", 0.0, 10.0),
        }
        fold_aucs = []
        for tr_idx, val_idx in splits:
            m = CatBoostClassifier(
                **params, random_seed=seed, verbose=0, allow_writing_files=False
            )
            m.fit(X_aug.iloc[tr_idx], y_aug[tr_idx])
            if len(np.unique(y_aug[val_idx])) == 2:
                fold_aucs.append(
                    roc_auc_score(y_aug[val_idx],
                                  m.predict_proba(X_aug.iloc[val_idx])[:, 1])
                )
        return float(np.mean(fold_aucs)) if fold_aucs else 0.0

    study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_iter, show_progress_bar=True)
    logger.info(
        f"Stage 1B MRF-aug best CV AUC: {study.best_value:.4f} | "
        f"params: {study.best_params}"
    )

    oof_proba = np.full(len(aug), np.nan)
    for tr_idx, val_idx in splits:
        m = CatBoostClassifier(
            **study.best_params, random_seed=seed, verbose=0, allow_writing_files=False
        )
        m.fit(X_aug.iloc[tr_idx], y_aug[tr_idx])
        oof_proba[val_idx] = m.predict_proba(X_aug.iloc[val_idx])[:, 1]

    oof_auc = roc_auc_score(y_aug, oof_proba)
    logger.info(f"Stage 1B MRF-aug OOF AUC: {oof_auc:.4f}")

    return oof_proba, y_aug, oof_auc, study


# =============================================================================
# Weight computation
# =============================================================================


def compute_weights(
    train_df: pd.DataFrame,
    bmca_oof: np.ndarray,
    mrf_aug_oof_by_subject: dict[str, float],
    alpha: float,
    variant: str,
) -> np.ndarray:
    """
    Compute normalised sample weights for each training subject.

    Variants
    --------
    B: MRF agreement — w_aug = 1 + alpha * P(MRF agrees with label)
       P(MRF agrees) = mrf_aug_oof if label=1, else 1 - mrf_aug_oof
    C: Disagreement  — w_aug = 1 + alpha * |bmca_oof - mrf_aug_oof|

    Primary subjects always receive w = 1.0.
    Weights are normalised to mean = 1 (preserving effective training scale).
    """
    n = len(train_df)
    weights = np.ones(n, dtype=float)

    labels = train_df[LABEL_COL].values.astype(float)
    subject_ids = train_df[SUBJECT_ID_COL].values
    is_aug = (train_df["analysis_set"] == "augmentation").values

    for i in np.where(is_aug)[0]:
        sid = subject_ids[i]
        if sid not in mrf_aug_oof_by_subject:
            continue
        mrf_score = mrf_aug_oof_by_subject[sid]

        if variant == "B":
            # P(MRF agrees with true label)
            p_agree = mrf_score if labels[i] == 1 else (1.0 - mrf_score)
            weights[i] = 1.0 + alpha * p_agree

        elif variant == "C":
            # |BMCA OOF - MRF OOF| — disagreement between the two views
            disagreement = abs(bmca_oof[i] - mrf_score)
            weights[i] = 1.0 + alpha * disagreement

    # Normalise to mean = 1 so the total loss scale is preserved
    weights /= weights.mean()
    return weights


# =============================================================================
# Stage 2 evaluation
# =============================================================================


def _evaluate_model(
    best_model: CatBoostClassifier,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    n_boot: int,
    seed: int,
) -> dict:
    eligible = _evaluation_eligible(test_df)
    X_test = eligible[feature_cols]
    y_test_arr = eligible[LABEL_COL].values.astype(float)
    groups_test = eligible[GROUP_COL].values

    y_score = best_model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test_arr, y_score)
    ci_low, ci_high = _bootstrap_auc(y_test_arr, y_score, groups_test, n_boot, seed)

    imp_df = (
        pd.DataFrame({
            "feature": feature_cols,
            "importance": best_model.get_feature_importance(),
        })
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

    return {
        "auc": auc,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "y_true": y_test_arr,
        "y_score": y_score,
        "groups": groups_test,
        "importance": imp_df,
    }


# =============================================================================
# Plotting
# =============================================================================


def _plot_roc(results: list[dict], output_path: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    colors = ["steelblue", "forestgreen", "firebrick"]
    n_pairs = int(results[0]["eval"]["y_true"].sum())
    for r, color in zip(results, colors):
        ev = r["eval"]
        RocCurveDisplay.from_predictions(
            ev["y_true"], ev["y_score"], ax=ax,
            name=f"{r['name']} (AUC = {ev['auc']:.3f})",
            color=color,
        )
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="Chance")
    ax.set_title(
        f"MRF Sample Reweighting — ROC Comparison\n"
        f"(primary test set, n = {n_pairs} pairs)"
    )
    ax.set_xlabel("1 − Specificity (FPR)")
    ax.set_ylabel("Sensitivity (TPR)")
    ax.grid(True, which="both", alpha=0.3)
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="lower right")
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
    output_dir: str = "results",
    plots_dir: str = "plots",
    n_iter: int = 50,
    n_iter_oof: int = 30,
    n_splits: int = 5,
    n_boot: int = 1000,
    seed: int = 0,
    alpha: float = 2.0,
    n_jobs: int = -1,
) -> dict:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    Path(plots_dir).mkdir(parents=True, exist_ok=True)

    bmca_train = pd.read_csv(bmca_train_path)
    bmca_test = pd.read_csv(bmca_test_path)
    mrf_train = pd.read_csv(mrf_train_path)

    feature_cols = _feature_cols(bmca_train)
    X_train = bmca_train[feature_cols]
    y_train = bmca_train[LABEL_COL].values.astype(float)
    groups_train = bmca_train[GROUP_COL].values
    val_mask = (bmca_train["analysis_set"] == "primary").values

    n_primary = int(val_mask.sum())
    n_aug = int((~val_mask).sum())
    logger.info(
        f"\n{'='*60}\n"
        f"Strategy 5: MRF-Informed Sample Reweighting (Exp 14-Adapted)\n"
        f"{'='*60}\n"
        f"Training subjects: {len(bmca_train)} ({n_primary} primary, {n_aug} augmentation)\n"
        f"BMCA features: {len(feature_cols)}\n"
        f"alpha: {alpha}\n"
        f"MRF weights applied to: augmentation subjects only"
    )

    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    # ------------------------------------------------------------------
    # Stage 1A: BMCA OOF (separate seed to prevent circularity)
    # ------------------------------------------------------------------
    logger.info("\n--- Stage 1A: BMCA OOF computation (for weight derivation) ---")
    oof_seed = seed + 100
    _, _, bmca_oof = _optuna_catboost(
        X_train, y_train, groups_train, cv, val_mask,
        weights=None, n_iter=n_iter_oof, seed=oof_seed,
        label="Stage 1A BMCA OOF",
    )
    logger.info(
        f"BMCA OOF stats — "
        f"primary: mean={bmca_oof[val_mask].mean():.3f}, "
        f"aug: mean={bmca_oof[~val_mask].mean():.3f}"
    )

    # ------------------------------------------------------------------
    # Stage 1B: MRF augmentation OOF
    # ------------------------------------------------------------------
    logger.info("\n--- Stage 1B: MRF augmentation OOF computation ---")
    mrf_aug_oof_arr, aug_labels, mrf_aug_oof_auc, s1b_study = _get_mrf_aug_oof(
        mrf_train, n_iter=n_iter, n_splits=n_splits, seed=oof_seed
    )
    if mrf_aug_oof_auc < 0.60:
        logger.warning(
            f"MRF aug OOF AUC = {mrf_aug_oof_auc:.4f} < 0.60. "
            "MRF signal for augmentation is weaker than expected (Exp 12 gate)."
        )

    # Map augmentation OOF predictions to subject_id
    aug_df = mrf_train[mrf_train["analysis_set"] == "augmentation"].copy()
    mrf_aug_oof_by_subject = dict(
        zip(aug_df[SUBJECT_ID_COL].values, mrf_aug_oof_arr)
    )

    # Diagnostic: weight distribution preview
    aug_mask_train = (~val_mask)
    bmca_aug_oof = bmca_oof[aug_mask_train]
    aug_subject_ids = bmca_train.loc[aug_mask_train, SUBJECT_ID_COL].values
    aug_mrf_scores = np.array([mrf_aug_oof_by_subject.get(s, np.nan) for s in aug_subject_ids])
    valid = ~np.isnan(aug_mrf_scores)
    logger.info(
        f"\nWeight diagnostic (augmentation subjects, alpha={alpha}):\n"
        f"  MRF aug OOF AUC:   {mrf_aug_oof_auc:.4f}\n"
        f"  |BMCA - MRF| mean: {np.abs(bmca_aug_oof[valid] - aug_mrf_scores[valid]).mean():.3f}, "
        f"max: {np.abs(bmca_aug_oof[valid] - aug_mrf_scores[valid]).max():.3f}\n"
        f"  MRF agree (label=1 mean): {aug_mrf_scores[valid][aug_labels[valid] == 1].mean():.3f}, "
        f"(label=0 mean): {1 - aug_mrf_scores[valid][aug_labels[valid] == 0].mean():.3f}"
    )

    # ------------------------------------------------------------------
    # Stage 2: Three BMCA models (baseline, B, C) — same seed
    # ------------------------------------------------------------------
    variant_configs = [
        {"name": "Baseline (unweighted)", "variant": None},
        {"name": "Variant B: MRF agreement", "variant": "B"},
        {"name": "Variant C: Disagreement", "variant": "C"},
    ]

    results = []
    for cfg in variant_configs:
        v = cfg["variant"]
        name = cfg["name"]

        if v is None:
            weights = None
        else:
            weights = compute_weights(bmca_train, bmca_oof, mrf_aug_oof_by_subject, alpha, v)

        if weights is not None:
            aug_w = weights[aug_mask_train]
            pri_w = weights[val_mask]
            logger.info(
                f"\n{name} weight stats:\n"
                f"  Augmentation — mean: {aug_w.mean():.3f}, "
                f"min: {aug_w.min():.3f}, max: {aug_w.max():.3f}\n"
                f"  Primary      — mean: {pri_w.mean():.3f}, "
                f"min: {pri_w.min():.3f}, max: {pri_w.max():.3f}"
            )

        logger.info(f"\n--- Stage 2: Training {name} ---")
        study, best_model, _ = _optuna_catboost(
            X_train, y_train, groups_train, cv, val_mask,
            weights=weights, n_iter=n_iter, seed=seed,
            label=name,
        )
        ev = _evaluate_model(best_model, bmca_test, feature_cols, n_boot, seed)

        logger.info(
            f"{name}  AUC = {ev['auc']:.3f}  "
            f"95% CI [{ev['ci_low']:.3f}, {ev['ci_high']:.3f}]  "
            f"CV = {study.best_value:.4f}"
        )
        results.append({
            "name": name,
            "variant": v,
            "study": study,
            "model": best_model,
            "eval": ev,
            "weights": weights,
        })

    baseline = results[0]

    # ------------------------------------------------------------------
    # Bootstrap comparisons vs baseline
    # ------------------------------------------------------------------
    diffs = []
    for r in results[1:]:
        diff = _bootstrap_auc_diff(
            baseline["eval"]["y_true"],
            r["eval"]["y_score"],
            baseline["eval"]["y_score"],
            baseline["eval"]["groups"],
            n_boot=10_000,
            seed=seed,
        )
        diff["comparison"] = f"{r['name']} vs Baseline"
        diffs.append(diff)
        logger.info(
            f"\n{r['name']} vs Baseline: "
            f"Δ = {diff['observed_diff']:+.4f}  "
            f"95% CI [{diff['ci_low']:+.4f}, {diff['ci_high']:+.4f}]  "
            f"p(Δ ≤ 0) = {diff['p_value']:.3f}"
        )

    # ------------------------------------------------------------------
    # Save outputs
    # ------------------------------------------------------------------
    _plot_roc(results, f"{plots_dir}/mrf_sample_reweighting_roc.pdf")

    metrics_rows = []
    for r in results:
        ev = r["eval"]
        row = {
            "model": r["name"],
            "variant": r["variant"] or "A",
            "alpha": alpha if r["variant"] else 0.0,
            "auc": round(ev["auc"], 4),
            "auc_ci_low_95": round(ev["ci_low"], 4),
            "auc_ci_high_95": round(ev["ci_high"], 4),
            "best_inner_cv_auc": round(r["study"].best_value, 4),
            "n_features": len(feature_cols),
            **{f"param_{k}": v for k, v in r["study"].best_params.items()},
        }
        metrics_rows.append(row)
    pd.DataFrame(metrics_rows).to_csv(
        f"{output_dir}/mrf_sample_reweighting_evaluation.csv", index=False
    )

    pd.DataFrame(diffs).to_csv(
        f"{output_dir}/mrf_sample_reweighting_bootstrap_diff.csv", index=False
    )

    pd.DataFrame([{
        "stage1b_mrf_aug_oof_auc": round(mrf_aug_oof_auc, 4),
        "alpha": alpha,
        "n_aug_subjects": n_aug,
        "n_primary_subjects": n_primary,
    }]).to_csv(f"{output_dir}/mrf_sample_reweighting_stage1.csv", index=False)

    for r in results:
        slug = (r["variant"] or "baseline").lower()
        r["eval"]["importance"].to_csv(
            f"{output_dir}/mrf_sample_reweighting_{slug}_importance.csv", index=False
        )

    joblib.dump({
        "bmca_oof": bmca_oof,
        "mrf_aug_oof": mrf_aug_oof_arr,
        "mrf_aug_oof_auc": mrf_aug_oof_auc,
        "results": [
            {
                "name": r["name"],
                "model": r["model"],
                "study": r["study"],
                "eval": {k: v for k, v in r["eval"].items() if k != "importance"},
                "weights": r["weights"],
            }
            for r in results
        ],
        "diffs": diffs,
    }, f"{output_dir}/mrf_sample_reweighting_models.joblib")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    logger.info(
        f"\n{'='*60}\n"
        f"SUMMARY: Strategy 5 — MRF Sample Reweighting (alpha={alpha})\n"
        f"{'='*60}\n"
        f"Stage 1B MRF aug OOF AUC: {mrf_aug_oof_auc:.4f}\n"
        f"\nBMCA evaluation (primary test set):\n"
        f"  {'Model':<35} {'AUC':>8} {'95% CI':>22} {'CV AUC':>8}\n"
        f"  {'-'*73}\n"
    )
    for r in results:
        ev = r["eval"]
        logger.info(
            f"  {r['name']:<35} {ev['auc']:>8.4f} "
            f"[{ev['ci_low']:.3f}, {ev['ci_high']:.3f}] "
            f"{r['study'].best_value:>8.4f}"
        )
    for d in diffs:
        logger.info(
            f"\n  {d['comparison']}:\n"
            f"    Δ AUC = {d['observed_diff']:+.4f}  "
            f"CI [{d['ci_low']:+.4f}, {d['ci_high']:+.4f}]  "
            f"p(Δ≤0) = {d['p_value']:.3f}"
        )
    logger.info(f"{'='*60}")

    return {
        "results": results,
        "diffs": diffs,
        "mrf_aug_oof_auc": mrf_aug_oof_auc,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Strategy 5: MRF-Informed Sample Reweighting (Exp 14-Adapted)"
    )
    parser.add_argument("--bmca_train", default="data/adni_bmca_features_train.csv")
    parser.add_argument("--bmca_test", default="data/adni_bmca_features_test.csv")
    parser.add_argument("--mrf_train", default="data/adni_mrf_features_train.csv")
    parser.add_argument("--output_dir", default="results")
    parser.add_argument("--plots_dir", default="plots")
    parser.add_argument("--n_iter", type=int, default=50,
                        help="Optuna trials for Stage 2 comparison models")
    parser.add_argument("--n_iter_oof", type=int, default=30,
                        help="Optuna trials for Stage 1 OOF computation")
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--n_boot", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--alpha", type=float, default=2.0,
                        help="Weight scaling factor: w_aug = 1 + alpha * f(scores)")
    args = parser.parse_args()

    run(
        bmca_train_path=args.bmca_train,
        bmca_test_path=args.bmca_test,
        mrf_train_path=args.mrf_train,
        output_dir=args.output_dir,
        plots_dir=args.plots_dir,
        n_iter=args.n_iter,
        n_iter_oof=args.n_iter_oof,
        n_splits=args.n_splits,
        n_boot=args.n_boot,
        seed=args.seed,
        alpha=args.alpha,
    )
