"""Evaluation protocols shared by the experiment scripts.

``run_nested_cv`` is the full-CV protocol promoted unchanged from
``run_cv_for_feature_set`` in ``scripts/model_strate_cv_evaluation.py``
(phase 3 of the code reorganization): outer folds rotate matched pairs through
test, an inner grouped CV tunes hyperparameters per fold, and out-of-fold
predictions accumulate over every pair.

``run_single_split`` is the legacy protocol shared by the single-split
experiment scripts (BMCA, MRF, ...): one Optuna tuning run over the fixed
training split via ``utils_model.train_model``, evaluation on the
``evaluation_eligible`` rows of the held-out test split, and a matched-pair
bootstrap CI. Kept so the early experimental record stays reproducible.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from dcf_adni.modeling.bootstrap import bootstrap_auc_ci
from dcf_adni.modeling.datasets import MatchedDataset
from dcf_adni.modeling.estimators import tune_and_fit_catboost
from dcf_adni.modeling.reporting import ExperimentResult
from dcf_adni.modeling.schema import (
    GROUP_COL,
    TRANSITION_LABEL_COL,
    evaluation_eligible,
    primary_mask,
)

logger = logging.getLogger(__name__)


def run_single_split(
    dataset: MatchedDataset,
    *,
    name: str,
    display_name: str,
    label_col: str = TRANSITION_LABEL_COL,
    n_iter: int = 50,
    n_splits: int = 5,
    n_boot: int = 1000,
    seed: int = 0,
    n_jobs: int = -1,
    gpu: bool = False,
) -> ExperimentResult:
    """Tune, fit, and evaluate one single-split experiment.

    Inner CV uses StratifiedGroupKFold(n_splits) with groups = matched pair ID,
    so both members of a matched pair always stay in the same fold. Augmentation
    pairs train in every fold but only primary rows count for validation
    scoring. Evaluation is restricted to ``evaluation_eligible == 1`` test rows
    with a matched-pair bootstrap 95% CI.
    """
    from dcf_adni.modeling.utils_model import train_model

    train_df, test_df = dataset.train, dataset.test
    feature_cols = list(dataset.feature_cols)

    for split_name, df in (("train", train_df), ("test", test_df)):
        if df[label_col].isna().any():
            raise ValueError(
                f"{split_name} split has NaN in label column {label_col!r}; "
                "choose the label column explicitly (see dcf_adni.modeling.schema)."
            )

    X_train = train_df[feature_cols]
    y_train = train_df[label_col].astype(float)
    groups_train = train_df[GROUP_COL]

    X_test = test_df[feature_cols]
    y_test = test_df[label_col].astype(float)

    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    logger.info(
        f"Training CatBoost on {display_name} features: "
        f"{len(feature_cols)} features, "
        f"{len(X_train)} train rows, {len(X_test)} test rows, "
        f"{n_iter} Optuna trials, {n_splits}-fold stratified group CV."
    )

    # Score inner CV folds on primary pairs only so the CV AUC is comparable
    # to the primary-only test AUC. Augmentation pairs are still used for
    # training within each fold; only the validation scoring is restricted.
    val_mask = primary_mask(train_df)

    study, best_model, _inner_splits = train_model(
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

    eligible = evaluation_eligible(test_df)
    X_eval = eligible[feature_cols]
    y_eval = eligible[label_col].values.astype(float)
    groups_eval = eligible[GROUP_COL].values

    y_score = best_model.predict_proba(X_eval)[:, 1]
    auc = roc_auc_score(y_eval, y_score)
    ci_low, ci_high = bootstrap_auc_ci(y_eval, y_score, groups_eval, n_boot=n_boot, seed=seed)

    logger.info(
        f"{display_name} CatBoost  AUC = {auc:.3f}  95% CI [{ci_low:.3f}, {ci_high:.3f}]"
    )

    importance_df = (
        pd.DataFrame(
            {"feature": feature_cols, "importance": best_model.get_feature_importance()}
        )
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    logger.info(
        f"\nTop 10 features by importance:\n{importance_df.head(10).to_string(index=False)}"
    )

    return ExperimentResult(
        name=name,
        display_name=display_name,
        auc=auc,
        ci_low=ci_low,
        ci_high=ci_high,
        y_true=y_eval,
        y_score=y_score,
        groups=groups_eval,
        importance=importance_df,
        study=study,
        model=best_model,
        feature_cols=feature_cols,
    )


def run_nested_cv(
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
        outer_cv.split(cv_df, y=cv_df[TRANSITION_COL], groups=cv_df[GROUP_COL])
    ):
        cv_train = cv_df.iloc[train_idx]
        cv_test = cv_df.iloc[test_idx]

        # Extra training data (augmentation for combined mode, empty otherwise)
        fold_train = pd.concat([cv_train, extra_train_df], ignore_index=True)

        X_train = fold_train[feature_cols]
        y_train = fold_train[TRANSITION_COL].astype(float)
        groups_train = fold_train[GROUP_COL]

        X_test = cv_test[feature_cols]
        y_test = cv_test[TRANSITION_COL].astype(float).values

        # In combined mode, only score on primary validation subjects within inner CV.
        # In primary_only/augmentation_only mode, score on all validation subjects.
        if training_mode == "combined":
            inner_scoring_mask = (fold_train["analysis_set"] == "primary").values
        else:
            inner_scoring_mask = np.ones(len(fold_train), dtype=bool)

        final_model, study = tune_and_fit_catboost(
            X_train,
            y_train,
            groups_train,
            inner_scoring_mask,
            n_inner=n_inner,
            n_iter=n_iter,
            seed=seed,
            cv_seed=seed + fold_idx,
            n_jobs=n_jobs,
        )

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
    y_all = cv_df[TRANSITION_COL].astype(float).values
    groups_all = cv_df[GROUP_COL].values
    oof_auc = roc_auc_score(y_all, oof_scores)
    ci_low, ci_high = bootstrap_auc_ci(y_all, oof_scores, groups_all, n_boot=2000, seed=seed)

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
