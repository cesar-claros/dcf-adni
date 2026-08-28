"""Matched-pair bootstrap statistics for AUC estimates.

Promoted unchanged from ``scripts/model_strate_cv_evaluation.py`` (phase 3 of
the code reorganization). Both functions resample matched pairs (group IDs)
with replacement, never individual subjects, so the case/control pairing is
preserved in every bootstrap replicate.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score


def bootstrap_auc_ci(
    y_true: np.ndarray,
    y_score: np.ndarray,
    groups: np.ndarray,
    n_boot: int = 2000,
    seed: int = 0,
) -> tuple[float, float]:
    """Bootstrap 95% CI for AUC by resampling matched pairs with replacement."""
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


def paired_bootstrap_auc_diff(
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
