"""
Experiment 16: Multi-Task Learning with MRF Auxiliary Loss
===========================================================

Uses MRF information as an *auxiliary training objective* rather than as input
features. A shared encoder processes BMCA features; the primary head predicts
CN→MCI for all subjects; an auxiliary head predicts CI→dementia for augmentation
subjects using the encoder output concatenated with MRF features.

Architecture
------------
  BMCA features (29) → SharedEncoder → PrimaryHead → P(transition)  [all subjects]

  Augmentation subjects only (during training):
  concat(SharedEncoder_output, MRF_features(30)) → AuxHead → P(transition)

Loss = L_primary(all subjects) + λ · L_aux(augmentation subjects only)

At test time: only SharedEncoder + PrimaryHead (BMCA features only).
MRF signal enters through gradient flow from L_aux during training.

Three models compared
---------------------
A: CatBoost BMCA-only (established baseline)
B: Neural BMCA-only (λ = 0, same architecture — architecture baseline)
C: Neural multi-task (λ tuned by Optuna — tests the multi-task hypothesis)

The cleanest test of the multi-task hypothesis is C vs B (same architecture,
isolates the effect of auxiliary MRF loss). The CatBoost comparison provides
absolute performance context.

Usage::

    python model_multitask_mrf_evaluation.py
    python model_multitask_mrf_evaluation.py --n_iter 100 --max_epochs 500
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
import torch
import torch.nn as nn
from catboost import CatBoostClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import RocCurveDisplay, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

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


# =============================================================================
# Bootstrap utilities
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
# Preprocessing: per-fold imputation + scaling for neural models
# =============================================================================


def _impute_and_scale(
    X_tr: np.ndarray, X_val: np.ndarray
) -> tuple[np.ndarray, np.ndarray, SimpleImputer, StandardScaler]:
    imp = SimpleImputer(strategy="median").fit(X_tr)
    X_tr_imp = imp.transform(X_tr)
    X_val_imp = imp.transform(X_val)
    scl = StandardScaler().fit(X_tr_imp)
    X_tr_sc = np.nan_to_num(scl.transform(X_tr_imp), nan=0.0).astype(np.float32)
    X_val_sc = np.nan_to_num(scl.transform(X_val_imp), nan=0.0).astype(np.float32)
    return X_tr_sc, X_val_sc, imp, scl


# =============================================================================
# PyTorch model
# =============================================================================


class MultiTaskNet(nn.Module):
    def __init__(
        self, bmca_dim: int, mrf_dim: int, hidden_dim: int, dropout: float
    ):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(bmca_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.primary_head = nn.Linear(hidden_dim, 1)
        self.aux_head = nn.Linear(hidden_dim + mrf_dim, 1)

    def forward_primary(self, x_bmca: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x_bmca)
        return self.primary_head(h).squeeze(-1)

    def forward_aux(self, x_bmca: torch.Tensor, x_mrf: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x_bmca)
        combined = torch.cat([h, x_mrf], dim=-1)
        return self.aux_head(combined).squeeze(-1)

    @torch.no_grad()
    def predict_proba(self, x_bmca: torch.Tensor) -> np.ndarray:
        self.eval()
        logits = self.forward_primary(x_bmca)
        return torch.sigmoid(logits).cpu().numpy()


# =============================================================================
# Training loop with early stopping
# =============================================================================


def _train_multitask(
    model: MultiTaskNet,
    X_bmca: torch.Tensor,
    X_mrf: torch.Tensor,
    y: torch.Tensor,
    is_aug: torch.Tensor,
    X_bmca_val: torch.Tensor | None,
    y_val_np: np.ndarray | None,
    val_primary_mask: np.ndarray | None,
    lr: float,
    weight_decay: float,
    lambda_aux: float,
    max_epochs: int = 300,
    patience: int = 30,
    seed: int = 0,
) -> tuple[float, int]:
    """
    Train multi-task model. Returns (best_val_auc, best_epoch).
    If no validation data, trains for max_epochs and returns (0.0, max_epochs).
    """
    torch.manual_seed(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss()

    best_val_auc = -1.0
    best_state = None
    best_epoch = 0
    wait = 0

    has_val = (
        X_bmca_val is not None
        and y_val_np is not None
        and val_primary_mask is not None
        and val_primary_mask.sum() >= 4
        and len(np.unique(y_val_np[val_primary_mask])) == 2
    )

    for epoch in range(max_epochs):
        model.train()
        optimizer.zero_grad()

        logits_primary = model.forward_primary(X_bmca)
        loss_primary = criterion(logits_primary, y)

        if lambda_aux > 0 and is_aug.any():
            logits_aux = model.forward_aux(X_bmca[is_aug], X_mrf[is_aug])
            loss_aux = criterion(logits_aux, y[is_aug])
            loss = loss_primary + lambda_aux * loss_aux
        else:
            loss = loss_primary

        loss.backward()
        optimizer.step()

        if has_val:
            proba = model.predict_proba(X_bmca_val)
            val_auc = roc_auc_score(y_val_np[val_primary_mask], proba[val_primary_mask])
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                best_epoch = epoch
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    break

    if best_state is not None:
        model.load_state_dict(best_state)
    else:
        best_epoch = max_epochs - 1

    return best_val_auc, best_epoch


# =============================================================================
# CatBoost Optuna baseline
# =============================================================================


def _optuna_catboost(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    groups_train: np.ndarray,
    cv: StratifiedGroupKFold,
    val_mask: np.ndarray,
    n_iter: int,
    seed: int,
    label: str = "",
) -> tuple[optuna.Study, CatBoostClassifier, np.ndarray]:
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
            m.fit(X_train.iloc[tr_idx], y_train[tr_idx])
            primary_in_val = val_mask[val_idx]
            primary_val_idx = val_idx[primary_in_val]
            if len(primary_val_idx) < 4 or len(np.unique(y_train[primary_val_idx])) < 2:
                continue
            proba = m.predict_proba(X_train.iloc[primary_val_idx])[:, 1]
            fold_aucs.append(roc_auc_score(y_train[primary_val_idx], proba))
        return float(np.mean(fold_aucs)) if fold_aucs else 0.0

    study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_iter, show_progress_bar=True)
    logger.info(f"{label} best CV AUC: {study.best_value:.4f}")

    oof_proba = np.full(len(X_train), np.nan)
    for tr_idx, val_idx in splits:
        m = CatBoostClassifier(
            **study.best_params, random_seed=seed, verbose=0, allow_writing_files=False
        )
        m.fit(X_train.iloc[tr_idx], y_train[tr_idx])
        oof_proba[val_idx] = m.predict_proba(X_train.iloc[val_idx])[:, 1]

    best_model = CatBoostClassifier(
        **study.best_params, random_seed=seed, verbose=0, allow_writing_files=False
    )
    best_model.fit(X_train, y_train)

    return study, best_model, oof_proba


# =============================================================================
# Neural Optuna search
# =============================================================================


def _optuna_neural(
    X_bmca_raw: np.ndarray,
    X_mrf_raw: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    is_aug: np.ndarray,
    val_mask: np.ndarray,
    cv: StratifiedGroupKFold,
    n_iter: int,
    seed: int,
    multitask: bool,
    bmca_dim: int,
    mrf_dim: int,
    max_epochs: int = 300,
    patience: int = 30,
    label: str = "",
) -> tuple[optuna.Study, dict, np.ndarray]:
    splits = list(cv.split(X_bmca_raw, y, groups))
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial: optuna.Trial) -> float:
        hidden_dim = trial.suggest_categorical("hidden_dim", [32, 64, 128])
        dropout = trial.suggest_float("dropout", 0.0, 0.5)
        lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
        lambda_aux = (
            trial.suggest_float("lambda_aux", 0.01, 10.0, log=True)
            if multitask
            else 0.0
        )

        fold_aucs = []
        for tr_idx, val_idx in splits:
            X_bmca_tr_sc, X_bmca_val_sc, _, _ = _impute_and_scale(
                X_bmca_raw[tr_idx], X_bmca_raw[val_idx]
            )
            X_mrf_tr_sc, _, _, _ = _impute_and_scale(
                X_mrf_raw[tr_idx], X_mrf_raw[val_idx]
            )

            X_bmca_tr_t = torch.tensor(X_bmca_tr_sc)
            X_bmca_val_t = torch.tensor(X_bmca_val_sc)
            X_mrf_tr_t = torch.tensor(X_mrf_tr_sc)
            y_tr_t = torch.tensor(y[tr_idx], dtype=torch.float32)
            is_aug_tr_t = torch.tensor(is_aug[tr_idx], dtype=torch.bool)

            val_primary = val_mask[val_idx]
            if val_primary.sum() < 4 or len(np.unique(y[val_idx][val_primary])) < 2:
                continue

            model = MultiTaskNet(bmca_dim, mrf_dim, hidden_dim, dropout)
            val_auc, _ = _train_multitask(
                model,
                X_bmca_tr_t, X_mrf_tr_t, y_tr_t, is_aug_tr_t,
                X_bmca_val_t, y[val_idx], val_primary,
                lr=lr, weight_decay=weight_decay, lambda_aux=lambda_aux,
                max_epochs=max_epochs, patience=patience, seed=seed,
            )
            fold_aucs.append(val_auc)

        return float(np.mean(fold_aucs)) if fold_aucs else 0.0

    study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_iter, show_progress_bar=True)
    logger.info(f"{label} best CV AUC: {study.best_value:.4f} | params: {study.best_params}")

    # Recompute OOF with best params + collect stopping epochs
    best_p = study.best_params
    oof_proba = np.full(len(X_bmca_raw), np.nan)
    stop_epochs = []

    for tr_idx, val_idx in splits:
        X_bmca_tr_sc, X_bmca_val_sc, _, _ = _impute_and_scale(
            X_bmca_raw[tr_idx], X_bmca_raw[val_idx]
        )
        X_mrf_tr_sc, _, _, _ = _impute_and_scale(
            X_mrf_raw[tr_idx], X_mrf_raw[val_idx]
        )

        X_bmca_tr_t = torch.tensor(X_bmca_tr_sc)
        X_bmca_val_t = torch.tensor(X_bmca_val_sc)
        X_mrf_tr_t = torch.tensor(X_mrf_tr_sc)
        y_tr_t = torch.tensor(y[tr_idx], dtype=torch.float32)
        is_aug_tr_t = torch.tensor(is_aug[tr_idx], dtype=torch.bool)

        val_primary = val_mask[val_idx]
        model = MultiTaskNet(bmca_dim, mrf_dim, best_p["hidden_dim"], best_p["dropout"])
        _, stop_ep = _train_multitask(
            model,
            X_bmca_tr_t, X_mrf_tr_t, y_tr_t, is_aug_tr_t,
            X_bmca_val_t, y[val_idx], val_primary,
            lr=best_p["lr"], weight_decay=best_p["weight_decay"],
            lambda_aux=best_p.get("lambda_aux", 0.0),
            max_epochs=max_epochs, patience=patience, seed=seed,
        )
        stop_epochs.append(stop_ep)
        oof_proba[val_idx] = model.predict_proba(X_bmca_val_t)

    median_stop = int(np.median(stop_epochs))
    best_p["_median_stop_epoch"] = median_stop
    logger.info(f"{label} OOF stop epochs: {stop_epochs} → median={median_stop}")

    return study, best_p, oof_proba


# =============================================================================
# Train final neural model on full training data
# =============================================================================


def _train_final_neural(
    X_bmca_train_raw: np.ndarray,
    X_mrf_train_raw: np.ndarray,
    X_bmca_test_raw: np.ndarray,
    X_mrf_test_raw: np.ndarray,
    y_train: np.ndarray,
    is_aug: np.ndarray,
    params: dict,
    seed: int,
    bmca_dim: int,
    mrf_dim: int,
) -> tuple[MultiTaskNet, np.ndarray]:
    # Impute + scale: fit on full train, transform both
    X_bmca_tr_sc, X_bmca_te_sc, _, _ = _impute_and_scale(
        X_bmca_train_raw, X_bmca_test_raw
    )
    X_mrf_tr_sc, _, _, _ = _impute_and_scale(X_mrf_train_raw, X_mrf_test_raw)

    X_bmca_tr_t = torch.tensor(X_bmca_tr_sc)
    X_bmca_te_t = torch.tensor(X_bmca_te_sc)
    X_mrf_tr_t = torch.tensor(X_mrf_tr_sc)
    y_tr_t = torch.tensor(y_train, dtype=torch.float32)
    is_aug_t = torch.tensor(is_aug, dtype=torch.bool)

    model = MultiTaskNet(bmca_dim, mrf_dim, params["hidden_dim"], params["dropout"])

    # Train for median stopping epoch (no early stopping — no validation set)
    n_epochs = params.get("_median_stop_epoch", 100) + 1
    _train_multitask(
        model,
        X_bmca_tr_t, X_mrf_tr_t, y_tr_t, is_aug_t,
        X_bmca_val=None, y_val_np=None, val_primary_mask=None,
        lr=params["lr"], weight_decay=params["weight_decay"],
        lambda_aux=params.get("lambda_aux", 0.0),
        max_epochs=n_epochs, patience=n_epochs + 1,  # disable early stopping
        seed=seed,
    )

    test_proba = model.predict_proba(X_bmca_te_t)
    return model, test_proba


# =============================================================================
# Feature importance for neural models
# =============================================================================


def _neural_feature_importance(
    model: MultiTaskNet, feature_names: list[str]
) -> pd.DataFrame:
    W = model.encoder[0].weight.detach().cpu().numpy()  # (hidden_dim, input_dim)
    importance = np.linalg.norm(W, axis=0)  # L2 norm per input feature
    importance = importance / importance.sum() * 100
    return (
        pd.DataFrame({"feature": feature_names, "importance": importance})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


# =============================================================================
# Plotting
# =============================================================================


def _plot_roc(results: list[dict], output_path: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    colors = ["steelblue", "forestgreen", "firebrick"]
    for r, color in zip(results, colors):
        ev = r["eval"]
        RocCurveDisplay.from_predictions(
            ev["y_true"], ev["y_score"], ax=ax,
            name=f"{r['name']} (AUC = {ev['auc']:.3f})",
            color=color,
        )
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="Chance")
    ax.set_title("Multi-Task MRF — ROC Comparison\n(primary test set)")
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
    mrf_test_path: str = "data/adni_mrf_features_test.csv",
    output_dir: str = "results",
    plots_dir: str = "plots",
    n_iter: int = 50,
    n_splits: int = 5,
    n_boot: int = 1000,
    seed: int = 0,
    max_epochs: int = 300,
    patience: int = 30,
) -> dict:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    Path(plots_dir).mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    bmca_train = pd.read_csv(bmca_train_path)
    bmca_test = pd.read_csv(bmca_test_path)
    mrf_train = pd.read_csv(mrf_train_path)
    mrf_test = pd.read_csv(mrf_test_path)

    bmca_feature_cols = _feature_cols(bmca_train)
    mrf_feature_cols = _feature_cols(mrf_train)
    bmca_dim = len(bmca_feature_cols)
    mrf_dim = len(mrf_feature_cols)

    y_train = bmca_train[LABEL_COL].values.astype(np.float32)
    groups_train = bmca_train[GROUP_COL].values
    val_mask = (bmca_train["analysis_set"] == "primary").values
    is_aug = (bmca_train["analysis_set"] == "augmentation").values

    # Raw numpy arrays for neural models
    X_bmca_train_raw = bmca_train[bmca_feature_cols].values.astype(np.float32)
    X_bmca_test_raw = bmca_test[bmca_feature_cols].values.astype(np.float32)
    X_mrf_train_raw = mrf_train[mrf_feature_cols].values.astype(np.float32)
    X_mrf_test_raw = mrf_test[mrf_feature_cols].values.astype(np.float32)

    # Test set evaluation targets
    eligible_test = _evaluation_eligible(bmca_test)
    y_test = eligible_test[LABEL_COL].values.astype(np.float32)
    groups_test = eligible_test[GROUP_COL].values
    eligible_idx = eligible_test.index

    n_primary = int(val_mask.sum())
    n_aug = int(is_aug.sum())
    logger.info(
        f"\n{'='*60}\n"
        f"Experiment 16: Multi-Task Learning with MRF Auxiliary Loss\n"
        f"{'='*60}\n"
        f"Training: {len(bmca_train)} subjects ({n_primary} primary, {n_aug} augmentation)\n"
        f"Test: {len(eligible_test)} subjects ({int(y_test.sum())} transition)\n"
        f"BMCA features: {bmca_dim} | MRF features: {mrf_dim}\n"
        f"Optuna trials: {n_iter} | Max epochs: {max_epochs} | Patience: {patience}"
    )

    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    all_results = []

    # ==================================================================
    # Model A: CatBoost BMCA-only (reference baseline)
    # ==================================================================
    logger.info("\n--- Model A: CatBoost BMCA-only ---")
    cb_study, cb_model, cb_oof = _optuna_catboost(
        bmca_train[bmca_feature_cols], y_train, groups_train,
        cv, val_mask, n_iter, seed, label="CatBoost BMCA-only",
    )
    cb_test_proba = cb_model.predict_proba(eligible_test[bmca_feature_cols])[:, 1]
    cb_auc = roc_auc_score(y_test, cb_test_proba)
    cb_ci = _bootstrap_auc(y_test, cb_test_proba, groups_test, n_boot, seed)
    cb_imp = (
        pd.DataFrame({
            "feature": bmca_feature_cols,
            "importance": cb_model.get_feature_importance(),
        })
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    logger.info(
        f"CatBoost BMCA-only: AUC={cb_auc:.4f} [{cb_ci[0]:.3f}, {cb_ci[1]:.3f}] "
        f"CV={cb_study.best_value:.4f}"
    )

    all_results.append({
        "name": "CatBoost BMCA-only",
        "eval": {
            "auc": cb_auc, "ci_low": cb_ci[0], "ci_high": cb_ci[1],
            "y_true": y_test, "y_score": cb_test_proba, "groups": groups_test,
        },
        "importance": cb_imp,
        "cv_auc": cb_study.best_value,
        "params": cb_study.best_params,
        "lambda_aux": None,
    })

    # ==================================================================
    # Model B: Neural BMCA-only (lambda=0)
    # ==================================================================
    logger.info("\n--- Model B: Neural BMCA-only (λ=0) ---")
    nn_bmca_study, nn_bmca_params, nn_bmca_oof = _optuna_neural(
        X_bmca_train_raw, X_mrf_train_raw, y_train, groups_train,
        is_aug, val_mask, cv, n_iter, seed,
        multitask=False, bmca_dim=bmca_dim, mrf_dim=mrf_dim,
        max_epochs=max_epochs, patience=patience,
        label="Neural BMCA-only",
    )
    nn_bmca_model, nn_bmca_test_proba_full = _train_final_neural(
        X_bmca_train_raw, X_mrf_train_raw,
        X_bmca_test_raw, X_mrf_test_raw,
        y_train, is_aug, nn_bmca_params, seed, bmca_dim, mrf_dim,
    )
    nn_bmca_test_proba = nn_bmca_test_proba_full[eligible_idx]
    nn_bmca_auc = roc_auc_score(y_test, nn_bmca_test_proba)
    nn_bmca_ci = _bootstrap_auc(y_test, nn_bmca_test_proba, groups_test, n_boot, seed)
    nn_bmca_imp = _neural_feature_importance(nn_bmca_model, bmca_feature_cols)
    logger.info(
        f"Neural BMCA-only: AUC={nn_bmca_auc:.4f} [{nn_bmca_ci[0]:.3f}, {nn_bmca_ci[1]:.3f}] "
        f"CV={nn_bmca_study.best_value:.4f}"
    )

    all_results.append({
        "name": "Neural BMCA-only",
        "eval": {
            "auc": nn_bmca_auc, "ci_low": nn_bmca_ci[0], "ci_high": nn_bmca_ci[1],
            "y_true": y_test, "y_score": nn_bmca_test_proba, "groups": groups_test,
        },
        "importance": nn_bmca_imp,
        "cv_auc": nn_bmca_study.best_value,
        "params": nn_bmca_params,
        "lambda_aux": 0.0,
    })

    # ==================================================================
    # Model C: Neural multi-task (lambda tuned)
    # ==================================================================
    logger.info("\n--- Model C: Neural multi-task (λ tuned) ---")
    nn_mt_study, nn_mt_params, nn_mt_oof = _optuna_neural(
        X_bmca_train_raw, X_mrf_train_raw, y_train, groups_train,
        is_aug, val_mask, cv, n_iter, seed,
        multitask=True, bmca_dim=bmca_dim, mrf_dim=mrf_dim,
        max_epochs=max_epochs, patience=patience,
        label="Neural multi-task",
    )
    nn_mt_model, nn_mt_test_proba_full = _train_final_neural(
        X_bmca_train_raw, X_mrf_train_raw,
        X_bmca_test_raw, X_mrf_test_raw,
        y_train, is_aug, nn_mt_params, seed, bmca_dim, mrf_dim,
    )
    nn_mt_test_proba = nn_mt_test_proba_full[eligible_idx]
    nn_mt_auc = roc_auc_score(y_test, nn_mt_test_proba)
    nn_mt_ci = _bootstrap_auc(y_test, nn_mt_test_proba, groups_test, n_boot, seed)
    nn_mt_imp = _neural_feature_importance(nn_mt_model, bmca_feature_cols)
    best_lambda = nn_mt_params.get("lambda_aux", 0.0)
    logger.info(
        f"Neural multi-task: AUC={nn_mt_auc:.4f} [{nn_mt_ci[0]:.3f}, {nn_mt_ci[1]:.3f}] "
        f"CV={nn_mt_study.best_value:.4f} | λ_aux={best_lambda:.4f}"
    )

    all_results.append({
        "name": "Neural multi-task",
        "eval": {
            "auc": nn_mt_auc, "ci_low": nn_mt_ci[0], "ci_high": nn_mt_ci[1],
            "y_true": y_test, "y_score": nn_mt_test_proba, "groups": groups_test,
        },
        "importance": nn_mt_imp,
        "cv_auc": nn_mt_study.best_value,
        "params": nn_mt_params,
        "lambda_aux": best_lambda,
    })

    # ==================================================================
    # Bootstrap comparisons
    # ==================================================================
    comparisons = [
        ("Neural BMCA-only vs CatBoost", nn_bmca_test_proba, cb_test_proba),
        ("Neural multi-task vs CatBoost", nn_mt_test_proba, cb_test_proba),
        ("Neural multi-task vs Neural BMCA-only", nn_mt_test_proba, nn_bmca_test_proba),
    ]
    diffs = []
    for comp_name, score_a, score_b in comparisons:
        d = _bootstrap_auc_diff(y_test, score_a, score_b, groups_test, 10_000, seed)
        d["comparison"] = comp_name
        diffs.append(d)
        logger.info(
            f"\n{comp_name}: Δ={d['observed_diff']:+.4f} "
            f"CI [{d['ci_low']:+.4f}, {d['ci_high']:+.4f}] "
            f"p(Δ≤0)={d['p_value']:.3f}"
        )

    # ==================================================================
    # Save outputs
    # ==================================================================
    _plot_roc(all_results, f"{plots_dir}/multitask_mrf_roc.pdf")

    # Evaluation CSV
    eval_rows = []
    for r in all_results:
        ev = r["eval"]
        row = {
            "model": r["name"],
            "auc": round(ev["auc"], 4),
            "auc_ci_low_95": round(ev["ci_low"], 4),
            "auc_ci_high_95": round(ev["ci_high"], 4),
            "best_inner_cv_auc": round(r["cv_auc"], 4),
            "n_features": bmca_dim,
            "lambda_aux": r["lambda_aux"],
        }
        for k, v in r["params"].items():
            if not k.startswith("_"):
                row[f"param_{k}"] = v
        eval_rows.append(row)
    pd.DataFrame(eval_rows).to_csv(
        f"{output_dir}/multitask_mrf_evaluation.csv", index=False
    )

    # Bootstrap CSV
    pd.DataFrame(diffs).to_csv(
        f"{output_dir}/multitask_mrf_bootstrap_diff.csv", index=False
    )

    # Importance CSVs
    slugs = ["catboost", "neural_bmca", "neural_multitask"]
    for r, slug in zip(all_results, slugs):
        r["importance"].to_csv(
            f"{output_dir}/multitask_mrf_{slug}_importance.csv", index=False
        )

    # Training curves: final multi-task model
    # Re-train to capture history
    X_bmca_tr_sc, _, _, _ = _impute_and_scale(X_bmca_train_raw, X_bmca_test_raw)
    X_mrf_tr_sc, _, _, _ = _impute_and_scale(X_mrf_train_raw, X_mrf_test_raw)
    X_bmca_tr_t = torch.tensor(X_bmca_tr_sc)
    X_mrf_tr_t = torch.tensor(X_mrf_tr_sc)
    y_tr_t = torch.tensor(y_train, dtype=torch.float32)
    is_aug_t = torch.tensor(is_aug, dtype=torch.bool)

    curve_model = MultiTaskNet(bmca_dim, mrf_dim, nn_mt_params["hidden_dim"], nn_mt_params["dropout"])
    n_ep = nn_mt_params.get("_median_stop_epoch", 100) + 1
    criterion = nn.BCEWithLogitsLoss()
    torch.manual_seed(seed)
    optimizer = torch.optim.Adam(
        curve_model.parameters(), lr=nn_mt_params["lr"],
        weight_decay=nn_mt_params["weight_decay"],
    )
    curve_rows = []
    for epoch in range(n_ep):
        curve_model.train()
        optimizer.zero_grad()
        lp = criterion(curve_model.forward_primary(X_bmca_tr_t), y_tr_t)
        la = criterion(
            curve_model.forward_aux(X_bmca_tr_t[is_aug_t], X_mrf_tr_t[is_aug_t]),
            y_tr_t[is_aug_t],
        )
        loss = lp + best_lambda * la
        loss.backward()
        optimizer.step()
        curve_rows.append({
            "epoch": epoch,
            "loss_primary": round(lp.item(), 6),
            "loss_aux": round(la.item(), 6),
            "loss_total": round(loss.item(), 6),
            "lambda_aux": round(best_lambda, 4),
        })
    pd.DataFrame(curve_rows).to_csv(
        f"{output_dir}/multitask_mrf_training_curves.csv", index=False
    )

    # Save models
    joblib.dump({
        "catboost_model": cb_model,
        "neural_bmca_params": nn_bmca_params,
        "neural_multitask_params": nn_mt_params,
        "cb_oof": cb_oof,
        "nn_bmca_oof": nn_bmca_oof,
        "nn_mt_oof": nn_mt_oof,
    }, f"{output_dir}/multitask_mrf_models.joblib")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    logger.info(
        f"\n{'='*60}\n"
        f"SUMMARY: Experiment 16 — Multi-Task Learning\n"
        f"{'='*60}\n"
        f"  {'Model':<30} {'AUC':>8} {'95% CI':>22} {'CV AUC':>8}\n"
        f"  {'-'*68}"
    )
    for r in all_results:
        ev = r["eval"]
        logger.info(
            f"  {r['name']:<30} {ev['auc']:>8.4f} "
            f"[{ev['ci_low']:.3f}, {ev['ci_high']:.3f}] "
            f"{r['cv_auc']:>8.4f}"
        )
    logger.info(f"\n  Best λ_aux: {best_lambda:.4f}")
    for d in diffs:
        logger.info(
            f"\n  {d['comparison']}:\n"
            f"    Δ AUC = {d['observed_diff']:+.4f}  "
            f"CI [{d['ci_low']:+.4f}, {d['ci_high']:+.4f}]  "
            f"p(Δ≤0) = {d['p_value']:.3f}"
        )
    logger.info(f"{'='*60}")

    return {"results": all_results, "diffs": diffs}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Experiment 16: Multi-Task Learning with MRF Auxiliary Objective"
    )
    parser.add_argument("--bmca_train", default="data/adni_bmca_features_train.csv")
    parser.add_argument("--bmca_test", default="data/adni_bmca_features_test.csv")
    parser.add_argument("--mrf_train", default="data/adni_mrf_features_train.csv")
    parser.add_argument("--mrf_test", default="data/adni_mrf_features_test.csv")
    parser.add_argument("--output_dir", default="results")
    parser.add_argument("--plots_dir", default="plots")
    parser.add_argument("--n_iter", type=int, default=50,
                        help="Optuna trials per model")
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--n_boot", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_epochs", type=int, default=300,
                        help="Max training epochs per trial")
    parser.add_argument("--patience", type=int, default=30,
                        help="Early stopping patience (epochs)")
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
        max_epochs=args.max_epochs,
        patience=args.patience,
    )
