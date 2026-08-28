"""Model tuning and fitting for one training fold.

``tune_and_fit_catboost`` is the Optuna-tuned CatBoost used by the nested-CV
protocol, promoted unchanged from ``scripts/model_strate_cv_evaluation.py``
(phase 3 of the code reorganization). The single-split protocol keeps using
``dcf_adni.modeling.utils_model.train_model``, which has its own search space;
the two tuners are intentionally not unified so that converted experiments
reproduce their pre-conversion numbers exactly.
"""

from __future__ import annotations

import numpy as np
import optuna
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold


def tune_and_fit_catboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    groups_train: pd.Series,
    val_mask: np.ndarray,
    *,
    n_inner: int = 5,
    n_iter: int = 50,
    seed: int = 0,
    cv_seed: int | None = None,
    n_jobs: int = 1,
) -> tuple[CatBoostClassifier, optuna.Study]:
    """Tune CatBoost with Optuna over an inner grouped CV, then refit on all rows.

    ``val_mask`` marks the rows that count for inner-CV validation scoring
    (primary pairs in combined training mode); all rows always train.
    ``seed`` seeds the TPE sampler and the CatBoost models; ``cv_seed``
    (default ``seed``) seeds the inner CV splitter, which the nested-CV
    protocol varies per outer fold.

    Returns the refitted best model and the Optuna study.
    """
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    if cv_seed is None:
        cv_seed = seed

    inner_cv = StratifiedGroupKFold(n_splits=n_inner, shuffle=True, random_state=cv_seed)

    def objective(trial: optuna.Trial) -> float:
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

            val_scoring = val_mask[inner_val_idx]
            if val_scoring.sum() == 0 or len(np.unique(y_iv.values[val_scoring])) < 2:
                continue

            model = CatBoostClassifier(**params)
            model.fit(X_it, y_it, verbose=0)
            preds = model.predict_proba(X_iv)[:, 1]
            inner_aucs.append(roc_auc_score(y_iv.values[val_scoring], preds[val_scoring]))

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

    return final_model, study
