"""
Hypothesis Testing Pipeline
============================

CLI-driven script for testing specific hypotheses about model combinations
using the same nested cross-validation approach as ``model_training.py``.

Available Hypotheses
--------------------
- **h1_stacking**: Test incremental value of MRF as a block via stacking.
  Trains BIOM-only and MRF-only models, generates out-of-fold predictions,
  then fits a second-stage Logistic Regression on both scores.

Usage::

    python hypothesis_testing.py --hypothesis h1_stacking \\
        --seed_split 0 --model_name catboost
"""

import argparse
import logging
from collections import Counter
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict

from src.utils_model import (
    WoETransformer,
    _encode_categoricals,
    create_model,
    deduplicate_rule_matrix,
    extract_rf_rule_matrix,
    FeatureImportanceScorer,
    feature_engineering,
    filter_rules_by_support,
    forward_select_rules_by_auc,
    score_rules_with_base_predictions,
    train_model,
)

logging.basicConfig(level=logging.INFO, format='%(name)s — %(message)s')
logger = logging.getLogger(__name__)

_CONFIG_PATH = Path(__file__).parent / 'configs' / 'model_training.yaml'


# =============================================================================
# Helpers for pre-loaded CV splits
# =============================================================================

class _PrecomputedSplitter:
    """Wraps a pre-computed list of (train_idx, val_idx) index pairs.

    Provides the minimal sklearn CV splitter interface required by
    ``train_model`` and ``cross_val_predict``.  All arguments to ``split``
    are ignored; the stored splits are returned as-is.
    """

    def __init__(self, splits):
        self._splits = list(splits)
        self.n_splits = len(self._splits)

    def split(self, X=None, y=None, groups=None):
        return iter(self._splits)

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits


def _load_splits(splits_file):
    """Load pre-computed outer fold splits from a hypothesis results file.

    Accepts joblib output from any hypothesis (h1, h2, or h3).  The file
    must contain either a ``'fold_results'`` key (H2/H3 format) or a
    ``'fold_splits'`` key (H1 format).  Each fold entry must contain
    ``'train_index'``, ``'test_index'``, and ``'biom_inner_splits'``.

    Args:
        splits_file (str or Path): Path to the ``.joblib`` results file.

    Returns:
        list[dict]: One dict per fold with keys ``'fold'``, ``'train_index'``,
            ``'test_index'``, and ``'inner_splits'``.

    Raises:
        KeyError: If the expected keys are missing from the file.
    """
    data = joblib.load(splits_file)
    fold_list = data.get('fold_results') or data.get('fold_splits')
    if fold_list is None:
        raise KeyError(
            f"Splits file '{splits_file}' contains neither 'fold_results' nor "
            f"'fold_splits'. Available keys: {list(data.keys())}"
        )
    result = []
    for fr in fold_list:
        inner = fr.get('biom_inner_splits') or fr.get('mrf_inner_splits')
        if inner is None:
            raise KeyError(
                f"Fold {fr.get('fold')} entry has no 'biom_inner_splits' or "
                "'mrf_inner_splits' key."
            )
        result.append({
            'fold': fr['fold'],
            'train_index': fr['train_index'],
            'test_index': fr['test_index'],
            'inner_splits': inner,
        })
    logger.info(
        f"Loaded {len(result)} pre-computed fold splits from '{splits_file}'."
    )
    return result


def _load_config(config_path=None):
    """Load pipeline configuration from YAML."""
    path = Path(config_path) if config_path else _CONFIG_PATH
    with open(path, 'r') as f:
        return yaml.safe_load(f)


# =============================================================================
# Hypothesis 1: MRF Incremental Value via Stacking
# =============================================================================

def _filter_features(df, mode):
    """
    Filter columns based on feature mode.

    Args:
        df (pd.DataFrame): DataFrame with raw + WoE columns.
        mode (str): One of ``'raw'``, ``'woe'``, ``'raw_woe'``.

    Returns:
        pd.DataFrame: Filtered DataFrame.
    """
    if mode == 'raw':
        return df[[c for c in df.columns if not c.endswith('_WOE')]]
    elif mode == 'woe':
        return df[[c for c in df.columns if c.endswith('_WOE')]]
    else:  # raw_woe
        return df


def run_h1_stacking(model_name, seed_split=None, feature_mode_biom='raw_woe',
                    feature_mode_mrf='raw_woe', config_path=None,
                    gpu=False, n_jobs=-1, splits_file=None, **kwargs):
    """
    Test incremental value of MRF features via stacking.

    For each outer fold:
      1. WoE transform (fresh per fold)
      2. Train BIOM-only model (Optuna inner CV) → OOF probabilities
      3. Train MRF-only model (Optuna inner CV) → OOF probabilities
      4. Fit Logistic Regression on ``[biom_oof, mrf_oof]``
      5. Predict on outer test set with first-stage models → stack → final
      6. Compare BIOM-only AUC vs MRF-only AUC vs Stacked AUC

    Args:
        model_name (str): Base model type (``'catboost'``, ``'xgboost'``, ``'rf'``).
        seed_split (int or None): Seed for outer StratifiedGroupKFold.
            Mutually exclusive with ``splits_file``.
        feature_mode_biom (str): Features to use for BIOM (raw, woe, raw_woe).
        feature_mode_mrf (str): Features to use for MRF (raw, woe, raw_woe).
        config_path (str or None): Path to YAML config.
        gpu (bool): Enable GPU training for CatBoost.
        n_jobs (int): Number of parallel jobs.
        splits_file (str or None): Path to a joblib results file whose
            pre-computed outer fold indices and inner splits are reused
            verbatim.  When provided ``seed_split`` is used only for
            output-file naming (and may be ``None``).
    """
    cfg = _load_config(config_path)
    pipeline_cfg = cfg.get('pipeline', {})
    n_iter = pipeline_cfg.get('n_iter', 50)
    n_splits = pipeline_cfg.get('n_splits', 5)
    seed_cv = pipeline_cfg.get('seed_cv', 0)
    seed_rf = pipeline_cfg.get('seed_rf', 0)
    seed_bayes = pipeline_cfg.get('seed_bayes', 0)

    woe_dict_biom = cfg['woe_dict_biom']
    woe_dict_mrf = cfg['woe_dict_mrf']
    categorical_biom = cfg['categorical_biom']
    categorical_mrf = cfg['categorical_mrf']

    logger.info(f"Feature mode BIOM: {feature_mode_biom}")
    logger.info(f"Feature mode MRF:  {feature_mode_mrf}")

    # Inner CV splitter (used by Optuna for hyperparameter tuning)
    cv_inner = StratifiedGroupKFold(
        n_splits=n_splits, shuffle=True, random_state=seed_cv,
    )

    # ----- Load data -----
    logger.info("Loading data...")
    joint_dataset_df = pd.read_csv(
        'data/joint_dataset.csv', index_col=0,
    ).set_index('subject_id')
    dataset_df = feature_engineering(joint_dataset_df)

    X_all = dataset_df.drop(['transition'], axis='columns')
    y_all = dataset_df['transition']
    groups_all = dataset_df['group']

    fold_aucs = {'biom': [], 'mrf': [], 'stacked': []}
    fold_splits = []

    # Build the outer iteration source and the output label
    if splits_file is not None:
        _preloaded = _load_splits(splits_file)
        n_splits = len(_preloaded)
        split_label = Path(splits_file).stem
        _outer_iter = (
            (f['train_index'], f['test_index'], f['inner_splits'])
            for f in _preloaded
        )
    else:
        outer_cv = StratifiedGroupKFold(
            n_splits=n_splits, shuffle=True, random_state=seed_split,
        )
        split_label = str(seed_split)
        _outer_iter = (
            (train_idx, test_idx, None)
            for train_idx, test_idx in outer_cv.split(X_all, y_all, groups_all)
        )

    for k, (train_index, test_index, _preloaded_inner) in enumerate(tqdm(
        _outer_iter,
        total=n_splits, desc='Outer CV folds', unit='fold',
    )):
        logger.info(f"\n{'='*60}")
        logger.info(f"Outer fold {k+1}/{n_splits}: "
                    f"Train={len(train_index)}, Test={len(test_index)}")
        logger.info(f"{'='*60}")

        # Use pre-loaded inner splits when available, otherwise fall back to
        # the freshly-generated StratifiedGroupKFold splitter.
        cv_inner_fold = (
            _PrecomputedSplitter(_preloaded_inner)
            if _preloaded_inner is not None
            else cv_inner
        )

        # Fresh WoE transformers per fold (no data leakage to outer test)
        woe_biom = WoETransformer(woe_dict_biom, categorical_biom)
        woe_mrf = WoETransformer(woe_dict_mrf, categorical_mrf)

        # ----- Step 1: WoE Transformation -----
        X_biom_train, X_biom_test, _, _, _ = \
            woe_biom.fit_transform_split(dataset_df, train_index, test_index)
        X_mrf_train, X_mrf_test, y_train, y_test, _ = \
            woe_mrf.fit_transform_split(dataset_df, train_index, test_index)

        # Filter features based on mode
        X_biom_train = _filter_features(X_biom_train, feature_mode_biom)
        X_biom_test = _filter_features(X_biom_test, feature_mode_biom)
        X_mrf_train = _filter_features(X_mrf_train, feature_mode_mrf)
        X_mrf_test = _filter_features(X_mrf_test, feature_mode_mrf)

        logger.info(f"  BIOM features: {X_biom_train.shape[1]}, "
                    f"MRF features: {X_mrf_train.shape[1]}")

        groups_train = dataset_df.iloc[train_index]['group']

        # Cast categorical variables: str → category
        cat_biom = [c for c in categorical_biom if c in X_biom_train.columns]
        cat_mrf = [c for c in categorical_mrf if c in X_mrf_train.columns]
        for df in [X_biom_train, X_biom_test]:
            if cat_biom:
                df[cat_biom] = df[cat_biom].astype(str).astype('category')
        for df in [X_mrf_train, X_mrf_test]:
            if cat_mrf:
                df[cat_mrf] = df[cat_mrf].astype(str).astype('category')

        # ----- Step 2: Train BIOM model + OOF predictions -----
        logger.info("Training BIOM model...")
        biom_study, biom_model, biom_inner_splits = train_model(
            X_biom_train, y_train, X_biom_test, y_test,
            model=model_name, seed_rf=seed_rf,
            seed_bayes=seed_bayes + 20, n_iter=n_iter,
            cv=cv_inner_fold, groups=groups_train,
            cat_vars=cat_biom or None, n_jobs=n_jobs, gpu=gpu,
        )
        # OOF predictions on training set (for stacking features)
        # Encode categoricals for the trained model
        X_biom_train_enc = _encode_categoricals(X_biom_train, model_name)
        X_biom_test_enc = _encode_categoricals(X_biom_test, model_name)
        biom_oof = cross_val_predict(
            biom_model, X_biom_train_enc, y_train,
            cv=cv_inner_fold, groups=groups_train,
            method='predict_proba', n_jobs=n_jobs,
        )[:, 1]
        # Test set predictions
        biom_test_proba = biom_model.predict_proba(X_biom_test_enc)[:, 1]
        biom_test_auc = roc_auc_score(y_test, biom_test_proba)
        logger.info(f"  BIOM test AUC: {biom_test_auc:.4f}")

        # ----- Step 3: Train MRF model + OOF predictions -----
        logger.info("Training MRF model...")
        mrf_study, mrf_model, mrf_inner_splits = train_model(
            X_mrf_train, y_train, X_mrf_test, y_test,
            model=model_name, seed_rf=seed_rf,
            seed_bayes=seed_bayes + 30, n_iter=n_iter,
            cv=cv_inner_fold, groups=groups_train,
            cat_vars=cat_mrf or None, n_jobs=n_jobs, gpu=gpu,
        )
        X_mrf_train_enc = _encode_categoricals(X_mrf_train, model_name)
        X_mrf_test_enc = _encode_categoricals(X_mrf_test, model_name)
        mrf_oof = cross_val_predict(
            mrf_model, X_mrf_train_enc, y_train,
            cv=cv_inner_fold, groups=groups_train,
            method='predict_proba', n_jobs=n_jobs,
        )[:, 1]
        mrf_test_proba = mrf_model.predict_proba(X_mrf_test_enc)[:, 1]
        mrf_test_auc = roc_auc_score(y_test, mrf_test_proba)
        logger.info(f"  MRF test AUC: {mrf_test_auc:.4f}")

        # ----- Step 4: Second-stage stacking model -----
        logger.info("Training stacking model...")
        X_stack_train = np.column_stack([biom_oof, mrf_oof])
        X_stack_test = np.column_stack([biom_test_proba, mrf_test_proba])

        stacker = LogisticRegression(random_state=seed_rf, max_iter=1000)
        stacker.fit(X_stack_train, y_train)

        stacked_test_proba = stacker.predict_proba(X_stack_test)[:, 1]
        stacked_test_auc = roc_auc_score(y_test, stacked_test_proba)
        logger.info(f"  Stacked test AUC: {stacked_test_auc:.4f}")

        # Log stacking coefficients
        biom_coef, mrf_coef = stacker.coef_[0]
        logger.info(f"  Stacking weights: BIOM={biom_coef:.3f}, MRF={mrf_coef:.3f}")

        fold_aucs['biom'].append(biom_test_auc)
        fold_aucs['mrf'].append(mrf_test_auc)
        fold_aucs['stacked'].append(stacked_test_auc)

        fold_splits.append({
            'fold': k,
            'train_index': train_index,
            'test_index': test_index,
            'biom_inner_splits': biom_inner_splits,
            'mrf_inner_splits': mrf_inner_splits,
        })

    # ----- Summary -----
    logger.info(f"\n{'='*60}")
    logger.info("H1 Stacking — Summary across outer folds")
    logger.info(f"{'='*60}")
    for name in ['biom', 'mrf', 'stacked']:
        aucs = fold_aucs[name]
        logger.info(f"  {name:>8s}: AUC = {np.mean(aucs):.3f} ± {np.std(aucs):.3f}")

    # ----- Boxplot -----
    prefix = f'plots/h1_stacking_{model_name}_B{feature_mode_biom}_M{feature_mode_mrf}_seed_{split_label}'
    auc_rows = []
    for name in ['biom', 'mrf', 'stacked']:
        for auc in fold_aucs[name]:
            auc_rows.append({'model': name.upper(), 'AUC': auc})

    auc_df = pd.DataFrame(auc_rows)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.boxplot(data=auc_df, x='model', y='AUC',
                color='.8', linecolor='#137', linewidth=0.75, ax=ax)
    sns.stripplot(data=auc_df, x='model', y='AUC', ax=ax, size=8, jitter=False)
    ax.set_title(f'H1: MRF Incremental Value (Stacking, {n_splits}-fold)')
    ax.set_ylabel('Test AUC (outer fold)')
    ax.set_xlabel('')
    fig.savefig(f'{prefix}_boxplot.pdf', bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Plot saved to {prefix}_boxplot.pdf")

    # ----- Save results -----
    results_path = f'results/h1_stacking_{model_name}_B{feature_mode_biom}_M{feature_mode_mrf}_seed_{split_label}.joblib'
    joblib.dump({
        'hypothesis': 'h1_stacking',
        'model_name': model_name,
        'feature_mode_biom': feature_mode_biom,
        'feature_mode_mrf': feature_mode_mrf,
        'seed_split': seed_split,
        'splits_file': splits_file,
        'split_label': split_label,
        'n_folds': n_splits,
        'fold_aucs': fold_aucs,
        'fold_splits': fold_splits,
    }, results_path)
    logger.info(f"Results saved to {results_path}")


# =============================================================================
# Hypothesis 2: Conditional MRF Feature Selection (Forward Selection)
# =============================================================================

def run_h2_forward_selection(model_name, seed_split=None,
                             feature_mode_biom='raw_woe',
                             feature_mode_mrf='raw_woe',
                             config_path=None, gpu=False, n_jobs=-1,
                             auc_threshold=0.005, splits_file=None, **kwargs):
    """
    Test which specific MRF features add incremental value over BIOM.

    Uses greedy forward selection: starting from a BIOM-only model, adds
    one MRF feature at a time (the one with the highest OOF AUC gain),
    stopping when no candidate improves AUC by more than ``auc_threshold``.

    The BIOM model's best hyperparameters (from Optuna) are reused during
    selection to keep it computationally feasible.

    Args:
        model_name (str): Base model type.
        seed_split (int or None): Seed for outer StratifiedGroupKFold.
            Mutually exclusive with ``splits_file``.
        feature_mode_biom (str): Features to use for BIOM (raw, woe, raw_woe).
        feature_mode_mrf (str): Features to use for MRF (raw, woe, raw_woe).
        config_path (str or None): Path to YAML config.
        gpu (bool): Enable GPU training for CatBoost.
        n_jobs (int): Number of parallel jobs.
        auc_threshold (float): Minimum AUC gain to keep a feature.
        splits_file (str or None): Path to a joblib results file whose
            pre-computed outer fold indices and inner splits are reused
            verbatim.  When provided ``seed_split`` is used only for
            output-file naming (and may be ``None``).
    """
    cfg = _load_config(config_path)
    pipeline_cfg = cfg.get('pipeline', {})
    n_iter = pipeline_cfg.get('n_iter', 50)
    n_splits = pipeline_cfg.get('n_splits', 5)
    seed_cv = pipeline_cfg.get('seed_cv', 0)
    seed_rf = pipeline_cfg.get('seed_rf', 0)
    seed_bayes = pipeline_cfg.get('seed_bayes', 0)

    woe_dict_biom = cfg['woe_dict_biom']
    woe_dict_mrf = cfg['woe_dict_mrf']
    categorical_biom = cfg['categorical_biom']
    categorical_mrf = cfg['categorical_mrf']

    logger.info(f"Feature mode BIOM: {feature_mode_biom}")
    logger.info(f"Feature mode MRF:  {feature_mode_mrf}")
    logger.info(f"AUC threshold: {auc_threshold}")

    cv_inner = StratifiedGroupKFold(
        n_splits=n_splits, shuffle=True, random_state=seed_cv,
    )

    # ----- Load data -----
    logger.info("Loading data...")
    joint_dataset_df = pd.read_csv(
        'data/joint_dataset.csv', index_col=0,
    ).set_index('subject_id')
    dataset_df = feature_engineering(joint_dataset_df)

    X_all = dataset_df.drop(['transition'], axis='columns')
    y_all = dataset_df['transition']
    groups_all = dataset_df['group']

    fold_results = []

    # Build outer iteration source and output label
    if splits_file is not None:
        _preloaded = _load_splits(splits_file)
        n_splits = len(_preloaded)
        split_label = Path(splits_file).stem
        _outer_iter = (
            (f['train_index'], f['test_index'], f['inner_splits'])
            for f in _preloaded
        )
    else:
        outer_cv = StratifiedGroupKFold(
            n_splits=n_splits, shuffle=True, random_state=seed_split,
        )
        split_label = str(seed_split)
        _outer_iter = (
            (train_idx, test_idx, None)
            for train_idx, test_idx in outer_cv.split(X_all, y_all, groups_all)
        )

    for k, (train_index, test_index, _preloaded_inner) in enumerate(tqdm(
        _outer_iter,
        total=n_splits, desc='Outer CV folds', unit='fold',
    )):
        logger.info(f"\n{'='*60}")
        logger.info(f"Outer fold {k+1}/{n_splits}")
        logger.info(f"{'='*60}")

        cv_inner_fold = (
            _PrecomputedSplitter(_preloaded_inner)
            if _preloaded_inner is not None
            else cv_inner
        )

        # Fresh WoE per fold
        woe_biom = WoETransformer(woe_dict_biom, categorical_biom)
        woe_mrf = WoETransformer(woe_dict_mrf, categorical_mrf)

        X_biom_train, X_biom_test, _, _, _ = \
            woe_biom.fit_transform_split(dataset_df, train_index, test_index)
        X_mrf_train, X_mrf_test, y_train, y_test, _ = \
            woe_mrf.fit_transform_split(dataset_df, train_index, test_index)

        # Filter by feature mode
        X_biom_train = _filter_features(X_biom_train, feature_mode_biom)
        X_biom_test = _filter_features(X_biom_test, feature_mode_biom)
        X_mrf_train = _filter_features(X_mrf_train, feature_mode_mrf)
        X_mrf_test = _filter_features(X_mrf_test, feature_mode_mrf)

        groups_train = dataset_df.iloc[train_index]['group']

        # Remove overlapping columns (e.g., subject_age in both)
        overlap = set(X_biom_train.columns) & set(X_mrf_train.columns)
        X_mrf_train = X_mrf_train.drop(columns=list(overlap))
        X_mrf_test = X_mrf_test.drop(columns=list(overlap))

        # Cast categoricals
        cat_biom = [c for c in categorical_biom if c in X_biom_train.columns]
        cat_mrf = [c for c in categorical_mrf if c in X_mrf_train.columns]
        for df in [X_biom_train, X_biom_test]:
            if cat_biom:
                df[cat_biom] = df[cat_biom].astype(str).astype('category')
        for df in [X_mrf_train, X_mrf_test]:
            if cat_mrf:
                df[cat_mrf] = df[cat_mrf].astype(str).astype('category')

        # ----- Step 1: Train BIOM baseline -----
        logger.info("Training BIOM baseline model...")
        biom_study, biom_model, biom_inner_splits = train_model(
            X_biom_train, y_train, X_biom_test, y_test,
            model=model_name, seed_rf=seed_rf,
            seed_bayes=seed_bayes + 20, n_iter=n_iter,
            cv=cv_inner_fold, groups=groups_train,
            cat_vars=cat_biom or None, n_jobs=n_jobs, gpu=gpu,
        )
        best_params = biom_study.best_params

        # Encode for predictions
        X_biom_train_enc = _encode_categoricals(X_biom_train, model_name)
        X_biom_test_enc = _encode_categoricals(X_biom_test, model_name)

        # BIOM-only OOF AUC (baseline)
        biom_oof = cross_val_predict(
            biom_model, X_biom_train_enc, y_train,
            cv=cv_inner_fold, groups=groups_train,
            method='predict_proba', n_jobs=n_jobs,
        )[:, 1]
        baseline_auc = roc_auc_score(y_train, biom_oof)
        logger.info(f"  BIOM baseline OOF AUC: {baseline_auc:.4f}")

        # BIOM test AUC
        biom_test_proba = biom_model.predict_proba(X_biom_test_enc)[:, 1]
        biom_test_auc = roc_auc_score(y_test, biom_test_proba)
        logger.info(f"  BIOM test AUC: {biom_test_auc:.4f}")

        # ----- Step 2: Forward selection of MRF features -----
        logger.info("Starting forward selection of MRF features...")
        mrf_candidates = list(X_mrf_train.columns)
        selected_features = []
        selection_history = []
        current_auc = baseline_auc

        step = 0
        while mrf_candidates:
            step += 1
            best_feat = None
            best_auc = current_auc

            for feat in mrf_candidates:
                # Combine BIOM + selected + candidate
                cols = list(X_biom_train.columns) + selected_features + [feat]
                X_combined_train = pd.concat(
                    [X_biom_train, X_mrf_train[selected_features + [feat]]],
                    axis=1,
                )
                X_combined_train_enc = _encode_categoricals(
                    X_combined_train, model_name,
                )

                # Create model with BIOM's best params (no re-tuning)
                cat_combined = cat_biom + [c for c in cat_mrf
                                           if c in selected_features + [feat]]
                m = create_model(
                    model_name, seed=seed_rf,
                    cat_vars=cat_combined or None, gpu=gpu,
                )
                m.set_params(**best_params)

                try:
                    oof_proba = cross_val_predict(
                        m, X_combined_train_enc, y_train,
                        cv=cv_inner_fold, groups=groups_train,
                        method='predict_proba', n_jobs=n_jobs,
                    )[:, 1]
                    candidate_auc = roc_auc_score(y_train, oof_proba)
                except Exception as e:
                    logger.warning(f"    Feature '{feat}' failed: {e}")
                    candidate_auc = 0.0

                if candidate_auc > best_auc:
                    best_auc = candidate_auc
                    best_feat = feat

            gain = best_auc - current_auc
            if best_feat is not None and gain >= auc_threshold:
                selected_features.append(best_feat)
                mrf_candidates.remove(best_feat)
                current_auc = best_auc
                selection_history.append({
                    'step': step,
                    'feature': best_feat,
                    'oof_auc': best_auc,
                    'gain': gain,
                })
                logger.info(f"  Step {step}: +{best_feat} → "
                            f"OOF AUC={best_auc:.4f} (Δ={gain:+.4f})")
            else:
                logger.info(f"  Step {step}: No feature improves AUC by "
                            f"≥{auc_threshold}. Stopping.")
                break

        # ----- Step 3: Evaluate on outer test -----
        if selected_features:
            X_final_train = pd.concat(
                [X_biom_train, X_mrf_train[selected_features]], axis=1,
            )
            X_final_test = pd.concat(
                [X_biom_test, X_mrf_test[selected_features]], axis=1,
            )
            X_final_train_enc = _encode_categoricals(X_final_train, model_name)
            X_final_test_enc = _encode_categoricals(X_final_test, model_name)

            cat_final = cat_biom + [c for c in cat_mrf if c in selected_features]
            final_model = create_model(
                model_name, seed=seed_rf,
                cat_vars=cat_final or None, gpu=gpu,
            )
            final_model.set_params(**best_params)
            final_model.fit(
                X_final_train_enc,
                y_train.values.squeeze(),
            )
            final_test_proba = final_model.predict_proba(X_final_test_enc)[:, 1]
            final_test_auc = roc_auc_score(y_test, final_test_proba)
        else:
            final_test_auc = biom_test_auc

        logger.info(f"  Selected {len(selected_features)} MRF features: "
                    f"{selected_features}")
        logger.info(f"  BIOM-only test AUC: {biom_test_auc:.4f}")
        logger.info(f"  BIOM+selected test AUC: {final_test_auc:.4f}")

        fold_results.append({
            'fold': k,
            'train_index': train_index,
            'test_index': test_index,
            'biom_inner_splits': biom_inner_splits,
            'selected_features': selected_features,
            'selection_history': selection_history,
            'biom_test_auc': biom_test_auc,
            'final_test_auc': final_test_auc,
            'baseline_oof_auc': baseline_auc,
        })

    # ----- Aggregate -----
    logger.info(f"\n{'='*60}")
    logger.info("H2 Forward Selection — Summary")
    logger.info(f"{'='*60}")

    biom_aucs = [r['biom_test_auc'] for r in fold_results]
    final_aucs = [r['final_test_auc'] for r in fold_results]
    logger.info(f"  BIOM-only:     AUC = {np.mean(biom_aucs):.3f} "
                f"± {np.std(biom_aucs):.3f}")
    logger.info(f"  BIOM+selected: AUC = {np.mean(final_aucs):.3f} "
                f"± {np.std(final_aucs):.3f}")

    # Feature selection frequency
    feat_counter = Counter()
    for r in fold_results:
        feat_counter.update(r['selected_features'])
    logger.info("\n  Feature selection frequency:")
    for feat, count in feat_counter.most_common():
        logger.info(f"    {feat}: {count}/{n_splits} folds")

    # ----- Plots -----
    prefix = f'plots/h2_forward_{model_name}_B{feature_mode_biom}_M{feature_mode_mrf}_seed_{split_label}'

    # 1. Boxplot: BIOM vs BIOM+selected
    auc_rows = []
    for auc in biom_aucs:
        auc_rows.append({'model': 'BIOM', 'AUC': auc})
    for auc in final_aucs:
        auc_rows.append({'model': 'BIOM+sel.MRF', 'AUC': auc})

    auc_df = pd.DataFrame(auc_rows)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.boxplot(data=auc_df, x='model', y='AUC',
                color='.8', linecolor='#137', linewidth=0.75, ax=ax)
    sns.stripplot(data=auc_df, x='model', y='AUC', ax=ax, size=8, jitter=False)
    ax.set_title(f'H2: Forward Selection ({n_splits}-fold)')
    ax.set_ylabel('Test AUC (outer fold)')
    ax.set_xlabel('')
    fig.savefig(f'{prefix}_boxplot.pdf', bbox_inches='tight')
    plt.close(fig)

    # 2. Feature frequency bar chart
    if feat_counter:
        feat_df = pd.DataFrame(
            feat_counter.most_common(),
            columns=['feature', 'frequency'],
        )
        fig, ax = plt.subplots(figsize=(10, max(4, len(feat_df) * 0.4)))
        sns.barplot(data=feat_df, y='feature', x='frequency', ax=ax,
                    color='steelblue')
        ax.set_xlim(0, n_splits)
        ax.set_xlabel(f'Selection frequency (out of {n_splits} folds)')
        ax.set_title('MRF Feature Selection Frequency')
        fig.savefig(f'{prefix}_frequency.pdf', bbox_inches='tight')
        plt.close(fig)

    logger.info(f"Plots saved to {prefix}_*.pdf")

    # ----- Save -----
    results_path = (f'results/h2_forward_{model_name}_B{feature_mode_biom}'
                    f'_M{feature_mode_mrf}_seed_{split_label}.joblib')
    joblib.dump({
        'hypothesis': 'h2_forward',
        'model_name': model_name,
        'feature_mode_biom': feature_mode_biom,
        'feature_mode_mrf': feature_mode_mrf,
        'seed_split': seed_split,
        'splits_file': splits_file,
        'split_label': split_label,
        'auc_threshold': auc_threshold,
        'fold_results': fold_results,
        'feature_frequency': dict(feat_counter),
    }, results_path)
    logger.info(f"Results saved to {results_path}")


# =============================================================================
# Hypothesis 3: Non-linear MRF interactions via Tree Leaves
# =============================================================================

def run_h3_tree_leaves(model_name, seed_split=None,
                       feature_mode_biom='raw_woe',
                       feature_mode_mrf='raw_woe',
                       config_path=None, gpu=False, n_jobs=-1,
                       leaf_min_support=0.05, leaf_top_k=100,
                       leaf_filter_method='combined',
                       splits_file=None, **kwargs):
    """
    Test incremental value of MRF non-linear baseline expansion via tree leaves.

    1. Train base model on MRF features independently.
    2. Extract leaf identities for each sample across all MRF trees.
    3. Filter leaves by minimum support in the training set.
    4. Screen top K leaves by association (target correlation, weight, or both).
    5. Fit L1-regularized Logistic Regression on [BIOM + selected_leaves]
       to enforce sparsity and find stable indicators.
    6. Evaluate combined model on held-out outer test fold.

    Args:
        model_name (str): Must be ``'catboost'``.
        seed_split (int or None): Seed for outer StratifiedGroupKFold.
            Mutually exclusive with ``splits_file``.
        splits_file (str or None): Path to a joblib results file whose
            pre-computed outer fold indices and inner splits are reused
            verbatim.  When provided ``seed_split`` is used only for
            output-file naming (and may be ``None``).
    """
    if model_name != 'catboost':
        raise ValueError("H3 tree leaves extraction is currently implemented "
                         "optimally for CatBoost. Please use --model_name catboost.")

    cfg = _load_config(config_path)
    pipeline_cfg = cfg.get('pipeline', {})
    n_iter = pipeline_cfg.get('n_iter', 50)
    n_splits = pipeline_cfg.get('n_splits', 5)
    seed_cv = pipeline_cfg.get('seed_cv', 0)
    seed_rf = pipeline_cfg.get('seed_rf', 0)
    seed_bayes = pipeline_cfg.get('seed_bayes', 0)

    categorical_biom = cfg['categorical_biom']
    categorical_mrf = cfg['categorical_mrf']
    woe_dict_mrf = cfg['woe_dict_mrf']
    woe_dict_biom = cfg['woe_dict_biom']

    logger.info(f"Feature mode BIOM: {feature_mode_biom}")
    logger.info(f"Feature mode MRF:  {feature_mode_mrf}")
    logger.info(f"Leaf min support: {leaf_min_support}")
    logger.info(f"Leaf top K: {leaf_top_k} (method: {leaf_filter_method})")

    cv_inner = StratifiedGroupKFold(
        n_splits=n_splits, shuffle=True, random_state=seed_cv,
    )

    logger.info("Loading data...")
    joint_dataset_df = pd.read_csv(
        'data/joint_dataset.csv', index_col=0,
    ).set_index('subject_id')
    dataset_df = feature_engineering(joint_dataset_df)

    X_all = dataset_df.drop(['transition'], axis='columns')
    y_all = dataset_df['transition']
    groups_all = dataset_df['group']

    fold_results = []
    feat_counter = Counter()

    # Build outer iteration source and output label
    if splits_file is not None:
        _preloaded = _load_splits(splits_file)
        n_splits = len(_preloaded)
        split_label = Path(splits_file).stem
        _outer_iter = (
            (f['train_index'], f['test_index'], f['inner_splits'])
            for f in _preloaded
        )
    else:
        outer_cv = StratifiedGroupKFold(
            n_splits=n_splits, shuffle=True, random_state=seed_split,
        )
        split_label = str(seed_split)
        _outer_iter = (
            (train_idx, test_idx, None)
            for train_idx, test_idx in outer_cv.split(X_all, y_all, groups_all)
        )

    for k, (train_index, test_index, _preloaded_inner) in enumerate(tqdm(
        _outer_iter,
        total=n_splits, desc='Outer CV folds', unit='fold',
    )):
        logger.info(f"\n{'='*60}\nOuter fold {k+1}/{n_splits}\n{'='*60}")

        cv_inner_fold = (
            _PrecomputedSplitter(_preloaded_inner)
            if _preloaded_inner is not None
            else cv_inner
        )

        # ----- Step 0: Data Preprocessing -----
        woe_biom = WoETransformer(woe_dict_biom, categorical_biom)
        X_biom_train, X_biom_test, _, _, _ = \
            woe_biom.fit_transform_split(dataset_df, train_index, test_index)

        woe_mrf = WoETransformer(woe_dict_mrf, categorical_mrf)
        X_mrf_train, X_mrf_test, y_train, y_test, _ = \
            woe_mrf.fit_transform_split(dataset_df, train_index, test_index)

        X_biom_train = _filter_features(X_biom_train, feature_mode_biom)
        X_biom_test = _filter_features(X_biom_test, feature_mode_biom)
        X_mrf_train = _filter_features(X_mrf_train, feature_mode_mrf)
        X_mrf_test = _filter_features(X_mrf_test, feature_mode_mrf)

        groups_train = dataset_df.iloc[train_index]['group']

        cat_biom = [c for c in categorical_biom if c in X_biom_train.columns]
        cat_mrf = [c for c in categorical_mrf if c in X_mrf_train.columns]
        for df in [X_biom_train, X_biom_test]:
            if cat_biom:
                df[cat_biom] = df[cat_biom].astype(str).astype('category')
        for df in [X_mrf_train, X_mrf_test]:
            if cat_mrf:
                df[cat_mrf] = df[cat_mrf].astype(str).astype('category')

        # Encode numeric representations for models that need it (LogReg)
        X_biom_train_enc = _encode_categoricals(X_biom_train, model_name)
        X_biom_test_enc = _encode_categoricals(X_biom_test, model_name)

        # ----- Step 1: Train BIOM baseline -----
        logger.info("Training BIOM baseline model...")
        biom_study, biom_model, biom_inner_splits = train_model(
            X_biom_train, y_train, X_biom_test, y_test,
            model=model_name, seed_rf=seed_rf,
            seed_bayes=seed_bayes + 20, n_iter=n_iter,
            cv=cv_inner_fold, groups=groups_train,
            cat_vars=cat_biom or None, n_jobs=n_jobs, gpu=gpu,
        )
        biom_test_proba = biom_model.predict_proba(X_biom_test_enc)[:, 1]
        biom_test_auc = roc_auc_score(y_test, biom_test_proba)
        logger.info(f"  BIOM-only test AUC: {biom_test_auc:.4f}")

        # ----- Step 2: Train MRF model & Extract Leaves -----
        logger.info("Training MRF model for leaf extraction...")
        mrf_study, mrf_model, mrf_inner_splits = train_model(
            X_mrf_train, y_train, X_mrf_test, y_test,
            model=model_name, seed_rf=seed_rf,
            seed_bayes=seed_bayes + 30, n_iter=n_iter,
            cv=cv_inner_fold, groups=groups_train,
            cat_vars=cat_mrf or None, n_jobs=n_jobs, gpu=gpu,
        )

        logger.info("Extracting leaf identities...")
        # FeatureImportanceScorer returns binary leaf membership dataframes
        X_mrf_train_enc = _encode_categoricals(X_mrf_train, model_name)
        X_mrf_test_enc = _encode_categoricals(X_mrf_test, model_name)
        
        lm_train, lm_test, _, leaf_stats_df = FeatureImportanceScorer.compute_leaf_correlation(
            mrf_model, 
            X_mrf_train_enc, y_train, 
            X_mrf_test_enc, y_test,
            X_mrf_test_enc, y_test, # We don't have all_test in this script wrapper, pass test twice
            model_type=model_name
        )

        # ----- Step 3: Support Filtering -----
        total_samples = len(lm_train)
        leaf_supports = lm_train.sum(axis=0) / total_samples
        valid_leaves = leaf_supports[leaf_supports >= leaf_min_support].index
        logger.info(f"  Leaves passing {leaf_min_support*100}% support: {len(valid_leaves)} / {len(lm_train.columns)}")
        
        lm_train_filtered = lm_train[valid_leaves]
        lm_test_filtered = lm_test[valid_leaves]
        stats_filtered = leaf_stats_df[leaf_stats_df['leaf'].isin(valid_leaves)]

        # ----- Step 4: Association Filtering -----
        if leaf_filter_method == 'target':
            stats_filtered = stats_filtered.sort_values(by='correlation_target', ascending=False, key=abs)
        elif leaf_filter_method == 'weight':
            stats_filtered = stats_filtered.sort_values(by='leaf_weight', ascending=False)
        elif leaf_filter_method == 'combined':
            stats_filtered = stats_filtered.sort_values(by='combined_score', ascending=False)
        else:
            raise ValueError(f"Unknown leaf_filter_method: {leaf_filter_method}")

        top_leaves = stats_filtered['leaf'].head(leaf_top_k).tolist()
        logger.info(f"  Retaining top {len(top_leaves)} leaves by '{leaf_filter_method}'")
        
        lm_train_top = lm_train_filtered[top_leaves]
        lm_test_top = lm_test_filtered[top_leaves]

        # ----- Step 5: L1 Sparse Selection & Final Model -----
        logger.info("Applying L1 LogisticRegression CV for sparse leaf selection...")
        # When feature_mode_biom='raw_woe', raw biomarker columns may contain
        # missing values that break logistic regression.  Use only the
        # WoE-transformed BIOM columns for the L1 selection stage.
        # With metric_missing='empirical' in WoETransformer, the WoE of a missing
        # value reflects its actual relationship to the target (MNAR signal encoded).
        if feature_mode_biom == 'raw_woe':
            X_biom_logreg_train = _filter_features(X_biom_train_enc, 'woe')
            X_biom_logreg_test = _filter_features(X_biom_test_enc, 'woe')
            logger.info(
                f"  LogReg: restricting BIOM to {X_biom_logreg_train.shape[1]} "
                "WoE-only columns to avoid missing values in raw biomarkers."
            )
        else:
            X_biom_logreg_train = X_biom_train_enc
            X_biom_logreg_test = X_biom_test_enc

        # Capture WoE/continuous BIOM columns before augmenting with binary indicators.
        # Only these columns need z-score scaling; the indicators (0/1) do not.
        biom_cols = X_biom_logreg_train.columns

        # Add explicit MNAR missingness indicator columns.
        # These are binary flags (1 = raw value was missing) derived from the raw BIOM
        # columns in X_biom_train.  They make the "not-measured" signal visible to L1
        # logistic regression so it can select which missingness patterns matter, even
        # when the raw feature itself cannot enter the model.
        if feature_mode_biom in ('raw', 'raw_woe'):
            _raw_biom_cols = [c for c in X_biom_train.columns if not c.endswith('_WOE')]
            _missing_cols = [c for c in _raw_biom_cols if X_biom_train[c].isna().any()]
            if _missing_cols:
                _miss_train = (
                    X_biom_train[_missing_cols].isna().astype(int)
                    .rename(columns=lambda c: f'{c}_MISSING')
                )
                _miss_test = (
                    X_biom_test[_missing_cols].isna().astype(int)
                    .rename(columns=lambda c: f'{c}_MISSING')
                )
                X_biom_logreg_train = pd.concat([X_biom_logreg_train, _miss_train], axis=1)
                X_biom_logreg_test = pd.concat([X_biom_logreg_test, _miss_test], axis=1)
                logger.info(
                    f"  LogReg: added {len(_missing_cols)} MNAR missingness indicators "
                    f"({_missing_cols})"
                )

        X_combined_train = pd.concat([X_biom_logreg_train, lm_train_top], axis=1)
        X_combined_test = pd.concat([X_biom_logreg_test, lm_test_top], axis=1)

        # Track the boundary between BIOM-related columns (WoE + optional MISSING
        # indicators) and leaf columns so we can slice coef_ correctly below.
        n_non_leaf_cols = X_biom_logreg_train.shape[1]

        # Scale only continuous BIOM (WoE) columns; leaves and MISSING indicators
        # are already on a [0, 1] scale and must not be standardised.
        biom_means = X_combined_train[biom_cols].mean()
        biom_stds = X_combined_train[biom_cols].std()
        # Prevent zero division
        biom_stds[biom_stds == 0] = 1.0 

        X_combined_train_scaled = X_combined_train.copy()
        X_combined_test_scaled = X_combined_test.copy()
        X_combined_train_scaled[biom_cols] = (X_combined_train[biom_cols] - biom_means) / biom_stds
        X_combined_test_scaled[biom_cols] = (X_combined_test[biom_cols] - biom_means) / biom_stds

        # Create list of tuples for StratifiedGroupKFold to pass to LogisticRegressionCV
        cv_inner_splits = list(cv_inner_fold.split(X_combined_train_scaled, y_train, groups_train))

        l1_model = LogisticRegressionCV(
            Cs=100, cv=cv_inner_splits, l1_ratios=(1,), solver='saga',
            scoring='roc_auc', random_state=seed_rf, n_jobs=n_jobs, max_iter=5000,
            use_legacy_attributes=False
        )
        l1_model.fit(X_combined_train_scaled, y_train)

        # Identify which MRF leaves survived L1 selection
        # (coef_[0] aligns with X_combined_train columns: WoE | MISSING | leaves)
        coefs = l1_model.coef_[0]
        # leaf coefficients start after all non-leaf columns (WoE + MISSING indicators)
        leaf_coefs = coefs[n_non_leaf_cols:]
        selected_leaves = [leaf for leaf, c in zip(top_leaves, leaf_coefs) if abs(c) > 1e-5]

        logger.info(f"  L1 selection kept {len(selected_leaves)} / {len(top_leaves)} MRF leaves.")
        feat_counter.update(selected_leaves)

        # ----- Step 6: Final CatBoost on BIOM + selected leaves -----
        # L1 was used only for feature selection; the final model reuses the
        # BIOM-tuned CatBoost hyperparameters and is trained on the full
        # BIOM feature set (CatBoost handles NaN natively) plus the leaves
        # that L1 identified as informative.
        logger.info(
            f"  Training final CatBoost on BIOM + {len(selected_leaves)} "
            "selected leaves..."
        )
        lm_train_selected = (lm_train_top[selected_leaves] if selected_leaves
                             else pd.DataFrame(index=X_biom_train_enc.index))
        lm_test_selected  = (lm_test_top[selected_leaves] if selected_leaves
                             else pd.DataFrame(index=X_biom_test_enc.index))

        X_final_train = pd.concat([X_biom_train_enc, lm_train_selected], axis=1)
        X_final_test  = pd.concat([X_biom_test_enc,  lm_test_selected],  axis=1)

        final_cb = create_model(model_name, seed=seed_rf, cat_vars=cat_biom or None, gpu=gpu)
        final_cb.set_params(**biom_study.best_params)
        final_cb.fit(X_final_train, y_train)

        final_test_proba = final_cb.predict_proba(X_final_test)[:, 1]
        final_test_auc = roc_auc_score(y_test, final_test_proba)

        logger.info(f"  BIOM-only test AUC: {biom_test_auc:.4f}")
        logger.info(f"  BIOM+Leaves (CatBoost) test AUC: {final_test_auc:.4f}")

        fold_results.append({
            'fold': k,
            'train_index': train_index,
            'test_index': test_index,
            'biom_inner_splits': biom_inner_splits,
            'top_K_leaves': top_leaves,
            'selected_leaves': selected_leaves,
            'leaf_coefficients': {leaf: c for leaf, c in zip(top_leaves, leaf_coefs) if abs(c) > 1e-5},
            'biom_test_auc': biom_test_auc,
            'final_test_auc': final_test_auc,
        })

    # ----- Aggregate & Summary -----
    logger.info(f"\n{'='*60}\nH3 Tree Leaves — Summary across outer folds\n{'='*60}")
    biom_aucs = [r['biom_test_auc'] for r in fold_results]
    final_aucs = [r['final_test_auc'] for r in fold_results]
    
    logger.info(f"  BIOM-only AUC:      {np.mean(biom_aucs):.3f} ± {np.std(biom_aucs):.3f}")
    logger.info(f"  BIOM+Leaves (CatBoost) AUC: {np.mean(final_aucs):.3f} ± {np.std(final_aucs):.3f}")

    if feat_counter:
        logger.info("Most frequently selected leaves:")
        for feat, count in feat_counter.most_common(10):
            logger.info(f"    {feat}: {count}/{n_splits} folds")

    # ----- Plots -----
    prefix = f'plots/h3_tree_leaves_{model_name}_B{feature_mode_biom}_M{feature_mode_mrf}_seed_{split_label}'

    # 1. Boxplot: BIOM vs BIOM+Leaves
    auc_rows = []
    for r in fold_results:
        auc_rows.append({'model': 'BIOM-only', 'AUC': r['biom_test_auc']})
        auc_rows.append({'model': 'BIOM+Leaves (CatBoost)', 'AUC': r['final_test_auc']})

    auc_df = pd.DataFrame(auc_rows)
    fig, ax = plt.subplots(figsize=(6, 6))
    sns.boxplot(data=auc_df, x='model', y='AUC',
                color='.8', linecolor='#137', linewidth=0.75, ax=ax)
    sns.stripplot(data=auc_df, x='model', y='AUC', ax=ax, size=8, jitter=False)
    ax.set_title(f'H3: Non-linear MRF Leaves Incremental Value ({n_splits}-fold)')
    ax.set_ylabel('Test AUC (outer fold)')
    ax.set_xlabel('')
    fig.savefig(f'{prefix}_boxplot.pdf', bbox_inches='tight')
    plt.close(fig)

    # ----- Save -----
    results_path = (f'results/h3_tree_leaves_{model_name}_B{feature_mode_biom}'
                    f'_M{feature_mode_mrf}_seed_{split_label}.joblib')
    joblib.dump({
        'hypothesis': 'h3_tree_leaves',
        'model_name': model_name,
        'feature_mode_biom': feature_mode_biom,
        'feature_mode_mrf': feature_mode_mrf,
        'seed_split': seed_split,
        'splits_file': splits_file,
        'split_label': split_label,
        'leaf_min_support': leaf_min_support,
        'leaf_top_k': leaf_top_k,
        'leaf_filter_method': leaf_filter_method,
        'fold_results': fold_results,
        'feature_frequency': dict(feat_counter),
    }, results_path)
    logger.info(f"Results saved to {results_path}")


# =============================================================================
# Hypothesis 4: RF RuleFit-style conditional rule selection
# =============================================================================

def run_h4_rulefit_rf(model_name, seed_split=None,
                      feature_mode_biom='raw_woe',
                      feature_mode_mrf='raw_woe',
                      config_path=None, gpu=False, n_jobs=-1,
                      rule_rf_n_estimators=300, rule_rf_max_depth=3,
                      rule_rf_min_samples_leaf=0.05,
                      rule_rf_max_features='sqrt',
                      rule_support_min=0.05, rule_support_max=0.5,
                      rule_top_k=50, rule_auc_threshold=0.002,
                      rule_max_selected=10,
                      splits_file=None, **kwargs):
    """
    Test incremental MRF value via explicit RF path rules.

    Per outer fold:
      1. Train the BIOM baseline with Optuna-tuned hyperparameters.
      2. Train a shallow RF on MRF only and extract terminal-node path rules.
      3. Filter/deduplicate rule indicators by support and identical activations.
      4. Screen rules by inner-CV AUC gain over BIOM.
      5. Greedily add the rules with the best conditional AUC gains.
      6. Refit the final model on BIOM + selected rules and evaluate on the
         held-out outer test fold.
    """
    cfg = _load_config(config_path)
    pipeline_cfg = cfg.get('pipeline', {})
    n_iter = pipeline_cfg.get('n_iter', 50)
    n_splits = pipeline_cfg.get('n_splits', 5)
    seed_cv = pipeline_cfg.get('seed_cv', 0)
    seed_rf = pipeline_cfg.get('seed_rf', 0)
    seed_bayes = pipeline_cfg.get('seed_bayes', 0)

    woe_dict_biom = cfg['woe_dict_biom']
    woe_dict_mrf = cfg['woe_dict_mrf']
    categorical_biom = cfg['categorical_biom']
    categorical_mrf = cfg['categorical_mrf']

    logger.info(f"Feature mode BIOM: {feature_mode_biom}")
    logger.info(f"Feature mode MRF:  {feature_mode_mrf}")
    logger.info(
        "RF rule generator: "
        f"n_estimators={rule_rf_n_estimators}, max_depth={rule_rf_max_depth}, "
        f"min_samples_leaf={rule_rf_min_samples_leaf}, max_features={rule_rf_max_features}"
    )
    logger.info(
        "Rule selection: "
        f"support=[{rule_support_min}, {rule_support_max}], top_k={rule_top_k}, "
        f"auc_threshold={rule_auc_threshold}, max_selected={rule_max_selected}"
    )

    cv_inner = StratifiedGroupKFold(
        n_splits=n_splits, shuffle=True, random_state=seed_cv,
    )

    logger.info("Loading data...")
    joint_dataset_df = pd.read_csv(
        'data/joint_dataset.csv', index_col=0,
    ).set_index('subject_id')
    dataset_df = feature_engineering(joint_dataset_df)

    X_all = dataset_df.drop(['transition'], axis='columns')
    y_all = dataset_df['transition']
    groups_all = dataset_df['group']

    fold_results = []
    feat_counter = Counter()

    if splits_file is not None:
        _preloaded = _load_splits(splits_file)
        n_splits = len(_preloaded)
        split_label = Path(splits_file).stem
        _outer_iter = (
            (f['train_index'], f['test_index'], f['inner_splits'])
            for f in _preloaded
        )
    else:
        outer_cv = StratifiedGroupKFold(
            n_splits=n_splits, shuffle=True, random_state=seed_split,
        )
        split_label = str(seed_split)
        _outer_iter = (
            (train_idx, test_idx, None)
            for train_idx, test_idx in outer_cv.split(X_all, y_all, groups_all)
        )

    for k, (train_index, test_index, _preloaded_inner) in enumerate(tqdm(
        _outer_iter,
        total=n_splits, desc='Outer CV folds', unit='fold',
    )):
        logger.info(f"\n{'='*60}\nOuter fold {k+1}/{n_splits}\n{'='*60}")

        cv_inner_fold = (
            _PrecomputedSplitter(_preloaded_inner)
            if _preloaded_inner is not None
            else cv_inner
        )

        woe_biom = WoETransformer(woe_dict_biom, categorical_biom)
        woe_mrf = WoETransformer(woe_dict_mrf, categorical_mrf)

        X_biom_train, X_biom_test, _, _, _ = \
            woe_biom.fit_transform_split(dataset_df, train_index, test_index)
        X_mrf_train, X_mrf_test, y_train, y_test, _ = \
            woe_mrf.fit_transform_split(dataset_df, train_index, test_index)

        X_biom_train = _filter_features(X_biom_train, feature_mode_biom)
        X_biom_test = _filter_features(X_biom_test, feature_mode_biom)
        X_mrf_train = _filter_features(X_mrf_train, feature_mode_mrf)
        X_mrf_test = _filter_features(X_mrf_test, feature_mode_mrf)

        groups_train = dataset_df.iloc[train_index]['group']

        overlap = set(X_biom_train.columns) & set(X_mrf_train.columns)
        if overlap:
            X_mrf_train = X_mrf_train.drop(columns=list(overlap))
            X_mrf_test = X_mrf_test.drop(columns=list(overlap))
            logger.info(f"  Dropped {len(overlap)} BIOM/MRF overlapping columns from rule generator.")

        cat_biom = [c for c in categorical_biom if c in X_biom_train.columns]
        cat_mrf = [c for c in categorical_mrf if c in X_mrf_train.columns]
        for df in [X_biom_train, X_biom_test]:
            if cat_biom:
                df[cat_biom] = df[cat_biom].astype(str).astype('category')
        for df in [X_mrf_train, X_mrf_test]:
            if cat_mrf:
                df[cat_mrf] = df[cat_mrf].astype(str).astype('category')

        logger.info("Training BIOM baseline model...")
        biom_study, biom_model, biom_inner_splits = train_model(
            X_biom_train, y_train, X_biom_test, y_test,
            model=model_name, seed_rf=seed_rf,
            seed_bayes=seed_bayes + 20, n_iter=n_iter,
            cv=cv_inner_fold, groups=groups_train,
            cat_vars=cat_biom or None, n_jobs=n_jobs, gpu=gpu,
        )

        X_biom_train_enc = _encode_categoricals(X_biom_train, model_name)
        X_biom_test_enc = _encode_categoricals(X_biom_test, model_name)
        biom_oof = cross_val_predict(
            biom_model, X_biom_train_enc, y_train,
            cv=biom_inner_splits, groups=groups_train,
            method='predict_proba', n_jobs=n_jobs,
        )[:, 1]
        baseline_oof_auc = roc_auc_score(y_train, biom_oof)
        biom_test_proba = biom_model.predict_proba(X_biom_test_enc)[:, 1]
        biom_test_auc = roc_auc_score(y_test, biom_test_proba)
        logger.info(f"  BIOM-only test AUC: {biom_test_auc:.4f}")
        logger.info(f"  Cached BIOM inner-CV AUC: {baseline_oof_auc:.4f}")

        logger.info("Training shallow RF on MRF for rule extraction...")
        X_mrf_train_rf = _encode_categoricals(X_mrf_train, 'rf')
        X_mrf_test_rf = _encode_categoricals(X_mrf_test, 'rf')
        rf_rule_model = create_model('rf', seed=seed_rf)
        rf_rule_model.set_params(
            n_estimators=rule_rf_n_estimators,
            max_depth=rule_rf_max_depth,
            min_samples_leaf=rule_rf_min_samples_leaf,
            max_features=rule_rf_max_features,
            n_jobs=n_jobs,
        )
        rf_rule_model.fit(X_mrf_train_rf, y_train.values.squeeze())

        logger.info("Extracting RF path rules...")
        rule_train, rule_test, rule_meta = extract_rf_rule_matrix(
            rf_rule_model, X_mrf_train_rf, X_mrf_test_rf,
        )
        raw_rule_count = int(rule_train.shape[1])
        logger.info(f"  Extracted {raw_rule_count} raw rule indicators.")

        rule_train, rule_test, rule_meta = deduplicate_rule_matrix(
            rule_train, rule_test, rule_meta,
        )
        dedup_rule_count = int(rule_train.shape[1])
        logger.info(f"  {dedup_rule_count} rule indicators remain after deduplication.")

        rule_train, rule_test, rule_meta = filter_rules_by_support(
            rule_train, rule_test, rule_meta,
            min_support=rule_support_min, max_support=rule_support_max,
        )
        for col in ['rule_id', 'rule', 'support']:
            if col not in rule_meta.columns:
                rule_meta[col] = pd.Series(dtype=object if col != 'support' else float)
        filtered_rule_count = int(rule_train.shape[1])
        logger.info(f"  {filtered_rule_count} rule indicators remain after support filtering.")

        if rule_train.empty:
            logger.info("  No candidate rules survived filtering. Keeping BIOM-only model.")
            selected_rule_ids = []
            selected_rules = []
            selection_history = []
            screening_df = pd.DataFrame(columns=['rule_id', 'rule', 'support', 'candidate_auc', 'delta_auc'])
            final_test_auc = biom_test_auc
            final_oof_auc = baseline_oof_auc
        else:
            logger.info("Cheap-screening rules with cached BIOM OOF scores...")
            baseline_oof_auc, screening_df = score_rules_with_base_predictions(
                biom_oof, rule_train, y_train,
                cv_splits=biom_inner_splits, seed=seed_rf,
            )
            screening_df = screening_df.merge(
                rule_meta[['rule_id', 'rule', 'support']],
                on='rule_id', how='left',
            )

            top_rule_ids = screening_df.dropna(subset=['delta_auc'])['rule_id'].head(rule_top_k).tolist()
            logger.info(
                f"  Retaining top {len(top_rule_ids)} cheap-screened rules "
                "for expensive forward selection."
            )

            selected_rule_ids, selection_history, final_oof_auc = forward_select_rules_by_auc(
                X_biom_train, rule_train, y_train, top_rule_ids,
                model_name=model_name, model_params=biom_study.best_params,
                cv=biom_inner_splits, groups=groups_train, seed=seed_rf,
                cat_vars=cat_biom or None, gpu=gpu, n_jobs=n_jobs,
                auc_threshold=rule_auc_threshold,
                max_selected=rule_max_selected,
            )

            selected_rules = (
                rule_meta.set_index('rule_id')
                .loc[selected_rule_ids, 'rule']
                .tolist()
                if selected_rule_ids else []
            )
            feat_counter.update(selected_rules)

            if selected_rule_ids:
                X_final_train = pd.concat([X_biom_train, rule_train[selected_rule_ids]], axis=1)
                X_final_test = pd.concat([X_biom_test, rule_test[selected_rule_ids]], axis=1)
                X_final_train_enc = _encode_categoricals(X_final_train, model_name)
                X_final_test_enc = _encode_categoricals(X_final_test, model_name)

                final_model = create_model(
                    model_name, seed=seed_rf, cat_vars=cat_biom or None, gpu=gpu,
                )
                if model_name in ('rf', 'xgboost'):
                    final_model.set_params(n_jobs=n_jobs)
                final_model.set_params(**biom_study.best_params)
                final_model.fit(X_final_train_enc, y_train.values.squeeze())
                final_test_proba = final_model.predict_proba(X_final_test_enc)[:, 1]
                final_test_auc = roc_auc_score(y_test, final_test_proba)
            else:
                logger.info("  No rules met the conditional gain threshold. Keeping BIOM-only model.")
                final_test_auc = biom_test_auc

        logger.info(f"  BIOM baseline inner-CV AUC: {baseline_oof_auc:.4f}")
        logger.info(f"  Selected {len(selected_rule_ids)} RF rules.")
        logger.info(f"  BIOM+Rules test AUC: {final_test_auc:.4f}")

        fold_results.append({
            'fold': k,
            'train_index': train_index,
            'test_index': test_index,
            'biom_inner_splits': biom_inner_splits,
            'n_rules_raw': raw_rule_count,
            'n_rules_dedup': dedup_rule_count,
            'n_rules_filtered': filtered_rule_count,
            'baseline_oof_auc': baseline_oof_auc,
            'final_oof_auc': final_oof_auc,
            'selected_rule_ids': selected_rule_ids,
            'selected_rules': selected_rules,
            'selection_history': selection_history,
            'screening_top_rules': screening_df.head(rule_top_k).to_dict('records'),
            'biom_test_auc': biom_test_auc,
            'final_test_auc': final_test_auc,
        })

    logger.info(f"\n{'='*60}\nH4 RuleFit RF — Summary across outer folds\n{'='*60}")
    biom_aucs = [r['biom_test_auc'] for r in fold_results]
    final_aucs = [r['final_test_auc'] for r in fold_results]
    logger.info(f"  BIOM-only AUC:   {np.mean(biom_aucs):.3f} ± {np.std(biom_aucs):.3f}")
    logger.info(f"  BIOM+Rules AUC:  {np.mean(final_aucs):.3f} ± {np.std(final_aucs):.3f}")

    if feat_counter:
        logger.info("Most frequently selected rules:")
        for feat, count in feat_counter.most_common(10):
            logger.info(f"    {feat}: {count}/{n_splits} folds")

    prefix = f'plots/h4_rulefit_rf_{model_name}_B{feature_mode_biom}_M{feature_mode_mrf}_seed_{split_label}'

    auc_rows = []
    for r in fold_results:
        auc_rows.append({'model': 'BIOM-only', 'AUC': r['biom_test_auc']})
        auc_rows.append({'model': 'BIOM+Rules', 'AUC': r['final_test_auc']})

    auc_df = pd.DataFrame(auc_rows)
    fig, ax = plt.subplots(figsize=(6, 6))
    sns.boxplot(data=auc_df, x='model', y='AUC',
                color='.8', linecolor='#137', linewidth=0.75, ax=ax)
    sns.stripplot(data=auc_df, x='model', y='AUC', ax=ax, size=8, jitter=False)
    ax.set_title(f'H4: RF RuleFit-style MRF Value ({n_splits}-fold)')
    ax.set_ylabel('Test AUC (outer fold)')
    ax.set_xlabel('')
    fig.savefig(f'{prefix}_boxplot.pdf', bbox_inches='tight')
    plt.close(fig)

    if feat_counter:
        feat_df = pd.DataFrame(
            feat_counter.most_common(15),
            columns=['rule', 'frequency'],
        )
        feat_df['label'] = feat_df['rule'].map(
            lambda x: x if len(x) <= 90 else f"{x[:87]}..."
        )
        fig, ax = plt.subplots(figsize=(10, max(4, len(feat_df) * 0.45)))
        sns.barplot(data=feat_df, y='label', x='frequency', ax=ax, color='steelblue')
        ax.set_xlim(0, n_splits)
        ax.set_xlabel(f'Selection frequency (out of {n_splits} folds)')
        ax.set_ylabel('Rule')
        ax.set_title('H4: RF rule selection frequency')
        fig.savefig(f'{prefix}_frequency.pdf', bbox_inches='tight')
        plt.close(fig)

    results_path = (f'results/h4_rulefit_rf_{model_name}_B{feature_mode_biom}'
                    f'_M{feature_mode_mrf}_seed_{split_label}.joblib')
    joblib.dump({
        'hypothesis': 'h4_rulefit_rf',
        'model_name': model_name,
        'feature_mode_biom': feature_mode_biom,
        'feature_mode_mrf': feature_mode_mrf,
        'seed_split': seed_split,
        'splits_file': splits_file,
        'split_label': split_label,
        'rule_rf_n_estimators': rule_rf_n_estimators,
        'rule_rf_max_depth': rule_rf_max_depth,
        'rule_rf_min_samples_leaf': rule_rf_min_samples_leaf,
        'rule_rf_max_features': rule_rf_max_features,
        'rule_support_min': rule_support_min,
        'rule_support_max': rule_support_max,
        'rule_top_k': rule_top_k,
        'rule_auc_threshold': rule_auc_threshold,
        'rule_max_selected': rule_max_selected,
        'fold_results': fold_results,
        'rule_frequency': dict(feat_counter),
    }, results_path)
    logger.info(f"Results saved to {results_path}")


# =============================================================================
# Hypothesis Dispatch
# =============================================================================

HYPOTHESES = {
    'h1_stacking': run_h1_stacking,
    'h2_forward': run_h2_forward_selection,
    'h3_tree_leaves': run_h3_tree_leaves,
    'h4_rulefit_rf': run_h4_rulefit_rf,
}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Hypothesis Testing Pipeline',
    )
    parser.add_argument(
        '--hypothesis', type=str, required=True,
        choices=list(HYPOTHESES.keys()),
        help=f"Hypothesis to test: {list(HYPOTHESES.keys())}",
    )
    split_source = parser.add_mutually_exclusive_group(required=True)
    split_source.add_argument(
        '--seed_split', type=int, default=None,
        help="Seed for the outer StratifiedGroupKFold (generates new splits).",
    )
    split_source.add_argument(
        '--splits_file', type=str, default=None,
        help="Path to a joblib results file (from any hypothesis run) whose "
             "pre-computed train/test indices and inner CV splits are reused "
             "verbatim.  Mutually exclusive with --seed_split.",
    )
    parser.add_argument(
        '--model_name', type=str, required=True,
        choices=['catboost', 'xgboost', 'rf'],
        help="Base model type",
    )
    parser.add_argument(
        '--gpu', action='store_true', default=False,
        help="Enable GPU training for CatBoost",
    )
    parser.add_argument(
        '--feature_mode_biom', type=str, default='raw_woe',
        choices=['raw', 'woe', 'raw_woe'],
        help="Feature selection for BIOM: 'raw' (original variables only), "
             "'woe' (WoE-transformed only), 'raw_woe' (both, default)",
    )
    parser.add_argument(
        '--feature_mode_mrf', type=str, default='raw_woe',
        choices=['raw', 'woe', 'raw_woe'],
        help="Feature selection for MRF candidates: 'raw', 'woe', 'raw_woe'",
    )
    parser.add_argument(
        '--n_jobs', type=int, default=None,
        help="Number of parallel jobs (default: 1 for GPU, -1 for CPU)",
    )
    # H3-specific arguments
    parser.add_argument(
        '--leaf_min_support', type=float, default=0.05,
        help="H3: Minimum training sample support needed to keep a leaf (default: 0.05)",
    )
    parser.add_argument(
        '--leaf_top_k', type=int, default=100,
        help="H3: Number of top leaves to keep after support filtering, before L1 (default: 100)",
    )
    parser.add_argument(
        '--leaf_filter_method', type=str, default='combined',
        choices=['target', 'weight', 'combined'],
        help="H3: Metric used to select top K leaves (target correlation, leaf weight, or combined score)",
    )
    # H4-specific arguments
    parser.add_argument(
        '--rule_rf_n_estimators', type=int, default=300,
        help="H4: Number of trees in the RF rule generator (default: 300)",
    )
    parser.add_argument(
        '--rule_rf_max_depth', type=int, default=3,
        help="H4: Maximum depth of the RF rule generator (default: 3)",
    )
    parser.add_argument(
        '--rule_rf_min_samples_leaf', type=float, default=0.05,
        help="H4: Minimum samples per RF leaf; sklearn accepts float as a training-set fraction (default: 0.05)",
    )
    parser.add_argument(
        '--rule_rf_max_features', type=str, default='sqrt',
        choices=['sqrt', 'log2'],
        help="H4: RF rule-generator max_features strategy (default: sqrt)",
    )
    parser.add_argument(
        '--rule_support_min', type=float, default=0.05,
        help="H4: Minimum train-fold rule support to retain (default: 0.05)",
    )
    parser.add_argument(
        '--rule_support_max', type=float, default=0.5,
        help="H4: Maximum train-fold rule support to retain (default: 0.5)",
    )
    parser.add_argument(
        '--rule_top_k', type=int, default=50,
        help="H4: Number of screened rules to keep before forward selection (default: 50)",
    )
    parser.add_argument(
        '--rule_auc_threshold', type=float, default=0.002,
        help="H4: Minimum incremental inner-CV AUC gain required to add a rule (default: 0.002)",
    )
    parser.add_argument(
        '--rule_max_selected', type=int, default=10,
        help="H4: Maximum number of rules to keep in the final set (default: 10)",
    )
    args = parser.parse_args()

    # Filter out argparse attributes to pass to hypothesis directly
    kwargs = vars(args)
    hypothesis = kwargs.pop('hypothesis')
    # Use GPU rule for n_jobs if it wasn't specified
    if kwargs.get('n_jobs') is None:
        kwargs['n_jobs'] = 1 if kwargs.get('gpu') else -1

    HYPOTHESES[hypothesis](**kwargs)
