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
    train_model,
    feature_engineering,
    extract_tree_leaves,
)

logging.basicConfig(level=logging.INFO, format='%(name)s — %(message)s')
logger = logging.getLogger(__name__)

_CONFIG_PATH = Path(__file__).parent / 'configs' / 'model_training.yaml'


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


def run_h1_stacking(model_name, seed_split, feature_mode_biom='raw_woe',
                    feature_mode_mrf='raw_woe', config_path=None,
                    gpu=False, n_jobs=-1):
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
        seed_split (int): Seed for outer StratifiedGroupKFold.
        feature_mode_biom (str): Features to use for BIOM (raw, woe, raw_woe).
        feature_mode_mrf (str): Features to use for MRF (raw, woe, raw_woe).
        config_path (str or None): Path to YAML config.
        gpu (bool): Enable GPU training for CatBoost.
        n_jobs (int): Number of parallel jobs.
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

    # Outer CV loop
    outer_cv = StratifiedGroupKFold(
        n_splits=n_splits, shuffle=True, random_state=seed_split,
    )
    X_all = dataset_df.drop(['transition'], axis='columns')
    y_all = dataset_df['transition']
    groups_all = dataset_df['group']

    fold_aucs = {'biom': [], 'mrf': [], 'stacked': []}
    fold_splits = []

    for k, (train_index, test_index) in enumerate(tqdm(
        outer_cv.split(X_all, y_all, groups_all),
        total=n_splits, desc='Outer CV folds', unit='fold',
    )):
        logger.info(f"\n{'='*60}")
        logger.info(f"Outer fold {k+1}/{n_splits}: "
                    f"Train={len(train_index)}, Test={len(test_index)}")
        logger.info(f"{'='*60}")

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
            cv=cv_inner, groups=groups_train,
            cat_vars=cat_biom or None, n_jobs=n_jobs, gpu=gpu,
        )
        # OOF predictions on training set (for stacking features)
        # Encode categoricals for the trained model
        X_biom_train_enc = _encode_categoricals(X_biom_train, model_name)
        X_biom_test_enc = _encode_categoricals(X_biom_test, model_name)
        biom_oof = cross_val_predict(
            biom_model, X_biom_train_enc, y_train,
            cv=cv_inner, groups=groups_train,
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
            cv=cv_inner, groups=groups_train,
            cat_vars=cat_mrf or None, n_jobs=n_jobs, gpu=gpu,
        )
        X_mrf_train_enc = _encode_categoricals(X_mrf_train, model_name)
        X_mrf_test_enc = _encode_categoricals(X_mrf_test, model_name)
        mrf_oof = cross_val_predict(
            mrf_model, X_mrf_train_enc, y_train,
            cv=cv_inner, groups=groups_train,
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
    prefix = f'plots/h1_stacking_{model_name}_B{feature_mode_biom}_M{feature_mode_mrf}_seed_{seed_split}'
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
    results_path = f'results/h1_stacking_{model_name}_B{feature_mode_biom}_M{feature_mode_mrf}_seed_{seed_split}.joblib'
    joblib.dump({
        'hypothesis': 'h1_stacking',
        'model_name': model_name,
        'feature_mode_biom': feature_mode_biom,
        'feature_mode_mrf': feature_mode_mrf,
        'seed_split': seed_split,
        'n_folds': n_splits,
        'fold_aucs': fold_aucs,
        'fold_splits': fold_splits,
    }, results_path)
    logger.info(f"Results saved to {results_path}")


# =============================================================================
# Hypothesis 2: Conditional MRF Feature Selection (Forward Selection)
# =============================================================================

def run_h2_forward_selection(model_name, seed_split,
                             feature_mode_biom='raw_woe',
                             feature_mode_mrf='raw_woe',
                             config_path=None, gpu=False, n_jobs=-1,
                             auc_threshold=0.005):
    """
    Test which specific MRF features add incremental value over BIOM.

    Uses greedy forward selection: starting from a BIOM-only model, adds
    one MRF feature at a time (the one with the highest OOF AUC gain),
    stopping when no candidate improves AUC by more than ``auc_threshold``.

    The BIOM model's best hyperparameters (from Optuna) are reused during
    selection to keep it computationally feasible.

    Args:
        model_name (str): Base model type.
        seed_split (int): Seed for outer StratifiedGroupKFold.
        feature_mode_biom (str): Features to use for BIOM (raw, woe, raw_woe).
        feature_mode_mrf (str): Features to use for MRF (raw, woe, raw_woe).
        config_path (str or None): Path to YAML config.
        gpu (bool): Enable GPU training for CatBoost.
        n_jobs (int): Number of parallel jobs.
        auc_threshold (float): Minimum AUC gain to keep a feature.
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

    outer_cv = StratifiedGroupKFold(
        n_splits=n_splits, shuffle=True, random_state=seed_split,
    )
    X_all = dataset_df.drop(['transition'], axis='columns')
    y_all = dataset_df['transition']
    groups_all = dataset_df['group']

    fold_results = []

    for k, (train_index, test_index) in enumerate(tqdm(
        outer_cv.split(X_all, y_all, groups_all),
        total=n_splits, desc='Outer CV folds', unit='fold',
    )):
        logger.info(f"\n{'='*60}")
        logger.info(f"Outer fold {k+1}/{n_splits}")
        logger.info(f"{'='*60}")

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
            cv=cv_inner, groups=groups_train,
            cat_vars=cat_biom or None, n_jobs=n_jobs, gpu=gpu,
        )
        best_params = biom_study.best_params

        # Encode for predictions
        X_biom_train_enc = _encode_categoricals(X_biom_train, model_name)
        X_biom_test_enc = _encode_categoricals(X_biom_test, model_name)

        # BIOM-only OOF AUC (baseline)
        biom_oof = cross_val_predict(
            biom_model, X_biom_train_enc, y_train,
            cv=cv_inner, groups=groups_train,
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
                        cv=cv_inner, groups=groups_train,
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
    prefix = f'plots/h2_forward_{model_name}_B{feature_mode_biom}_M{feature_mode_mrf}_seed_{seed_split}'

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
                    f'_M{feature_mode_mrf}_seed_{seed_split}.joblib')
    joblib.dump({
        'hypothesis': 'h2_forward',
        'model_name': model_name,
        'feature_mode_biom': feature_mode_biom,
        'feature_mode_mrf': feature_mode_mrf,
        'seed_split': seed_split,
        'auc_threshold': auc_threshold,
        'fold_results': fold_results,
        'feature_frequency': dict(feat_counter),
    }, results_path)
    logger.info(f"Results saved to {results_path}")


# =============================================================================
# Hypothesis 3: Non-linear MRF interactions via Tree Leaves
# =============================================================================

def run_h3_tree_leaves(model_name, seed_split,
                       feature_mode_biom='raw_woe',
                       feature_mode_mrf='raw_woe',
                       config_path=None, gpu=False, n_jobs=-1,
                       leaf_min_support=0.05, leaf_top_k=100,
                       leaf_filter_method='combined', **kwargs):
    """
    Test incremental value of MRF non-linear baseline expansion via tree leaves.

    1. Train base model on MRF features independently.
    2. Extract leaf identities for each sample across all MRF trees.
    3. Filter leaves by minimum support in the training set.
    4. Screen top K leaves by association (target correlation, weight, or both).
    5. Fit L1-regularized Logistic Regression on [BIOM + selected_leaves]
       to enforce sparsity and find stable indicators.
    6. Evaluate combined model on held-out outer test fold.
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

    outer_cv = StratifiedGroupKFold(
        n_splits=n_splits, shuffle=True, random_state=seed_split,
    )
    X_all = dataset_df.drop(['transition'], axis='columns')
    y_all = dataset_df['transition']
    groups_all = dataset_df['group']

    fold_results = []
    feat_counter = Counter()

    for k, (train_index, test_index) in enumerate(tqdm(
        outer_cv.split(X_all, y_all, groups_all),
        total=n_splits, desc='Outer CV folds', unit='fold',
    )):
        logger.info(f"\n{'='*60}\nOuter fold {k+1}/{n_splits}\n{'='*60}")

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
            cv=cv_inner, groups=groups_train,
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
            cv=cv_inner, groups=groups_train,
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
        X_combined_train = pd.concat([X_biom_train_enc, lm_train_top], axis=1)
        X_combined_test = pd.concat([X_biom_test_enc, lm_test_top], axis=1)

        # We must scale features for L1 Logistic Regression
        # We only scale the biom features, leaves are binary [0, 1]
        biom_cols = X_biom_train_enc.columns
        biom_means = X_combined_train[biom_cols].mean()
        biom_stds = X_combined_train[biom_cols].std()
        # Prevent zero division
        biom_stds[biom_stds == 0] = 1.0 

        X_combined_train_scaled = X_combined_train.copy()
        X_combined_test_scaled = X_combined_test.copy()
        X_combined_train_scaled[biom_cols] = (X_combined_train[biom_cols] - biom_means) / biom_stds
        X_combined_test_scaled[biom_cols] = (X_combined_test[biom_cols] - biom_means) / biom_stds

        # Create list of tuples for StratifiedGroupKFold to pass to LogisticRegressionCV
        cv_inner_splits = list(cv_inner.split(X_combined_train_scaled, y_train, groups_train))

        l1_model = LogisticRegressionCV(
            Cs=10, cv=cv_inner_splits, penalty='l1', solver='liblinear',
            scoring='roc_auc', random_state=seed_rf, n_jobs=n_jobs, max_iter=2000
        )
        l1_model.fit(X_combined_train_scaled, y_train)

        # Identify which MRF leaves survived L1 selection
        # (coef_[0] aligns with X_combined_train columns)
        coefs = l1_model.coef_[0]
        # the first len(biom_cols) are biom, the rest are the MRF leaves
        leaf_coefs = coefs[len(biom_cols):]
        selected_leaves = [leaf for leaf, c in zip(top_leaves, leaf_coefs) if abs(c) > 1e-5]

        logger.info(f"  L1 selection kept {len(selected_leaves)} / {len(top_leaves)} MRF leaves.")
        feat_counter.update(selected_leaves)

        final_test_proba = l1_model.predict_proba(X_combined_test_scaled)[:, 1]
        final_test_auc = roc_auc_score(y_test, final_test_proba)

        logger.info(f"  BIOM-only test AUC: {biom_test_auc:.4f}")
        logger.info(f"  BIOM+Leaves (L1) test AUC: {final_test_auc:.4f}")

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
    logger.info(f"  BIOM+Leaves AUC:    {np.mean(final_aucs):.3f} ± {np.std(final_aucs):.3f}")

    if feat_counter:
        logger.info("Most frequently selected leaves:")
        for feat, count in feat_counter.most_common(10):
            logger.info(f"    {feat}: {count}/{n_splits} folds")

    # ----- Plots -----
    prefix = f'plots/h3_tree_leaves_{model_name}_B{feature_mode_biom}_M{feature_mode_mrf}_seed_{seed_split}'

    # 1. Boxplot: BIOM vs BIOM+Leaves
    auc_rows = []
    for r in fold_results:
        auc_rows.append({'model': 'BIOM-only', 'AUC': r['biom_test_auc']})
        auc_rows.append({'model': 'BIOM+Leaves', 'AUC': r['final_test_auc']})

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
                    f'_M{feature_mode_mrf}_seed_{seed_split}.joblib')
    joblib.dump({
        'hypothesis': 'h3_tree_leaves',
        'model_name': model_name,
        'feature_mode_biom': feature_mode_biom,
        'feature_mode_mrf': feature_mode_mrf,
        'seed_split': seed_split,
        'leaf_min_support': leaf_min_support,
        'leaf_top_k': leaf_top_k,
        'leaf_filter_method': leaf_filter_method,
        'fold_results': fold_results,
        'feature_frequency': dict(feat_counter),
    }, results_path)
    logger.info(f"Results saved to {results_path}")


# =============================================================================
# Hypothesis Dispatch
# =============================================================================

HYPOTHESES = {
    'h1_stacking': run_h1_stacking,
    'h2_forward': run_h2_forward_selection,
    'h3_tree_leaves': run_h3_tree_leaves,
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
    parser.add_argument(
        '--seed_split', type=int, required=True,
        help="Seed for the outer StratifiedGroupKFold",
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
    args = parser.parse_args()

    # Filter out argparse attributes to pass to hypothesis directly
    kwargs = vars(args)
    hypothesis = kwargs.pop('hypothesis')
    # Use GPU rule for n_jobs if it wasn't specified
    if kwargs.get('n_jobs') is None:
        kwargs['n_jobs'] = 1 if kwargs.get('gpu') else -1

    HYPOTHESES[hypothesis](**kwargs)
