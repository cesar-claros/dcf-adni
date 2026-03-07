"""
ADNI Model Training Pipeline
============================

This script trains and evaluates six model configurations for predicting
cognitive transition (CN → MCI/AD) in ADNI subjects:

1. **LIBRA** — Lifestyle for Brain Health risk score as a single feature
2. **BIOM** — Biomarker features (CSF markers, cognitive tests)
3. **MRF** — Modifiable Risk Factor features (lifestyle, demographics)
4. **BIOM+MRF** — Combined biomarker + risk factor features
5. **BIOM+rMRF** — Biomarker + rule-extracted features (leaf memberships from MRF tree)
6. **BIOM+sMRF** — Biomarker + top MRF features selected via DCG importance

Pipeline overview
-----------------
Using a single stratified group split (``StratifiedGroupKFold``):
  1. Apply WoE (Weight of Evidence) transformation to create BIOM and MRF feature sets
  2. Merge feature sets to create BIOM+MRF
  3. Build "augmented test set" by adding remaining unmatched CN controls (all label=0)
  4. Compute partial LIBRA scores from the combined feature set
  5. Train each model variant via Bayesian hyperparameter search (``BayesSearchCV``)
  6. Collect cross-validation predictions and test scores
  7. Extract tree rules from the MRF model for rMRF and sMRF variants
  8. Generate ROC curve plots
  9. Serialize all results to joblib

Known limitations
-----------------
- Cross-validation predictions (``cross_val_predict``) reuse the same splitter
  as the inner Bayesian search, introducing mild optimistic bias. For unbiased
  estimates, nested cross-validation would be needed.
- The augmented test set is class-imbalanced (all additional samples are controls).
  Metrics on this set should be interpreted with care.

Configuration::

    configs/model_training.yaml — WoE parameters, categorical variables, pipeline settings

Usage::

    python model_training.py --seed_split 0 --model_name catboost
"""

import argparse
import logging
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from catboost import Pool
from tqdm import tqdm
from sklearn.metrics import RocCurveDisplay, roc_auc_score
from sklearn.model_selection import (
    StratifiedGroupKFold,
    cross_validate,
)


from src.utils_model import (
    WoETransformer,
    TreeRuleExtractor,
    FeatureImportanceScorer,
    create_model,
    train_model,
    feature_engineering,
    calculate_libra_revised,
)

logging.basicConfig(level=logging.INFO, format='%(name)s — %(message)s')
logger = logging.getLogger(__name__)


# =============================================================================
# Configuration Loading
# =============================================================================

_CONFIG_PATH = Path(__file__).parent / 'configs' / 'model_training.yaml'


def _load_config(config_path=None):
    """
    Load WoE parameters, categorical variables, and pipeline settings from YAML.

    Args:
        config_path (str or Path or None): Path to the YAML config file.
            Defaults to ``configs/model_training.yaml``.

    Returns:
        dict: Parsed configuration dictionary.
    """
    path = Path(config_path) if config_path else _CONFIG_PATH
    with open(path, 'r') as f:
        return yaml.safe_load(f)




# =============================================================================
# Model Training Pipeline
# =============================================================================

class ModelTrainingPipeline:
    """
    Orchestrates the full model training and evaluation pipeline.

    This class encapsulates the repeated pattern of:
      1. WoE transformation
      2. Feature set assembly
      3. Model training via Bayesian search
      4. Cross-validation prediction collection
      5. ROC plot generation
      6. Result serialization

    Attributes:
        model_name (str): Model type ('catboost', 'xgboost', 'rf').
        seed_split (int): Seed for outer train/test split.
        param_space (dict): Hyperparameter search space for the chosen model.
        n_iter (int): Number of Bayesian search iterations.
        n_splits (int): Number of outer CV folds.
        n_rules (int): Number of top leaf memberships for rMRF.
        n_subset (int): Number of top DCG features for sMRF.
    """

    def __init__(self, model_name, seed_split, config_path=None,
                 gpu=False, n_jobs=-1):
        """
        Args:
            model_name (str): One of ``'catboost'``, ``'xgboost'``, ``'rf'``.
            seed_split (int): Random seed for the outer StratifiedGroupKFold.
            config_path (str or Path or None): Path to YAML config file.
                Defaults to ``configs/model_training.yaml``.
            gpu (bool): Enable GPU training for CatBoost.
            n_jobs (int): Number of parallel jobs for BayesSearchCV and CV
                functions. Use 1 when GPU is enabled to avoid CUDA OOM.
        """
        self.model_name = model_name
        self.seed_split = seed_split
        self.gpu = gpu
        self.n_jobs = n_jobs

        # Load configuration from YAML
        cfg = _load_config(config_path)
        pipeline_cfg = cfg.get('pipeline', {})

        self.n_iter = pipeline_cfg.get('n_iter', 50)
        self.n_splits = pipeline_cfg.get('n_splits', 5)
        self.n_rules = pipeline_cfg.get('n_rules', 100)
        self.n_subset = pipeline_cfg.get('n_subset', 30)
        self.seed_cv = pipeline_cfg.get('seed_cv', 0)
        self.seed_rf = pipeline_cfg.get('seed_rf', 0)
        self.seed_bayes = pipeline_cfg.get('seed_bayes', 0)

        # WoE configuration from YAML
        self.woe_dict_biom = cfg['woe_dict_biom']
        self.woe_dict_mrf = cfg['woe_dict_mrf']
        self.categorical_biom = cfg['categorical_biom']
        self.categorical_mrf = cfg['categorical_mrf']
        self.model_colors = cfg.get('plot_colors', {})

        if model_name not in ('catboost', 'xgboost', 'rf'):
            raise ValueError(f"Unknown model '{model_name}'. Choose from: catboost, xgboost, rf")

        # Inner CV splitter (used by BayesSearchCV for hyperparameter tuning)
        self.cv_inner = StratifiedGroupKFold(
            n_splits=self.n_splits, shuffle=True, random_state=self.seed_cv
        )

    def run(self):
        """
        Execute the full nested cross-validation pipeline.

        **Outer loop** (this method): K-fold split for unbiased evaluation.
        Each outer fold produces a train/test partition. The test set is
        never seen during hyperparameter tuning.

        **Inner loop** (inside ``BayesSearchCV``): K-fold split within the
        outer training set for hyperparameter optimization.

        This nested design eliminates the optimistic bias that occurs when
        the same CV splitter is used for both tuning and evaluation.

        Both loops use ``StratifiedGroupKFold`` to preserve class balance
        and ensure subjects with the same group stay in the same set.
        """
        # Load and engineer features
        logger.info("Loading data...")
        joint_dataset_df = pd.read_csv(
            'data/joint_dataset.csv', index_col=0
        ).set_index('subject_id')
        remaining_test_df = pd.read_csv(
            'data/remaining_test.csv', index_col=0
        ).set_index('subject_id')

        dataset_df = feature_engineering(joint_dataset_df)
        additional_test_df = feature_engineering(remaining_test_df)

        # Outer CV loop — each fold gives an unbiased test evaluation
        outer_cv = StratifiedGroupKFold(
            n_splits=self.n_splits, shuffle=True, random_state=self.seed_split
        )
        X_all = dataset_df.drop(['transition'], axis='columns')
        y_all = dataset_df['transition']
        groups_all = dataset_df['group']

        all_fold_results = []
        for k, (train_index, test_index) in enumerate(tqdm(
            outer_cv.split(X_all, y_all, groups_all),
            total=self.n_splits, desc='Outer CV folds', unit='fold',
        )):
            logger.info(f"\n{'='*60}")
            logger.info(f"Outer fold {k+1}/{self.n_splits}: "
                        f"Train={len(train_index)}, Test={len(test_index)}")
            logger.info(f"{'='*60}")

            # Fresh WoE transformers per fold to prevent data leakage
            self.woe_biom = WoETransformer(self.woe_dict_biom, self.categorical_biom)
            self.woe_mrf = WoETransformer(self.woe_dict_mrf, self.categorical_mrf)

            fold_results = self._train_fold(
                k, dataset_df, additional_test_df, train_index, test_index
            )
            all_fold_results.append(fold_results)

        # Aggregate and save
        self._aggregate_results(all_fold_results)
        self._save_results(all_fold_results)

    def _train_fold(self, fold_k, dataset_df, additional_test_df,
                    train_index, test_index):
        """
        Train all six model variants for a single outer CV fold.

        Args:
            fold_k (int): Fold index.
            dataset_df (pd.DataFrame): Full engineered dataset.
            additional_test_df (pd.DataFrame): Remaining unmatched controls.
            train_index, test_index: Positional indices for this fold.

        Returns:
            dict: All results, models, predictions, and metadata for this fold.
        """
        # Categorical variable lists from config
        categorical_biom = self.categorical_biom
        categorical_mrf = self.categorical_mrf

        groups_train = dataset_df.iloc[train_index]['group']

        # ----- Step 1: WoE Transformation -----
        X_mrf_train, X_mrf_test, y_train, y_test, bp_mrf = \
            self.woe_mrf.fit_transform_split(dataset_df, train_index, test_index)
        X_biom_train, X_biom_test, _, _, bp_biom = \
            self.woe_biom.fit_transform_split(dataset_df, train_index, test_index)

        # ----- Step 2: Merge BIOM + MRF -----
        # Remove duplicate columns that appear in both BIOM and MRF (e.g. subject_age)
        repeated_vars = list(
            set(X_mrf_train.columns).intersection(set(X_biom_train.columns))
        )
        X_biom_mrf_train = pd.merge(
            X_biom_train, X_mrf_train.drop(columns=repeated_vars),
            left_index=True, right_index=True,
        )
        X_biom_mrf_test = pd.merge(
            X_biom_test, X_mrf_test.drop(columns=repeated_vars),
            left_index=True, right_index=True,
        )

        # ----- Step 3: Build augmented test set -----
        X_nt_mrf_test = self.woe_mrf.transform_external(additional_test_df)
        X_nt_biom_test = self.woe_biom.transform_external(additional_test_df)
        X_nt_biom_mrf_test = pd.merge(
            X_nt_biom_test, X_nt_mrf_test,
            left_index=True, right_index=True,
        )
        y_nt_test = pd.Series(
            np.zeros(X_nt_biom_mrf_test.shape[0]),
            index=X_nt_biom_mrf_test.index,
            name='transition',
        )

        y_all_test = pd.concat([y_test, y_nt_test], axis=0).astype(int)
        X_biom_mrf_all_test = pd.concat([X_biom_mrf_test, X_nt_biom_mrf_test], axis=0)
        X_biom_all_test = pd.concat([X_biom_test, X_nt_biom_test], axis=0)
        X_mrf_all_test = pd.concat([X_mrf_test, X_nt_mrf_test], axis=0)

        # ----- Step 4: Compute LIBRA scores -----
        libra_train = X_biom_mrf_train.apply(calculate_libra_revised, axis=1).to_frame()
        libra_test = X_biom_mrf_test.apply(calculate_libra_revised, axis=1).to_frame()
        libra_all_test = X_biom_mrf_all_test.apply(calculate_libra_revised, axis=1).to_frame()

        # ----- Step 5: Cast categorical variables to str -----
        cat_vars_biom_mrf = list(
            set(categorical_biom).union(set(categorical_mrf))
        )
        for df in [X_biom_mrf_train, X_biom_mrf_test, X_biom_mrf_all_test]:
            df[cat_vars_biom_mrf] = df[cat_vars_biom_mrf].astype(str).astype('category')
        for df in [X_biom_train, X_biom_test, X_biom_all_test]:
            df[categorical_biom] = df[categorical_biom].astype(str).astype('category')
        for df in [X_mrf_train, X_mrf_test, X_mrf_all_test]:
            df[categorical_mrf] = df[categorical_mrf].astype(str).astype('category')

        # ----- Step 6: Train all models -----
        results = {}
        model_steps = [
            'LIBRA', 'BIOM+MRF', 'BIOM', 'MRF',
            'Rule extraction', 'BIOM+rMRF', 'BIOM+sMRF',
        ]
        pbar = tqdm(total=len(model_steps), desc='Training models', unit='model')

        # 6a. LIBRA model
        pbar.set_description('Training LIBRA')
        pbar.update(1)
        libra_study, libra_model, libra_inner_splits = self._train_single_model(
            X_train=libra_train, y_train=y_train,
            X_test=libra_test, y_test=y_test,
            groups=groups_train, variant_seed=0,
        )
        results['libra'] = self._evaluate_on_test(
            libra_study, libra_model, libra_test, y_test, libra_inner_splits,
        )

        # 6b. BIOM+MRF model
        pbar.set_description('Training BIOM+MRF')
        pbar.update(1)
        biom_mrf_study, biom_mrf_model, biom_mrf_inner_splits = self._train_single_model(
            X_train=X_biom_mrf_train, y_train=y_train, 
            X_test=X_biom_mrf_test, y_test=y_test,
            groups=groups_train, cat_vars=cat_vars_biom_mrf,
            variant_seed=10,
        )
        results['biom_mrf'] = self._evaluate_on_test(
            biom_mrf_study, biom_mrf_model, X_biom_mrf_test, y_test, biom_mrf_inner_splits,
        )

        # 6c. BIOM model
        pbar.set_description('Training BIOM')
        pbar.update(1)
        biom_study, biom_model, biom_inner_splits = self._train_single_model(
            X_train=X_biom_train, y_train=y_train,
            X_test=X_biom_test, y_test=y_test,
            groups=groups_train, cat_vars=categorical_biom,
            variant_seed=20,
        )
        results['biom'] = self._evaluate_on_test(
            biom_study, biom_model, X_biom_test, y_test, biom_inner_splits,
        )

        # 6d. MRF model
        pbar.set_description('Training MRF')
        pbar.update(1)
        mrf_study, mrf_model, mrf_inner_splits = self._train_single_model(
            X_train=X_mrf_train, y_train=y_train,
            X_test=X_mrf_test, y_test=y_test,
            groups=groups_train, cat_vars=categorical_mrf,
            variant_seed=30,
        )
        results['mrf'] = self._evaluate_on_test(
            mrf_study, mrf_model, X_mrf_test, y_test, mrf_inner_splits,
        )

        # 6e. Extract tree rules from MRF model for rMRF/sMRF
        pbar.set_description('Extracting tree rules')
        pbar.update(1)
        mrf_train_pool = Pool(
            data=X_mrf_train, label=y_train,
            cat_features=categorical_mrf,
        )
        extractor = TreeRuleExtractor(mrf_model, model_type='catboost')
        all_trees, all_features, unique_trees, unique_features = \
            extractor.extract_all_rules(X_mrf_train, train_pool=mrf_train_pool)

        lm_train, lm_test, lm_all_test, correlation = \
            FeatureImportanceScorer.compute_leaf_correlation(
                mrf_model,
                X_mrf_train, y_train,
                X_mrf_test, y_test,
                X_mrf_all_test, y_all_test,
                model_type='catboost',
            )
        leaf_counts = mrf_model.get_tree_leaf_counts()
        dcg_importance = FeatureImportanceScorer.dcg_score(
            all_features, correlation, unique_features, leaf_counts,
        )

        # 6f. BIOM+rMRF model (biomarker + top leaf memberships, no RFE)
        pbar.set_description('Training BIOM+rMRF')
        pbar.update(1)
        X_biom_rmrf_train = pd.merge(
            X_biom_train, lm_train.iloc[:, :self.n_rules],
            left_index=True, right_index=True,
        )
        X_biom_rmrf_test = pd.merge(
            X_biom_test, lm_test.iloc[:, :self.n_rules],
            left_index=True, right_index=True,
        )
        X_biom_rmrf_all_test = pd.merge(
            X_biom_all_test, lm_all_test.iloc[:, :self.n_rules],
            left_index=True, right_index=True,
        )
        biom_rmrf_study, biom_rmrf_model, biom_rmrf_inner_splits = self._train_single_model(
            X_train=X_biom_rmrf_train, y_train=y_train,
            X_test=X_biom_rmrf_test, y_test=y_test,
            groups=groups_train, cat_vars=categorical_biom,
            variant_seed=40,
        )
        results['biom_rmrf'] = self._evaluate_on_test(
            biom_rmrf_study, biom_rmrf_model, X_biom_rmrf_test, y_test, biom_rmrf_inner_splits,
        )
        biom_rmrf_cv = cross_validate(
            biom_rmrf_model, X_biom_rmrf_train, y_train,
            groups=groups_train, cv=self.cv_inner,
            n_jobs=self.n_jobs,
            return_train_score=True, return_estimator=True, return_indices=True,
        )

        # 6g. BIOM+sMRF model (biomarker + top DCG-selected MRF features, no RFE)
        pbar.set_description('Training BIOM+sMRF')
        pbar.update(1)
        top_vars = dcg_importance.index[:self.n_subset]
        repeated_smrf = list(set(top_vars).intersection(set(X_biom_train.columns)))
        top_vars = top_vars.drop(repeated_smrf)

        X_biom_smrf_train = pd.merge(
            X_biom_train, X_mrf_train[top_vars],
            left_index=True, right_index=True,
        )
        X_biom_smrf_test = pd.merge(
            X_biom_test, X_mrf_test[top_vars],
            left_index=True, right_index=True,
        )
        X_biom_smrf_all_test = pd.merge(
            X_biom_all_test, X_mrf_all_test[top_vars],
            left_index=True, right_index=True,
        )
        biom_smrf_study, biom_smrf_model, biom_smrf_inner_splits = self._train_single_model(
            X_train=X_biom_smrf_train, y_train=y_train,
            X_test=X_biom_smrf_test, y_test=y_test,
            groups=groups_train, cat_vars=categorical_biom,
            variant_seed=50,
        )
        results['biom_smrf'] = self._evaluate_on_test(
            biom_smrf_study, biom_smrf_model, X_biom_smrf_test, y_test, biom_smrf_inner_splits,
        )

        pbar.close()

        # ----- Step 7: Plot ROC curves -----
        logger.info("Generating ROC plots...")
        model_data = {
            'libra':     (libra_model, libra_train, libra_test, libra_all_test),
            'biom':      (biom_model, X_biom_train, X_biom_test, X_biom_all_test),
            'mrf':       (mrf_model, X_mrf_train, X_mrf_test, X_mrf_all_test),
            'biom_mrf':  (biom_mrf_model, X_biom_mrf_train, X_biom_mrf_test, X_biom_mrf_all_test),
            'biom_rmrf': (biom_rmrf_model, X_biom_rmrf_train, X_biom_rmrf_test, X_biom_rmrf_all_test),
            'biom_smrf': (biom_smrf_model, X_biom_smrf_train, X_biom_smrf_test, X_biom_smrf_all_test),
        }
        self._plot_all_roc(fold_k, model_data, results, y_train, y_test, y_all_test)

        # ----- Step 8: Assemble fold results -----
        return {
            'fold_k': fold_k,
            'train_index': train_index,
            'test_index': test_index,
            'results': results,
            'model_data': model_data,
            'y_train': y_train,
            'y_test': y_test,
            'y_all_test': y_all_test,
            'lm_train': lm_train,
            'lm_test': lm_test,
            'lm_all_test': lm_all_test,
            'all_trees': all_trees,
            'all_features': all_features,
            'unique_trees': unique_trees,
            'unique_features': unique_features,
            'corr': correlation,
            'biom_rmrf_cv': biom_rmrf_cv,
        }

    def _train_single_model(self, X_train, y_train, X_test, y_test,
                            groups=None, cat_vars=None, variant_seed=0):
        """
        Train a single model variant via Bayesian hyperparameter search.

        The inner CV (``self.cv_inner``) is used by ``BayesSearchCV`` for
        hyperparameter tuning. The outer test set (X_test, y_test) is only
        used for final evaluation — never seen during tuning.

        Args:
            X_train, y_train: Training data (outer fold).
            X_test, y_test: Test data (outer fold — truly unseen).
            groups: Group labels for inner CV.
            cat_vars: Categorical feature list (CatBoost only).
            variant_seed (int): Offset added to ``seed_bayes`` so each model
                variant explores a different region of hyperparameter space.

        Returns:
            tuple: ``(study, best_model, inner_splits)``
        """
        return train_model(
            X_train, y_train, X_test, y_test,
            model=self.model_name,
            seed_rf=self.seed_rf, seed_bayes=self.seed_bayes + variant_seed,
            n_iter=self.n_iter, cv=self.cv_inner,
            groups=groups, cat_vars=cat_vars, n_jobs=self.n_jobs,
            gpu=self.gpu,
        )

    def _evaluate_on_test(self, study, best_model, X_test, y_test, inner_splits=None):
        """
        Evaluate a trained model on the outer fold's held-out test set.

        This provides an **unbiased** evaluation — the test set was never
        seen during hyperparameter tuning (inner CV) or model training.

        Args:
            study: Completed Optuna study.
            best_model: Best model refitted on full outer training set.
            X_test: Outer fold test features.
            y_test: Outer fold test labels.

        Returns:
            dict with 'test_proba', 'test_auc', 'best_params', 'inner_cv_score'.
        """
        y_proba = best_model.predict_proba(X_test)[:, 1]
        test_auc = roc_auc_score(y_test, y_proba)
        logger.info(f"  Outer test AUC: {test_auc:.4f} "
                    f"(inner best: {study.best_value:.4f})")
        return {
            'test_proba': y_proba,
            'test_auc': test_auc,
            'best_params': study.best_params,
            'inner_cv_score': study.best_value,
            'inner_splits': inner_splits,
        }

    def _plot_roc(self, ax, model_data, results, y_true, plot_type='test'):
        """
        Plot ROC curves for all model variants on a single axes.

        Args:
            ax: Matplotlib axes.
            model_data (dict): Maps model name → (best_model, X_train, X_test, X_all_test).
            results (dict): Maps model name → evaluation results dict.
            y_true: True labels for the plot.
            plot_type (str): 'test' or 'all_test'.
        """
        model_names = list(model_data.keys())
        last_model = model_names[-1]

        for name in model_names:
            best_model, X_train, X_test, X_all_test = model_data[name]
            color = self.model_colors.get(name, 'gray')
            display_name = name.upper().replace('_', '+')
            is_last = (name == last_model)

            if plot_type == 'test':
                y_score = results[name]['test_proba']
            elif plot_type == 'all_test':
                y_score = best_model.predict_proba(X_all_test)[:, 1]

            RocCurveDisplay.from_predictions(
                y_true, y_score,
                ax=ax, name=display_name,
                plot_chance_level=is_last,
                curve_kwargs={'color': color},
            )

        ax.minorticks_on()
        ax.grid(which='both')
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlabel('1-Specificity (FPR)')
        ax.set_ylabel('Sensitivity (TPR)')

    def _plot_all_roc(self, fold_k, model_data, results,
                      y_train, y_test, y_all_test):
        """Generate ROC curve plots for a single outer fold (test + augmented test)."""
        prefix = f'plots/{self.model_name}_fold_{fold_k}_seed_{self.seed_split}'

        # Held-out test ROC
        fig, ax = plt.subplots(figsize=(8, 6))
        self._plot_roc(ax, model_data, results, y_test, plot_type='test')
        ax.set_title(f'ROC Curve (Outer Fold {fold_k} — Test Set)')
        fig.savefig(f'{prefix}_testroc.pdf', bbox_inches='tight')
        plt.close(fig)

        # Augmented test ROC
        fig, ax = plt.subplots(figsize=(8, 6))
        self._plot_roc(ax, model_data, results, y_all_test, plot_type='all_test')
        ax.set_title(f'ROC Curve (Outer Fold {fold_k} — Augmented Test Set)')
        fig.savefig(f'{prefix}_alltestroc.pdf', bbox_inches='tight')
        plt.close(fig)

    def _aggregate_results(self, all_fold_results):
        """
        Aggregate test AUCs across all outer folds and generate summary plots.

        Reports mean ± std per model and creates:
          - A boxplot comparing test AUC distributions across models
          - A summary ROC plot (mean performance)
        """
        model_names = ['libra', 'biom', 'mrf', 'biom_mrf', 'biom_rmrf', 'biom_smrf']

        # Collect per-fold AUCs
        summary = {}
        for name in model_names:
            aucs = [fold['results'][name]['test_auc'] for fold in all_fold_results]
            inner_scores = [fold['results'][name]['inner_cv_score'] for fold in all_fold_results]
            summary[name] = {
                'test_aucs': aucs,
                'inner_scores': inner_scores,
                'mean_auc': np.mean(aucs),
                'std_auc': np.std(aucs),
            }
            logger.info(
                f"{name:>12s}: test AUC = {np.mean(aucs):.3f} ± {np.std(aucs):.3f}  "
                f"(inner CV: {np.mean(inner_scores):.3f} ± {np.std(inner_scores):.3f})"
            )

        # Boxplot of test AUCs across folds
        prefix = f'plots/{self.model_name}_nested_seed_{self.seed_split}'
        auc_rows = []
        for name in model_names:
            for auc in summary[name]['test_aucs']:
                auc_rows.append({'model': name.upper().replace('_', '+'), 'AUC': auc})

        auc_df = pd.DataFrame(auc_rows)
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.boxplot(data=auc_df, x='model', y='AUC',
                    color='.8', linecolor='#137', linewidth=0.75, ax=ax)
        sns.stripplot(data=auc_df, x='model', y='AUC', ax=ax, size=8, jitter=False)
        ax.set_title(f'Nested CV — Test AUC per Outer Fold ({self.n_splits}-fold)')
        ax.set_ylabel('Test AUC (outer fold)')
        ax.set_xlabel('')
        fig.savefig(f'{prefix}_nested_boxplot.pdf', bbox_inches='tight')
        plt.close(fig)

        self._summary = summary

    def _save_results(self, all_fold_results):
        """Save all fold results and aggregated summary to a joblib file."""
        path = (f'results/{self.model_name}_nested_seed_{self.seed_split}'
                f'_seedcv_{self.seed_cv}_results.joblib')
        results_dict = {
            'n_folds': self.n_splits,
            'model_name': self.model_name,
            'seed_split': self.seed_split,
            'summary': getattr(self, '_summary', {}),
            'folds': all_fold_results,
        }
        joblib.dump(results_dict, path)
        logger.info(f"All fold results saved to {path}")


# =============================================================================
# CLI Entry Point
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train ADNI dementia risk prediction models"
    )
    parser.add_argument(
        '--seed_split', type=int, required=True,
        help="Seed for the outer train/test split",
    )
    parser.add_argument(
        '--model_name', type=str, required=True,
        choices=['catboost', 'xgboost', 'rf'],
        help="Model type to train",
    )
    parser.add_argument(
        '--gpu', action='store_true', default=False,
        help="Enable GPU training for CatBoost (requires NVIDIA CUDA)",
    )
    parser.add_argument(
        '--n_jobs', type=int, default=None,
        help="Number of parallel jobs for BayesSearchCV and CV functions. "
             "Defaults to 1 when --gpu is set, -1 otherwise",
    )
    args = parser.parse_args()

    # Default n_jobs: 1 for GPU (avoid CUDA OOM), -1 for CPU (full parallelism)
    n_jobs = args.n_jobs if args.n_jobs is not None else (1 if args.gpu else -1)

    pipeline = ModelTrainingPipeline(
        model_name=args.model_name,
        seed_split=args.seed_split,
        gpu=args.gpu,
        n_jobs=n_jobs,
    )
    pipeline.run()