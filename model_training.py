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
from sklearn.metrics import RocCurveDisplay
from sklearn.model_selection import (
    StratifiedGroupKFold,
    cross_val_predict,
    cross_val_score,
    cross_validate,
)
from skopt.space import Real, Integer

from src.utils_model import (
    WoETransformer,
    TreeRuleExtractor,
    FeatureImportanceScorer,
    create_model,
    train_model,
    search_rules,
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


# Model hyperparameter search spaces (skopt types are not YAML-serializable)
PARAM_SPACES = {
    'catboost': {
        'iterations':          Integer(100, 1000),
        'learning_rate':       Real(1e-3, 1.0, 'log-uniform'),
        'depth':               Integer(3, 10),
        'l2_leaf_reg':         Real(1, 10, 'uniform'),
        'bagging_temperature': Real(0.0, 1.0, 'uniform'),
        'border_count':        Integer(32, 255),
    },
    'xgboost': {
        'n_estimators':     Integer(100, 1000),
        'learning_rate':    Real(0.01, 0.3, 'log-uniform'),
        'max_depth':        Integer(4, 20),
        'subsample':        Real(0.5, 1.0, 'uniform'),
        'colsample_bytree': Real(0.5, 1.0, 'uniform'),
        'gamma':            Real(0.0, 5.0, 'uniform'),
        'reg_alpha':        Real(0.0, 10.0, 'uniform'),
        'reg_lambda':       Real(1.0, 10.0, 'uniform'),
    },
    'rf': {
        'n_estimators':      Integer(100, 300),
        'max_depth':         Integer(4, 20),
        'min_samples_split': Integer(5, 20),
        'min_samples_leaf':  Integer(5, 20),
        'max_features':      ['sqrt', 'log2'],
    },
}


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

    def __init__(self, model_name, seed_split, config_path=None):
        """
        Args:
            model_name (str): One of ``'catboost'``, ``'xgboost'``, ``'rf'``.
            seed_split (int): Random seed for the outer StratifiedGroupKFold.
            config_path (str or Path or None): Path to YAML config file.
                Defaults to ``configs/model_training.yaml``.
        """
        self.model_name = model_name
        self.seed_split = seed_split

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

        if model_name not in PARAM_SPACES:
            raise ValueError(f"Unknown model '{model_name}'. Choose from: {list(PARAM_SPACES.keys())}")
        self.param_space = PARAM_SPACES[model_name]

        # WoE transformers (initialized from YAML config)
        self.woe_biom = WoETransformer(self.woe_dict_biom, self.categorical_biom)
        self.woe_mrf = WoETransformer(self.woe_dict_mrf, self.categorical_mrf)

        # Cross-validation splitters
        self.sgkf = StratifiedGroupKFold(
            n_splits=self.n_splits, shuffle=True, random_state=seed_split
        )
        self.cv_group_train = StratifiedGroupKFold(
            n_splits=self.n_splits, shuffle=True, random_state=self.seed_cv
        )

    def run(self):
        """
        Execute the full training pipeline.

        Loads data, obtains a single stratified group split, trains all six
        model variants, generates ROC plots, and saves results.

        The outer split uses ``StratifiedGroupKFold`` to produce a single
        train/test partition that respects both class balance (stratified)
        and subject grouping (no subject appears in both train and test).
        Only the first fold is used — ``next(iter(...))`` cleanly retrieves
        it without a loop+break pattern.
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

        # Single stratified group split — takes the first fold
        # StratifiedGroupKFold ensures:
        #   - Stratification: class proportions are preserved in both sets
        #   - Grouping: subjects with the same 'group' stay in the same set
        train_index, test_index = next(iter(
            self.sgkf.split(
                dataset_df.drop(['transition'], axis='columns'),
                dataset_df['transition'],
                dataset_df['group'],
            )
        ))

        logger.info(f"Train: {len(train_index)} samples, Test: {len(test_index)} samples")
        results = self._train_fold(
            0, dataset_df, additional_test_df, train_index, test_index
        )
        self._save_results(0, results)

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
            df[cat_vars_biom_mrf] = df[cat_vars_biom_mrf].astype(str)
        for df in [X_biom_train, X_biom_test, X_biom_all_test]:
            df[categorical_biom] = df[categorical_biom].astype(str)
        for df in [X_mrf_train, X_mrf_test, X_mrf_all_test]:
            df[categorical_mrf] = df[categorical_mrf].astype(str)

        # ----- Step 6: Train all models -----
        results = {}

        # 6a. LIBRA model
        logger.info("Training LIBRA model...")
        libra_bs, libra_score = self._train_single_model(
            X_train=libra_train, y_train=y_train,
            X_test=libra_test, y_test=y_test,
            groups=groups_train,
        )
        results['libra'] = self._collect_cv_results(
            libra_bs, libra_train, y_train, groups_train,
            libra_score,
        )

        # 6b. BIOM+MRF model
        logger.info("Training BIOM+MRF model...")
        biom_mrf_bs, biom_mrf_score = self._train_single_model(
            X_train=X_biom_mrf_train, y_train=y_train, 
            X_test=X_biom_mrf_test, y_test=y_test,
            groups=groups_train, cat_vars=cat_vars_biom_mrf,
        )
        results['biom_mrf'] = self._collect_cv_results(
            biom_mrf_bs, X_biom_mrf_train, y_train, groups_train,
            biom_mrf_score,
        )

        # 6c. BIOM model
        logger.info("Training BIOM model...")
        biom_bs, biom_score = self._train_single_model(
            X_train=X_biom_train, y_train=y_train,
            X_test=X_biom_test, y_test=y_test,
            groups=groups_train, cat_vars=categorical_biom,
        )
        results['biom'] = self._collect_cv_results(
            biom_bs, X_biom_train, y_train, groups_train,
            biom_score,
        )

        # 6d. MRF model
        logger.info("Training MRF model...")
        mrf_bs, mrf_score = self._train_single_model(
            X_train=X_mrf_train, y_train=y_train,
            X_test=X_mrf_test, y_test=y_test,
            groups=groups_train, cat_vars=categorical_mrf,
        )
        results['mrf'] = self._collect_cv_results(
            mrf_bs, X_mrf_train, y_train, groups_train,
            mrf_score,
        )

        # 6e. Extract tree rules from MRF model for rMRF/sMRF
        logger.info("Extracting tree rules from MRF model...")
        mrf_train_pool = Pool(
            data=X_mrf_train, label=y_train,
            cat_features=categorical_mrf,
        )
        extractor = TreeRuleExtractor(mrf_bs.best_estimator_, model_type='catboost')
        all_trees, all_features, unique_trees, unique_features = \
            extractor.extract_all_rules(X_mrf_train, train_pool=mrf_train_pool)

        lm_train, lm_test, lm_all_test, correlation = \
            FeatureImportanceScorer.compute_leaf_correlation(
                mrf_bs.best_estimator_,
                X_mrf_train, y_train,
                X_mrf_test, y_test,
                X_mrf_all_test, y_all_test,
                model_type='catboost',
            )
        leaf_counts = mrf_bs.best_estimator_.get_tree_leaf_counts()
        dcg_importance = FeatureImportanceScorer.dcg_score(
            all_features, correlation, unique_features, leaf_counts,
        )

        # 6f. BIOM+rMRF model (biomarker + top rule-based leaf memberships)
        logger.info("Training BIOM+rMRF model...")
        biom_rmrf_bs, biom_rmrf_score = search_rules(
            X_biom_train, lm_train.iloc[:, :self.n_rules], y_train,
            X_biom_test, lm_test.iloc[:, :self.n_rules], y_test,
            self.param_space, model=self.model_name,
            seed_rf=self.seed_rf, seed_bayes=self.seed_bayes,
            n_iter=self.n_iter, cv=self.cv_group_train,
            groups=groups_train, cat_vars=categorical_biom,
        )
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
        results['biom_rmrf'] = self._collect_cv_results(
            biom_rmrf_bs, X_biom_rmrf_train, y_train, groups_train,
            biom_rmrf_score,
        )
        biom_rmrf_cv = cross_validate(
            biom_rmrf_bs.best_estimator_, X_biom_rmrf_train, y_train,
            groups=groups_train, cv=self.cv_group_train, n_jobs=-1,
            return_train_score=True, return_estimator=True, return_indices=True,
        )

        # 6g. BIOM+sMRF model (biomarker + top DCG-selected MRF features)
        logger.info("Training BIOM+sMRF model...")
        top_vars = dcg_importance.index[:self.n_subset]
        repeated_smrf = list(set(top_vars).intersection(set(X_biom_train.columns)))
        top_vars = top_vars.drop(repeated_smrf)

        biom_smrf_bs, biom_smrf_score = search_rules(
            X_biom_train, X_mrf_train[top_vars], y_train,
            X_biom_test, X_mrf_test[top_vars], y_test,
            self.param_space, model=self.model_name,
            seed_rf=self.seed_rf, seed_bayes=self.seed_bayes,
            n_iter=self.n_iter, cv=self.cv_group_train,
            groups=groups_train, cat_vars=categorical_biom,
        )
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
        results['biom_smrf'] = self._collect_cv_results(
            biom_smrf_bs, X_biom_smrf_train, y_train, groups_train,
            biom_smrf_score,
        )

        # ----- Step 7: Plot ROC curves -----
        logger.info("Generating ROC plots...")
        model_data = {
            'libra':     (libra_bs, libra_train, libra_test, libra_all_test),
            'biom':      (biom_bs, X_biom_train, X_biom_test, X_biom_all_test),
            'mrf':       (mrf_bs, X_mrf_train, X_mrf_test, X_mrf_all_test),
            'biom_mrf':  (biom_mrf_bs, X_biom_mrf_train, X_biom_mrf_test, X_biom_mrf_all_test),
            'biom_rmrf': (biom_rmrf_bs, X_biom_rmrf_train, X_biom_rmrf_test, X_biom_rmrf_all_test),
            'biom_smrf': (biom_smrf_bs, X_biom_smrf_train, X_biom_smrf_test, X_biom_smrf_all_test),
        }
        self._plot_all_roc(fold_k, model_data, results, y_train, y_test, y_all_test)
        self._plot_score_boxplot(fold_k, results)

        # ----- Step 8: Assemble full results dict -----
        return {
            'train_index': train_index,
            'test_index': test_index,
            'lm_train': lm_train,
            'lm_test': lm_test,
            'lm_all_test': lm_all_test,
            'all_trees': all_trees,
            'all_features': all_features,
            'unique_trees': unique_trees,
            'unique_features': unique_features,
            'corr': correlation,
            'cv_group_train': self.cv_group_train,
            'biom_rmrf_cv': biom_rmrf_cv,
            'X_biom_rmrf_train': X_biom_rmrf_train,
            'X_biom_rmrf_test': X_biom_rmrf_test,
            **{f'{name}_bayes_search': model_data[name][0] for name in model_data},
            **{f'{name}_score': results[name]['test_score'] for name in results},
            **{f'{name}_cv_scores': results[name]['cv_scores'] for name in results},
            **{f'{name}_all_fold_predictions': results[name]['fold_predictions'] for name in results},
        }

    def _train_single_model(self, X_train, y_train, X_test, y_test,
                            groups=None, cat_vars=None):
        """
        Train a single model variant via Bayesian hyperparameter search.

        Args:
            X_train, y_train: Training data.
            X_test, y_test: Test data.
            groups: Group labels for grouped CV.
            cat_vars: Categorical feature list (CatBoost only).

        Returns:
            tuple: ``(bayes_search, test_score)``
        """
        return train_model(
            X_train, y_train, X_test, y_test,
            self.param_space, model=self.model_name,
            seed_rf=self.seed_rf, seed_bayes=self.seed_bayes,
            n_iter=self.n_iter, cv=self.cv_group_train,
            groups=groups, cat_vars=cat_vars,
        )

    def _collect_cv_results(self, bayes_search, X_train, y_train,
                            groups, test_score):
        """
        Collect cross-validation predictions and scores for a trained model.

        Args:
            bayes_search: Fitted BayesSearchCV object.
            X_train, y_train: Training data.
            groups: Group labels for CV.
            test_score: Test set accuracy.

        Returns:
            dict: Contains 'fold_predictions', 'cv_scores', and 'test_score'.
        """
        fold_predictions = cross_val_predict(
            bayes_search.best_estimator_, X_train, y_train,
            groups=groups, cv=self.cv_group_train,
            method='predict_proba', n_jobs=-1,
        )
        cv_scores = cross_val_score(
            bayes_search.best_estimator_, X_train, y_train,
            groups=groups, cv=self.cv_group_train, n_jobs=-1,
        )
        return {
            'fold_predictions': fold_predictions,
            'cv_scores': cv_scores,
            'test_score': test_score,
        }

    def _plot_roc(self, ax, model_data, results, y_true, plot_type='cv'):
        """
        Plot ROC curves for all model variants on a single axes.

        Args:
            ax: Matplotlib axes.
            model_data (dict): Maps model name → (bayes_search, X_train, X_test, X_all_test).
            results (dict): Maps model name → CV results dict.
            y_true: True labels for the plot.
            plot_type (str): 'cv' for CV predictions, 'test' or 'all_test' for estimator-based.
        """
        model_names = list(model_data.keys())
        last_model = model_names[-1]

        for name in model_names:
            bs, X_train, X_test, X_all_test = model_data[name]
            color = self.model_colors.get(name, 'gray')
            display_name = name.upper().replace('_', '+')
            is_last = (name == last_model)

            if plot_type == 'cv':
                RocCurveDisplay.from_predictions(
                    y_true, results[name]['fold_predictions'][:, 1],
                    ax=ax, name=display_name, color=color,
                    plot_chance_level=is_last,
                )
            elif plot_type == 'test':
                RocCurveDisplay.from_estimator(
                    bs.best_estimator_, X_test, y_true,
                    ax=ax, name=display_name, color=color,
                    plot_chance_level=is_last,
                )
            elif plot_type == 'all_test':
                RocCurveDisplay.from_estimator(
                    bs.best_estimator_, X_all_test, y_true,
                    ax=ax, name=display_name, color=color,
                    plot_chance_level=is_last,
                )

        ax.minorticks_on()
        ax.grid(which='both')
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlabel('1-Specificity (FPR)')
        ax.set_ylabel('Sensitivity (TPR)')

    def _plot_all_roc(self, fold_k, model_data, results,
                      y_train, y_test, y_all_test):
        """Generate all three ROC curve plots (CV, test, augmented test)."""
        prefix = f'plots/{self.model_name}_split_{fold_k}_seed_{self.seed_split}_seedcv_{self.seed_cv}'

        # CV predictions ROC
        fig, ax = plt.subplots(figsize=(8, 6))
        self._plot_roc(ax, model_data, results, y_train, plot_type='cv')
        ax.set_title('ROC Curve (CV Test Sets)')
        fig.savefig(f'{prefix}_cvroc.pdf', bbox_inches='tight')
        plt.close(fig)

        # Held-out test ROC
        fig, ax = plt.subplots(figsize=(8, 6))
        self._plot_roc(ax, model_data, results, y_test, plot_type='test')
        ax.set_title('ROC Curve (Test Set)')
        fig.savefig(f'{prefix}_testroc.pdf', bbox_inches='tight')
        plt.close(fig)

        # Augmented test ROC
        fig, ax = plt.subplots(figsize=(8, 6))
        self._plot_roc(ax, model_data, results, y_all_test, plot_type='all_test')
        ax.set_title('ROC Curve (Test Set Augmented)')
        fig.savefig(f'{prefix}_alltestroc.pdf', bbox_inches='tight')
        plt.close(fig)

    def _plot_score_boxplot(self, fold_k, results):
        """Generate a boxplot comparing CV scores across model variants."""
        prefix = f'plots/{self.model_name}_split_{fold_k}_seed_{self.seed_split}_seedcv_{self.seed_cv}'

        scores_data = [
            {'model': name, 'scores': results[name]['cv_scores']}
            for name in results
        ]
        test_data = [
            {'model': name, 'scores': results[name]['test_score']}
            for name in results
        ]

        scores_df = pd.DataFrame(scores_data).explode('scores').reset_index(drop=True)
        test_df = pd.DataFrame(test_data)

        fig, ax = plt.subplots(figsize=(12, 8))
        sns.boxplot(data=scores_df, x='model', y='scores',
                    color='.8', linecolor='#137', linewidth=0.75, ax=ax)
        sns.stripplot(data=scores_df, x='model', y='scores', ax=ax, color='gray')
        sns.stripplot(data=test_df, x='model', y='scores', ax=ax, color='red')
        fig.savefig(f'{prefix}_boxplot.pdf', bbox_inches='tight')
        plt.close(fig)

    def _save_results(self, fold_k, results_dict):
        """Save results to a joblib file."""
        path = f'results/{self.model_name}_split_{fold_k}_seed_{self.seed_split}_seedcv_{self.seed_cv}_results.joblib'
        joblib.dump(results_dict, path)
        logger.info(f"Results saved to {path}")


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
    args = parser.parse_args()

    pipeline = ModelTrainingPipeline(
        model_name=args.model_name,
        seed_split=args.seed_split,
    )
    pipeline.run()