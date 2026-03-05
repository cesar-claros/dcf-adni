"""
Model Training Utilities for DCF-ADNI Project
===============================================

This module provides utility classes and functions for the DCF-ADNI model
training pipeline, which evaluates dementia risk prediction models using
ADNI (Alzheimer's Disease Neuroimaging Initiative) data.

The pipeline compares six model configurations:
  - **LIBRA**: Lifestyle for Brain Health (LIfestyle for BRAin health) risk score
  - **BIOM**: Biomarker-based features (CSF markers, cognitive scores)
  - **MRF**: Modifiable Risk Factor features (lifestyle, demographics)
  - **BIOM+MRF**: Combined biomarker + risk factor features
  - **BIOM+rMRF**: Biomarker + rule-extracted MRF features (from tree leaf memberships)
  - **BIOM+sMRF**: Biomarker + selected top MRF features (via DCG importance)

Key classes:
  - :class:`WoETransformer` — Weight-of-Evidence binning and transformation
  - :class:`TreeRuleExtractor` — Rule extraction from RF / XGBoost / CatBoost
  - :class:`FeatureImportanceScorer` — DCG-based, MDI, and SHAP importance scoring

Key functions:
  - :func:`create_model` — Factory for CatBoost / XGBoost / RandomForest classifiers
  - :func:`train_model` — Bayesian hyperparameter search + evaluation
  - :func:`search_rules` — Pipeline with RFE-based feature selection + Bayesian search
  - :func:`feature_engineering` — Derived features for BIOM and MRF models
  - :func:`calculate_libra_revised` — Partial LIBRA risk score computation
"""

import json
import re
import logging

import numpy as np
import pandas as pd
import shap
import joblib
from scipy.stats import chi2_contingency
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn import set_config
from skopt import BayesSearchCV
from tqdm import tqdm
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from optbinning import BinningProcess

logger = logging.getLogger(__name__)


# =============================================================================
# Model Factory
# =============================================================================

class _CatBoostWrapper(BaseEstimator):
    """
    Sklearn-compatible CatBoostClassifier wrapper.

    CatBoost internally reorders the ``cat_features`` parameter during
    ``__init__``, which breaks ``sklearn.base.clone()`` (used by
    ``BayesSearchCV``, ``cross_val_predict``, etc.).

    This wrapper avoids the issue entirely by storing hyperparameters as
    plain attributes (via ``BaseEstimator``) and only instantiating the
    real ``CatBoostClassifier`` inside ``fit()``. Attribute access to
    CatBoost-specific properties (e.g., ``tree_count_``,
    ``calc_leaf_indexes``) is transparently proxied to the fitted model.
    """

    def __init__(self, cat_features=None, verbose=0, random_state=None,
                 iterations=None, learning_rate=None, depth=None,
                 l2_leaf_reg=None, bagging_temperature=None,
                 border_count=None, task_type=None):
        self.cat_features = cat_features
        self.verbose = verbose
        self.random_state = random_state
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.depth = depth
        self.l2_leaf_reg = l2_leaf_reg
        self.bagging_temperature = bagging_temperature
        self.border_count = border_count
        self.task_type = task_type

    def _build_catboost(self):
        """Create a CatBoostClassifier from current parameter values."""
        params = {
            'cat_features': self.cat_features,
            'verbose': self.verbose,
            'random_state': self.random_state,
            'iterations': self.iterations,
            'learning_rate': self.learning_rate,
            'depth': self.depth,
            'l2_leaf_reg': self.l2_leaf_reg,
            'bagging_temperature': self.bagging_temperature,
            'border_count': self.border_count,
            'task_type': self.task_type,
        }
        # Only pass non-None params to CatBoost
        return CatBoostClassifier(**{k: v for k, v in params.items() if v is not None})

    def fit(self, X, y, **kwargs):
        self.model_ = self._build_catboost()
        self.model_.fit(X, y, **kwargs)
        self.classes_ = self.model_.classes_
        return self

    def predict(self, X):
        return self.model_.predict(X)

    def predict_proba(self, X):
        return self.model_.predict_proba(X)

    def score(self, X, y):
        return self.model_.score(X, y)

    def __getattr__(self, name):
        """Proxy attribute access to the fitted CatBoost model."""
        # Avoid infinite recursion on dunder lookups (pickling, copying)
        if name.startswith('__') and name.endswith('__'):
            raise AttributeError(name)
        try:
            model = object.__getattribute__(self, 'model_')
            return getattr(model, name)
        except AttributeError:
            raise AttributeError(
                f"'{type(self).__name__}' has no attribute '{name}'"
            )


def create_model(model_name, seed=0, cat_vars=None, gpu=False):
    """
    Factory function that creates a configured classifier instance.

    Centralizes model instantiation to avoid duplicating if/elif blocks
    across training and search functions.

    Args:
        model_name (str): One of ``'rf'``, ``'xgboost'``, or ``'catboost'``.
        seed (int): Random seed for reproducibility.
        cat_vars (list[str] or None): Categorical feature names (CatBoost only).
        gpu (bool): Enable GPU training for CatBoost (requires NVIDIA CUDA).

    Returns:
        A scikit-learn-compatible classifier instance.

    Raises:
        ValueError: If ``model_name`` is not recognized.
    """
    if model_name == 'rf':
        return RandomForestClassifier(
            max_samples=1.0,
            bootstrap=True,
            oob_score=True,
            random_state=seed,
        )
    elif model_name == 'xgboost':
        return XGBClassifier(
            objective='binary:logistic',
            eval_metric='logloss',
            use_label_encoder=False,
            random_state=seed,
        )
    elif model_name == 'catboost':
        return _CatBoostWrapper(
            verbose=0,
            random_state=seed,
            cat_features=cat_vars,
            task_type='GPU' if gpu else None,
        )
    else:
        raise ValueError(
            f"Unknown model '{model_name}'. Choose from 'rf', 'xgboost', 'catboost'."
        )


# =============================================================================
# Weight of Evidence (WoE) Transformer
# =============================================================================

class WoETransformer:
    """
    Encapsulates Weight-of-Evidence (WoE) binning and transformation.

    WoE converts each feature into a monotonic score reflecting its predictive
    power for the binary target. The ``optbinning.BinningProcess`` is fit on
    training data only to avoid data leakage.

    The WoE values are **negated** so that higher values correspond to higher
    risk (transition), matching the convention used in the ADNI pipeline.

    Attributes:
        woe_dict (dict): Feature-specific binning parameters
            (e.g., monotonic trend constraints).
        categorical_variables (list[str]): Names of categorical features.
        binning_process_ (BinningProcess or None): Fitted binning process
            (set after ``fit_transform_split``).
    """

    def __init__(self, woe_dict, categorical_variables):
        """
        Args:
            woe_dict (dict): Maps feature names to binning parameter dicts.
                Example: ``{'age': {'monotonic_trend': 'ascending'}}``.
            categorical_variables (list[str]): Categorical feature names.
        """
        self.woe_dict = woe_dict
        self.categorical_variables = categorical_variables
        self.all_variables = list(
            set(woe_dict.keys()).union(set(categorical_variables))
        )
        self.binning_process_ = None

    def fit_transform_split(self, dataset_df, train_idx, test_idx):
        """
        Fit WoE binning on training split and transform both splits.

        For each feature, the method produces:
          - The original raw value
          - A WoE-transformed version (suffixed ``_WOE``)

        Constant columns (zero variance after WoE) are dropped.

        Args:
            dataset_df (pd.DataFrame): Full dataset with target column ``'transition'``.
            train_idx (array-like): Positional indices for training rows.
            test_idx (array-like): Positional indices for test rows.

        Returns:
            tuple: ``(X_train, X_test, y_train, y_test, binning_process)``
                where X includes both raw and WoE features.
        """
        X_train_raw = dataset_df.iloc[train_idx][self.all_variables]
        y_train = dataset_df.loc[X_train_raw.index, 'transition']
        X_test_raw = dataset_df.iloc[test_idx][self.all_variables]
        y_test = dataset_df.loc[X_test_raw.index, 'transition']

        # Fit binning process on training data ONLY
        self.binning_process_ = BinningProcess(
            self.all_variables,
            categorical_variables=self.categorical_variables,
            binning_fit_params=self.woe_dict,
        )
        X_train_WOE = self.binning_process_.fit_transform(
            X_train_raw, y_train, metric='woe'
        )
        X_train_WOE = -1 * X_train_WOE.add_suffix('_WOE')

        X_test_WOE = self.binning_process_.transform(X_test_raw, metric='woe')
        X_test_WOE = -1 * X_test_WOE.add_suffix('_WOE')

        # Merge raw + WoE features
        X_train = pd.merge(X_train_raw, X_train_WOE, left_index=True, right_index=True)
        X_test = pd.merge(X_test_raw, X_test_WOE, left_index=True, right_index=True)

        # Drop constant columns
        drop_columns = X_train.columns[X_train.nunique() == 1]
        X_train = X_train.drop(columns=drop_columns)
        X_test = X_test.drop(columns=drop_columns)

        return X_train, X_test, y_train, y_test, self.binning_process_

    def transform_external(self, external_df):
        """
        Transform an external dataset using the already-fitted binning process.

        This is used for the "remaining test set" (unmatched controls) that
        are not part of the matched train/test splits.

        Args:
            external_df (pd.DataFrame): External data with features matching
                ``binning_process_.variable_names``.

        Returns:
            pd.DataFrame: Merged raw + WoE features for the external data.

        Raises:
            RuntimeError: If called before ``fit_transform_split()``.
        """
        if self.binning_process_ is None:
            raise RuntimeError(
                "BinningProcess not fitted. Call fit_transform_split() first."
            )

        bp = self.binning_process_
        X_WOE = -1 * bp.transform(
            external_df[bp.variable_names], metric='woe'
        )
        X_WOE = X_WOE.add_suffix('_WOE')
        X_out = pd.merge(
            external_df[list(self.woe_dict.keys())],
            X_WOE,
            left_index=True, right_index=True,
        )
        return X_out


# =============================================================================
# Statistical Helpers
# =============================================================================

def cramers_v(x, y):
    """
    Compute Cramér's V statistic for categorical–categorical association.

    Cramér's V is a measure of association between two nominal variables,
    giving a value between 0 and 1 (inclusive). It is based on the chi-squared
    statistic from a contingency table.

    Args:
        x (pd.Series): First categorical variable.
        y (pd.Series): Second categorical variable.

    Returns:
        float: Cramér's V statistic (0 = no association, 1 = perfect).
    """
    confusion_matrix = pd.crosstab(x, y)
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    min_dim = min(confusion_matrix.shape) - 1

    if min_dim == 0 or n == 0:
        return 0

    return np.sqrt(chi2 / (n * min_dim))


def calculate_dcg(relevance_scores, k=None):
    """
    Compute Discounted Cumulative Gain (DCG).

    DCG is a measure of ranking quality. It penalizes relevant items appearing
    lower in a ranked list by dividing each item's relevance by the logarithm
    of its position.

    Args:
        relevance_scores (list or np.ndarray): Relevance scores in ranked order.
        k (int or None): Position cutoff. If None, uses the full list.

    Returns:
        float: The DCG score.

    Raises:
        TypeError: If ``relevance_scores`` is not a list or array.
        ValueError: If ``k`` is not a positive integer.
    """
    if not isinstance(relevance_scores, (list, np.ndarray)):
        raise TypeError("relevance_scores must be a list or numpy array.")
    if k is not None and (not isinstance(k, int) or k <= 0):
        raise ValueError("k must be a positive integer or None.")

    if k is None:
        k = len(relevance_scores)
    else:
        k = min(k, len(relevance_scores))

    dcg = 0.0
    for i in range(k):
        dcg += relevance_scores[i] / np.log2(i + 2)
    return dcg


# =============================================================================
# Tree Rule Extraction
# =============================================================================

class TreeRuleExtractor:
    """
    Extracts decision rules and leaf paths from tree-based models.

    Supports RandomForest (sklearn), XGBoost, and CatBoost (oblivious trees).
    The extracted rules describe the path from root to each leaf node,
    enabling rule-based feature importance analysis.

    Usage::

        extractor = TreeRuleExtractor(model, model_type='catboost')
        rules_dict, features_dict, all_rules, unique_features = \\
            extractor.extract_all_rules(X_train, train_pool=pool)
    """

    def __init__(self, model, model_type='catboost'):
        """
        Args:
            model: Fitted tree-based classifier.
            model_type (str): One of ``'rf'``, ``'xgboost'``, ``'catboost'``.
        """
        self.model = model
        self.model_type = model_type

    def extract_all_rules(self, X_train, train_pool=None, seed=None):
        """
        Extract all decision rules from every tree in the model.

        For CatBoost, the model is serialized to JSON to access the
        oblivious tree structure. For RF/XGBoost, the tree structure
        is accessed via the respective APIs.

        Args:
            X_train (pd.DataFrame): Training features (for feature name resolution).
            train_pool (catboost.Pool or None): CatBoost Pool (required for
                CatBoost models using ``_get_tree_splits``).
            seed (int or None): Seed identifier for CatBoost JSON export path.

        Returns:
            tuple: ``(all_tree_rules_dict, all_tree_features_dict, all_rules_df, unique_features_df)``
                - ``all_tree_rules_dict``: Maps leaf IDs to rule lists
                - ``all_tree_features_dict``: Maps tree IDs to feature lists
                - ``all_rules_df``: DataFrame of rule counts
                - ``unique_features_df``: DataFrame of feature counts
        """
        if self.model_type == 'catboost' and train_pool is not None:
            return self._extract_catboost_rules_via_pool(train_pool)
        elif self.model_type == 'catboost':
            return self._extract_catboost_rules_via_json(X_train, seed)
        elif self.model_type == 'rf':
            return self._extract_rf_rules(X_train)
        elif self.model_type == 'xgboost':
            return self._extract_xgboost_rules(X_train)
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")

    # --- CatBoost via internal Pool API ---

    def _extract_catboost_rules_via_pool(self, train_pool):
        """Extract rules from CatBoost using the internal C++ object."""
        tree_leaf_counts = self.model.get_tree_leaf_counts()
        cb_object = self.model._object
        tree_count = cb_object._get_tree_count()

        all_tree_rules_dict = {}
        all_tree_features_dict = {}

        for tree_idx in tqdm(range(tree_count), total=tree_count, desc="Extracting CatBoost rules"):
            split_data = cb_object._get_tree_splits(tree_idx, train_pool)
            for k in range(tree_leaf_counts[tree_idx]):
                leaf_path = self._get_leaf_path(k, split_data)
                all_tree_rules_dict[f'tree_{tree_idx}-leaf_{k}'] = leaf_path
            features = self._get_unique_features(split_data)
            all_tree_features_dict[f'tree_{tree_idx}'] = features

        all_rules_df, unique_features_df = self._aggregate_rules_and_features(
            all_tree_rules_dict, all_tree_features_dict
        )
        return all_tree_rules_dict, all_tree_features_dict, all_rules_df, unique_features_df

    # --- CatBoost via JSON export ---

    def _extract_catboost_rules_via_json(self, X_train, seed=None):
        """Extract rules from CatBoost by exporting model to JSON."""
        seed_label = seed if seed is not None else 0
        path = f'results/catboost_model_{seed_label}.json'
        logger.info(f'Saving CatBoost model to {path}...')
        self.model.save_model(path, format='json')

        with open(path, 'r') as f:
            catboost_json = json.load(f)

        trees_list = catboost_json.get('oblivious_trees')
        if not trees_list:
            raise ValueError("JSON file does not contain 'oblivious_trees'.")

        feature_names = list(X_train.columns)
        all_tree_rules_dict = {}
        all_tree_features_dict = {}

        for i, tree_json in tqdm(enumerate(trees_list), total=len(trees_list), desc="Processing CatBoost trees"):
            rules, nodes, thresholds, directions, variables = \
                self._get_catboost_oblivious_rules(tree_json, feature_names)
            for leaf_idx, rule in enumerate(rules):
                all_tree_rules_dict[f'tree_{i}-leaf_{nodes[leaf_idx]}'] = rule
            # Extract unique features from this tree's variables
            unique_feats = list(set(v for var_list in variables for v in var_list))
            all_tree_features_dict[f'tree_{i}'] = unique_feats

        all_rules_df, unique_features_df = self._aggregate_rules_and_features(
            all_tree_rules_dict, all_tree_features_dict
        )
        return all_tree_rules_dict, all_tree_features_dict, all_rules_df, unique_features_df

    # --- RandomForest ---

    def _extract_rf_rules(self, X_train):
        """Extract rules from each tree in a RandomForest."""
        from sklearn.tree import _tree

        all_tree_rules_dict = {}
        all_tree_features_dict = {}

        for i, tree in enumerate(self.model.estimators_):
            rules, nodes, _, _, variables = self._get_sklearn_tree_rules(
                tree, X_train.columns
            )
            for leaf_idx, rule in enumerate(rules):
                all_tree_rules_dict[f'tree_{i}-leaf_{nodes[leaf_idx]}'] = rule
            unique_feats = list(set(v for var_list in variables for v in var_list))
            all_tree_features_dict[f'tree_{i}'] = unique_feats

        all_rules_df, unique_features_df = self._aggregate_rules_and_features(
            all_tree_rules_dict, all_tree_features_dict
        )
        return all_tree_rules_dict, all_tree_features_dict, all_rules_df, unique_features_df

    # --- XGBoost ---

    def _extract_xgboost_rules(self, X_train):
        """Extract rules from each booster in an XGBoost model."""
        booster = self.model.get_booster()
        num_trees = len(booster.get_dump())
        all_tree_rules_dict = {}
        all_tree_features_dict = {}

        for i in range(num_trees):
            rules, nodes, _, _, variables = self._get_xgboost_tree_rules(
                booster, i, X_train.columns
            )
            for leaf_idx, rule in enumerate(rules):
                all_tree_rules_dict[f'tree_{i}-leaf_{nodes[leaf_idx]}'] = rule
            unique_feats = list(set(v for var_list in variables for v in var_list))
            all_tree_features_dict[f'tree_{i}'] = unique_feats

        all_rules_df, unique_features_df = self._aggregate_rules_and_features(
            all_tree_rules_dict, all_tree_features_dict
        )
        return all_tree_rules_dict, all_tree_features_dict, all_rules_df, unique_features_df

    # --- Aggregation helper ---

    @staticmethod
    def _aggregate_rules_and_features(all_tree_rules_dict, all_tree_features_dict):
        """Aggregate rule and feature counts across all trees."""
        all_rules = []
        for x in all_tree_rules_dict:
            all_rules.extend(all_tree_rules_dict[x])
        all_rules_df = pd.Series(all_rules).value_counts().to_frame(name='counts')

        all_features = []
        for x in all_tree_features_dict:
            all_features.extend(all_tree_features_dict[x])
        unique_features_df = pd.Series(all_features).value_counts().to_frame(name='counts')

        return all_rules_df, unique_features_df

    # --- Low-level tree parsing methods ---

    @staticmethod
    def _get_sklearn_tree_rules(tree, feature_names):
        """Parse decision paths from a single sklearn DecisionTree."""
        from sklearn.tree import _tree

        tree_ = tree.tree_
        feature_name = [
            feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
            for i in tree_.feature
        ]
        paths, nodes, thresholds_list, directions_list, variables_list = [], [], [], [], []

        def recurse(node, path):
            if tree_.feature[node] != _tree.TREE_UNDEFINED:
                name = feature_name[node]
                threshold = tree_.threshold[node]
                recurse(tree_.children_left[node], path + [f"({name} <= {threshold:.2f})"])
                recurse(tree_.children_right[node], path + [f"({name} > {threshold:.2f})"])
            else:
                direction = [-1 if '<' in p else 1 for p in path]
                threshold = [p[1:-1].split(' ')[2] for p in path]
                var = [p[1:-1].split(' ')[0] for p in path]
                paths.append(path)
                nodes.append(node)
                thresholds_list.append(threshold)
                directions_list.append(direction)
                variables_list.append(var)

        recurse(0, [])
        return paths, nodes, thresholds_list, directions_list, variables_list

    @staticmethod
    def _get_xgboost_tree_rules(booster, booster_id, feature_names):
        """Parse decision paths from a single XGBoost booster tree."""
        dump = booster.get_dump(with_stats=False, dump_format='text')
        tree_text = dump[booster_id]
        paths, nodes, thresholds_list, directions_list, variables_list = [], [], [], [], []
        lines = tree_text.strip().split('\n')

        def recurse(node_id, path):
            line = next((l for l in lines if l.strip().startswith(f'{node_id}:')), None)
            if line is None:
                return

            if re.search(r'leaf', line):
                direction = [-1 if '<' in p else 1 for p in path]
                threshold = [p.split(' ')[2].strip(']') for p in path]
                var = [p.split(' ')[0].strip('[') for p in path]
                paths.append(path)
                nodes.append(node_id)
                thresholds_list.append(threshold)
                directions_list.append(direction)
                variables_list.append(var)
            else:
                match = re.search(r'\[(.*)<(.*)\]', line)
                if match:
                    variable = match.group(1)
                    threshold = float(match.group(2))
                    left_id = int(re.search(r'yes=(\d+)', line).group(1))
                    recurse(left_id, path + [f"[{variable} < {threshold:.2f}]"])
                    right_id = int(re.search(r'no=(\d+)', line).group(1))
                    recurse(right_id, path + [f"[{variable} >= {threshold:.2f}]"])

        recurse(0, [])
        return paths, nodes, thresholds_list, directions_list, variables_list

    @staticmethod
    def _get_catboost_oblivious_rules(tree_data, feature_names):
        """Parse decision paths from a CatBoost oblivious tree JSON structure."""
        splits = tree_data['splits']
        leaf_values = tree_data['leaf_values']
        paths, nodes, thresholds_list, directions_list, variables_list = [], [], [], [], []
        num_leaves = len(leaf_values)

        for i in range(num_leaves):
            current_path, current_thresholds, current_directions, current_variables = [], [], [], []

            for j, split in enumerate(splits):
                split_type = split['split_type']
                is_right_branch = (i >> j) & 1

                if split_type == 'FloatFeature':
                    feature_index = split['float_feature_index']
                    variable = feature_names[feature_index]
                    threshold = split['border']
                    if is_right_branch:
                        current_path.append(f"[{variable} > {threshold}]")
                        current_directions.append(1)
                    else:
                        current_path.append(f"[{variable} <= {threshold}]")
                        current_directions.append(-1)
                    current_thresholds.append(threshold)
                    current_variables.append(variable)

                elif split_type in ('OneHotFeature', 'BinarizedFeature'):
                    feature_index = split['cat_feature_index']
                    variable = feature_names[feature_index]
                    value = split['value']
                    if is_right_branch:
                        current_path.append(f"[{variable} != {value}]")
                        current_directions.append(1)
                    else:
                        current_path.append(f"[{variable} == {value}]")
                        current_directions.append(-1)
                    current_thresholds.append(value)
                    current_variables.append(variable)

                elif split_type == 'OnlineCtr':
                    feature_index = split['cat_feature_index']
                    variable = feature_names[feature_index]
                    threshold = split['border']
                    if is_right_branch:
                        current_path.append(f"[{variable} (OnlineCtr) > {threshold}]")
                        current_directions.append(1)
                    else:
                        current_path.append(f"[{variable} (OnlineCtr) <= {threshold}]")
                        current_directions.append(-1)
                    current_thresholds.append(threshold)
                    current_variables.append(variable)

            paths.append(current_path)
            nodes.append(i)
            thresholds_list.append(current_thresholds)
            directions_list.append(current_directions)
            variables_list.append(current_variables)

        return paths, nodes, thresholds_list, directions_list, variables_list

    @staticmethod
    def _get_leaf_path(leaf_idx, split_data):
        """
        Reconstruct the path of split conditions for a CatBoost leaf.

        Uses the binary representation of the leaf index to determine
        which branch (left/right) was taken at each split level.

        Args:
            leaf_idx (int): The leaf index within the tree.
            split_data (list): Split condition strings from CatBoost C++ object.

        Returns:
            list[str]: Description of each split operation on the path.
        """
        border_string = 'border='
        bin_string = 'bin='
        value_string = 'value='
        path = []
        depth = len(split_data)
        if depth == 0:
            return path

        binary_path = format(leaf_idx, f'0{depth}b')
        for level, decision in enumerate(binary_path):
            split_info = split_data[level]
            if border_string in split_info.split(',')[-1]:
                strings = split_info.split(', ' + border_string)
                feature_name, threshold = strings[0], strings[1]
                operations = ['<=', '>']
            elif bin_string in split_info.split(',')[-1]:
                strings = split_info.split(', ' + bin_string)
                feature_name, threshold = strings[0], strings[1]
                operations = ['<=', '>']
            elif value_string in split_info.split(',')[-1]:
                strings = split_info.split(', ' + value_string)
                feature_name, threshold = strings[0], strings[1]
                operations = ['!=', '=']
            else:
                continue

            path.append(f"{feature_name} {operations[int(decision)]} {threshold}")
        return path

    @staticmethod
    def _get_unique_features(split_data):
        """Extract unique feature names from CatBoost split data strings."""
        unique_features = []
        pattern = r"\{([^}]+)\}"
        for x in split_data:
            ctrs = re.findall(pattern, x)
            if len(ctrs) == 0:
                feature = x.split(',')
                unique_features.append(feature[0])
            else:
                feature = ctrs[0].split(',')
                if len(feature) == 1:
                    unique_features.append(feature[0])
                else:
                    for i, k in enumerate(feature):
                        feat = k.split(' ')
                        if i == 0:
                            unique_features.append(feat[0])
                        else:
                            unique_features.append(feat[1])
        return list(set(unique_features))


# =============================================================================
# Feature Importance Scoring
# =============================================================================

class FeatureImportanceScorer:
    """
    Multi-method feature importance scoring.

    Computes importance using:
      - **Leaf correlation**: Cramér's V between leaf membership and target
      - **nDCG**: Normalized Discounted Cumulative Gain based on how often
        a feature appears in highly-correlated leaves
      - **MDI**: Mean Decrease in Impurity (built-in tree importance)
      - **SHAP**: SHapley Additive exPlanations

    This is used to identify the most discriminative MRF features
    for the BIOM+sMRF model variant.
    """

    @staticmethod
    def compute_leaf_correlation(model, X_train, y_train, X_test, y_test,
                                 X_all_test, y_all_test, model_type='catboost'):
        """
        Compute leaf membership matrices and rank leaves by correlation with target.

        Each leaf in each tree becomes a binary feature indicating whether a
        sample falls into that leaf. Leaves are ranked by Cramér's V with the
        target variable.

        Args:
            model: Fitted tree-based classifier.
            X_train, X_test, X_all_test: Feature DataFrames.
            y_train, y_test, y_all_test: Target Series.
            model_type (str): ``'rf'``, ``'xgboost'``, or ``'catboost'``.

        Returns:
            tuple: ``(lm_train, lm_test, lm_all_test, correlation_df)``
                - ``lm_*``: Binary leaf membership DataFrames, columns ordered
                  by descending correlation.
                - ``correlation_df``: DataFrame with columns ``['leaf', 'correlation_target']``.
        """
        if model_type == 'rf':
            n_trees = len(model.estimators_)
        elif model_type == 'xgboost':
            n_trees = model.n_estimators
        elif model_type == 'catboost':
            n_trees = model.tree_count_
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        tree_leaves_arrays = []
        for X_data in [X_train, X_all_test]:
            if model_type == 'catboost':
                tree_leaf_pred = model.calc_leaf_indexes(X_data)
            else:
                tree_leaf_pred = model.apply(X_data)

            tree_dict = {f'tree_{k}': tree_leaf_pred[:, k] for k in range(n_trees)}
            tree_leaves_dict = {
                f'{tree}-leaf_{int(leaf)}': (tree_dict[tree] == leaf).astype(int)
                for tree in tree_dict
                for leaf in np.unique(tree_dict[tree])
            }
            tree_leaves_arrays.append(
                pd.DataFrame(tree_leaves_dict, index=X_data.index)
            )

        # Merge, fill missing leaves with 0
        leaf_membership_df = pd.concat(tree_leaves_arrays).fillna(0).astype(int)
        lm_train = leaf_membership_df.loc[y_train.index]
        lm_all_test = leaf_membership_df.loc[y_all_test.index]
        lm_test = lm_all_test.loc[y_test.index]

        # Rank leaves by Cramér's V with target
        correlation_df = pd.DataFrame([
            {'leaf': col, 'correlation_target': cramers_v(lm_train[col], y_train)}
            for col in lm_train.columns
        ]).sort_values(by='correlation_target', ascending=False)

        ordered_columns = correlation_df['leaf']
        lm_train = lm_train[ordered_columns]
        lm_test = lm_test[ordered_columns]
        lm_all_test = lm_all_test[ordered_columns]

        return lm_train, lm_test, lm_all_test, correlation_df

    @staticmethod
    def dcg_score(all_trees, correlation, unique_features, leaf_counts):
        """
        Compute nDCG-based feature importance.

        For each feature, identifies which leaves use that feature, retrieves
        those leaves' correlation ranks, and computes a normalized DCG score.
        Features appearing in more highly-correlated leaves get higher scores.

        Args:
            all_trees (dict): Maps tree IDs to feature lists (from ``TreeRuleExtractor``).
            correlation (pd.DataFrame): Leaf correlation data with ``'leaf'`` and
                ``'correlation_target'`` columns.
            unique_features (pd.DataFrame): Feature counts (from ``TreeRuleExtractor``).
            leaf_counts (array-like): Number of leaves per tree.

        Returns:
            pd.Series: nDCG scores indexed by feature name, sorted descending.
        """
        rules_expanded = pd.concat(
            [pd.Series(all_trees[x], name=x) for x in all_trees], axis=1
        )
        correlation_leaf_target = correlation.set_index('leaf')
        correlation_leaf_target['rank'] = correlation_leaf_target[
            'correlation_target'
        ].rank(ascending=True, method='min').astype(int)

        vars_expanded = {}
        for col in rules_expanded.columns:
            tree_idx = int(col.split('_')[1])
            for j in range(leaf_counts[tree_idx]):
                vars_expanded[f'{col}-leaf_{j}'] = rules_expanded[col]
        vars_expanded = pd.DataFrame(vars_expanded)

        unique_features_names = unique_features.reset_index()['index']
        vars_bool = pd.concat(
            [unique_features_names.isin(vars_expanded[col]).rename(col)
             for col in vars_expanded.columns],
            axis=1,
        )
        vars_bool = vars_bool.set_index(unique_features.index).T
        vars_bool = pd.merge(
            vars_bool, correlation_leaf_target['rank'],
            left_index=True, right_index=True, how='inner',
        )

        vars_dcg = {}
        for var in unique_features.index:
            ranks_list = vars_bool[vars_bool[var]]['rank'].to_list()
            ideal_rank = np.arange(
                start=correlation_leaf_target.shape[0], stop=1, step=-1
            )
            dcg = calculate_dcg(ranks_list)
            ideal_dcg = calculate_dcg(ideal_rank)
            vars_dcg[var] = dcg / ideal_dcg if ideal_dcg != 0 else 0

        return pd.Series(vars_dcg, name='nDCG').sort_values(ascending=False)


# =============================================================================
# Training Functions
# =============================================================================

def train_model(X_train, y_train, X_test, y_test, param_space,
                model='rf', seed_rf=0, seed_bayes=0, cv=10,
                n_iter=100, groups=None, cat_vars=None, n_jobs=-1,
                gpu=False):
    """
    Perform Bayesian hyperparameter search and evaluate on a test fold.

    Uses ``BayesSearchCV`` from scikit-optimize to find optimal hyperparameters
    via cross-validation within the training set.

    Args:
        X_train, y_train: Training features and target.
        X_test, y_test: Test features and target.
        param_space (dict): Hyperparameter search space (e.g., ``Integer``, ``Real``).
        model (str): Model type — ``'rf'``, ``'xgboost'``, or ``'catboost'``.
        seed_rf (int): Seed for the model's random state.
        seed_bayes (int): Seed for the Bayesian search.
        cv: Cross-validation splitter or integer.
        n_iter (int): Number of Bayesian search iterations.
        groups (array-like or None): Group labels for grouped cross-validation.
        cat_vars (list[str] or None): Categorical feature names (CatBoost).
        n_jobs (int): Number of parallel jobs for BayesSearchCV.
            Use 1 for CatBoost GPU to avoid OOM.

    Returns:
        tuple: ``(bayes_search, test_score)``
            - ``bayes_search``: Fitted ``BayesSearchCV`` object.
            - ``test_score``: Accuracy on the test set.
    """
    m = create_model(model, seed=seed_rf, cat_vars=cat_vars, gpu=gpu)
    bayes_search = BayesSearchCV(
        m, param_space, n_iter=n_iter, cv=cv,
        n_jobs=n_jobs, verbose=0, random_state=seed_bayes,
    )

    if groups is None:
        bayes_search.fit(X_train, y_train.values.squeeze())
    else:
        bayes_search.fit(X=X_train, y=y_train.values.squeeze(), groups=groups)

    logger.info(f"Best Parameters: {bayes_search.best_params_}")
    test_score = bayes_search.best_estimator_.score(X_test, y_test.values.squeeze())
    logger.info(f"Test Set Accuracy: {test_score}")
    return bayes_search, test_score


def search_rules(df1_train, df2_train, y_train, df1_test, df2_test, y_test,
                 param_space, model='rf', seed_rf=0, seed_bayes=0,
                 cv=10, n_iter=100, groups=None, cat_vars=None, n_jobs=-1,
                 gpu=False):
    """
    Train a model with RFE-based feature selection on a second feature set.

    Builds a pipeline:
      1. ``ColumnTransformer`` applies RFE to ``df2`` features while passing
         ``df1`` features through unchanged.
      2. A classifier (RF / XGBoost / CatBoost) is trained on the combined features.

    Bayesian search jointly optimizes the number of selected features and
    the model hyperparameters.

    This is used for the BIOM+rMRF model (where ``df1`` = BIOM features and
    ``df2`` = leaf membership features from the MRF model).

    Args:
        df1_train, df2_train: Primary (passed through) and secondary (selected) features.
        y_train: Training target.
        df1_test, df2_test: Test counterparts.
        y_test: Test target.
        param_space (dict): Model hyperparameter search space.
        model (str): Model type.
        seed_rf, seed_bayes (int): Random seeds.
        cv: Cross-validation splitter.
        n_iter (int): Bayesian search iterations.
        groups (array-like or None): Group labels for CV.
        cat_vars (list[str] or None): Categorical features.
        n_jobs (int): Number of parallel jobs for BayesSearchCV.

    Returns:
        tuple: ``(bayes_search, test_score)``
    """
    set_config(transform_output="pandas")
    X_combined = pd.merge(df1_train, df2_train, left_index=True, right_index=True)
    df2_features = df2_train.columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ('feature_selection',
             RFE(estimator=RandomForestClassifier(random_state=seed_rf),
                 n_features_to_select=2),
             df2_features),
        ],
        remainder='passthrough',
        verbose_feature_names_out=False,
    )

    param_grid = {
        'preprocessing__feature_selection__n_features_to_select': (1, len(df2_features)),
    }
    param_grid.update({f'model__{k}': param_space[k] for k in param_space})

    m = create_model(model, seed=seed_rf, cat_vars=cat_vars, gpu=gpu)
    pipeline = Pipeline([
        ('preprocessing', preprocessor),
        ('model', m),
    ])

    bayes_search = BayesSearchCV(
        pipeline, param_grid, n_iter=n_iter, cv=cv,
        n_jobs=n_jobs, verbose=0, random_state=seed_bayes,
    )

    if groups is None:
        bayes_search.fit(X=X_combined, y=y_train.values.squeeze())
    else:
        bayes_search.fit(X=X_combined, y=y_train.values.squeeze(), groups=groups)

    logger.info(f"Best Parameters: {bayes_search.best_params_}")
    X_test = pd.merge(df1_test, df2_test, left_index=True, right_index=True)
    test_score = bayes_search.best_estimator_.score(X_test, y_test.values.squeeze())
    logger.info(f"Test Set Accuracy: {test_score}")

    return bayes_search, test_score


# =============================================================================
# Feature Engineering & LIBRA Score
# =============================================================================

def feature_engineering(joint_dataset_df):
    """
    Create derived features for both BIOM and MRF model groups.

    **Biomarker ratios** (BIOM):
      - ``PTAU/ABETA42``: Phosphorylated tau to amyloid-beta ratio
      - ``TAU/ABETA42``: Total tau to amyloid-beta ratio
      - ``medical_current``: Number of current medical conditions
      - ``medical_old``: Number of past medical conditions

    **Sociodemographic / lifestyle** (MRF):
      - ``married``: Binary married indicator
      - ``lives_alone``: Binary (PTHOME in {3, 4})
      - ``retired``: Binary retirement indicator
      - ``homeowner``: Binary homeowner indicator
      - ``social_isolation``: Sum of married + lives_alone + retired
      - ``education_retired``: Education × retirement interaction
      - ``years_retired``: Years since retirement
      - ``married_homeowner``: Marriage × homeownership interaction
      - ``retired_lives_alone``: Retired + living alone indicator
      - ``SES_score``: Z-normalized socioeconomic score

    Args:
        joint_dataset_df (pd.DataFrame): Dataset with raw ADNI columns.

    Returns:
        pd.DataFrame: Copy of input with new derived columns added.
    """
    df = joint_dataset_df.copy()

    # Biomarker ratios
    df['PTAU/ABETA42'] = df['PTAU'] / df['ABETA42']
    df['TAU/ABETA42'] = df['TAU'] / df['ABETA42']
    df['medical_current'] = df['MHNUM'] * (df['MHCUR'] == 1)
    df['medical_old'] = df['MHNUM'] * (df['MHCUR'] == 0)

    # Sociodemographic features
    df['married'] = (df['PTMARRY'] == 1).astype(int)
    df['lives_alone'] = df['PTHOME'].isin([3, 4]).astype(int)
    df['retired'] = (df['PTNOTRT'] == 1).astype(int)
    df['homeowner'] = df['PTHOME'].isin([1, 2]).astype(int)
    df['social_isolation'] = df['married'] + df['lives_alone'] + df['retired']
    df['education_retired'] = df['PTEDUCAT'] * df['retired']
    df['years_retired'] = (
        pd.to_datetime(df['subject_date']).dt.year - df['PTRTYR']
    ) * df['retired']
    df['married_homeowner'] = df['married'] * df['homeowner']
    df['retired_lives_alone'] = (
        (df['retired'] == 1) & (df['lives_alone'] == 1)
    ).astype(int)

    SES_vars = ['PTEDUCAT', 'homeowner']
    df['SES_score'] = df[SES_vars].apply(
        lambda x: (x - x.mean()) / x.std(), axis=0
    ).mean(axis=1)

    return df


def calculate_libra_revised(row):
    """
    Compute a partial LIBRA (LIfestyle for BRAin health) risk score.

    The LIBRA score aggregates modifiable dementia risk and protective factors
    based on the Lancet Commission framework. Each factor has a weight
    reflecting its relative contribution to dementia risk.

    **Risk factors** (increase score):
      - Depression (GDS ≥ 10): +2.1
      - Hypertension (SBP ≥ 130 or DBP ≥ 80): +1.6
      - Obesity (BMI ≥ 30): +1.6
      - Smoking (MH16ASMOK > 0): +1.5
      - High cholesterol (RCT20 ≥ 240 mg/dL): +1.4
      - Diabetes (Glucose ≥ 126 mg/dL): +1.3

    **Protective factors** (decrease score):
      - Low-to-moderate alcohol (1–2 drinks/day proxy): -1.0
      - High cognitive activity (education > 12 years proxy): -3.2

    .. warning::
        Alcohol use is proxied by ``MH14AALCH`` which measures consumption
        during abuse periods, not regular moderate use. This is a known
        limitation of the available ADNI variables.

    Args:
        row (pd.Series): A single row of subject data.

    Returns:
        float: Partial LIBRA score (higher = more risk).
    """
    score = 0.0
    weights = {
        'depression': 2.1,
        'hypertension': 1.6,
        'obesity': 1.6,
        'smoking': 1.5,
        'high_cholesterol': 1.4,
        'diabetes': 1.3,
        'low_mod_alcohol': -1.0,
        'high_cognitive_activity': -3.2,
    }

    # Risk factors
    if row['GDTOTAL'] >= 10:
        score += weights['depression']
    if row['VSBPSYS'] >= 130 or row['VSBPDIA'] >= 80:
        score += weights['hypertension']
    if row['BMI'] >= 30:
        score += weights['obesity']
    if row['MH16ASMOK'] > 0:
        score += weights['smoking']
    if row['RCT20'] >= 240:
        score += weights['high_cholesterol']
    if row['GLUCOSE'] >= 126:
        score += weights['diabetes']

    # Protective factors
    if 1 <= row['MH14AALCH'] <= 2:
        score += weights['low_mod_alcohol']
    if row['PTEDUCAT'] > 12:
        score += weights['high_cognitive_activity']

    return round(score, 2)


# =============================================================================
# Descriptive Statistics Helper
# =============================================================================

def get_stats(X_train, woe_dict):
    """
    Compute summary statistics for WoE and categorical features.

    Args:
        X_train (pd.DataFrame): Training features.
        woe_dict (dict): WoE feature configuration.

    Returns:
        pd.DataFrame: Combined statistics with count, mean, min, max.
    """
    stats_woe = X_train[sorted(list(woe_dict.keys()))].describe().T[
        ['count', 'mean', 'min', 'max']
    ]
    cat = sorted(list(set(X_train.columns) - set(list(woe_dict.keys()))))
    stats_cat = X_train[cat].count(axis=0).to_frame(name='count')
    stats_cat['unique values'] = X_train[cat].apply(
        lambda col: sorted([np.round(x, 3) for x in col.unique() if ~np.isnan(x)])
    )
    stats_cat = pd.concat(
        [stats_cat, X_train[cat].describe().T[['min', 'max']]], axis=1
    )
    stats = pd.concat([stats_woe, stats_cat])
    stats['count'] = stats['count'].astype(int)
    stats[['mean', 'min', 'max']] = stats[['mean', 'min', 'max']].round(3)
    return stats