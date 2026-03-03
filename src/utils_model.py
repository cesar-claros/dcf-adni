#%%
import pandas as pd
import numpy as np
from optbinning import BinningProcess
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold
from skopt import BayesSearchCV
from sklearn.tree import _tree
from scipy.stats import chi2_contingency
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.compose import ColumnTransformer # Import ColumnTransformer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import RFE
from sklearn.feature_selection import SequentialFeatureSelector # Import SequentialFeatureSelector
from sklearn.base import BaseEstimator, TransformerMixin
import joblib
import shap
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
import re
import json
from tqdm import tqdm
from sklearn import set_config
#%%
def transform_WOE(joint_dataset_df, woe_dict, categorical_variables, idx_train, idx_test):
    all_variables = list(set(woe_dict.keys()).union(set(categorical_variables)))
    X_train_raw = joint_dataset_df.iloc[idx_train]
    X_train_raw = X_train_raw[all_variables]
    y_train = joint_dataset_df.loc[X_train_raw.index]['transition']
    X_test_raw = joint_dataset_df.iloc[idx_test]
    X_test_raw = X_test_raw[all_variables]
    y_test = joint_dataset_df.loc[X_test_raw.index]['transition']
    binning_process = BinningProcess(all_variables,
                                        categorical_variables=categorical_variables,
                                        binning_fit_params=woe_dict,
                                        )
    # Fit and transform training set
    X_train_WOE = binning_process.fit_transform(X_train_raw,y_train, metric='woe')
    X_train_WOE = X_train_WOE.add_suffix('_WOE')
    # Transform test set
    X_test_WOE = binning_process.transform(X_test_raw, metric='woe')
    X_test_WOE = X_test_WOE.add_suffix('_WOE')
    X_train_WOE = -1*X_train_WOE
    X_test_WOE = -1*X_test_WOE
    X_train = pd.merge(left=X_train_raw, right=X_train_WOE, left_index=True, right_index=True )
    X_test = pd.merge(left=X_test_raw, right=X_test_WOE, left_index=True, right_index=True )
    drop_columns = X_train.columns[X_train.nunique()==1]
    X_train = X_train.drop(columns=drop_columns)
    X_test = X_test.drop(columns=drop_columns)
    # classifier = RandomForestClassifier(max_samples=0.90, bootstrap=True, oob_score=True, random_state=seed_rf, ccp_alpha=0.0)
    # grid_classifier = GridSearchCV(classifier, grid, cv=5)
    # grid_classifier.fit(X_train, y_train.values.ravel())
    # best_classifier = grid_classifier.best_estimator_
    # test_score = best_classifier.score(X_test, y_test)
    return X_train, X_test, y_train, y_test, binning_process
#%%
def get_stats(X_train, woe_dict):
    stats_woe = X_train[sorted(list(woe_dict.keys()))].describe().T[['count','mean','min','max']]
    cat = sorted(list(set(X_train.columns)-set(list(woe_dict.keys()))))
    stats_cat = X_train[cat].count(axis=0).to_frame(name='count')
    stats_cat['unique values'] = X_train[cat].apply(lambda col: sorted([np.round(x,3) for x in col.unique() if ~np.isnan(x)]))
    stats_cat = pd.concat([stats_cat,X_train[cat].describe().T[['min','max']]],axis=1)
    stats = pd.concat([stats_woe,stats_cat])
    stats['count'] = stats['count'].astype(int)
    stats[['mean','min','max']] = stats[['mean','min','max']].round(3)
    return stats
#%%
def calculate_dcg(relevance_scores, k=None):
    """
    Calculates the Discounted Cumulative Gain (DCG) for a list of relevance scores.

    Args:
        relevance_scores (list or np.array): A list or array of relevance scores
                                             for items in a ranked list.
        k (int, optional): The position up to which to calculate DCG.
                           If None, DCG is calculated for the entire list.

    Returns:
        float: The DCG score.
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
        # Relevance score of the item at position i (0-indexed)
        relevance = relevance_scores[i]
        # Logarithmic discount factor
        discount = np.log2(i + 2)  # i + 1 for 1-indexed position
        dcg += relevance / discount
    return dcg
#%%
def get_tree_rules(tree, feature_names):
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]

    paths = []
    nodes = []
    thresholds = []
    directions = []
    variables = []
    def recurse(node, path):
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]

            # Left child
            recurse(tree_.children_left[node], path + [f"({name} <= {threshold:.2f})"])
            # Right child
            recurse(tree_.children_right[node], path + [f"({name} > {threshold:.2f})"])
        else:
            direction = [-1 if '<' in p else 1 for p in path]
            threshold = [p[1:-1].split(' ')[2] for p in path]
            var = [p[1:-1].split(' ')[0] for p in path]
            # Leaf node
            paths.append(path)
            nodes.append(node)
            thresholds.append(threshold)
            directions.append(direction)
            variables.append(var)

    recurse(0, [])
    return paths, nodes, thresholds, directions, variables
#%%
def get_xgboost_tree_rules(model, booster_id, feature_names):
    """
    Extracts the rules from a single XGBoost decision tree (booster).

    Args:
        model: The trained XGBoost model.
        booster_id: The index of the tree (booster) to extract rules from.
        feature_names: A list of feature names used in the model.
        
    Returns:
        A tuple containing lists of paths, nodes, thresholds, directions, and variables.
    """
    dump = model.get_dump(with_stats=False, dump_format='text')
    tree_text = dump[booster_id]
    
    paths = []
    nodes = []
    thresholds = []
    directions = []
    variables = []
    
    lines = tree_text.strip().split('\n')
    
    def recurse(node_id, path):
        line = next((l for l in lines if l.strip().startswith(f'{node_id}:')), None)
        if line is None:
            return  # This shouldn't happen
            
        is_leaf = re.search(r'leaf', line)
        if is_leaf:
            # This is a leaf node
            direction = [-1 if '<' in p else 1 for p in path]
            threshold = [p.split(' ')[2].strip(']') for p in path]
            var = [p.split(' ')[0].strip('[') for p in path]
            
            paths.append(path)
            nodes.append(node_id)
            thresholds.append(threshold)
            directions.append(direction)
            variables.append(var)
        else:
            # This is a non-leaf node, extract condition
            match = re.search(r'\[(.*)<(.*)\]', line)
            if match:
                print(line)
                # var_index = int(match.group(1).replace('f', ''))
                variable = match.group(1)
                # variable = feature_names[var_index]
                threshold = float(match.group(2))
                
                # Left child (less than)
                left_id = int(re.search(r'yes=(\d+)', line).group(1))
                recurse(left_id, path + [f"[{variable} < {threshold:.2f}]"])
                
                # Right child (greater than or equal to)
                right_id = int(re.search(r'no=(\d+)', line).group(1))
                recurse(right_id, path + [f"[{variable} >= {threshold:.2f}]"])
                
    recurse(0, [])
    return paths, nodes, thresholds, directions, variables

#%%
def cramers_v(x, y):
    """Calculates Cramér's V statistic for categorical-categorical association.
    
    Args:
        x (pd.Series): First categorical variable.
        y (pd.Series): Second categorical variable.
    
    Returns:
        float: Cramér's V statistic.
    """
    confusion_matrix = pd.crosstab(x, y)
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    minimum_dimension = min(confusion_matrix.shape) - 1
    
    # Avoid division by zero
    if minimum_dimension == 0 or n == 0:
      return 0
    
    return np.sqrt(chi2 / (n * minimum_dimension))
#%%

def rule_correlation(best_classifier, correlation_leaf_target, X_train, y_train, model='rf', seed=None):
    correlation_leaf_target = correlation_leaf_target.copy()
    all_tree_rules_dict = {}
    rule_corr = []
    if model == 'rf':
        for i, tree in enumerate(best_classifier.estimators_):
            rules,nodes,thresholds,directions,variables = get_tree_rules(tree, X_train.columns)
            # all_tree_rules.append(rules)
            # print(f"\nTree {i} rules:")
            for leaf_idx, rule in enumerate(rules):
                # tree_leaf = f'tree_{i}-leaf_{leaf_idx}': f'{rule}'
                all_tree_rules_dict.update( {f'tree_{i}-leaf_{nodes[leaf_idx]}': rule} )
                for r in rule:
                    split_txt = r[1:-1].split(' ')
                    var = split_txt[0]
                    threshold = float(split_txt[2])
                    direction = -1 if '<=' in split_txt[2] else 1
                    evaluation = (X_train[var]<=threshold) if direction==-1 else X_train[var]>threshold
                    y_corr = cramers_v(evaluation,y_train)
                    rule_corr.append( {'rule':r,'correlation':y_corr} )
    elif model=='xgboost':
        booster = best_classifier.get_booster()
        num_trees = len(booster.get_dump())
        for i in range(num_trees):
            rules, nodes, thresholds, directions, variables = get_xgboost_tree_rules(booster, i, X_train.columns)

            for leaf_idx, rule in enumerate(rules):
                # Update the dictionary with the rule for the current leaf
                all_tree_rules_dict.update({f'tree_{i}-leaf_{nodes[leaf_idx]}': rule})
                
                for r_full in rule:
                    # Parse the rule string for variable, threshold, and direction
                    # The format from get_xgboost_tree_rules is "[variable < threshold]" or "[variable >= threshold]"
                    var = r_full.split(' ')[0][1:]  # Extracts variable name, e.g., 'Age' from '[Age'
                    direction_str = r_full.split(' ')[1]  # Extracts '<' or '>='
                    threshold = float(r_full.split(' ')[2][:-1]) # Extracts threshold, e.g., '30.5' from '30.5]'
                    # Determine the evaluation based on the direction
                    evaluation = (X_train[var] < threshold) if direction_str == '<' else (X_train[var] >= threshold)
                    # Calculate correlation and append to the list
                    y_corr = cramers_v(evaluation, y_train)
                    rule_corr.append({'rule': r_full, 'correlation': y_corr})
    elif model=='catboost':
        assert isinstance(seed,int)
        path = f'results/catboost_model_{seed}.json'
        print(f'Saving catboost model in {path}...')
        best_classifier.save_model(path, format='json')
        print(f'Loading catboost model from {path}...')
        with open(path, 'r') as f:
            catboost_json = json.load(f)

        trees_list = catboost_json.get('oblivious_trees')

        if not trees_list:
            raise ValueError("The JSON file does not contain 'oblivious_trees'.")
        feature_names = list(X_train.columns)
        for i, tree_json in tqdm((enumerate(trees_list)), total=len(trees_list), desc="Processing items"):
            rules, nodes, thresholds, directions, variables = get_catboost_oblivious_rules(tree_json, feature_names)
            unique_rules = list(set([x for r in rules for x in r]))
            for leaf_idx, rule in enumerate(rules):
                all_tree_rules_dict.update({f'tree_{i}-leaf_{nodes[leaf_idx]}': rule})
                
            for r_full in unique_rules:
                parts = r_full[1:-1].split(' ')
                var = parts[0]
                operator = parts[1]
                threshold_or_value = parts[2]
                
                if operator == '<=':
                    threshold = float(threshold_or_value)
                    evaluation = (X_train[var] <= threshold)
                elif operator == '>':
                    threshold = float(threshold_or_value)
                    evaluation = (X_train[var] > threshold)
                elif operator == '==':
                    evaluation = (X_train[var].astype(str) == threshold_or_value)
                elif operator == '!=':
                    evaluation = (X_train[var].astype(str) != threshold_or_value)
                
                y_corr = cramers_v(evaluation, y_train)
                rule_corr.append({'rule': r_full, 'correlation': y_corr})
    # print(all_tree_rules_dict)
    univariate_rules_df = pd.DataFrame(rule_corr).drop_duplicates()
    univariate_rules_df['var'] = univariate_rules_df['rule'].apply(lambda x: x[1:-1].split(' ')[0])
    univariate_rules_df = univariate_rules_df.drop(columns='rule').drop_duplicates().sort_values(by='correlation',ascending=False)
    univariate_rules_df = univariate_rules_df.drop_duplicates(subset='var').reset_index(drop=True)
    univariate_rules_df = univariate_rules_df.round(3)
    rules = [all_tree_rules_dict[tl] for tl in correlation_leaf_target['leaf']]
    correlation_leaf_target['rules'] = rules
    # rules_expanded = [f'rule_{i+1}' for i in range(best_classifier.max_depth)]
    rules_expanded_df = pd.DataFrame(correlation_leaf_target['rules'].to_list(), index=correlation_leaf_target.index)
    rules_expanded_df.columns = [f'rule_{i+1}' for i in range(len(rules_expanded_df.columns))]
    rules_expanded_df.index = correlation_leaf_target['leaf']
    correlation_leaf_target = correlation_leaf_target.set_index('leaf')
    correlation_leaf_target = pd.merge(left=correlation_leaf_target, right=rules_expanded_df, left_index=True, right_index=True)
    # rules_expanded_ = pd.DataFrame(correlation_leaf_target['rules'].to_list(), index=correlation_leaf_target.index)
    # # correlation_leaf_target = correlation_leaf_target.head(k_rules)
    correlation_leaf_target['rank'] = correlation_leaf_target['correlation_target'].rank(ascending=True,method='min').astype(int)
    # rules_expanded = [f'rule_{i+1}' for i in range(best_classifier.max_depth)]
    vars_ = rules_expanded_df.map(lambda x: x[1:-1].split(' ')[0], na_action='ignore').T
    vars_bool_df = pd.concat([univariate_rules_df['var'].isin(vars_[col]).rename(col) for col in vars_.columns],axis=1)
    vars_bool_df = vars_bool_df.set_index(univariate_rules_df['var']).T
    vars_bool_df['rank'] = correlation_leaf_target['rank']
    
    vars_dcg = {}
    for var in univariate_rules_df['var']:
        ranks_list = vars_bool_df[vars_bool_df[var]]['rank'].to_list()
        ideal_rank = np.arange(start=correlation_leaf_target.shape[0],stop=1,step=-1)
        dcg = calculate_dcg(ranks_list)
        ideal_dcg = calculate_dcg(ideal_rank)
        vars_dcg.update( {var: dcg/ideal_dcg if ideal_dcg!=0 else 0} )
    vars_dcg_df = pd.Series(vars_dcg,name='nDCG').sort_values(ascending=False)
    # Get MDI feature importances
    feature_importances = best_classifier.feature_importances_
    feature_names = X_train.columns
    importance_mdi_df = pd.DataFrame({'variable':feature_names,'MDI':feature_importances}).sort_values(by=['MDI'],ascending=False)
    importance_mdi_df = importance_mdi_df.set_index('variable').sort_values(by='MDI', ascending=False)
    # SHAP
    explainer = shap.TreeExplainer(best_classifier)
    # Calculate SHAP values for the test set
    shap_values = explainer.shap_values(X_train.values)
    # For binary classification, focus on the positive class (index 1)
    if model=='rf':
        shap_values_positive_class = shap_values[:,:,1]
    elif model=='xgboost' or model=='catboost':
        shap_values_positive_class = shap_values
    shap_df = pd.DataFrame({'feature':X_train.columns,'mean_|SHAP|':np.abs(shap_values_positive_class).mean(axis=0)}).set_index('feature')
    # Scores
    all_scores_df = pd.concat([univariate_rules_df.set_index('var')['correlation'],vars_dcg_df,importance_mdi_df['MDI'],shap_df],axis=1)
    all_scores_df = all_scores_df.iloc[np.linalg.norm(all_scores_df,axis=1).argsort()[::-1]]
    all_scores_df = all_scores_df.dropna()
    return all_scores_df


#%%
# vars_df = pd.read_csv('adni_variables.csv')
def feature_engineering(joint_dataset_df):
    joint_dataset_df = joint_dataset_df.copy()
    # Feature engineering
    # Model A
    joint_dataset_df['PTAU/ABETA42'] = joint_dataset_df['PTAU']/joint_dataset_df['ABETA42']
    joint_dataset_df['TAU/ABETA42'] = joint_dataset_df['TAU']/joint_dataset_df['ABETA42']
    joint_dataset_df['medical_current'] = joint_dataset_df['MHNUM'] * (joint_dataset_df['MHCUR']==1)
    joint_dataset_df['medical_old'] = joint_dataset_df['MHNUM'] * (joint_dataset_df['MHCUR']==0)
    biomarker_vars = ['ABETA42', 'PTAU', 'TAU', 'BAT126', 'RCT392']

    # Model B
    joint_dataset_df['married'] = (joint_dataset_df['PTMARRY'] == 1).astype(int)
    joint_dataset_df['lives_alone'] = joint_dataset_df['PTHOME'].isin([3, 4]).astype(int)
    joint_dataset_df['retired'] = (joint_dataset_df['PTNOTRT'] == 1).astype(int)
    joint_dataset_df['homeowner'] = joint_dataset_df['PTHOME'].isin([1, 2]).astype(int)
    joint_dataset_df['social_isolation'] = (
        joint_dataset_df['married'] +
        joint_dataset_df['lives_alone'] +  # assuming 3,4 = lives alone/assisted
        joint_dataset_df['retired']
    ) 
    joint_dataset_df['education_retired'] = joint_dataset_df['PTEDUCAT'] * joint_dataset_df['retired']
    joint_dataset_df['years_retired'] = (pd.to_datetime(joint_dataset_df['subject_date']).dt.year - joint_dataset_df['PTRTYR']) * joint_dataset_df['retired']
    joint_dataset_df['married_homeowner'] = joint_dataset_df['married'] * joint_dataset_df['homeowner']
    joint_dataset_df['retired_lives_alone'] = ((joint_dataset_df['retired'] == 1) & (joint_dataset_df['lives_alone'] == 1)).astype(int)
    SES_vars = ['PTEDUCAT', 'homeowner']
    joint_dataset_df['SES_score'] = joint_dataset_df[SES_vars].apply(lambda x: (x - x.mean()) / x.std(), axis=0).mean(axis=1)
    return joint_dataset_df

#%%
def leaf_correlation(best_classifier, X_train, y_train, X_test, y_test, X_all_test, y_all_test, model='rf'):
    if model =='rf':
        n_trees = len(best_classifier.estimators_)
    elif model =='xgboost':
        n_trees = best_classifier.n_estimators
    elif model =='catboost':
        n_trees = best_classifier.tree_count_
    tree_leaves_arrays = []
    for X_data,y_data in [(X_train, y_train), (X_all_test, y_all_test)]:
        # for label in [0,1]:
            # X = X_data[(y_data==label)]
        X = X_data
        if model=='catboost':
            tree_leaf_pred = best_classifier.calc_leaf_indexes(X)
        else:
            tree_leaf_pred = best_classifier.apply(X)
        tree_dict = {f'tree_{k}':tree_leaf_pred[:,k] for k in range(n_trees)}
        # print(tree_dict)
        # tree_dict.update({'id':X.index})
        tree_leaves_dict = {f'{tree}-leaf_{int(leaf)}':(tree_dict[tree]==leaf).astype(int) for tree in tree_dict for leaf in np.unique(tree_dict[tree])}
        tree_leaves_dict_df = pd.DataFrame(tree_leaves_dict, index=X.index)
        tree_leaves_arrays.append(tree_leaves_dict_df)
    # print(tree_leaves_arrays)
    leaf_membership_df = pd.concat(tree_leaves_arrays).fillna(0).astype(int)
    leaf_membership_train = leaf_membership_df.loc[y_train.index]
    leaf_membership_all_test = leaf_membership_df.loc[y_all_test.index] 
    leaf_membership_test = leaf_membership_all_test.loc[y_test.index]
    correlation_leaf_target = pd.DataFrame([{'leaf':col,
                                 'correlation_target':cramers_v(leaf_membership_train[col],y_train)} for col in leaf_membership_train.columns]).sort_values(by='correlation_target', ascending=False)
    ordered_columns = correlation_leaf_target['leaf']
    leaf_membership_train = leaf_membership_train[ordered_columns]
    leaf_membership_test = leaf_membership_test[ordered_columns]
    leaf_membership_all_test = leaf_membership_all_test[ordered_columns]

    return leaf_membership_train, leaf_membership_test, leaf_membership_all_test, correlation_leaf_target
#%%
def train_model(X_cv_train, y_cv_train, X_cv_test, y_cv_test, param_space, model='rf', seed_rf=0, seed_bayes=0, cv=10, n_iter=100, groups=None, cat_vars=None):
    # Define the RandomForest model
    if model=='rf':
        m = RandomForestClassifier(max_samples=1.0, bootstrap=True, oob_score=True, random_state=seed_rf)
    elif model=='xgboost':
        m = XGBClassifier(
                    objective='binary:logistic',
                    eval_metric='logloss', # Specify evaluation metric for early stopping
                    use_label_encoder=False, # Suppress the warning
                    random_state=seed_rf
                )
    elif model=='catboost':
        m = CatBoostClassifier(
                    verbose=0,  # Suppress verbosity during fitting
                    random_state=seed_rf,
                    cat_features=cat_vars,
                )

    # Set up the Bayesian optimization search
    bayes_search = BayesSearchCV(m, param_space, n_iter=n_iter, cv=cv, n_jobs=-1, verbose=0, random_state=seed_bayes)
    # Fit the model
    if groups is None:
        bayes_search.fit(X_cv_train, y_cv_train.values.squeeze())
    else:
        bayes_search.fit(X = X_cv_train, y = y_cv_train.values.squeeze(), groups = groups)
    # Get the best parameters
    print("Best Parameters:", bayes_search.best_params_)
    # Evaluate the best model on the test set
    test_score = bayes_search.best_estimator_.score(X_cv_test, y_cv_test.values.squeeze())
    print("CV Test Set Accuracy:", test_score)
    return bayes_search, test_score
#%%
class RandomFeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, n_features_to_select=None,):
        self.n_features_to_select = n_features_to_select
        # self.random_state = random_state
        self.selected_features_ = None # Store the actual features chosen

    def fit(self, X, y=None):
        # np.random.seed(self.random_state)
        # Ensure n_features_to_select doesn't exceed the available features
        if self.n_features_to_select is None or self.n_features_to_select > X.shape[1]:
            self.selected_features_ = X.columns.tolist() # Select all features if None or exceeding
        else:
            self.selected_features_ = X.columns[:self.n_features_to_select].tolist() # Select all features if None or exceeding
            # self.selected_features_ = np.random.choice(
            #     X.columns,
            #     self.n_features_to_select,
            #     replace=False
            # ).tolist()
        return self

    def transform(self, X, y=None):
        return X[self.selected_features_]
#%%
class LeafCorrelationTransformation(BaseEstimator, TransformerMixin):
    def __init__(self, n_features_to_select=None,):
        self.n_features_to_select = n_features_to_select
        # self.random_state = random_state
        self.selected_features_ = None # Store the actual features chosen

    def fit(self, X, y=None):
        # np.random.seed(self.random_state)
        # Ensure n_features_to_select doesn't exceed the available features
        if self.n_features_to_select is None or self.n_features_to_select > X.shape[1]:
            self.selected_features_ = X.columns.tolist() # Select all features if None or exceeding
        else:
            self.selected_features_ = X.columns[:self.n_features_to_select].tolist() # Select all features if None or exceeding
            # self.selected_features_ = np.random.choice(
            #     X.columns,
            #     self.n_features_to_select,
            #     replace=False
            # ).tolist()
        return self

    def transform(self, X, y=None):
        return X[self.selected_features_]
#%%
def search_rules(df1_train, df2_train, y_cv_train, df1_test, df2_test, y_cv_test, param_space, model='rf', seed_rf=0, seed_bayes=0, cv=10, n_iter=100, groups=None, cat_vars=None):
    set_config(transform_output="pandas")
    X_df1_df2 = pd.merge(left=df1_train,right=df2_train,left_index=True,right_index=True)
    df2_features = df2_train.columns.tolist()
    # Define the ColumnTransformer to apply SelectKBest only to df2's features
    preprocessor = ColumnTransformer(
        transformers = [
            ('feature_selection', RFE(estimator=RandomForestClassifier(random_state=seed_rf),n_features_to_select=2), df2_features),
        ],
        remainder = 'passthrough', # Passes df1's features through without transformation
        verbose_feature_names_out = False
    )
    # Define the hyperparameter grid
    param_grid = {
        'preprocessing__feature_selection__n_features_to_select': (1,len(df2_features)),  # Number of features to select from df2
    }
    param_grid.update({f'model__{k}':param_space[k] for k in param_space })
    if model=='rf':
        m = RandomForestClassifier(
                    max_samples=1.0, 
                    bootstrap=True, 
                    oob_score=True, 
                    random_state=seed_rf)
    elif model=='xgboost':
        m = XGBClassifier(
                    objective='binary:logistic',
                    eval_metric='logloss', # Specify evaluation metric for early stopping
                    use_label_encoder=False, # Suppress the warning
                    random_state=seed_rf
                )
    elif model=='catboost':
        m = CatBoostClassifier(
                    verbose=0,  # Suppress verbosity during fitting
                    random_state=seed_rf,
                    cat_features=cat_vars,
                )
    # Define the pipeline steps
    pipeline_steps = [
        ('preprocessing', preprocessor),
        ('model', m)
    ]
    pipeline = Pipeline(pipeline_steps)
    # Set up the Bayesian optimization search
    bayes_search = BayesSearchCV(pipeline, param_grid, n_iter=n_iter, cv=cv, n_jobs=-1, verbose=0, random_state=seed_bayes)
    # Fit the model
    if groups is None:
        bayes_search.fit(X=X_df1_df2, y=y_cv_train.values.squeeze())
    else:
        bayes_search.fit(X = X_df1_df2, y = y_cv_train.values.squeeze(), groups=groups)
    # Get the best parameters
    print("Best Parameters:", bayes_search.best_params_)
    # Evaluate the best model on the test set
    X_test = pd.merge(left=df1_test,right=df2_test,left_index=True,right_index=True)
    test_score = bayes_search.best_estimator_.score(X_test, y_cv_test.values.squeeze())
    print("CV Test Set Accuracy:", test_score)
    
    return bayes_search, test_score
#%%
# def search_rules(df1_train, df2_train, y_cv_train, df1_test, df2_test, y_cv_test, param_space, model='rf', seed_rf=0, seed_bayes=0, cv=10, n_iter=100, groups=None, cat_vars=None):
#     # Identify the columns from df1 and df2
#     # df1_features = df1_train.columns.tolist()
#     df2_features = df2_train.columns.tolist()
#     # print(df2_features)

#     # Define the ColumnTransformer to apply SelectKBest only to df2's features
#     preprocessor = ColumnTransformer(
#         transformers=[
#             # ('feature_selection', RandomFeatureSelector(), df2_features), # Applies custom selector to df2_features
#             # ('select_features_from_df2', RFECV(estimator=RandomForestClassifier(random_state=42)), df2_features), # Applies SelectKBest to df2_features
#             ('feature_selection', RFE(estimator=RandomForestClassifier(random_state=seed_rf)), df2_features),
#         #     ('sequential_feature_selection',
#         #  SequentialFeatureSelector(estimator=RandomForestClassifier(random_state=42), # SFS needs an estimator
#         #                            direction='forward', # 'forward' or 'backward'
#         #                            cv=3, # Cross-validation folds for feature selection
#         #                            scoring='accuracy'), # Scoring metric for feature selection
#         #  df2_features),
#         ],
#         remainder='passthrough' # Passes df1's features through without transformation
#     )
#     # Define the hyperparameter grid
#     param_grid = {
#         'preprocessing__feature_selection__n_features_to_select': (1,len(df2_features)),  # Number of features to select from df2
#         # 'preprocessing__select_features_from_df2__min_features_to_select': (1,len(df2_features)),
#         # 'preprocessing__sequential_feature_selection__n_features_to_select': (1,len(df2_features)),
#     }
#     param_grid.update({f'model__{k}':param_space[k] for k in param_space })
#     if model=='rf':
#         m = RandomForestClassifier(
#                     max_samples=1.0, 
#                     bootstrap=True, 
#                     oob_score=True, 
#                     random_state=seed_rf)
#     elif model=='xgboost':
#         m = XGBClassifier(
#                     objective='binary:logistic',
#                     eval_metric='logloss', # Specify evaluation metric for early stopping
#                     use_label_encoder=False, # Suppress the warning
#                     random_state=seed_rf
#                 )
#     elif model=='catboost':
#         m = CatBoostClassifier(
#                     verbose=0,  # Suppress verbosity during fitting
#                     random_state=seed_rf,
#                     cat_features=cat_vars,
#                 )
#     # Define the pipeline steps
#     pipeline_steps = [
#         ('preprocessing', preprocessor),
#         ('model', m)
#     ]
#     pipeline = Pipeline(pipeline_steps)
#     # Set up the Bayesian optimization search
#     bayes_search = BayesSearchCV(pipeline, param_grid, n_iter=n_iter, cv=cv, n_jobs=-1, verbose=0, random_state=seed_bayes)
#     # Fit the model
#     if groups is None:
#         bayes_search.fit(pd.merge(left=df1_train,right=df2_train,left_index=True,right_index=True), y_cv_train.values.squeeze())
#     else:
#         bayes_search.fit(X = pd.merge( left=df1_train,right=df2_train,left_index=True,right_index=True), y = y_cv_train.values.squeeze(), groups=groups)
#     # Get the best parameters
#     print("Best Parameters:", bayes_search.best_params_)
#     # Evaluate the best model on the test set
#     X_test = pd.merge(left=df1_test,right=df2_test,left_index=True,right_index=True)
#     test_score = bayes_search.best_estimator_.score(X_test, y_cv_test.values.squeeze())
#     print("CV Test Set Accuracy:", test_score)
    
#     return bayes_search, test_score

def calculate_libra_revised(row):
    """
    Calculates a partial LIBRA score based.
    """
    score = 0
    
    # Define weights for each factor
    weights = {
        'depression': 2.1,
        'hypertension': 1.6,
        'obesity': 1.6,
        'smoking': 1.5,
        'high_cholesterol': 1.4,
        'diabetes': 1.3,
        'low_mod_alcohol': -1.0,
        'high_cognitive_activity': -3.2
    }

    # --- Risk Factors (add to score) ---

    # 1. Depression: GDS score >= 10
    if row['GDTOTAL'] >= 10:
        score += weights['depression']

    # 2. Hypertension: Systolic >= 130 mmHg OR Diastolic >= 80 mmHg
    if row['VSBPSYS'] >= 130 or row['VSBPDIA'] >= 80:
        score += weights['hypertension']

    # 3. Obesity: BMI >= 30
    # Assumption: VSWEIGHT is in kg and VSHEIGHT is in cm.
    # bmi = calculate_bmi(row['VSWEIGHT'], row['VSHEIGHT'])
    bmi = row['BMI']
    if bmi >= 30:
        score += weights['obesity']

    # 4. Smoking: Any history of smoking adds risk.
    # Assumption: MH16ASMOK > 0 indicates a significant smoking history.
    if row['MH16ASMOK'] > 0:
        score += weights['smoking']
        
    # 5. High Cholesterol: Total Cholesterol >= 240mg/dL
    # Assumption: A total cholesterol level of 240 mg/dL or higher is considered high.
    if row['RCT20'] >= 240:
        score += weights['high_cholesterol']
        
    # 6. Diabetes: Glucose >= 126 mg/dL
    # Assumption: A fasting glucose level of 126 mg/dL or higher indicates diabetes.
    if row['GLUCOSE'] >= 126:
        score += weights['diabetes']

    # --- Protective Factors (subtract from score) ---

    # 7. Low-to-Moderate Alcohol Use: 1-2 drinks per day (proxy)
    # WARNING: This is a weak proxy. The variable measures consumption during periods of ABUSE,
    # not regular moderate use. Using it this way assumes that 1-2 drinks/day during such a
    # period is equivalent to general low-to-moderate consumption (7-14 drinks/week).
    if 1 <= row['MH14AALCH'] <= 2:
        score += weights['low_mod_alcohol']
    
    # 8. High Cognitive Activity: > 12 years of education (proxy)
    # Assumption: More than 12 years of education serves as a proxy for high cognitive activity/reserve.
    if row['PTEDUCAT'] > 12:
        score += weights['high_cognitive_activity']
        
    return round(score, 2)

# Calculate BMI and add it as a column for reference
# df['BMI'] = df.apply(lambda row: calculate_bmi(row['VSWEIGHT'], row['VSHEIGHT']), axis=1)

# Apply the function to calculate the revised LIBRA score
# df['LIBRA_score_revised'] = df.apply(calculate_libra_revised, axis=1)

# print("--- Revised Partial LIBRA Score Calculation ---")
# print(df)

def calculate_bmi(weight_kg, height_cm):
    """Calculates BMI from weight in kg and height in cm."""
    if height_cm == 0:
        return np.nan
    return weight_kg / ((height_cm / 100) ** 2)

#%%
def get_catboost_oblivious_rules(tree_data, feature_names):
    """
    Extracts rules from an oblivious CatBoost tree's JSON structure.
    
    Args:
        tree_data: The JSON dictionary for a single oblivious tree.
        feature_names: A list of feature names.
    
    Returns:
        A tuple containing lists of paths, nodes, thresholds, directions, and variables.
    """
    splits = tree_data['splits']
    leaf_values = tree_data['leaf_values']
    
    paths = []
    nodes = []
    thresholds = []
    directions = []
    variables = []
    
    num_leaves = len(leaf_values)
    
    for i in range(num_leaves):
        current_path = []
        current_thresholds = []
        current_directions = []
        current_variables = []
        
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
            
            elif split_type == 'OneHotFeature' or split_type == 'BinarizedFeature':
                # Note: CatBoost may use different split types for categorical features.
                # Adjust the key names if your data uses something other than 'cat_feature_index'
                feature_index = split['cat_feature_index']
                variable = feature_names[feature_index]
                value = split['value'] # Or 'cat_feature_value' depending on version
                
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
                # OnlineCtr is a special form of binarization. The split is `ctr > threshold`.
                if is_right_branch:
                    current_path.append(f"[{variable} (OnlineCtr) > {threshold}]")
                    current_directions.append(1)
                else:
                    current_path.append(f"[{variable} (OnlineCtr) <= {threshold}]")
                    current_directions.append(-1)
                current_thresholds.append(threshold)
                current_variables.append(variable)
                
            # You might need to add more elif blocks for other split types if they appear.
            
        paths.append(current_path)
        nodes.append(i)
        thresholds.append(current_thresholds)
        directions.append(current_directions)
        variables.append(current_variables)
        
    return paths, nodes, thresholds, directions, variables

# %%
def all_tree_rules(best_classifier, X_train, y_train, model='rf', seed=None):
    # correlation_leaf_target = correlation_leaf_target.copy()
    all_tree_rules_dict = {}
    rule_corr = []
    if model=='catboost':
        assert isinstance(seed,int)
        path = f'results/catboost_model_{seed}.json'
        print(f'Saving catboost model in {path}...')
        best_classifier.save_model(path, format='json')
        print(f'Loading catboost model from {path}...')
        with open(path, 'r') as f:
            catboost_json = json.load(f)

        trees_list = catboost_json.get('oblivious_trees')

        if not trees_list:
            raise ValueError("The JSON file does not contain 'oblivious_trees'.")
        feature_names = list(X_train.columns)
        for i, tree_json in tqdm((enumerate(trees_list)), total=len(trees_list), desc="Processing items"):
            rules, nodes, thresholds, directions, variables = get_catboost_oblivious_rules(tree_json, feature_names)
            unique_rules = list(set([x for r in rules for x in r]))
            for leaf_idx, rule in enumerate(rules):
                all_tree_rules_dict.update({f'tree_{i}-leaf_{nodes[leaf_idx]}': rule})
                
    return all_tree_rules_dict
#%%
def dcg_score(all_trees, correlation, unique_features, leaf_counts, leaf_weights):
    rules_expanded = pd.concat([pd.Series(all_trees[x],name=x) for x in all_trees], axis=1)
    correlation_leaf_target = correlation.set_index('leaf')
    correlation_leaf_target['rank_corr'] = correlation_leaf_target['correlation_target'].rank(ascending=True,method='min').astype(int)
    leaf_weights_df = pd.Series(leaf_weights,index=[f'tree_{k}-leaf_{j}' for k in range(len(leaf_counts)) for j in range(leaf_counts[k]) ]).to_frame(name='weight')
    leaf_weights_df = leaf_weights_df.loc[correlation_leaf_target.index]
    leaf_weights_df['rank_weight'] = leaf_weights_df['weight'].rank(ascending=True,method='min').astype(int)
    vars_expanded = {}
    for col in rules_expanded.columns:
        tree_idx = int(col.split('_')[1])
        for j in range( leaf_counts[tree_idx]):
            vars_expanded.update({f'{col}-leaf_{j}':rules_expanded[col]})
    #  rules_expanded.map(remove_threshold, na_action='ignore')
    vars_expanded = pd.DataFrame(vars_expanded)
    unique_features_names = unique_features.reset_index()['index']
    vars_bool = pd.concat([unique_features_names.isin(vars_expanded[col]).rename(col) for col in vars_expanded.columns],axis=1)
    vars_bool = vars_bool.set_index(unique_features.index).T
    vars_bool = pd.merge(left=vars_bool, right=correlation_leaf_target['rank_corr'], left_index=True, right_index=True, how='inner')
    vars_bool = pd.merge(left=vars_bool, right=leaf_weights_df['rank_weight'], left_index=True, right_index=True, how='inner')
    vars_dcg = {}
    for var in unique_features.index:
        ranks_list = vars_bool[vars_bool[var]]['rank_corr'].to_list()
        ideal_rank = np.arange(start=correlation_leaf_target.shape[0],stop=1,step=-1)
        dcg = calculate_dcg(ranks_list)
        ideal_dcg = calculate_dcg(ideal_rank)
        vars_dcg.update( {var: dcg/ideal_dcg if ideal_dcg!=0 else 0} )
    vars_dcg_score_corr = pd.Series(vars_dcg,name='nDCG').sort_values(ascending=False)

    vars_dcg = {}
    for var in unique_features.index:
        ranks_list = vars_bool[vars_bool[var]]['rank_weight'].to_list()
        ideal_rank = np.arange(start=correlation_leaf_target.shape[0],stop=1,step=-1)
        dcg = calculate_dcg(ranks_list)
        ideal_dcg = calculate_dcg(ideal_rank)
        vars_dcg.update( {var: dcg/ideal_dcg if ideal_dcg!=0 else 0} )
    vars_dcg_score_weight = pd.Series(vars_dcg,name='nDCG').sort_values(ascending=False)

    return vars_dcg_score_corr, vars_dcg_score_weight
#%%
def get_unique_features(split_data):
    unique_features = []
    pattern = r"\{([^}]+)\}"
    for x in split_data:
        ctrs = re.findall(pattern, x)
        if len(ctrs)==0:
            feature = x.split(',')
            unique_features.append(feature[0])
        else:
            feature = ctrs[0].split(',')
            if len(feature)==1:
                unique_features.append(feature[0])
            else:
                for i,k in enumerate(feature):
                    feat = k.split(' ')
                    if i==0:
                        unique_features.append(feat[0])
                    else:
                        unique_features.append(feat[1])
    return list(set(unique_features))
#%%
def get_leaf_path(leaf_idx, split_data):
    """
    Reconstructs the full path of split conditions for a given leaf.

    Args:
        tree_idx (int): The index of the tree.
        leaf_idx (int): The index of the leaf within that tree.
        split_data (list): The list of split conditions from the C++ object.
        depth (int): The maximum depth of the tree.
        feature_names (list): The names of the features.

    Returns:
        list: A list of string descriptions for each split operation.
    """
    border_string = 'border='
    bin_string = 'bin='
    value_string = 'value='
    path = []
    # features = []
    depth = len(split_data)
    if depth==0:
        return path
    else:
        binary_path = format(leaf_idx, f'0{depth}b')
    
    for level, decision in enumerate(binary_path):
        split_info = split_data[level]
        if border_string in split_info.split(',')[-1]:
            strings = split_info.split(', '+border_string)
            feature_name = strings[0]
            threshold = strings[1]
            operations = ['<=','>']
        elif bin_string in split_info.split(',')[-1]:
            strings = split_info.split(', '+bin_string)
            feature_name = strings[0]
            threshold = strings[1]
            operations = ['<=','>']
        elif value_string in split_info.split(',')[-1]:
            strings = split_info.split(', '+value_string)
            feature_name = strings[0]
            threshold = strings[1]
            operations = ['!=','=']
        
        path.append(f"{feature_name} {operations[int(decision)]} {threshold}")

    return path

#%%
def get_all_trees_rules(best_estimator, train_pool):
    tree_leaf_counts = best_estimator.get_tree_leaf_counts()
    cb_object = best_estimator._object
    tree_count = cb_object._get_tree_count()
    all_tree_rules_dict = {}
    all_tree_features_dict = {}
    for tree_idx in tqdm((range(tree_count)), total=tree_count, desc="Processing items"):
        split_data = cb_object._get_tree_splits(tree_idx, train_pool)
        leaf_values = cb_object._get_tree_leaf_values(tree_idx)
        # print(split_data)
        for k in range(tree_leaf_counts[tree_idx]):
            leaf_path = get_leaf_path(k, split_data)
            all_tree_rules_dict.update( {f'tree_{tree_idx}-leaf_{k}': leaf_path} )
        features = get_unique_features(split_data)    
        all_tree_features_dict.update( {f'tree_{tree_idx}': features} )
    all_rules = []
    for x in all_tree_rules_dict:
        all_rules.extend(all_tree_rules_dict[x])
    all_rules = pd.Series(all_rules).value_counts().to_frame(name='counts')
    all_features = []
    for x in all_tree_features_dict:
        all_features.extend(all_tree_features_dict[x])
    all_features = pd.Series(all_features).value_counts().to_frame(name='counts')
    # unique_rules = list(set(all_rules))
    return all_tree_rules_dict, all_tree_features_dict, all_rules, all_features
# %%