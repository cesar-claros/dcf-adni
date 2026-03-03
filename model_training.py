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
import argparse
from src.utils_model import *
from sklearn.linear_model import LogisticRegression
from skopt.space import Real, Integer, Categorical
from sklearn.model_selection import cross_val_predict, cross_val_score, cross_validate
from sklearn.metrics import RocCurveDisplay
import json
import re
from catboost import Pool

#%%
# WoE configs
woe_dict_biom = {
                'ABETA42': {"monotonic_trend":'auto_asc_desc' },
                'BAT126': {"monotonic_trend":'auto_asc_desc' },
                'PTAU': {"monotonic_trend":'auto_asc_desc' },
                'RCT14': {"monotonic_trend":'auto_asc_desc' },
                'RCT392': {"monotonic_trend":'auto_asc_desc' },
                'TAU': {"monotonic_trend":'auto_asc_desc' },
                'FAQTOTAL': {"monotonic_trend":'auto_asc_desc' },
                'LDELTOTAL': {"monotonic_trend":'auto_asc_desc' },
                'LIMMTOTAL': {"monotonic_trend":'auto_asc_desc' },
                'MMSCORE': {"monotonic_trend":'auto_asc_desc' },
                'TOTAL13': {"monotonic_trend":'auto_asc_desc' },
                'TOTSCORE': {"monotonic_trend":'auto_asc_desc' },
                'TRAASCOR': {"monotonic_trend":'auto_asc_desc' },
                'GDTOTAL': {"monotonic_trend":'auto_asc_desc' },
                'HMT40': {"monotonic_trend":'auto_asc_desc' },
                'NPIDSEV': {"monotonic_trend":'auto_asc_desc' },
                'NPIDTOT': {"monotonic_trend":'auto_asc_desc' },
                'BSXCHRON': {"monotonic_trend":'auto_asc_desc' },
                'BSXSEVER': {"monotonic_trend":'auto_asc_desc' },
                'HMSCORE': {"monotonic_trend":'auto_asc_desc' },
                'DXCONFID': {"monotonic_trend":'auto_asc_desc' },
                'subject_age': {"monotonic_trend":'ascending' },
                # 'BSXSYMNO':{'cat_unknown':0},
                'PTAU/ABETA42': {"monotonic_trend":'auto_asc_desc' },
                'TAU/ABETA42': {"monotonic_trend":'auto_asc_desc' },
            }
categorical_variables_biom = ['FAQFINAN','FAQTRAVL','NXGAIT','BCDPMOOD','NPID',
                        'NPID8','BCHDACHE','BCMUSCLE','BCVISION','BSXSYMNO',
                        'MHCUR','MHNUM','PTGENDER','medical_old','medical_current']
woe_dict_mrf = {
                'years_retired': {"monotonic_trend":'auto_asc_desc' },
                'GLUCOSE': {"monotonic_trend":'auto_asc_desc' },
                'NPIKTOT': {"monotonic_trend":'auto_asc_desc' },
                'RCT1408': {"monotonic_trend":'auto_asc_desc' },
                'RCT19': {"monotonic_trend":'auto_asc_desc' },
                'RCT20': {"monotonic_trend":'auto_asc_desc' },
                'VSBPDIA': {"monotonic_trend":'auto_asc_desc' },
                'VSBPSYS': {"monotonic_trend":'auto_asc_desc' },
                'BMI': {"monotonic_trend":'auto_asc_desc' },
                'MH14AALCH': {"monotonic_trend":'auto_asc_desc' },
                'MH16ASMOK': {"monotonic_trend":'auto_asc_desc' },
                'MH16CSMOK': {"monotonic_trend":'auto_asc_desc' },
                'subject_age': {"monotonic_trend":'ascending' },
                'PTEDUCAT': {"monotonic_trend":'descending' },
                'social_isolation': {"monotonic_trend":'auto_asc_desc' },
                'education_retired': {"monotonic_trend":'descending' },
                'NPIK9A': {"monotonic_trend":'auto_asc_desc' },
                'NPIK9B': {"monotonic_trend":'auto_asc_desc' },
                'NPIK9C': {"monotonic_trend":'auto_asc_desc' },
            }
categorical_variables_mrf = ['PTHOME','PTMARRY','PTNOTRT','PTPLANG','PTGENDER',
                            'HMHYPERT','NPIK','NPIK1','NPIK2','NPIK4','NPIK6',
                            'MH14ALCH','MH16SMOK','BCINSOMN',
                            'MH12RENA','MH4CARD','NXAUDITO','PXHEART','PXPERIPH',
                            'lives_alone','married_homeowner',
                            'retired_lives_alone','homeowner']
#%%
def train_test_splits(seed_split:int, model_name:str):
    # Settings
    n_iter = 50
    n_seeds = 1
    # cv = 5
    n_splits = 5
    n_repeats = 1
    n_rules = 100
    n_subset = 30

    seed_cv = 0 # For consistency
    seed_rf = 0
    seed_lr = 0
    seed_bayes = 0
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed_split)
    cv_group_train = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed_cv)
    param_space_catboost = {
        'iterations': Integer(100, 1000),
        'learning_rate': Real(1e-3, 1.0, 'log-uniform'),
        'depth': Integer(3, 10),
        'l2_leaf_reg': Real(1, 10, 'uniform'),
        'bagging_temperature': Real(0.0, 1.0, 'uniform'),
        'border_count': Integer(32, 255)
    }

    param_space_xgboost = {
        'n_estimators': Integer(100, 1000),
        'learning_rate': Real(0.01, 0.3, 'log-uniform'),
        'max_depth': Integer(4, 20),
        'subsample': Real(0.5, 1.0, 'uniform'),
        'colsample_bytree': Real(0.5, 1.0, 'uniform'),
        'gamma': Real(0.0, 5.0, 'uniform'),
        'reg_alpha': Real(0.0, 10.0, 'uniform'),
        'reg_lambda': Real(1.0, 10.0, 'uniform')
    }

    param_space_rf = {
        'n_estimators': Integer(100, 300),  # Range of number of trees
        'max_depth': Integer(4, 20),        # Depth of trees
        'min_samples_split': Integer(5, 20),  # Minimum samples required to split a node
        'min_samples_leaf': Integer(5, 20),   # Minimum samples required to be at a leaf node
        'max_features': ['sqrt', 'log2']  # Features to consider for best split
    }

    if model_name=='xgboost':
        param_space = param_space_xgboost
    elif model_name=='catboost':
        param_space = param_space_catboost
    elif model_name=='rf':
        param_space = param_space_rf
    # Read data
    joint_dataset_df = pd.read_csv('joint_dataset.csv', index_col=0).set_index('subject_id')
    remaining_test_df = pd.read_csv('remaining_test.csv', index_col=0).set_index('subject_id')
    dataset_df = feature_engineering(joint_dataset_df)
    additional_test_df = feature_engineering(remaining_test_df)
    # results_dict = {}

    for k, (train_index, test_index) in enumerate(sgkf.split(dataset_df.drop(['transition'],axis='columns'), dataset_df['transition'], dataset_df['group'])):
        X_mrf_train, X_mrf_test, y_train, y_test, bp_mrf = transform_WOE(dataset_df, woe_dict_mrf, categorical_variables_mrf, train_index, test_index)
        X_biom_train, X_biom_test, _, _, bp_biom = transform_WOE(dataset_df, woe_dict_biom, categorical_variables_biom, train_index, test_index)
        repeated_vars = list(set(X_mrf_train.columns).intersection(set(X_biom_train.columns)))
        X_biom_mrf_train = pd.merge(left=X_biom_train, right=X_mrf_train.drop(columns=repeated_vars), left_index=True, right_index=True)
        X_biom_mrf_test = pd.merge(left=X_biom_test, right=X_mrf_test.drop(columns=repeated_vars), left_index=True, right_index=True)
        
        # Additional test samples
        X_nt_mrf_WOE_test = -1*bp_mrf.transform(additional_test_df[bp_mrf.variable_names], metric='woe')
        X_nt_mrf_WOE_test = X_nt_mrf_WOE_test.add_suffix('_WOE')
        X_nt_mrf_test = pd.merge(left=additional_test_df[woe_dict_mrf.keys()], right=X_nt_mrf_WOE_test, left_index=True, right_index=True )
        X_nt_biom_WOE_test = -1*bp_biom.transform(additional_test_df[bp_biom.variable_names], metric='woe')
        X_nt_biom_WOE_test = X_nt_biom_WOE_test.add_suffix('_WOE')
        X_nt_biom_test = pd.merge(left=additional_test_df[woe_dict_biom.keys()], right=X_nt_biom_WOE_test, left_index=True, right_index=True )
        X_nt_biom_mrf_test = pd.merge(left=X_nt_biom_test, right=X_nt_mrf_test, left_index=True, right_index=True)
        y_nt_test = pd.Series(np.zeros(X_nt_biom_mrf_test.shape[0]), index=X_nt_biom_mrf_test.index, name='transition')
        # Augmented test set
        y_all_test = pd.concat([y_test,y_nt_test], axis=0).astype(int)
        # Include all NT samples
        X_biom_mrf_all_test = pd.concat([X_biom_mrf_test,X_nt_biom_mrf_test],axis=0)
        X_biom_all_test = pd.concat([X_biom_test,X_nt_biom_test],axis=0)
        X_mrf_all_test = pd.concat([X_mrf_test,X_nt_mrf_test],axis=0)
        # Compute partial LIBRA scores
        libra_train = X_biom_mrf_train.apply(calculate_libra_revised, axis=1).to_frame()
        libra_test = X_biom_mrf_test.apply(calculate_libra_revised, axis=1).to_frame()
        libra_all_test = X_biom_mrf_all_test.apply(calculate_libra_revised, axis=1).to_frame()
        # Groups in training set
        groups_train = dataset_df.iloc[train_index]['group']
        # break
        # Train LIBRA model
        print('LIBRA model')
        for seed_libra in range(n_seeds):
            # seed_libra = 0
            print('seed_libra:',seed_libra)
            libra_bayes_search, libra_score = train_model(libra_train, y_train, libra_test, y_test, param_space, model=model_name, seed_rf=seed_rf, seed_bayes=seed_bayes+seed_libra, n_iter=n_iter, cv=cv_group_train, groups=groups_train)
        libra_all_fold_predictions = cross_val_predict(libra_bayes_search.best_estimator_, libra_train, y_train, groups=groups_train, cv=cv_group_train, method='predict_proba', n_jobs=-1)
        libra_cv_scores = cross_val_score(libra_bayes_search.best_estimator_, libra_train, y_train, groups=groups_train, cv=cv_group_train, n_jobs=-1)
        # Train BIOM+MRF model
        # Transforming categorical variables
        cat_vars_biom_mrf = list(set(categorical_variables_biom).union(set(categorical_variables_mrf)))
        X_biom_mrf_train[cat_vars_biom_mrf] = X_biom_mrf_train[cat_vars_biom_mrf].astype(str)
        X_biom_mrf_test[cat_vars_biom_mrf] = X_biom_mrf_test[cat_vars_biom_mrf].astype(str)
        X_biom_mrf_all_test[cat_vars_biom_mrf] = X_biom_mrf_all_test[cat_vars_biom_mrf].astype(str)
        print('BIOM+MRF model')
        for seed_biom_mrf in range(n_seeds):
            # seed_biom_mrf = 0
            print('seed_biom_mrf:',seed_biom_mrf)
            biom_mrf_bayes_search, biom_mrf_score = train_model(X_biom_mrf_train, y_train, X_biom_mrf_test, y_test, param_space, model=model_name, seed_rf=seed_rf, seed_bayes=seed_bayes+seed_biom_mrf, n_iter=n_iter, cv=cv_group_train, groups=groups_train, cat_vars=cat_vars_biom_mrf )
        biom_mrf_all_fold_predictions = cross_val_predict(biom_mrf_bayes_search.best_estimator_, X_biom_mrf_train, y_train, groups=groups_train, cv=cv_group_train, method='predict_proba', n_jobs=-1)
        biom_mrf_cv_scores = cross_val_score(biom_mrf_bayes_search.best_estimator_, X_biom_mrf_train, y_train, groups=groups_train, cv=cv_group_train, n_jobs=-1)
        
        # Train BIOM model
        print('BIOM model')
        # Transforming categorical variables
        X_biom_train[categorical_variables_biom] = X_biom_train[categorical_variables_biom].astype(str)
        X_biom_test[categorical_variables_biom] = X_biom_test[categorical_variables_biom].astype(str)
        X_biom_all_test[categorical_variables_biom] = X_biom_all_test[categorical_variables_biom].astype(str)
        for seed_biom in range(n_seeds):
            # seed_biom = 5
            print('seed_biom:',seed_biom)
            biom_bayes_search, biom_score  = train_model(X_biom_train, y_train, X_biom_test, y_test, param_space, model=model_name, seed_rf=seed_rf, seed_bayes=seed_bayes+seed_biom, n_iter=n_iter, cv=cv_group_train, groups=groups_train, cat_vars=categorical_variables_biom)
        biom_all_fold_predictions = cross_val_predict(biom_bayes_search.best_estimator_, X_biom_train, y_train, groups=groups_train, cv=cv_group_train, method='predict_proba', n_jobs=-1)
        biom_cv_scores = cross_val_score(biom_bayes_search.best_estimator_, X_biom_train, y_train, groups=groups_train, cv=cv_group_train, n_jobs=-1)
        
        # Train MRF model
        print('MRF model')
        # Transforming categorical variables
        X_mrf_train[categorical_variables_mrf] = X_mrf_train[categorical_variables_mrf].astype(str)
        X_mrf_test[categorical_variables_mrf] = X_mrf_test[categorical_variables_mrf].astype(str)
        X_mrf_all_test[categorical_variables_mrf] = X_mrf_all_test[categorical_variables_mrf].astype(str)
        # Pool for interpreation purposes
        mrf_train_pool = Pool(
                            data=X_mrf_train,
                            label=y_train,
                            cat_features=categorical_variables_mrf
                        )

        for seed_mrf in range(n_seeds):
            # seed_mrf = 0
            print('seed_mrf:',seed_mrf)
            mrf_bayes_search, mrf_score  = train_model(X_mrf_train, y_train, X_mrf_test, y_test, param_space, model=model_name, seed_rf=seed_rf, seed_bayes=seed_bayes+seed_mrf, n_iter=n_iter, cv=cv_group_train, groups=groups_train, cat_vars=categorical_variables_mrf)
        mrf_all_fold_predictions = cross_val_predict(mrf_bayes_search.best_estimator_, X_mrf_train, y_train, groups=groups_train, cv=cv_group_train, method='predict_proba', n_jobs=-1)
        mrf_cv_scores = cross_val_score(mrf_bayes_search.best_estimator_, X_mrf_train, y_train, groups=groups_train, cv=cv_group_train, n_jobs=-1)
        
        all_trees, all_features, unique_trees, unique_features = get_all_trees_rules(mrf_bayes_search.best_estimator_, mrf_train_pool)
        lm_train, lm_test, lm_all_test, correlation = leaf_correlation(mrf_bayes_search.best_estimator_, X_mrf_train, y_train, X_mrf_test, y_test, X_mrf_all_test, y_all_test, model='catboost')
        leaf_counts = mrf_bayes_search.best_estimator_.get_tree_leaf_counts()
        dcg_importance = dcg_score(all_features, correlation, unique_features, leaf_counts)
                
        # Train rMRF model
        print('BIOM+rMRF model')
        for seed_biom_rmrf in range(n_seeds):
        # seed_biom_rmrf = 9
            print('seed_biom_rmrf:',seed_biom_rmrf)
            biom_rmrf_bayes_search, biom_rmrf_score  = search_rules(X_biom_train,lm_train.iloc[:,:n_rules],y_train,X_biom_test,lm_test.iloc[:,:n_rules],y_test,param_space, model=model_name, seed_rf=seed_rf, seed_bayes=seed_bayes+seed_biom_rmrf, n_iter=n_iter, cv=cv_group_train, groups=groups_train, cat_vars=categorical_variables_biom)
        X_biom_rmrf_train = pd.merge(left=X_biom_train,right=lm_train.iloc[:,:n_rules],left_index=True,right_index=True)
        biom_rmrf_all_fold_predictions = cross_val_predict(biom_rmrf_bayes_search.best_estimator_, X_biom_rmrf_train, y_train, groups=groups_train, cv=cv_group_train, method='predict_proba', n_jobs=-1)
        biom_rmrf_cv_scores = cross_val_score(biom_rmrf_bayes_search.best_estimator_, X_biom_rmrf_train, y_train, groups=groups_train, cv=cv_group_train, n_jobs=-1)
        biom_rmrf_cv = cross_validate(biom_rmrf_bayes_search.best_estimator_, X_biom_rmrf_train, y_train, groups=groups_train, cv=cv_group_train, n_jobs=-1,
                                        return_train_score=True, return_estimator=True, return_indices=True,)
        X_biom_rmrf_test = pd.merge(left=X_biom_test,right=lm_test.iloc[:,:n_rules],left_index=True,right_index=True)
        X_biom_rmrf_all_test = pd.merge(left=X_biom_all_test,right=lm_all_test.iloc[:,:n_rules],left_index=True,right_index=True)

        # Train sMRF model
        print('BIOM+sMRF model')
        # dcg_importance = dcg_score(all_features, correlation, unique_features, leaf_counts)
        for seed_biom_smrf in range(n_seeds):
            # seed_biom_smrf = 9
            print('seed_biom_smrf:',seed_biom_smrf)
            top_vars = dcg_importance.index[:n_subset]
            repeated_vars = list(set(top_vars).intersection(set(X_biom_train.columns)))
            top_vars = top_vars.drop(repeated_vars)
            # biom_smrf_bayes_search, biom_smrf_score  = search_rules(X_biom_train,X_mrf_train[top_vars],y_train,X_biom_test,X_mrf_test[top_vars],y_test,param_space, model=model_name, seed_rf=seed_rf, seed_bayes=seed_bayes+seed_biom_smrf, n_iter=n_iter, cv=cv_group_train, groups=groups_train, cat_vars=list( set(categorical_variables_biom).union( set(top_vars).intersection(set(categorical_variables_mrf)) )))
            biom_smrf_bayes_search, biom_smrf_score  = search_rules(X_biom_train,X_mrf_train[top_vars],y_train,X_biom_test,X_mrf_test[top_vars],y_test,param_space, model=model_name, seed_rf=seed_rf, seed_bayes=seed_bayes+seed_biom_smrf, n_iter=n_iter, cv=cv_group_train, groups=groups_train, cat_vars=categorical_variables_biom)
        X_biom_smrf_train = pd.merge(left=X_biom_train,right=X_mrf_train[top_vars],left_index=True,right_index=True)
        biom_smrf_all_fold_predictions = cross_val_predict(biom_smrf_bayes_search.best_estimator_, X_biom_smrf_train, y_train, groups=groups_train, cv=cv_group_train, method='predict_proba', n_jobs=-1)
        biom_smrf_cv_scores = cross_val_score(biom_smrf_bayes_search.best_estimator_, X_biom_smrf_train, y_train, groups=groups_train, cv=cv_group_train, n_jobs=-1)
        X_biom_smrf_test = pd.merge(left=X_biom_test,right=X_mrf_test[top_vars],left_index=True,right_index=True)
        X_biom_smrf_all_test = pd.merge(left=X_biom_all_test,right=X_mrf_all_test[top_vars],left_index=True,right_index=True)
        
        test_scores = [
                    {'model': 'libra', 
                    'scores': libra_score},
                    {'model': 'biom_mrf', 
                    'scores': biom_mrf_score},
                    {'model': 'biom', 
                    'scores': biom_score},
                    {'model': 'mrf', 
                    'scores': mrf_score},
                    {'model': 'biom_rmrf', 
                    'scores': biom_rmrf_score},
                    {'model': 'biom_smrf', 
                    'scores': biom_smrf_score},]
        test_scores_df = pd.DataFrame(test_scores)

        # Plot ROC considering cross-validation folds
        fig, ax = plt.subplots(figsize=(8, 6))
        RocCurveDisplay.from_predictions(y_train, libra_all_fold_predictions[:,1], ax=ax, name='LIBRA', color='brown')
        RocCurveDisplay.from_predictions(y_train, mrf_all_fold_predictions[:,1], ax=ax, name='MRF', color='cyan')
        RocCurveDisplay.from_predictions(y_train, biom_all_fold_predictions[:,1], ax=ax, name='BIOM', color='magenta')
        RocCurveDisplay.from_predictions(y_train, biom_mrf_all_fold_predictions[:,1], ax=ax, name='BIOM+MRF', color='blue')
        RocCurveDisplay.from_predictions(y_train, biom_smrf_all_fold_predictions[:,1], ax=ax, name='BIOM+sMRF', color='green')
        RocCurveDisplay.from_predictions(y_train, biom_rmrf_all_fold_predictions[:,1], ax=ax, name='BIOM+rMRF', color='red', plot_chance_level=True)
        # RocCurveDisplay.from_estimator(classifier_biom_cluster, X_test_biom_cluster, y_test, ax=ax, name='BIOM+WOE(clusters)')
        ax.minorticks_on()
        ax.grid(which='both')
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlabel('1-Specificity (FPR)')
        ax.set_ylabel('Sensitivity (TPR)')
        ax.set_title('ROC Curve (CV Test Sets)')
        fig.savefig(f'plots/{model_name}_split_{k}_seed_{seed_split}_seedcv_{seed_cv}_cvroc.pdf', bbox_inches='tight')

        # Plot ROC considering test set
        fig, ax = plt.subplots(figsize=(8, 6))
        RocCurveDisplay.from_estimator(libra_bayes_search.best_estimator_, libra_test, y_test, ax=ax, name='LIBRA', color='brown')
        RocCurveDisplay.from_estimator(mrf_bayes_search.best_estimator_, X_mrf_test, y_test, ax=ax, name='MRF', color='cyan')
        RocCurveDisplay.from_estimator(biom_bayes_search.best_estimator_, X_biom_test, y_test, ax=ax, name='BIOM', color='magenta')
        RocCurveDisplay.from_estimator(biom_mrf_bayes_search.best_estimator_, X_biom_mrf_test, y_test, ax=ax, name='BIOM+MRF', color='blue')
        RocCurveDisplay.from_estimator(biom_smrf_bayes_search.best_estimator_, X_biom_smrf_test, y_test, ax=ax, name='BIOM+sMRF', color='green')
        RocCurveDisplay.from_estimator(biom_rmrf_bayes_search.best_estimator_, X_biom_rmrf_test, y_test, ax=ax, name='BIOM+rMRF', color='red', plot_chance_level=True)
        # RocCurveDisplay.from_estimator(classifier_biom_cluster, X_test_biom_cluster, y_test, ax=ax, name='BIOM+WOE(clusters)')
        ax.minorticks_on()
        ax.grid(which='both')
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlabel('1-Specificity (FPR)')
        ax.set_ylabel('Sensitivity (TPR)')
        ax.set_title('ROC Curve (Test Set)')
        fig.savefig(f'plots/{model_name}_split_{k}_seed_{seed_split}_seedcv_{seed_cv}_testroc.pdf', bbox_inches='tight')

        # Plot ROC considering combined test set
        fig, ax = plt.subplots(figsize=(8, 6))
        RocCurveDisplay.from_estimator(libra_bayes_search.best_estimator_, libra_all_test, y_all_test, ax=ax, name='LIBRA', color='brown')
        RocCurveDisplay.from_estimator(mrf_bayes_search.best_estimator_, X_mrf_all_test, y_all_test, ax=ax, name='MRF', color='cyan')
        RocCurveDisplay.from_estimator(biom_bayes_search.best_estimator_, X_biom_all_test, y_all_test, ax=ax, name='BIOM', color='magenta')
        RocCurveDisplay.from_estimator(biom_mrf_bayes_search.best_estimator_, X_biom_mrf_all_test, y_all_test, ax=ax, name='BIOM+MRF', color='blue')
        RocCurveDisplay.from_estimator(biom_smrf_bayes_search.best_estimator_, X_biom_smrf_all_test, y_all_test, ax=ax, name='BIOM+sMRF', color='green')
        RocCurveDisplay.from_estimator(biom_rmrf_bayes_search.best_estimator_, X_biom_rmrf_all_test, y_all_test, ax=ax, name='BIOM+rMRF', color='red', plot_chance_level=True)
        # RocCurveDisplay.from_estimator(classifier_biom_cluster, X_test_biom_cluster, y_test, ax=ax, name='BIOM+WOE(clusters)')
        ax.minorticks_on()
        ax.grid(which='both')
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlabel('1-Specificity (FPR)')
        ax.set_ylabel('Sensitivity (TPR)')
        ax.set_title('ROC Curve (Test Set Augmented)')
        fig.savefig(f'plots/{model_name}_split_{k}_seed_{seed_split}_seedcv_{seed_cv}_alltestroc.pdf', bbox_inches='tight')

        scores_splits = [
                    {'model': 'libra', 
                    'scores': libra_cv_scores},
                    {'model': 'biom_mrf', 
                    'scores': biom_mrf_cv_scores},
                    {'model': 'biom', 
                    'scores': biom_cv_scores},
                    {'model': 'mrf', 
                    'scores': mrf_cv_scores},
                    {'model': 'biom_rmrf', 
                    'scores': biom_rmrf_cv_scores},
                    {'model': 'biom_smrf', 
                    'scores': biom_smrf_cv_scores},]
        scores_splits_df = pd.DataFrame(scores_splits).explode('scores').reset_index(drop=True)
        fig, ax = plt.subplots(figsize=(12,8))
        ax = sns.boxplot(data=scores_splits_df, x="model", y="scores", color=".8", linecolor="#137", linewidth=.75, ax=ax)
        ax = sns.stripplot(data=scores_splits_df, x="model", y="scores", ax=ax, color='gray')
        ax = sns.stripplot(data=test_scores_df, x="model", y="scores", ax=ax, color='red')
        fig.savefig(f'plots/{model_name}_split_{k}_seed_{seed_split}_seedcv_{seed_cv}_boxplot.pdf', bbox_inches='tight')

        results_dict = {
                'train_index':train_index,
                'test_index':test_index,
                'lm_train':lm_train,
                'lm_test':lm_test,
                'lm_all_test':lm_all_test,
                'libra_all_fold_predictions':libra_all_fold_predictions,
                'libra_cv_scores':libra_cv_scores,
                'libra_score':libra_score,
                'biom_mrf_all_fold_predictions':biom_mrf_all_fold_predictions,
                'biom_mrf_cv_scores':biom_mrf_cv_scores,
                'biom_mrf_score':biom_mrf_score,
                'biom_all_fold_predictions':biom_all_fold_predictions,
                'biom_cv_scores':biom_cv_scores,
                'biom_score':biom_score,
                'mrf_all_fold_predictions':mrf_all_fold_predictions,
                'mrf_cv_scores':mrf_cv_scores,
                'mrf_score':mrf_score,
                'biom_rmrf_all_fold_predictions':biom_rmrf_all_fold_predictions,
                'biom_rmrf_cv_scores':biom_rmrf_cv_scores,
                'biom_rmrf_score':biom_rmrf_score,
                'biom_smrf_all_fold_predictions':biom_smrf_all_fold_predictions,
                'biom_smrf_cv_scores':biom_smrf_cv_scores,
                'biom_smrf_score':biom_smrf_score,
                'biom_rmrf_cv':biom_rmrf_cv,
                'X_biom_rmrf_train':X_biom_rmrf_train,
                'X_biom_rmrf_test':X_biom_rmrf_test,
                'all_trees':all_trees,
                'all_features':all_features, 
                'unique_trees':unique_trees,
                'unique_features':unique_features,
                'corr':correlation,
                'cv_group_train':cv_group_train,
                'libra_bayes_search':libra_bayes_search,
                'biom_mrf_bayes_search':biom_mrf_bayes_search,
                'biom_bayes_search':biom_bayes_search,
                'mrf_bayes_search':mrf_bayes_search,
                'biom_rmrf_bayes_search':biom_rmrf_bayes_search,
                'biom_smrf_bayes_search':biom_smrf_bayes_search,
            }
        joblib.dump(results_dict, f'results/{model_name}_split_{k}_seed_{seed_split}_seedcv_{seed_cv}_results.joblib')
        break
#%%
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Uncertainty evaluation")
    parser.add_argument('--seed_split', type=str, required=True, help="Seed number for partition")
    parser.add_argument('--model_name', type=str, required=True, help="Model")
    # parser.add_argument('--split', type=str, required=True, help="Split number for partition")
    # parser.add_argument('--cv_splits', type=str, required=True, help="Number of splits for CV")    
    # parser.add_argument('--save_idx', required=True, action=argparse.BooleanOptionalAction, help="Save idx flag")
    args = parser.parse_args()
    seed_split = int(args.seed_split)
    model_name = args.model_name
    # split = int(args.split)
    # cv_splits = int(args.cv_splits)
    # save_idx = args.save_idx
    train_test_splits(seed_split, model_name)
#%%
# seed_split = 0
# model_name = 'catboost'
#%%
# train_test_splits(seed_split,model_name)
#%%
# Settings
# n_iter = 50
# n_seeds = 1
# # cv = 5
# n_splits = 5
# n_repeats = 1
# n_rules = 100
# n_subset = 30

# seed_cv = 0 # For consistency
# seed_rf = 0
# seed_lr = 0
# seed_bayes = 0
# sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed_split)
# cv_group_train = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed_cv)
# param_space_catboost = {
#     'iterations': Integer(100, 1000),
#     'learning_rate': Real(1e-3, 1.0, 'log-uniform'),
#     'depth': Integer(4, 12),
#     'l2_leaf_reg': Real(1, 10, 'uniform'),
#     'bagging_temperature': Real(0.0, 1.0, 'uniform'),
#     'border_count': Integer(32, 255)
# }

# param_space_xgboost = {
#     'n_estimators': Integer(100, 1000),
#     'learning_rate': Real(0.01, 0.3, 'log-uniform'),
#     'max_depth': Integer(4, 20),
#     'subsample': Real(0.5, 1.0, 'uniform'),
#     'colsample_bytree': Real(0.5, 1.0, 'uniform'),
#     'gamma': Real(0.0, 5.0, 'uniform'),
#     'reg_alpha': Real(0.0, 10.0, 'uniform'),
#     'reg_lambda': Real(1.0, 10.0, 'uniform')
# }

# param_space_rf = {
#     'n_estimators': Integer(100, 300),  # Range of number of trees
#     'max_depth': Integer(4, 20),        # Depth of trees
#     'min_samples_split': Integer(5, 20),  # Minimum samples required to split a node
#     'min_samples_leaf': Integer(5, 20),   # Minimum samples required to be at a leaf node
#     'max_features': ['sqrt', 'log2']  # Features to consider for best split
# }

# if model_name=='xgboost':
#     param_space = param_space_xgboost
# elif model_name=='catboost':
#     param_space = param_space_catboost
# elif model_name=='rf':
#     param_space = param_space_rf
# # Read data
# joint_dataset_df = pd.read_csv('joint_dataset.csv', index_col=0).set_index('subject_id')
# remaining_test_df = pd.read_csv('remaining_test.csv', index_col=0).set_index('subject_id')
# dataset_df = feature_engineering(joint_dataset_df)
# additional_test_df = feature_engineering(remaining_test_df)
# # results_dict = {}

# for k, (train_index, test_index) in enumerate(sgkf.split(dataset_df.drop(['transition'],axis='columns'), dataset_df['transition'], dataset_df['group'])):
#     X_mrf_train, X_mrf_test, y_train, y_test, bp_mrf = transform_WOE(dataset_df, woe_dict_mrf, categorical_variables_mrf, train_index, test_index)
#     X_biom_train, X_biom_test, _, _, bp_biom = transform_WOE(dataset_df, woe_dict_biom, categorical_variables_biom, train_index, test_index)
#     repeated_vars = list(set(X_mrf_train.columns).intersection(set(X_biom_train.columns)))
#     X_biom_mrf_train = pd.merge(left=X_biom_train, right=X_mrf_train.drop(columns=repeated_vars), left_index=True, right_index=True)
#     X_biom_mrf_test = pd.merge(left=X_biom_test, right=X_mrf_test.drop(columns=repeated_vars), left_index=True, right_index=True)
    
#     # Additional test samples
#     X_nt_mrf_WOE_test = -1*bp_mrf.transform(additional_test_df[bp_mrf.variable_names], metric='woe')
#     X_nt_mrf_WOE_test = X_nt_mrf_WOE_test.add_suffix('_WOE')
#     X_nt_mrf_test = pd.merge(left=additional_test_df[woe_dict_mrf.keys()], right=X_nt_mrf_WOE_test, left_index=True, right_index=True )
#     X_nt_biom_WOE_test = -1*bp_biom.transform(additional_test_df[bp_biom.variable_names], metric='woe')
#     X_nt_biom_WOE_test = X_nt_biom_WOE_test.add_suffix('_WOE')
#     X_nt_biom_test = pd.merge(left=additional_test_df[woe_dict_biom.keys()], right=X_nt_biom_WOE_test, left_index=True, right_index=True )
#     X_nt_biom_mrf_test = pd.merge(left=X_nt_biom_test, right=X_nt_mrf_test, left_index=True, right_index=True)
#     y_nt_test = pd.Series(np.zeros(X_nt_biom_mrf_test.shape[0]), index=X_nt_biom_mrf_test.index, name='transition')
#     # Augmented test set
#     y_all_test = pd.concat([y_test,y_nt_test], axis=0).astype(int)
#     # Include all NT samples
#     X_biom_mrf_all_test = pd.concat([X_biom_mrf_test,X_nt_biom_mrf_test],axis=0)
#     X_biom_all_test = pd.concat([X_biom_test,X_nt_biom_test],axis=0)
#     X_mrf_all_test = pd.concat([X_mrf_test,X_nt_mrf_test],axis=0)
#     # Compute partial LIBRA scores
#     libra_train = X_biom_mrf_train.apply(calculate_libra_revised, axis=1).to_frame()
#     libra_test = X_biom_mrf_test.apply(calculate_libra_revised, axis=1).to_frame()
#     libra_all_test = X_biom_mrf_all_test.apply(calculate_libra_revised, axis=1).to_frame()
#     # Groups in training set
#     groups_train = dataset_df.iloc[train_index]['group']
#     # break
#     # Train LIBRA model
#     print('LIBRA model')
#     for seed_libra in range(n_seeds):
#         # seed_libra = 0
#         print('seed_libra:',seed_libra)
#         # libra_bayes_search, libra_score = train_model(libra_train, y_train, libra_test, y_test, param_space, model=model_name, seed_rf=seed_rf, seed_bayes=seed_bayes+seed_libra, n_iter=n_iter, cv=cv_group_train, groups=groups_train)
#     # libra_all_fold_predictions = cross_val_predict(libra_bayes_search.best_estimator_, libra_train, y_train, groups=groups_train, cv=cv_group_train, method='predict_proba', n_jobs=-1)
#     # libra_cv_scores = cross_val_score(libra_bayes_search.best_estimator_, libra_train, y_train, groups=groups_train, cv=cv_group_train, n_jobs=-1)
#     # Train BIOM+MRF model
#     # Transforming categorical variables
#     cat_vars_biom_mrf = list(set(categorical_variables_biom).union(set(categorical_variables_mrf)))
#     X_biom_mrf_train[cat_vars_biom_mrf] = X_biom_mrf_train[cat_vars_biom_mrf].astype(str)
#     X_biom_mrf_test[cat_vars_biom_mrf] = X_biom_mrf_test[cat_vars_biom_mrf].astype(str)
#     print('BIOM+MRF model')
#     for seed_biom_mrf in range(n_seeds):
#         # seed_biom_mrf = 0
#         print('seed_biom_mrf:',seed_biom_mrf)
#         # biom_mrf_bayes_search, biom_mrf_score = train_model(X_biom_mrf_train, y_train, X_biom_mrf_test, y_test, param_space, model=model_name, seed_rf=seed_rf, seed_bayes=seed_bayes+seed_biom_mrf, n_iter=n_iter, cv=cv_group_train, groups=groups_train, cat_vars=cat_vars_biom_mrf )
#     # biom_mrf_all_fold_predictions = cross_val_predict(biom_mrf_bayes_search.best_estimator_, X_biom_mrf_train, y_train, groups=groups_train, cv=cv_group_train, method='predict_proba', n_jobs=-1)
#     # biom_mrf_cv_scores = cross_val_score(biom_mrf_bayes_search.best_estimator_, X_biom_mrf_train, y_train, groups=groups_train, cv=cv_group_train, n_jobs=-1)
    
#     # Train BIOM model
#     print('BIOM model')
#     # Transforming categorical variables
#     X_biom_train[categorical_variables_biom] = X_biom_train[categorical_variables_biom].astype(str)
#     X_biom_test[categorical_variables_biom] = X_biom_test[categorical_variables_biom].astype(str)
#     X_biom_all_test[categorical_variables_biom] = X_biom_all_test[categorical_variables_biom].astype(str)
#     for seed_biom in range(n_seeds):
#         # seed_biom = 5
#         print('seed_biom:',seed_biom)
#         # biom_bayes_search, biom_score  = train_model(X_biom_train, y_train, X_biom_test, y_test, param_space, model=model_name, seed_rf=seed_rf, seed_bayes=seed_bayes+seed_biom, n_iter=n_iter, cv=cv_group_train, groups=groups_train, cat_vars=categorical_variables_biom)
#     # biom_all_fold_predictions = cross_val_predict(biom_bayes_search.best_estimator_, X_biom_train, y_train, groups=groups_train, cv=cv_group_train, method='predict_proba', n_jobs=-1)
#     # biom_cv_scores = cross_val_score(biom_bayes_search.best_estimator_, X_biom_train, y_train, groups=groups_train, cv=cv_group_train, n_jobs=-1)
    
#     # Train MRF model
#     print('MRF model')
#     # Transforming categorical variables
#     X_mrf_train[categorical_variables_mrf] = X_mrf_train[categorical_variables_mrf].astype(str)
#     X_mrf_test[categorical_variables_mrf] = X_mrf_test[categorical_variables_mrf].astype(str)
#     X_mrf_all_test[categorical_variables_mrf] = X_mrf_all_test[categorical_variables_mrf].astype(str)
#     # Pool for interpreation purposes
#     mrf_train_pool = Pool(
#                         data=X_mrf_train,
#                         label=y_train,
#                         cat_features=categorical_variables_mrf
#                     )

#     for seed_mrf in range(n_seeds):
#         # seed_mrf = 0
#         print('seed_mrf:',seed_mrf)
#         mrf_bayes_search, mrf_score  = train_model(X_mrf_train, y_train, X_mrf_test, y_test, param_space, model=model_name, seed_rf=seed_rf, seed_bayes=seed_bayes+seed_mrf, n_iter=n_iter, cv=cv_group_train, groups=groups_train, cat_vars=categorical_variables_mrf)
#     mrf_all_fold_predictions = cross_val_predict(mrf_bayes_search.best_estimator_, X_mrf_train, y_train, groups=groups_train, cv=cv_group_train, method='predict_proba', n_jobs=-1)
#     mrf_cv_scores = cross_val_score(mrf_bayes_search.best_estimator_, X_mrf_train, y_train, groups=groups_train, cv=cv_group_train, n_jobs=-1)

#     all_trees, all_features, unique_trees, unique_features = get_all_trees_rules(mrf_bayes_search.best_estimator_, mrf_train_pool)
#     lm_train, lm_test, lm_all_test, correlation = leaf_correlation(mrf_bayes_search.best_estimator_, X_mrf_train, y_train, X_mrf_test, y_test, X_mrf_all_test, y_all_test, model='catboost')
#     leaf_counts = mrf_bayes_search.best_estimator_.get_tree_leaf_counts()
#     dcg_importance = dcg_score(all_features, correlation, unique_features, leaf_counts)
#     break
#%%
# lm_train, lm_test, lm_all_test, correlation = leaf_correlation(mrf_bayes_search.best_estimator_, X_mrf_train, y_train, X_mrf_test, y_test, X_mrf_all_test, y_all_test, model='catboost')
# leaf_counts = mrf_bayes_search.best_estimator_.get_tree_leaf_counts()
# dcg_importance = dcg_score(all_features, correlation, unique_features, leaf_counts)
#%%
# Train rMRF model
# print('BIOM+rMRF model')
# for seed_biom_rmrf in range(n_seeds):
# # seed_biom_rmrf = 9
#     print('seed_biom_rmrf:',seed_biom_rmrf)
#     biom_rmrf_bayes_search, biom_rmrf_score  = search_rules(X_biom_train,lm_train.iloc[:,:n_rules],y_train,X_biom_test,lm_test.iloc[:,:n_rules],y_test,param_space, model=model_name, seed_rf=seed_rf, seed_bayes=seed_bayes+seed_biom_rmrf, n_iter=n_iter, cv=cv_group_train, groups=groups_train, cat_vars=categorical_variables_biom)
# X_biom_rmrf_train = pd.merge(left=X_biom_train,right=lm_train.iloc[:,:n_rules],left_index=True,right_index=True)
# biom_rmrf_all_fold_predictions = cross_val_predict(biom_rmrf_bayes_search.best_estimator_, X_biom_rmrf_train, y_train, groups=groups_train, cv=cv_group_train, method='predict_proba', n_jobs=-1)
# biom_rmrf_cv_scores = cross_val_score(biom_rmrf_bayes_search.best_estimator_, X_biom_rmrf_train, y_train, groups=groups_train, cv=cv_group_train, n_jobs=-1)
# biom_rmrf_cv = cross_validate(biom_rmrf_bayes_search.best_estimator_, X_biom_rmrf_train, y_train, groups=groups_train, cv=cv_group_train, n_jobs=-1,
#                                 return_train_score=True, return_estimator=True, return_indices=True,)
# X_biom_rmrf_test = pd.merge(left=X_biom_test,right=lm_test.iloc[:,:n_rules],left_index=True,right_index=True)
# X_biom_rmrf_all_test = pd.merge(left=X_biom_all_test,right=lm_all_test.iloc[:,:n_rules],left_index=True,right_index=True)

#%%
# Train sMRF model
# print('BIOM+sMRF model')
# # dcg_importance = dcg_score(all_features, correlation, unique_features, leaf_counts)
# for seed_biom_smrf in range(n_seeds):
#     # seed_biom_smrf = 9
#     print('seed_biom_smrf:',seed_biom_smrf)
#     top_vars = dcg_importance.index[:n_subset]
#     repeated_vars = list(set(top_vars).intersection(set(X_biom_train.columns)))
#     top_vars = top_vars.drop(repeated_vars)
#     # biom_smrf_bayes_search, biom_smrf_score  = search_rules(X_biom_train,X_mrf_train[top_vars],y_train,X_biom_test,X_mrf_test[top_vars],y_test,param_space, model=model_name, seed_rf=seed_rf, seed_bayes=seed_bayes+seed_biom_smrf, n_iter=n_iter, cv=cv_group_train, groups=groups_train, cat_vars=list( set(categorical_variables_biom).union( set(top_vars).intersection(set(categorical_variables_mrf)) )))
#     biom_smrf_bayes_search, biom_smrf_score  = search_rules(X_biom_train,X_mrf_train[top_vars],y_train,X_biom_test,X_mrf_test[top_vars],y_test,param_space, model=model_name, seed_rf=seed_rf, seed_bayes=seed_bayes+seed_biom_smrf, n_iter=n_iter, cv=cv_group_train, groups=groups_train, cat_vars=categorical_variables_biom)
# X_biom_smrf_train = pd.merge(left=X_biom_train,right=X_mrf_train[top_vars],left_index=True,right_index=True)
# biom_smrf_all_fold_predictions = cross_val_predict(biom_smrf_bayes_search.best_estimator_, X_biom_smrf_train, y_train, groups=groups_train, cv=cv_group_train, method='predict_proba', n_jobs=-1)
# biom_smrf_cv_scores = cross_val_score(biom_smrf_bayes_search.best_estimator_, X_biom_smrf_train, y_train, groups=groups_train, cv=cv_group_train, n_jobs=-1)
# X_biom_smrf_test = pd.merge(left=X_biom_test,right=X_mrf_test[top_vars],left_index=True,right_index=True)
# X_biom_smrf_all_test = pd.merge(left=X_biom_all_test,right=X_mrf_all_test[top_vars],left_index=True,right_index=True)
#%%
# test_scores = [
#             # {'model': 'libra', 
#             # 'scores': libra_score},
#             # {'model': 'biom_mrf', 
#             # 'scores': biom_mrf_score},
#             # {'model': 'biom', 
#             # 'scores': biom_score},
#             {'model': 'mrf', 
#             'scores': mrf_score},
#             {'model': 'biom_rmrf', 
#             'scores': biom_rmrf_score},
#             {'model': 'biom_smrf', 
#             'scores': biom_smrf_score},]
# test_scores_df = pd.DataFrame(test_scores)

# # Plot ROC considering cross-validation folds
# fig, ax = plt.subplots(figsize=(8, 6))
# # RocCurveDisplay.from_predictions(y_train, libra_all_fold_predictions[:,1], ax=ax, name='LIBRA', color='brown')
# RocCurveDisplay.from_predictions(y_train, mrf_all_fold_predictions[:,1], ax=ax, name='MRF', color='cyan')
# # RocCurveDisplay.from_predictions(y_train, biom_all_fold_predictions[:,1], ax=ax, name='BIOM', color='magenta')
# # RocCurveDisplay.from_predictions(y_train, biom_mrf_all_fold_predictions[:,1], ax=ax, name='BIOM+MRF', color='blue')
# RocCurveDisplay.from_predictions(y_train, biom_smrf_all_fold_predictions[:,1], ax=ax, name='BIOM+sMRF', color='green')
# RocCurveDisplay.from_predictions(y_train, biom_rmrf_all_fold_predictions[:,1], ax=ax, name='BIOM+rMRF', color='red', plot_chance_level=True)
# # RocCurveDisplay.from_estimator(classifier_biom_cluster, X_test_biom_cluster, y_test, ax=ax, name='BIOM+WOE(clusters)')
# ax.minorticks_on()
# ax.grid(which='both')
# ax.set_aspect('equal', adjustable='box')
# ax.set_xlabel('1-Specificity (FPR)')
# ax.set_ylabel('Sensitivity (TPR)')
# ax.set_title('ROC Curve (CV Test Sets)')
# fig.savefig(f'plots/{model_name}_split_{k}_seed_{seed_split}_seedcv_{seed_cv}_cvroc.pdf', bbox_inches='tight')
#%%
# Plot ROC considering test set
# fig, ax = plt.subplots(figsize=(8, 6))
# # RocCurveDisplay.from_estimator(libra_bayes_search.best_estimator_, libra_test, y_test, ax=ax, name='LIBRA', color='brown')
# RocCurveDisplay.from_estimator(mrf_bayes_search.best_estimator_, X_mrf_test, y_test, ax=ax, name='MRF', color='cyan')
# # RocCurveDisplay.from_estimator(biom_bayes_search.best_estimator_, X_biom_test, y_test, ax=ax, name='BIOM', color='magenta')
# # RocCurveDisplay.from_estimator(biom_mrf_bayes_search.best_estimator_, X_biom_mrf_test, y_test, ax=ax, name='BIOM+MRF', color='blue')
# RocCurveDisplay.from_estimator(biom_smrf_bayes_search.best_estimator_, X_biom_smrf_test, y_test, ax=ax, name='BIOM+sMRF', color='green')
# RocCurveDisplay.from_estimator(biom_rmrf_bayes_search.best_estimator_, X_biom_rmrf_test, y_test, ax=ax, name='BIOM+rMRF', color='red', plot_chance_level=True)
# # RocCurveDisplay.from_estimator(classifier_biom_cluster, X_test_biom_cluster, y_test, ax=ax, name='BIOM+WOE(clusters)')
# ax.minorticks_on()
# ax.grid(which='both')
# ax.set_aspect('equal', adjustable='box')
# ax.set_xlabel('1-Specificity (FPR)')
# ax.set_ylabel('Sensitivity (TPR)')
# ax.set_title('ROC Curve (Test Set)')
# fig.savefig(f'plots/{model_name}_split_{k}_seed_{seed_split}_seedcv_{seed_cv}_testroc.pdf', bbox_inches='tight')
#%%
# Plot ROC considering combined test set
# fig, ax = plt.subplots(figsize=(8, 6))
# # RocCurveDisplay.from_estimator(libra_bayes_search.best_estimator_, libra_all_test, y_all_test, ax=ax, name='LIBRA', color='brown')
# RocCurveDisplay.from_estimator(mrf_bayes_search.best_estimator_, X_mrf_all_test, y_all_test, ax=ax, name='MRF', color='cyan')
# # RocCurveDisplay.from_estimator(biom_bayes_search.best_estimator_, X_biom_all_test, y_all_test, ax=ax, name='BIOM', color='magenta')
# # RocCurveDisplay.from_estimator(biom_mrf_bayes_search.best_estimator_, X_biom_mrf_all_test, y_all_test, ax=ax, name='BIOM+MRF', color='blue')
# RocCurveDisplay.from_estimator(biom_smrf_bayes_search.best_estimator_, X_biom_smrf_all_test, y_all_test, ax=ax, name='BIOM+sMRF', color='green')
# RocCurveDisplay.from_estimator(biom_rmrf_bayes_search.best_estimator_, X_biom_rmrf_all_test, y_all_test, ax=ax, name='BIOM+rMRF', color='red', plot_chance_level=True)
# # RocCurveDisplay.from_estimator(classifier_biom_cluster, X_test_biom_cluster, y_test, ax=ax, name='BIOM+WOE(clusters)')
# ax.minorticks_on()
# ax.grid(which='both')
# ax.set_aspect('equal', adjustable='box')
# ax.set_xlabel('1-Specificity (FPR)')
# ax.set_ylabel('Sensitivity (TPR)')
# ax.set_title('ROC Curve (Test Set Augmented)')
# fig.savefig(f'plots/{model_name}_split_{k}_seed_{seed_split}_seedcv_{seed_cv}_alltestroc.pdf', bbox_inches='tight')
#%%
# scores_splits = [
#             # {'model': 'libra', 
#             # 'scores': libra_cv_scores},
#             # {'model': 'biom_mrf', 
#             # 'scores': biom_mrf_cv_scores},
#             # {'model': 'biom', 
#             # 'scores': biom_cv_scores},
#             {'model': 'mrf', 
#             'scores': mrf_cv_scores},
#             {'model': 'biom_rmrf', 
#             'scores': biom_rmrf_cv_scores},
#             {'model': 'biom_smrf', 
#             'scores': biom_smrf_cv_scores},]
# scores_splits_df = pd.DataFrame(scores_splits).explode('scores').reset_index(drop=True)
# fig, ax = plt.subplots(figsize=(12,8))
# ax = sns.boxplot(data=scores_splits_df, x="model", y="scores", color=".8", linecolor="#137", linewidth=.75, ax=ax)
# ax = sns.stripplot(data=scores_splits_df, x="model", y="scores", ax=ax, color='gray')
# ax = sns.stripplot(data=test_scores_df, x="model", y="scores", ax=ax, color='red')
# fig.savefig(f'plots/{model_name}_split_{k}_seed_{seed_split}_seedcv_{seed_cv}_boxplot.pdf', bbox_inches='tight')
#%%
# results_dict = {
#         'train_index':train_index,
#         'test_index':test_index,
#         'lm_train':lm_train,
#         'lm_test':lm_test,
#         'lm_all_test':lm_all_test,
#         # 'libra_all_fold_predictions':libra_all_fold_predictions,
#         # 'libra_cv_scores':libra_cv_scores,
#         # 'libra_score':libra_score,
#         # 'biom_mrf_all_fold_predictions':biom_mrf_all_fold_predictions,
#         # 'biom_mrf_cv_scores':biom_mrf_cv_scores,
#         # 'biom_mrf_score':biom_mrf_score,
#         # 'biom_all_fold_predictions':biom_all_fold_predictions,
#         # 'biom_cv_scores':biom_cv_scores,
#         # 'biom_score':biom_score,
#         'mrf_all_fold_predictions':mrf_all_fold_predictions,
#         'mrf_cv_scores':mrf_cv_scores,
#         'mrf_score':mrf_score,
#         'biom_rmrf_all_fold_predictions':biom_rmrf_all_fold_predictions,
#         'biom_rmrf_cv_scores':biom_rmrf_cv_scores,
#         'biom_rmrf_score':biom_rmrf_score,
#         'biom_smrf_all_fold_predictions':biom_smrf_all_fold_predictions,
#         'biom_smrf_cv_scores':biom_smrf_cv_scores,
#         'biom_smrf_score':biom_smrf_score,
#         'biom_rmrf_cv':biom_rmrf_cv,
#         'X_biom_rmrf_train':X_biom_rmrf_train,
#         'X_biom_rmrf_test':X_biom_rmrf_test,
#         'all_trees':all_trees,
#         'all_features':all_features, 
#         'unique_trees':unique_trees,
#         'unique_features':unique_features,
#         'correlation':correlation,
#         'cv_group_train':cv_group_train,
#         # 'libra_bayes_search':libra_bayes_search,
#         # 'biom_mrf_bayes_search':biom_mrf_bayes_search,
#         # 'biom_bayes_search':biom_bayes_search,
#         'mrf_bayes_search':mrf_bayes_search,
#         'biom_rmrf_bayes_search':biom_rmrf_bayes_search,
#         'biom_smrf_bayes_search':biom_smrf_bayes_search,
#     }
# joblib.dump(results_dict, f'results/{model_name}_split_{k}_seed_{seed_split}_seedcv_{seed_cv}_results.joblib')
#%%
# # Settings
# n_iter = 50
# n_seeds = 1
# # cv = 5
# n_splits = 5
# n_repeats = 1
# n_rules = 100
# n_subset = 30

# seed_cv = 0 # For consistency
# seed_rf = 0
# seed_lr = 0
# seed_bayes = 0
# sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed_split)
# cv_group_train = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed_cv)
# param_space_catboost = {
#     'iterations': Integer(100, 1000),
#     'learning_rate': Real(1e-3, 1.0, 'log-uniform'),
#     'depth': Integer(4, 12),
#     'l2_leaf_reg': Real(1, 10, 'uniform'),
#     'bagging_temperature': Real(0.0, 1.0, 'uniform'),
#     'border_count': Integer(32, 255)
# }

# param_space_xgboost = {
#     'n_estimators': Integer(100, 1000),
#     'learning_rate': Real(0.01, 0.3, 'log-uniform'),
#     'max_depth': Integer(4, 20),
#     'subsample': Real(0.5, 1.0, 'uniform'),
#     'colsample_bytree': Real(0.5, 1.0, 'uniform'),
#     'gamma': Real(0.0, 5.0, 'uniform'),
#     'reg_alpha': Real(0.0, 10.0, 'uniform'),
#     'reg_lambda': Real(1.0, 10.0, 'uniform')
# }

# param_space_rf = {
#     'n_estimators': Integer(100, 300),  # Range of number of trees
#     'max_depth': Integer(4, 20),        # Depth of trees
#     'min_samples_split': Integer(5, 20),  # Minimum samples required to split a node
#     'min_samples_leaf': Integer(5, 20),   # Minimum samples required to be at a leaf node
#     'max_features': ['sqrt', 'log2']  # Features to consider for best split
# }

# if model_name=='xgboost':
#     param_space = param_space_xgboost
# elif model_name=='catboost':
#     param_space = param_space_catboost
# elif model_name=='rf':
#     param_space = param_space_rf
# # Read data
# joint_dataset_df = pd.read_csv('joint_dataset.csv', index_col=0).set_index('subject_id')
# remaining_test_df = pd.read_csv('remaining_test.csv', index_col=0).set_index('subject_id')
# dataset_df = feature_engineering(joint_dataset_df)
# additional_test_df = feature_engineering(remaining_test_df)
# results_dict = {}
#%%
# for k, (train_index, test_index) in enumerate(sgkf.split(dataset_df.drop(['transition'],axis='columns'), dataset_df['transition'], dataset_df['group'])):
#     X_mrf_train, X_mrf_test, y_train, y_test, bp_mrf = transform_WOE(dataset_df, woe_dict_mrf, categorical_variables_mrf, train_index, test_index)
#     X_biom_train, X_biom_test, _, _, bp_biom = transform_WOE(dataset_df, woe_dict_biom, categorical_variables_biom, train_index, test_index)
#     repeated_vars = list(set(X_mrf_train.columns).intersection(set(X_biom_train.columns)))
#     X_biom_mrf_train = pd.merge(left=X_biom_train, right=X_mrf_train.drop(columns=repeated_vars), left_index=True, right_index=True)
#     X_biom_mrf_test = pd.merge(left=X_biom_test, right=X_mrf_test.drop(columns=repeated_vars), left_index=True, right_index=True)
#     # Transforming categorical variables
#     X_mrf_train[categorical_variables_mrf] = X_mrf_train[categorical_variables_mrf].astype(str)
#     X_mrf_test[categorical_variables_mrf] = X_mrf_test[categorical_variables_mrf].astype(str)
#     X_biom_train[categorical_variables_biom] = X_biom_train[categorical_variables_biom].astype(str)
#     X_biom_test[categorical_variables_biom] = X_biom_test[categorical_variables_biom].astype(str)
#     # Pool for tree interpreation purposes
#     train_pool = Pool(
#                         data=X_mrf_train,
#                         label=y_train,
#                         cat_features=categorical_variables_mrf
#                     )
#     # Additional test samples
#     X_nt_mrf_WOE_test = -1*bp_mrf.transform(additional_test_df[bp_mrf.variable_names], metric='woe')
#     X_nt_mrf_WOE_test = X_nt_mrf_WOE_test.add_suffix('_WOE')
#     X_nt_mrf_test = pd.merge(left=additional_test_df[woe_dict_mrf.keys()], right=X_nt_mrf_WOE_test, left_index=True, right_index=True )
#     X_nt_biom_WOE_test = -1*bp_biom.transform(additional_test_df[bp_biom.variable_names], metric='woe')
#     X_nt_biom_WOE_test = X_nt_biom_WOE_test.add_suffix('_WOE')
#     X_nt_biom_test = pd.merge(left=additional_test_df[woe_dict_biom.keys()], right=X_nt_biom_WOE_test, left_index=True, right_index=True )
#     X_nt_biom_mrf_test = pd.merge(left=X_nt_biom_test, right=X_nt_mrf_test, left_index=True, right_index=True)
#     y_nt_test = pd.Series(np.zeros(X_nt_biom_mrf_test.shape[0]), index=X_nt_biom_mrf_test.index, name='transition')
#     # Augmented test set
#     y_all_test = pd.concat([y_test,y_nt_test], axis=0).astype(int)
#     # Include all NT samples
#     X_biom_mrf_all_test = pd.concat([X_biom_mrf_test,X_nt_biom_mrf_test],axis=0)
#     X_biom_all_test = pd.concat([X_biom_test,X_nt_biom_test],axis=0)
#     X_mrf_all_test = pd.concat([X_mrf_test,X_nt_mrf_test],axis=0)
#     # Compute partial LIBRA scores
#     libra_train = X_biom_mrf_train.apply(calculate_libra_revised, axis=1).to_frame()
#     libra_test = X_biom_mrf_test.apply(calculate_libra_revised, axis=1).to_frame()
#     libra_all_test = X_biom_mrf_all_test.apply(calculate_libra_revised, axis=1).to_frame()
#     # Groups in training set
#     groups_train = dataset_df.iloc[train_index]['group']
#     break
#     # Train LIBRA model
#     print('LIBRA model')
#     for seed_libra in range(n_seeds):
#         # seed_libra = 0
#         print('seed_libra:',seed_libra)
#         libra_bayes_search, libra_score = train_model(libra_train, y_train, libra_test, y_test, param_space, model=model_name, seed_rf=seed_rf, seed_bayes=seed_bayes+seed_libra, n_iter=n_iter, cv=cv_group_train, groups=groups_train)
#     libra_all_fold_predictions = cross_val_predict(libra_bayes_search.best_estimator_, libra_train, y_train, groups=groups_train, cv=cv_group_train, method='predict_proba', n_jobs=-1)
#     libra_cv_scores = cross_val_score(libra_bayes_search.best_estimator_, libra_train, y_train, groups=groups_train, cv=cv_group_train, n_jobs=-1)
#     # Train BIOM+MRF model
#     # Transforming categorical variables
#     cat_vars_biom_mrf = list(set(categorical_variables_biom).union(set(categorical_variables_mrf)))
#     X_biom_mrf_train[cat_vars_biom_mrf] = X_biom_mrf_train[cat_vars_biom_mrf].astype(str)
#     X_biom_mrf_test[cat_vars_biom_mrf] = X_biom_mrf_test[cat_vars_biom_mrf].astype(str)
#     print('BIOM+MRF model')
#     for seed_biom_mrf in range(n_seeds):
#         # seed_biom_mrf = 0
#         print('seed_biom_mrf:',seed_biom_mrf)
#         biom_mrf_bayes_search, biom_mrf_score = train_model(X_biom_mrf_train, y_train, X_biom_mrf_test, y_test, param_space, model=model_name, seed_rf=seed_rf, seed_bayes=seed_bayes+seed_biom_mrf, n_iter=n_iter, cv=cv_group_train, groups=groups_train, cat_vars=cat_vars_biom_mrf )
#     biom_mrf_all_fold_predictions = cross_val_predict(biom_mrf_bayes_search.best_estimator_, X_biom_mrf_train, y_train, groups=groups_train, cv=cv_group_train, method='predict_proba', n_jobs=-1)
#     biom_mrf_cv_scores = cross_val_score(biom_mrf_bayes_search.best_estimator_, X_biom_mrf_train, y_train, groups=groups_train, cv=cv_group_train, n_jobs=-1)
#     break

# %%
# # Train BIOM model
# print('BIOM model')
# for seed_biom in range(n_seeds):
#     # seed_biom = 5
#     print('seed_biom:',seed_biom)
#     biom_bayes_search, biom_score  = train_model(X_biom_train, y_train, X_biom_test, y_test, param_space, model=model_name, seed_rf=seed_rf, seed_bayes=seed_bayes+seed_biom, n_iter=n_iter, cv=cv_group_train, groups=groups_train, cat_vars=categorical_variables_biom)
# biom_all_fold_predictions = cross_val_predict(biom_bayes_search.best_estimator_, X_biom_train, y_train, groups=groups_train, cv=cv_group_train, method='predict_proba', n_jobs=-1)
# biom_cv_scores = cross_val_score(biom_bayes_search.best_estimator_, X_biom_train, y_train, groups=groups_train, cv=cv_group_train, n_jobs=-1)
#%%
# Train MRF model
# print('MRF model')
# # Transforming categorical variables
# X_mrf_train[categorical_variables_mrf] = X_mrf_train[categorical_variables_mrf].astype(str)
# X_mrf_test[categorical_variables_mrf] = X_mrf_test[categorical_variables_mrf].astype(str)
# for seed_mrf in range(n_seeds):
#     # seed_mrf = 0
#     print('seed_mrf:',seed_mrf)
#     mrf_bayes_search, mrf_score  = train_model(X_mrf_train, y_train, X_mrf_test, y_test, param_space, model=model_name, seed_rf=seed_rf, seed_bayes=seed_bayes+seed_mrf, n_iter=n_iter, cv=cv_group_train, groups=groups_train, cat_vars=categorical_variables_mrf)
# mrf_all_fold_predictions = cross_val_predict(mrf_bayes_search.best_estimator_, X_mrf_train, y_train, groups=groups_train, cv=cv_group_train, method='predict_proba', n_jobs=-1)
# mrf_cv_scores = cross_val_score(mrf_bayes_search.best_estimator_, X_mrf_train, y_train, groups=groups_train, cv=cv_group_train, n_jobs=-1)
# all_trees, all_features, unique_trees, unique_features = get_all_trees_rules(mrf_bayes_search.best_estimator_, train_pool)
# lm_train, lm_test, correlation = leaf_correlation(mrf_bayes_search.best_estimator_, X_mrf_train, y_train, X_mrf_test, y_test, model='catboost')
# leaf_counts = mrf_bayes_search.best_estimator_.get_tree_leaf_counts()
# dcg_importance = dcg_score(all_features, correlation, unique_features, leaf_counts)
#%%
# # Train rMRF model
# print('BIOM+rMRF model')
# for seed_biom_rmrf in range(n_seeds):
# # seed_biom_rmrf = 9
#     print('seed_biom_rmrf:',seed_biom_rmrf)
#     biom_rmrf_bayes_search, biom_rmrf_score  = search_rules(X_biom_train,lm_train.iloc[:,:n_rules],y_train,X_biom_test,lm_test.iloc[:,:n_rules],y_test,param_space, model=model_name, seed_rf=seed_rf, seed_bayes=seed_bayes+seed_biom_rmrf, n_iter=n_iter, cv=cv_group_train, groups=groups_train, cat_vars=categorical_variables_biom)
# X_biom_rmrf_train = pd.merge(left=X_biom_train,right=lm_train.iloc[:,:n_rules],left_index=True,right_index=True)
# biom_rmrf_all_fold_predictions = cross_val_predict(biom_rmrf_bayes_search.best_estimator_, X_biom_rmrf_train, y_train, groups=groups_train, cv=cv_group_train, method='predict_proba', n_jobs=-1)
# biom_rmrf_cv_scores = cross_val_score(biom_rmrf_bayes_search.best_estimator_, X_biom_rmrf_train, y_train, groups=groups_train, cv=cv_group_train, n_jobs=-1)
# biom_rmrf_cv = cross_validate(biom_rmrf_bayes_search.best_estimator_, X_biom_rmrf_train, y_train, groups=groups_train, cv=cv_group_train, n_jobs=-1,
#                                 return_train_score=True, return_estimator=True, return_indices=True,)
# X_biom_rmrf_test = pd.merge(left=X_biom_test,right=lm_test.iloc[:,:n_rules],left_index=True,right_index=True)
# X_biom_rmrf_all_test = pd.merge(left=X_biom_all_test,right=lm_all_test[lm_test.columns].iloc[:,:n_rules],left_index=True,right_index=True)
#%%
# path = 'catboost_model.json'
# mrf_bayes_search.best_estimator_.save_model(path,format='json',pool=train_pool)
#%%
# with open(path, 'r') as f:
#     catboost_json = json.load(f)

# trees_list = catboost_json.get('oblivious_trees')
#%%
# import re
# pattern = r"\{([^}]+)\}"
#%%
# # Access the internal C++ object
# cb_object = mrf_bayes_search.best_estimator_._object
# # Get the total number of trees
# tree_count = cb_object._get_tree_count()
# tree_idx = 229
# # all_feats = []
# # for tree_idx in range(tree_count):
#     # Get the split conditions for all trees
# split_data = cb_object._get_tree_splits(tree_idx, train_pool)
#     # re.findall(pattern, my_string)
#     # cat = [x for x in split_data if '{' in x.split(',')[0] ]
#     # feats = [re.findall(pattern, x) for x in split_data if '{' in x and '}' in x ]
#     # print(feats)
#     # Get the leaf values for all trees
# leaf_values = cb_object._get_tree_leaf_values(tree_idx)



# #%%
# def dcg_score(all_trees, correlation, unique_features, leaf_counts):
#     rules_expanded = pd.concat([pd.Series(all_trees[x],name=x) for x in all_trees], axis=1)
#     correlation_leaf_target = correlation.set_index('leaf')
#     correlation_leaf_target['rank'] = correlation_leaf_target['correlation_target'].rank(ascending=True,method='min').astype(int)
#     vars_expanded = {}
#     for col in rules_expanded.columns:
#         tree_idx = int(col.split('_')[1])
#         for j in range( leaf_counts[tree_idx]):
#             vars_expanded.update({f'{col}-leaf_{j}':rules_expanded[col]})
#     #  rules_expanded.map(remove_threshold, na_action='ignore')
#     vars_expanded = pd.DataFrame(vars_expanded)
#     vars_bool = pd.concat([unique_features.reset_index()['index'].isin(vars_expanded[col]).rename(col) for col in vars_expanded.columns],axis=1)
#     vars_bool = vars_bool.set_index(unique_features.index).T
#     vars_bool = pd.merge(left=vars_bool, right=correlation_leaf_target['rank'], left_index=True, right_index=True, how='inner')

#     vars_dcg = {}
#     for var in unique_features.index:
#         ranks_list = vars_bool[vars_bool[var]]['rank'].to_list()
#         ideal_rank = np.arange(start=correlation_leaf_target.shape[0],stop=1,step=-1)
#         dcg = calculate_dcg(ranks_list)
#         ideal_dcg = calculate_dcg(ideal_rank)
#         vars_dcg.update( {var: dcg/ideal_dcg if ideal_dcg!=0 else 0} )
#     vars_dcg_score = pd.Series(vars_dcg,name='nDCG').sort_values(ascending=False)
#     return vars_dcg_score
#%%
# Train sMRF model
# print('BIOM+sMRF model')
# dcg_importance = dcg_score(all_features, correlation, unique_features, leaf_counts)
# for seed_biom_smrf in range(n_seeds):
#     # seed_biom_smrf = 9
#     print('seed_biom_smrf:',seed_biom_smrf)
#     top_vars = dcg_importance.index[:n_subset]
#     repeated_vars = list(set(top_vars).intersection(set(X_biom_train.columns)))
#     top_vars = top_vars.drop(repeated_vars)
#     # biom_smrf_bayes_search, biom_smrf_score  = search_rules(X_biom_train,X_mrf_train[top_vars],y_train,X_biom_test,X_mrf_test[top_vars],y_test,param_space, model=model_name, seed_rf=seed_rf, seed_bayes=seed_bayes+seed_biom_smrf, n_iter=n_iter, cv=cv_group_train, groups=groups_train, cat_vars=list( set(categorical_variables_biom).union( set(top_vars).intersection(set(categorical_variables_mrf)) )))
#     biom_smrf_bayes_search, biom_smrf_score  = search_rules(X_biom_train,X_mrf_train[top_vars],y_train,X_biom_test,X_mrf_test[top_vars],y_test,param_space, model=model_name, seed_rf=seed_rf, seed_bayes=seed_bayes+seed_biom_smrf, n_iter=n_iter, cv=cv_group_train, groups=groups_train, cat_vars=categorical_variables_biom)
# X_biom_smrf_train = pd.merge(left=X_biom_train,right=X_mrf_train[top_vars],left_index=True,right_index=True)
# biom_smrf_all_fold_predictions = cross_val_predict(biom_smrf_bayes_search.best_estimator_, X_biom_smrf_train, y_train, groups=groups_train, cv=cv_group_train, method='predict_proba', n_jobs=-1)
# biom_smrf_cv_scores = cross_val_score(biom_smrf_bayes_search.best_estimator_, X_biom_smrf_train, y_train, groups=groups_train, cv=cv_group_train, n_jobs=-1)
# X_biom_smrf_test = pd.merge(left=X_biom_test,right=X_mrf_test[top_vars],left_index=True,right_index=True)
# X_biom_smrf_all_test = pd.merge(left=X_biom_all_test,right=X_mrf_all_test[top_vars],left_index=True,right_index=True)
# %%
# # Create a sample dataset
# data = {
#     'Feature_A': [25, 40, 35, 28, 50],
#     'Feature_B': ['a', 'b', 'a', 'b', 'b']
# }
# labels = [0, 1, 0, 1, 0]
# df = pd.DataFrame(data)
# #%%
# from catboost import Pool
# train_pool = Pool(
#     data=df,
#     label=labels,
#     cat_features=['Feature_B']
# )
#%%
# Train a minimal CatBoost model with a single tree
# model = CatBoostClassifier(iterations=1, depth=2, random_seed=42, verbose=0, cat_features=['Feature_B'])
# model.fit(df, labels, )
# # Access the leaf values for the single tree
# leaf_values = model._object._get_tree_leaf_values(0)
# split_data = model._object._get_tree_splits(0,train_pool)

# # For a binary classifier, this returns a list of log-odds contributions per leaf.
# print("Leaf values for the single tree (log-odds):")
# print(leaf_values)
#%%
# print(split_data)
# %%
# get_leaf_path(1, split_data)
# %%
# from sklearn import set_config
# def search_rules(df1_train, df2_train, y_cv_train, df1_test, df2_test, y_cv_test, param_space, model='rf', seed_rf=0, seed_bayes=0, cv=10, n_iter=100, groups=None, cat_vars=None):
#     set_config(transform_output="pandas")
#     X_df1_df2 = pd.merge(left=df1_train,right=df2_train,left_index=True,right_index=True)
#     df2_features = df2_train.columns.tolist()
#     # Define the ColumnTransformer to apply SelectKBest only to df2's features
#     preprocessor = ColumnTransformer(
#         transformers = [
#             ('feature_selection', RFE(estimator=RandomForestClassifier(random_state=seed_rf),n_features_to_select=2), df2_features),
#         ],
#         remainder = 'passthrough', # Passes df1's features through without transformation
#         verbose_feature_names_out = False
#     )
#     # Define the hyperparameter grid
#     param_grid = {
#         'preprocessing__feature_selection__n_features_to_select': (1,len(df2_features)),  # Number of features to select from df2
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
#         bayes_search.fit(X=X_df1_df2, y=y_cv_train.values.squeeze())
#     else:
#         bayes_search.fit(X = X_df1_df2, y = y_cv_train.values.squeeze(), groups=groups)
#     # Get the best parameters
#     print("Best Parameters:", bayes_search.best_params_)
#     # Evaluate the best model on the test set
#     X_test = pd.merge(left=df1_test,right=df2_test,left_index=True,right_index=True)
#     test_score = bayes_search.best_estimator_.score(X_test, y_cv_test.values.squeeze())
#     print("CV Test Set Accuracy:", test_score)
    
#     return bayes_search, test_score
# %%
#%%
# def get_leaf_path(leaf_idx, split_data):
#     """
#     Reconstructs the full path of split conditions for a given leaf.

#     Args:
#         tree_idx (int): The index of the tree.
#         leaf_idx (int): The index of the leaf within that tree.
#         split_data (list): The list of split conditions from the C++ object.
#         depth (int): The maximum depth of the tree.
#         feature_names (list): The names of the features.

#     Returns:
#         list: A list of string descriptions for each split operation.
#     """
#     border_string = 'border='
#     bin_string = 'bin='
#     value_string = 'value='
#     path = []
#     # features = []
#     depth = len(split_data)
#     if depth==0:
#         return path
#     else:
#         binary_path = format(leaf_idx, f'0{depth}b')
    
#     for level, decision in enumerate(binary_path):
#         split_info = split_data[level]
#         if border_string in split_info.split(',')[-1]:
#             strings = split_info.split(', '+border_string)
#             feature_name = strings[0]
#             threshold = strings[1]
#             operations = ['<=','>']
#         elif bin_string in split_info.split(',')[-1]:
#             strings = split_info.split(', '+bin_string)
#             feature_name = strings[0]
#             threshold = strings[1]
#             operations = ['<=','>']
#         elif value_string in split_info.split(',')[-1]:
#             strings = split_info.split(', '+value_string)
#             feature_name = strings[0]
#             threshold = strings[1]
#             operations = ['!=','=']
        
#         path.append(f"{feature_name} {operations[int(decision)]} {threshold}")

#     return path

#%%
# def get_all_trees_rules(best_estimator, train_pool):
#     tree_leaf_counts = best_estimator.get_tree_leaf_counts()
#     cb_object = best_estimator._object
#     tree_count = cb_object._get_tree_count()
#     all_tree_rules_dict = {}
#     all_tree_features_dict = {}
#     for tree_idx in tqdm((range(tree_count)), total=tree_count, desc="Processing items"):
#         split_data = cb_object._get_tree_splits(tree_idx, train_pool)
#         leaf_values = cb_object._get_tree_leaf_values(tree_idx)
#         # print(split_data)
#         for k in range(tree_leaf_counts[tree_idx]):
#             leaf_path = get_leaf_path(k, split_data)
#             all_tree_rules_dict.update( {f'tree_{tree_idx}-leaf_{k}': leaf_path} )
#         features = get_unique_features(split_data)    
#         all_tree_features_dict.update( {f'tree_{tree_idx}': features} )
#     all_rules = []
#     for x in all_tree_rules_dict:
#         all_rules.extend(all_tree_rules_dict[x])
#     all_rules = pd.Series(all_rules).value_counts().to_frame(name='counts')
#     all_features = []
#     for x in all_tree_features_dict:
#         all_features.extend(all_tree_features_dict[x])
#     all_features = pd.Series(all_features).value_counts().to_frame(name='counts')
#     # unique_rules = list(set(all_rules))
#     return all_tree_rules_dict, all_tree_features_dict, all_rules, all_features
# %%
# indices = joblib.load(f'indices/split_{k}_seed_{seed_split}_indices.joblib')
# indices = joblib.load('results/train_test_indices.joblib')
# train_index = indices[f'split_{k}_seed_{seed_split}']['train_index']
# test_index = indices[f'split_{k}_seed_{seed_split}']['test_index']
# train_index = indices['train_index']
# test_index = indices['test_index']