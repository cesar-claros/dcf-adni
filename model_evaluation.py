#%%
import pandas as pd
import numpy as np
# from optbinning import BinningProcess
import matplotlib.pyplot as plt
# from sklearn.model_selection import GridSearchCV
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
# from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
# from sklearn.metrics import roc_auc_score
# from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold
# from skopt import BayesSearchCV
# from sklearn.tree import _tree
# from scipy.stats import chi2_contingency
# from sklearn.model_selection import StratifiedGroupKFold
# from sklearn.compose import ColumnTransformer # Import ColumnTransformer
# from sklearn.feature_selection import SelectKBest, f_classif
# from sklearn.pipeline import Pipeline
# from sklearn.feature_selection import RFE
# from sklearn.feature_selection import SequentialFeatureSelector # Import SequentialFeatureSelector
# from sklearn.base import BaseEstimator, TransformerMixin
import joblib
import shap
# import argparse
from src.utils_model import *
# from sklearn.linear_model import LogisticRegression
from skopt.space import Real, Integer, Categorical
from sklearn.model_selection import cross_val_predict, cross_val_score
from sklearn.metrics import RocCurveDisplay
import json
from matplotlib.lines import Line2D
from sklearn.model_selection import cross_validate
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
                'BSXSYMNO':{'cat_unknown':0},
                'PTAU/ABETA42': {"monotonic_trend":'auto_asc_desc' },
                'TAU/ABETA42': {"monotonic_trend":'auto_asc_desc' },
            }
categorical_variables_biom = ['FAQFINAN','FAQTRAVL','NXGAIT','BCDPMOOD','NPID',
                        'NPID8','BCHDACHE','BCMUSCLE','BCVISION','BSXSYMNO',
                        'MHCUR','MHNUM','PTGENDER']
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
            }
categorical_variables_mrf = ['PTHOME','PTMARRY','PTNOTRT','PTPLANG','PTGENDER',
                            'HMHYPERT','NPIK','NPIK1','NPIK2','NPIK4','NPIK6',
                            'NPIK9A','NPIK9B','NPIK9C','MH14ALCH','MH16SMOK','BCINSOMN',
                            'MH12RENA','MH4CARD','NXAUDITO','PXHEART','PXPERIPH',
                            'lives_alone','married_homeowner',
                            'retired_lives_alone','homeowner_flag']
#%%
model_name = 'catboost'
k = 0
seed_split = 36
#%%
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
# sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed_split)
# cv_group_train = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed_cv)
param_space_catboost = {
    'iterations': Integer(100, 1000),
    'learning_rate': Real(1e-3, 1.0, 'log-uniform'),
    'depth': Integer(4, 12),
    'l2_leaf_reg': Real(1, 10, 'uniform'),
    'bagging_temperature': Real(0.0, 1.0, 'uniform'),
    'border_count': Integer(32, 255)
}

param_space = param_space_catboost
#%%
# Read data
joint_dataset_df = pd.read_csv('data/joint_dataset.csv', index_col=0).set_index('subject_id')
remaining_test_df = pd.read_csv('data/remaining_test.csv', index_col=0).set_index('subject_id')
dataset_df = feature_engineering(joint_dataset_df)
additional_test_df = feature_engineering(remaining_test_df)
# %%
results_dict = joblib.load(f'model/{model_name}_split_{k}_seed_{seed_split}_seedcv_{seed_cv}_results.joblib')
# %%
train_index = results_dict['train_index']
test_index = results_dict['test_index']
groups_train = dataset_df.iloc[train_index]['group']
cv_group_train = results_dict['cv_group_train']
libra_bayes_search = results_dict['libra_bayes_search']
biom_mrf_bayes_search = results_dict['biom_mrf_bayes_search']
biom_bayes_search = results_dict['biom_bayes_search']
mrf_bayes_search = results_dict['mrf_bayes_search']
biom_rmrf_bayes_search = results_dict['biom_rmrf_bayes_search']
biom_smrf_bayes_search = results_dict['biom_smrf_bayes_search']

#%%
fig, ax = plt.subplots(figsize=(5,5))
n_bins = np.arange(48,94,2)
sns.histplot(pd.concat([remaining_test_df['subject_age'],joint_dataset_df.iloc[test_index]['subject_age']]), bins=n_bins, stat='probability', element='step',ax=ax, label='Augmented Test Set',alpha=0.25, linewidth=2, color='blue')
sns.histplot(joint_dataset_df.iloc[train_index]['subject_age'], bins=n_bins, stat='probability', element='step',ax=ax, label='Training Set',alpha=0.75, fill=False, color='black', linewidth=2)
sns.histplot(joint_dataset_df.iloc[test_index]['subject_age'], bins=n_bins, stat='probability', element='step',ax=ax, label='Test Set',alpha=0.25, linewidth=2, color='orange')
ax.set_ylabel('Proportion')
ax.set_xlabel('Age')
ax.legend()
fig.tight_layout()

#%%
fig, ax = plt.subplots(figsize=(5,5))
sns.histplot(remaining_test_df['subject_age'], bins=n_bins, stat='count', color='gray', ax=ax, label='Remaining subjects', element="step")
sns.histplot(joint_dataset_df['subject_age'], bins=n_bins, stat='count', color='green', ax=ax, label='Paired data', element="step")
sns.histplot(joint_dataset_df.iloc[train_index]['subject_age'], bins=n_bins, stat='count', color='blue', ax=ax, label='Training set', element="step")
sns.histplot(joint_dataset_df.iloc[test_index]['subject_age'], bins=n_bins, stat='count', color='orange', ax=ax, label='Test set', element="step")
ax.set_xlabel('Age')
ax.set_ylabel('Counts')
ax.legend()
fig.tight_layout()

#%%
X_mrf_train, X_mrf_test, y_train, y_test, bp_mrf = transform_WOE(dataset_df, woe_dict_mrf, categorical_variables_mrf, train_index, test_index)
X_biom_train, X_biom_test, _, _, bp_biom = transform_WOE(dataset_df, woe_dict_biom, categorical_variables_biom, train_index, test_index)
X_biom_mrf_train = pd.merge(left=X_biom_train, right=X_mrf_train, left_index=True, right_index=True)
X_biom_mrf_test = pd.merge(left=X_biom_test, right=X_mrf_test, left_index=True, right_index=True)
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
# %%
# Compute partial LIBRA scores
libra_train = X_biom_mrf_train.apply(calculate_libra_revised, axis=1).to_frame()
libra_test = X_biom_mrf_test.apply(calculate_libra_revised, axis=1).to_frame()
# %%
libra_all_fold_predictions = cross_val_predict(libra_bayes_search.best_estimator_, libra_train, y_train, groups=groups_train, cv=cv_group_train, method='predict_proba', n_jobs=-1)
libra_cv_scores = cross_val_score(libra_bayes_search.best_estimator_, libra_train, y_train, groups=groups_train, cv=cv_group_train, n_jobs=-1)
libra_score = libra_bayes_search.best_estimator_.score(libra_test,y_test)
# %%
biom_mrf_all_fold_predictions = cross_val_predict(biom_mrf_bayes_search.best_estimator_, X_biom_mrf_train, y_train, groups=groups_train, cv=cv_group_train, method='predict_proba', n_jobs=-1)
biom_mrf_cv_scores = cross_val_score(biom_mrf_bayes_search.best_estimator_, X_biom_mrf_train, y_train, groups=groups_train, cv=cv_group_train, n_jobs=-1)
biom_mrf_score = biom_mrf_bayes_search.best_estimator_.score(X_biom_mrf_test,y_test)
#%%
biom_all_fold_predictions = cross_val_predict(biom_bayes_search.best_estimator_, X_biom_train, y_train, groups=groups_train, cv=cv_group_train, method='predict_proba', n_jobs=-1)
biom_cv_scores = cross_val_score(biom_bayes_search.best_estimator_, X_biom_train, y_train, groups=groups_train, cv=cv_group_train, n_jobs=-1)
biom_score = biom_bayes_search.best_estimator_.score(X_biom_test, y_test)
#%%
mrf_all_fold_predictions = cross_val_predict(mrf_bayes_search.best_estimator_, X_mrf_train, y_train, groups=groups_train, cv=cv_group_train, method='predict_proba', n_jobs=-1)
mrf_cv_scores = cross_val_score(mrf_bayes_search.best_estimator_, X_mrf_train, y_train, groups=groups_train, cv=cv_group_train, n_jobs=-1)
mrf_score = mrf_bayes_search.best_estimator_.score(X_mrf_test, y_test)
#%%
lm_train, lm_test, corr = leaf_correlation(mrf_bayes_search.best_estimator_, X_mrf_train, y_train, X_mrf_test, y_test, model=model_name)
_, lm_all_test, _ = leaf_correlation(mrf_bayes_search.best_estimator_, X_mrf_train, y_train, X_mrf_all_test, y_all_test, model=model_name)
X_biom_rmrf_train = pd.merge(left=X_biom_train,right=lm_train.iloc[:,:n_rules],left_index=True,right_index=True)
#%%
biom_rmrf_all_fold_predictions = cross_val_predict(biom_rmrf_bayes_search.best_estimator_, X_biom_rmrf_train, y_train, groups=groups_train, cv=cv_group_train, method='predict_proba', n_jobs=-1)
biom_rmrf_cv_scores = cross_val_score(biom_rmrf_bayes_search.best_estimator_, X_biom_rmrf_train, y_train, groups=groups_train, cv=cv_group_train, n_jobs=-1)
X_biom_rmrf_test = pd.merge(left=X_biom_test,right=lm_test.iloc[:,:n_rules],left_index=True,right_index=True)
X_biom_rmrf_all_test = pd.merge(left=X_biom_all_test,right=lm_all_test[lm_test.columns].iloc[:,:n_rules],left_index=True,right_index=True)
biom_rmrf_score = biom_rmrf_bayes_search.best_estimator_.score(X_biom_rmrf_test, y_test)
#%%
# mrf_ranked_vars = rule_correlation(mrf_bayes_search.best_estimator_, corr, X_mrf_train, y_train, model=model_name, seed=seed_split+k)
#%%
# mrf_ranked_vars = mrf_ranked_vars.dropna()
#%%
# top_vars = mrf_ranked_vars.index[:]
top_vars = X_mrf_train.columns
# repeated_vars = list(set(top_vars).intersection(set(X_biom_train.columns)))
repeated_vars = list(set(X_mrf_train.columns).intersection(set(X_biom_train.columns)))
top_vars = top_vars.drop(repeated_vars)
X_biom_smrf_train = pd.merge(left=X_biom_train,right=X_mrf_train[top_vars],left_index=True,right_index=True)
# X_biom_smrf_train = pd.merge(left=X_biom_train,right=X_mrf_train,left_index=True,right_index=True)
biom_smrf_all_fold_predictions = cross_val_predict(biom_smrf_bayes_search.best_estimator_, X_biom_smrf_train, y_train, groups=groups_train, cv=cv_group_train, method='predict_proba', n_jobs=-1)
biom_smrf_cv_scores = cross_val_score(biom_smrf_bayes_search.best_estimator_, X_biom_smrf_train, y_train, groups=groups_train, cv=cv_group_train, n_jobs=-1)
X_biom_smrf_test = pd.merge(left=X_biom_test,right=X_mrf_test[top_vars],left_index=True,right_index=True)
X_biom_smrf_all_test = pd.merge(left=X_biom_all_test,right=X_mrf_all_test[top_vars],left_index=True,right_index=True)
# X_biom_smrf_test = pd.merge(left=X_biom_test,right=X_mrf_test,left_index=True,right_index=True)
# X_biom_smrf_all_test = pd.merge(left=X_biom_all_test,right=X_mrf_all_test,left_index=True,right_index=True)
biom_smrf_score = biom_smrf_bayes_search.best_estimator_.score(X_biom_smrf_test, y_test)


#%%
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
#%%
# Plot ROC considering cross-validation folds
fig, ax = plt.subplots(figsize=(8, 6))
RocCurveDisplay.from_predictions(y_train, biom_mrf_all_fold_predictions[:,1], ax=ax, name='BIOM+MRF', color='blue')
RocCurveDisplay.from_predictions(y_train, biom_rmrf_all_fold_predictions[:,1], ax=ax, name='BIOM+rMRF', color='red')
RocCurveDisplay.from_predictions(y_train, biom_smrf_all_fold_predictions[:,1], ax=ax, name='BIOM+sMRF', color='green')
# RocCurveDisplay.from_estimator(classifier_biom_cluster, X_test_biom_cluster, y_test, ax=ax, name='BIOM+WOE(clusters)')
RocCurveDisplay.from_predictions(y_train, biom_all_fold_predictions[:,1], ax=ax, name='BIOM', color='magenta')
RocCurveDisplay.from_predictions(y_train, mrf_all_fold_predictions[:,1], ax=ax, name='MRF', plot_chance_level=True, color='cyan')
ax.minorticks_on()
ax.grid(which='both')
ax.set_aspect('equal', adjustable='box')
ax.set_xlabel('1-Specificity (FPR)')
ax.set_ylabel('Sensitivity (TPR)')
ax.set_title('ROC Curve (CV Test Sets)')
#%%
# Plot ROC considering test set
fig, ax = plt.subplots(figsize=(8, 6))
RocCurveDisplay.from_estimator(biom_mrf_bayes_search.best_estimator_, X_biom_mrf_test, y_test, ax=ax, name='BIOM+MRF', color='blue')
RocCurveDisplay.from_estimator(biom_rmrf_bayes_search.best_estimator_, X_biom_rmrf_test, y_test, ax=ax, name='BIOM+rMRF', color='red')
RocCurveDisplay.from_estimator(biom_smrf_bayes_search.best_estimator_, X_biom_smrf_test, y_test, ax=ax, name='BIOM+sMRF', color='green')
# RocCurveDisplay.from_estimator(classifier_biom_cluster, X_test_biom_cluster, y_test, ax=ax, name='BIOM+WOE(clusters)')
RocCurveDisplay.from_estimator(biom_bayes_search.best_estimator_, X_biom_test, y_test, ax=ax, name='BIOM', color='magenta')
RocCurveDisplay.from_estimator(mrf_bayes_search.best_estimator_, X_mrf_test, y_test, ax=ax, name='MRF', plot_chance_level=True, color='cyan')
ax.minorticks_on()
ax.grid(which='both')
ax.set_aspect('equal', adjustable='box')
ax.set_xlabel('1-Specificity (FPR)')
ax.set_ylabel('Sensitivity (TPR)')
ax.set_title('ROC Curve (Test Set)')
#%%
# Plot ROC considering combined test set
fig, ax = plt.subplots(figsize=(8, 6))
RocCurveDisplay.from_estimator(biom_mrf_bayes_search.best_estimator_, X_biom_mrf_all_test, y_all_test, ax=ax, name='BIOM+MRF', color='blue')
RocCurveDisplay.from_estimator(biom_rmrf_bayes_search.best_estimator_, X_biom_rmrf_all_test, y_all_test, ax=ax, name='BIOM+rMRF', color='red')
RocCurveDisplay.from_estimator(biom_smrf_bayes_search.best_estimator_, X_biom_smrf_all_test, y_all_test, ax=ax, name='BIOM+sMRF', color='green')
# RocCurveDisplay.from_estimator(classifier_biom_cluster, X_test_biom_cluster, y_test, ax=ax, name='BIOM+WOE(clusters)')
RocCurveDisplay.from_estimator(biom_bayes_search.best_estimator_, X_biom_all_test, y_all_test, ax=ax, name='BIOM', color='magenta')
RocCurveDisplay.from_estimator(mrf_bayes_search.best_estimator_, X_mrf_all_test, y_all_test, ax=ax, name='MRF', plot_chance_level=True, color='cyan')
ax.minorticks_on()
ax.grid(which='both')
ax.set_aspect('equal', adjustable='box')
ax.set_xlabel('1-Specificity (FPR)')
ax.set_ylabel('Sensitivity (TPR)')
ax.set_title('ROC Curve (Test Set Augmented)')
#%%
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
# Create custom legend handles using Line2D
legend_elements = [
    Line2D([0], [0], marker='o', color='w', label='CV Test Set',
           markerfacecolor='gray', markersize=10),
    Line2D([0], [0], marker='o', color='w', label='Test Set',
           markerfacecolor='red', markersize=10)
]

# Add the custom legend to the plot
ax.legend(handles=legend_elements)
ax.set_ylabel('Accuracy')
ax.set_xlabel('Input Feature Groups')
# %%
biom_rmrf_cv = cross_validate(biom_rmrf_bayes_search.best_estimator_, X_biom_rmrf_train, y_train, groups=groups_train, cv=cv_group_train, n_jobs=-1,
                                        return_train_score=True, return_estimator=True, return_indices=True,)
# %%
importances_kfold = []
shap_values_kfold = []
shap_values_extended_kfold = []
shap_values_mean_kfold = []
features = list(X_biom_rmrf_train) 
for k in range(cv_group_train.n_splits):
    model_kfold = biom_rmrf_cv['estimator'][k]
    X_train_kfold = X_biom_rmrf_train.iloc[biom_rmrf_cv['indices']['train'][k]]
    X_test_kfold = X_biom_rmrf_train.iloc[biom_rmrf_cv['indices']['test'][k]]
    # Feature importance by model
    feature_importances = model_kfold.named_steps['random_forest'].feature_importances_
    importances = pd.DataFrame(index=features)
    importances['importance'] = feature_importances
    importances['rank'] = importances['importance'].rank(ascending=False).values
    importances_kfold.append(importances)
    # SHAP values
    # Set up explainer using the model and feature values from training set
    explainer = shap.TreeExplainer(model_kfold.named_steps['random_forest'], X_train_kfold)
    # Get (and store) Shapley values along with base and feature values
    shap_values_extended = explainer(X_test_kfold)
    shap_values = shap_values_extended.values
    shap_values_extended_kfold.append(shap_values_extended)
    shap_values_kfold.append(shap_values_extended.values)
    shap_values_mean = pd.DataFrame(index=features)
    shap_values_mean['mean_shap'] = np.mean(shap_values, axis=0)
    shap_values_mean['abs_mean_shap'] = np.abs(shap_values_mean)
    shap_values_mean['mean_abs_shap'] = np.mean(np.abs(shap_values), axis=0)
    shap_values_mean['rank'] = shap_values_mean['mean_abs_shap'].rank(
        ascending=False).values
    shap_values_mean.sort_index()
    shap_values_mean_kfold.append(shap_values_mean)
# %%
top_k = 30
#%%
# Initialise DataFrame (stores mean of the absolute SHAP values for each kfold)
mean_abs_shap = pd.DataFrame()
# For each k-fold split
for k in range(cv_group_train.n_splits):
    # mean of the absolute SHAP values for each k-fold split
    mean_abs_shap[f'{k}'] = shap_values_mean_kfold[k]['mean_abs_shap']
mean_abs_shap_summary = pd.DataFrame()
mean_abs_shap_summary['min'] = mean_abs_shap.min(axis=1)
mean_abs_shap_summary['median'] = mean_abs_shap.median(axis=1) 
mean_abs_shap_summary['max'] = mean_abs_shap.max(axis=1)
mean_abs_shap_summary.sort_values('median', inplace=True, ascending=False)    
top_features_shap = list(mean_abs_shap_summary.head(top_k).index)
#%%
fig = plt.figure(figsize=(10,6))
ax1 = fig.add_subplot(111)
ax1.violinplot(mean_abs_shap.loc[top_features_shap].T,
               showmedians=True,
               widths=1)
ax1.set_ylim(0)
labels = top_features_shap
ax1.set_xticks(np.arange(1, len(labels) + 1))
ax1.set_xticklabels(labels, rotation=45, ha='right')
ax1.grid(which='both')
ax1.set_ylabel('|SHAP value| (log odds)')
ax1.set_title(f'Top {top_k} SHAP values across CV Test Sets')

#%%
# Initialise DataFrame (stores feature importance values for each kfold)
importances_df = pd.DataFrame()
# For each k-fold
for k in range(5):
    # feature importance value for each k-fold split
    importances_df[f'{k}'] = importances_kfold[k]['importance']
importances_summary = pd.DataFrame()
importances_summary['min'] = importances_df.min(axis=1)
importances_summary['median'] = importances_df.median(axis=1) 
importances_summary['max'] = importances_df.max(axis=1)
importances_summary.sort_values('median', inplace=True, ascending=False)
top_features_importances = list(importances_summary.head(top_k).index)
#%%
fig = plt.figure(figsize=(10,6))
ax1 = fig.add_subplot(111)
ax1.violinplot(importances_summary.loc[top_features_importances].T,
              showmedians=True,
              widths=1)
ax1.set_ylim(0)
labels = top_features_importances
ax1.set_xticks(np.arange(1, len(labels) + 1))
ax1.set_xticklabels(labels, rotation=45, ha='right')
ax1.grid(which='both')
ax1.set_ylabel('Importance')
ax1.set_title(f'Top {top_k} Feature Importance across CV Test Sets')

# %%
shap_importance = pd.DataFrame()
shap_importance['Shap'] = mean_abs_shap_summary['median']
shap_importance = shap_importance.merge(
    importances_summary['median'], left_index=True, right_index=True)
shap_importance.rename(columns={'median':'Importance'}, inplace=True)
shap_importance.sort_values('Shap', inplace=True, ascending=False)

fig = plt.figure(figsize=(6,6))
ax1 = fig.add_subplot(111)
ax1.scatter(shap_importance['Shap'],
            shap_importance['Importance'])
ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.set_xlabel('SHAP value (median of the k-folds [mean |Shap|])')
ax1.set_ylabel('Importance values (median of the k-folds)')
ax1.set_title('SHAP values vs Feature Importance')
ax1.grid()

# %%
# X_mrf_train.columns

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
# %%
all_tree_rules_dict = all_tree_rules(mrf_bayes_search.best_estimator_, X_mrf_train, y_train, model=model_name, seed=seed_split)

# %%
results_scores_dict = { 'libra':    {'libra_all_fold_predictions':libra_all_fold_predictions,
                                    'libra_cv_scores':libra_cv_scores,
                                    'libra_score':libra_score},
                        'biom_mrf': {'biom_mrf_all_fold_predictions':biom_mrf_all_fold_predictions,
                                    'biom_mrf_cv_scores':biom_mrf_cv_scores,
                                    'biom_mrf_score':biom_mrf_score},
                        'biom':     {'biom_all_fold_predictions':biom_all_fold_predictions,
                                    'biom_cv_scores':biom_cv_scores,
                                    'biom_score':biom_score},
                        'mrf':      {'mrf_all_fold_predictions':mrf_all_fold_predictions,
                                    'mrf_cv_scores':mrf_cv_scores,
                                    'mrf_score':mrf_score},
                        'biom_rmrf':{'biom_rmrf_all_fold_predictions':biom_rmrf_all_fold_predictions,
                                    'biom_rmrf_cv_scores':biom_rmrf_cv_scores,
                                    'biom_rmrf_score':biom_rmrf_score},
                        'biom_smrf':{'biom_smrf_all_fold_predictions':biom_smrf_all_fold_predictions,
                                    'biom_smrf_cv_scores':biom_smrf_cv_scores,
                                    'biom_smrf_score':biom_smrf_score},
                        'biom_rmrf_cv':biom_rmrf_cv,
                        'X_biom_rmrf_train':X_biom_rmrf_train,
                        'X_biom_rmrf_test':X_biom_rmrf_test,
                        'all_tree_rules_dict':all_tree_rules_dict        
                        }
# %%
# joblib.dump(results_scores_dict, f'results/results_scores_dict.joblib')
# %%
def set_ax(ax, category_list, feat, rotation=0):
    '''
    ax [matplotlib axis object] = matplotlib axis object
    category_list [list] = used for the xtick labels (the grouping of the data)
    rotation [integer] = xtick label rotation
    feat [string] = used in the axis label, the feature that is being plotted
    
    resource: 
    https://matplotlib.org/3.1.0/gallery/statistics/customized_violin.html
    '''
    # Set the axes ranges and axes labels
    ax.get_xaxis().set_tick_params(direction='out')
    ax.xaxis.set_ticks_position('bottom')
    ax.set_xticks(np.arange(1, len(category_list) + 1))
    category_list_labels =  [np.round(c,3) for c in category_list] 
    ax.set_xticklabels(category_list_labels, rotation=rotation, fontsize=10)
    ax.set_xlim(0.25, len(category_list) + 0.75)
    ax.set_ylabel(f'SHAP values for {feat}', fontsize=12)
    ax.set_xlabel(f'Feature values for {feat}', fontsize=12)
    return(ax)
# %%
shap_values_extended = shap_values_extended_kfold[0]
feat_to_show = top_features_shap[:30]
#%%
fig = plt.figure(figsize=(20,20))
# for each feature, prepare the data for the violin plot.
# data either already in categories, or if there's more than 50 unique values
# for a feature then assume it needs to be binned, and a violin for each bin
for n, feat in enumerate(feat_to_show):    
    feature_data = shap_values_extended[:, feat].data
    feature_shap = shap_values_extended[:, feat].values

    # if feature has more that 50 unique values, then assume it needs to be 
    # binned (other assume they are unique categories)
    if np.unique(feature_data).shape[0] > 50:
        # bin the data, create a violin per bin
        
        # settings for the plot
        rotation = 45
        step = 30
        n_bins = min(11, np.int((feature_data.max())/step))
        
        # create list of bin values
        bin_list = [(i*step) for i in range(n_bins)]
        bin_list.append(feature_data.max())

        # create list of bins (the unique categories)
        category_list =  [f'{i*step}-{(i+1)*step}' for i in range(n_bins-1)]
        category_list.append(f'{(n_bins-1)*step}+')

        # bin the feature data
        feature_data = pd.cut(feature_data, bin_list, labels=category_list)

    else:
        # create a violin per unique value
        
        # settings for the plot
        rotation = 90
        
        # create list of unique categories in the feature data
        category_list = np.unique(feature_data)
        category_list = [i for i in category_list if ~np.isnan(i)]

    # create a list, each entry contains the corresponsing SHAP value for that 
    # category (or bin). A violin will represent each list.    
    shap_per_category = []
    for category in category_list:
        mask = feature_data == category
        shap_per_category.append(feature_shap[mask])

    # create violin plot
    ax = fig.add_subplot(6,5,n+1)
    ax.violinplot(shap_per_category, showmedians=True, widths=0.9)
    
    # Add line at Shap = 0
    feature_values = shap_values_extended[:, feat].data
    ax.plot([0, len(feature_values)], [0,0],c='0.5')   

    # customise the axes
    ax = set_ax(ax, category_list, feat, rotation=rotation)
    plt.subplots_adjust(bottom=0.15, wspace=0.05)
    
    # Adjust stroke severity tickmarks
    if feat == 'Stroke severity':
        ax.set_xticks(np.arange(0, len(category_list), 2))
        ax.set_xticklabels(category_list[0::2])   
    
    # Add title
    ax.set_title(feat)
    
plt.tight_layout(pad=2)
    
# fig.savefig(
# f'output/xgb_thrombolysis_shap_violin.jpg', dpi=300,
#  bbox_inches='tight', pad_inches=0.2)
# %%
