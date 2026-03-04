# DCF-ADNI: Dementia Cognitive Forecasting with ADNI Data

A machine learning pipeline for predicting cognitive transition (CN → MCI/AD) using data from the Alzheimer's Disease Neuroimaging Initiative (ADNI). The project compares **biomarker-based** vs. **modifiable risk factor-based** models to evaluate the added value of lifestyle and demographic features in early dementia risk prediction.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Project Structure](#project-structure)
- [Pipeline: From Raw Data to Trained Model](#pipeline-from-raw-data-to-trained-model)
  - [Phase 1 — Data Preprocessing](#phase-1--data-preprocessing)
  - [Phase 2 — Feature Engineering](#phase-2--feature-engineering)
  - [Phase 3 — Model Training & Evaluation](#phase-3--model-training--evaluation)
- [Six Model Configurations](#six-model-configurations)
- [Configuration](#configuration)
- [Usage](#usage)
- [Known Limitations & Considerations](#known-limitations--considerations)
- [Dependencies](#dependencies)

---

## Project Overview

The central research question is: **Can modifiable risk factors (lifestyle, demographics, clinical measures) improve prediction of cognitive decline beyond established biomarkers?**

The pipeline:
1. Loads and cleans raw ADNI clinical + MRI data
2. Identifies subjects who transitioned from Cognitively Normal (CN) to Mild Cognitive Impairment (MCI) or Alzheimer's Disease (AD)
3. Matches transition subjects with stable CN controls on age, sex, and APOE genotype
4. Trains six model variants to compare biomarker-only, risk-factor-only, and combined approaches
5. Evaluates with ROC curves, cross-validation scores, and tree rule analysis

---

## Project Structure

```
dcf-adni/
├── data_preprocessing.py          # Entry point: raw ADNI data → clean datasets
├── model_training.py              # Entry point: clean datasets → trained models
├── model_evaluation.py            # Post-hoc model evaluation and analysis
├── main.py                        # Hydra-based entry point for preprocessing
├── extract_demo_subset.py         # Generates a small data subset for demos
├── preprocessing_demo.ipynb       # Jupyter notebook demoing preprocessing steps
│
├── src/
│   ├── data_preprocessing.py      # ADNIPreprocess class (core preprocessing logic)
│   ├── utils_model.py             # Model training utilities (WoE, rule extraction, etc.)
│   └── utils.py                   # General utilities
│
├── configs/
│   ├── config.yaml                # Hydra top-level config
│   ├── model_training.yaml        # WoE params, categorical vars, pipeline settings
│   └── preprocessing_pipeline/    # Hydra preprocessing pipeline configs
│
├── data/                          # Input CSVs and output datasets
├── model/                         # Saved model artifacts
└── README.md
```

---

## Pipeline: From Raw Data to Trained Model

### Phase 1 — Data Preprocessing

**Script:** `data_preprocessing.py` → uses `src/data_preprocessing.py` (`ADNIPreprocess` class)

The preprocessing pipeline transforms raw ADNI exports into analysis-ready, matched cohort datasets.

#### Step 1.1 — Load Raw Data

```python
preprocessor = ADNIPreprocess(
    data_path="data/All_Subjects_My_Table_03Jul2025.csv",
    mri_path="data/UCSFFSX7_20Jun2025.csv",
)
```

- **Main table**: ADNI Subjects table — clinical assessments, demographics, neuropsychological scores, laboratory results, medical history, and diagnoses across multiple visits
- **MRI table**: FreeSurfer-derived volumetric brain measures (UCSFFSX7)
- Columns are coerced to their correct data types (`Int64`, `float64`, `string`) using a pre-defined type mapping for 100+ variables

#### Step 1.2 — Encode Multi-Hot Variables

Several ADNI columns store multiple values in a single cell using pipe (`|`) delimiters (e.g., `"1|3|5"` for a patient with conditions 1, 3, and 5). These are expanded into binary indicator columns:

| Original Column | Expanded |
|---|---|
| `NPID` (neuropsychiatric symptoms) | `NPID1`, `NPID2`, ..., `NPID12` |
| `BSXSYMNO` (behavioral symptoms) | `BSXSYMNO1`, `BSXSYMNO2`, ... |
| `NXGAIT` (gait abnormalities) | `NXGAIT1`, `NXGAIT2`, ... |
| Other multi-hot columns | Similar binary expansion |

#### Step 1.3 — Clean Numeric Columns

- Force-coerce string-contaminated numeric columns (e.g., `"<200"` → `NaN`)
- Compute coefficient of variation (CV) for lab values where applicable
- Compute `HEIGHT` (cm), `WEIGHT` (kg), and `BMI` from raw ADNI fields

#### Step 1.4 — Filter Cognitively Normal Subjects

Starting from the full ADNI dataset, filter to subjects whose **first diagnosis** is CN:

```
All ADNI subjects
   └── First visit diagnosis == CN
        ├── Multiple visits (≥2 diagnoses) → candidates for transition/control
        └── Single visit only → excluded (insufficient follow-up)
```

#### Step 1.5 — Build Transition Cohort

Identify CN subjects who **later transitioned** to MCI or AD:

1. Among CN subjects with multiple visits, find those whose **last diagnosis** is MCI or AD
2. Locate the first visit where the diagnosis changed from CN → non-CN (**transition point**)
3. Extract clinical features from the **visit just before** the transition (the last CN visit)
4. This gives us the "snapshot" of modifiable and biomarker features at the moment just before decline

**Result:** `transition_df` — one row per transitioning subject, features from their last CN visit

#### Step 1.6 — Build No-Transition Control Cohort

Identify CN subjects who **remained stable** throughout all visits:

1. Among CN subjects with multiple visits, find those whose diagnosis stayed CN at every visit
2. Each subject contributes one "snapshot" row — randomly selected from a CN visit representing a stable assessment
3. Compute follow-up duration to ensure sufficient observation time

**Result:** `no_transition_df` — one row per stable CN subject

#### Step 1.7 — Match Cohorts

Match each **transition subject** to the **closest stable control** subject using nearest-neighbor matching on:

- **Age** (continuous, Euclidean distance)
- **Sex** (exact match)
- **APOE genotype** (exact match, based on allele combinations)

Matching is done **without replacement** — each control can be matched to at most one transition subject. This creates:

- `joint_dataset.csv` — the **matched dataset** (equal numbers of transition and control subjects) used for model training and evaluation
- `remaining_test.csv` — the **unmatched controls** (stable CN subjects not selected as matches) used as an additional test set to evaluate specificity

#### Step 1.8 — Feature Selection

Apply `VarianceThreshold` (threshold = 0.01) to remove near-zero-variance features — columns where almost all values are identical provide no discriminative signal.

#### Step 1.9 — Export

Save the final datasets:

| File | Description | Contents |
|---|---|---|
| `data/joint_dataset.csv` | Matched cohort | Transition + matched control subjects |
| `data/remaining_test.csv` | Unmatched controls | Stable CN subjects not used as matches |

---

### Phase 2 — Feature Engineering

**Script:** `model_training.py` → calls `feature_engineering()` from `src/utils_model.py`

Before model training, derived features are created from the raw clinical variables:

#### Step 2.1 — Biomarker Ratios (BIOM group)

| Feature | Formula | Rationale |
|---|---|---|
| `PTAU/ABETA42` | Phospho-tau / Amyloid-β42 | Key biomarker ratio for AD pathology |
| `TAU/ABETA42` | Total tau / Amyloid-β42 | Indicates neurodegeneration relative to amyloid burden |
| `medical_current` | `MHNUM × (MHCUR == 1)` | Count of current medical conditions |
| `medical_old` | `MHNUM × (MHCUR == 0)` | Count of past medical conditions |

#### Step 2.2 — Sociodemographic & Lifestyle Features (MRF group)

| Feature | Formula | Rationale |
|---|---|---|
| `married` | `PTMARRY == 1` | Social support indicator |
| `lives_alone` | `PTHOME ∈ {3, 4}` | Social isolation risk |
| `retired` | `PTNOTRT == 1` | Cognitive activity reduction risk |
| `homeowner` | `PTHOME ∈ {1, 2}` | Socioeconomic proxy |
| `social_isolation` | `married + lives_alone + retired` | Composite social risk score |
| `education_retired` | `PTEDUCAT × retired` | Education–retirement interaction |
| `years_retired` | `(visit_year - PTRTYR) × retired` | Duration of retirement |
| `married_homeowner` | `married × homeowner` | Combined SES indicator |
| `retired_lives_alone` | `retired AND lives_alone` | High-risk social pattern |
| `SES_score` | Z-normalized `[PTEDUCAT, homeowner]` | Standardized socioeconomic score |

---

### Phase 3 — Model Training & Evaluation

**Script:** `model_training.py` → `ModelTrainingPipeline` class

#### Step 3.1 — Stratified Group Train/Test Split

A single train/test split is created using `StratifiedGroupKFold`:

- **Stratification**: Preserves the transition/control class ratio in both sets
- **Group-aware**: Ensures subjects sharing a `group` ID stay together (prevents data leakage from matched pairs)

```python
train_index, test_index = next(iter(
    StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed_split)
    .split(X, y, groups)
))
```

#### Step 3.2 — Weight of Evidence (WoE) Transformation

Each continuous feature is binned and converted to a **Weight of Evidence** score using `optbinning.BinningProcess`:

1. **Fit on training data only** — the binning thresholds are learned from the training set to avoid data leakage
2. **Transform both sets** — training and test data are converted to WoE scores using the fitted bins
3. **Monotonic constraints** — each feature's WoE trend can be constrained (e.g., `ascending` for age, meaning higher age → higher WoE → higher risk)

The result for each feature group is a DataFrame containing both the **original raw values** and the **WoE-transformed values** (suffixed with `_WOE`), giving the model access to both representations.

Two separate WoE transformers are fitted:

| Transformer | Features | Categorical vars |
|---|---|---|
| `WoETransformer(woe_dict_biom)` | 24 continuous BIOM features | 15 categorical BIOM features |
| `WoETransformer(woe_dict_mrf)` | 19 continuous MRF features | 23 categorical MRF features |

#### Step 3.3 — Feature Set Assembly

Six feature sets are constructed from the WoE-transformed data:

```
WoE BIOM features ──────────────────────────┬──→ X_biom
                                             │
WoE MRF features ───────────────────────┬────┤──→ X_mrf
                                        │    │
                                   X_biom + X_mrf ──→ X_biom_mrf
                                        │
          calculate_libra_revised(X_biom_mrf) ──→ libra (single LIBRA score)
                                        │
              MRF model → tree rules → leaf memberships ──→ X_biom + lm_top ──→ X_biom_rmrf
                                        │
              MRF model → DCG importance → top features ──→ X_biom + top_mrf ──→ X_biom_smrf
```

#### Step 3.4 — Augmented Test Set Construction

The test set is augmented with the **remaining unmatched controls** (`remaining_test.csv`):

- These subjects are all stable CN (label = 0 by definition)
- They are WoE-transformed using the already-fitted binning process
- The augmented test set allows evaluation of model specificity on a broader, more realistic control population

> **Note:** The augmented test set is class-imbalanced (many more controls than transition cases). Metrics on this set should be interpreted differently from the balanced matched test set.

#### Step 3.5 — Model Training (Bayesian Hyperparameter Search)

Each model variant is trained using `BayesSearchCV` (scikit-optimize):

1. **Inner cross-validation** (`StratifiedGroupKFold`, 5 folds): Used by `BayesSearchCV` to evaluate candidate hyperparameter configurations
2. **Bayesian optimization** (50 iterations): Explores the hyperparameter space efficiently using Gaussian Process surrogate models
3. **Best model selection**: The configuration with the highest inner CV score is selected

Supported model types (configured at CLI):

| Model | Key Hyperparameters |
|---|---|
| **CatBoost** | iterations, learning_rate, depth, l2_leaf_reg, bagging_temperature |
| **XGBoost** | n_estimators, learning_rate, max_depth, subsample, colsample_bytree |
| **Random Forest** | n_estimators, max_depth, min_samples_split, min_samples_leaf |

#### Step 3.6 — Tree Rule Extraction (for rMRF and sMRF)

After training the MRF model, its internal decision tree rules are extracted:

1. **Leaf membership matrix**: For each sample, determine which leaf node it falls into in each tree. This creates a binary matrix where each column represents a tree-leaf combination
2. **Leaf correlation ranking**: Rank leaves by Cramér's V correlation with the target variable. Leaves that better discriminate transition vs. control are ranked higher
3. **nDCG feature importance**: For each MRF feature, compute a normalized Discounted Cumulative Gain score based on how often it appears in highly-correlated leaves

These are used to build:
- **BIOM+rMRF**: BIOM features + top-100 leaf memberships (the most discriminative leaf nodes from the MRF model)
- **BIOM+sMRF**: BIOM features + top-30 MRF features (selected by DCG importance)

#### Step 3.7 — Cross-Validation Predictions & Evaluation

For each trained model:
- `cross_val_predict` — out-of-fold probability predictions for ROC analysis
- `cross_val_score` — per-fold accuracy scores for variability assessment
- Test set accuracy — evaluation on the held-out matched test set

#### Step 3.8 — ROC Curve Generation

Three ROC curve plots are generated per fold:

| Plot | Data | Purpose |
|---|---|---|
| CV ROC | Out-of-fold predictions on training set | Unbiased (within fold) performance estimate |
| Test ROC | Predictions on matched test set | Generalization to unseen matched data |
| Augmented Test ROC | Predictions on test + unmatched controls | Specificity evaluation on broader population |

#### Step 3.9 — Results Serialization

All results are saved to a single `.joblib` file containing:
- Trained models (`BayesSearchCV` objects)
- Train/test indices
- CV predictions and scores
- Leaf membership matrices
- Tree rules and feature importance scores

---

## Six Model Configurations

| Model | Features | Purpose |
|---|---|---|
| **LIBRA** | A single computed LIBRA risk score | Baseline: established dementia risk index |
| **BIOM** | CSF biomarkers + cognitive scores | Biomarker-only reference model |
| **MRF** | Lifestyle + demographics + clinical | Risk factor-only model |
| **BIOM+MRF** | All features combined | Full model (are combined features better?) |
| **BIOM+rMRF** | BIOM + rule-extracted leaf memberships | Do MRF tree patterns add value to BIOM? |
| **BIOM+sMRF** | BIOM + top DCG-selected MRF features | Do the most important MRF features add value? |

---

## Configuration

### WoE & Pipeline Settings

All Weight of Evidence parameters, categorical variable lists, and pipeline settings are stored in `configs/model_training.yaml`:

```yaml
# WoE binning constraints
woe_dict_biom:
  ABETA42: { monotonic_trend: auto_asc_desc }
  subject_age: { monotonic_trend: ascending }
  ...

# Pipeline tuning
pipeline:
  n_iter: 50         # Bayesian search iterations
  n_splits: 5        # CV folds
  n_rules: 100       # Leaf memberships for rMRF
  n_subset: 30       # Top features for sMRF
```

### Preprocessing Settings

Preprocessing is configured via Hydra (`configs/preprocessing_pipeline/preprocessing.yaml`).

---

## Usage

### Step 1: Preprocess Raw Data

```bash
python data_preprocessing.py
```

This runs the full `ADNIPreprocess` pipeline and produces `data/joint_dataset.csv` and `data/remaining_test.csv`.

Alternatively, use the Hydra entry point:

```bash
python main.py preprocessing_pipeline=preprocessing
```

### Step 2: Train Models

```bash
python model_training.py --seed_split 0 --model_name catboost
```

| Argument | Options | Description |
|---|---|---|
| `--seed_split` | Any integer | Controls the outer train/test split |
| `--model_name` | `catboost`, `xgboost`, `rf` | Which model type to train |

Results are saved to `results/` and plots to `plots/`.

### Step 3: Explore Preprocessing (Optional)

```bash
jupyter notebook preprocessing_demo.ipynb
```

An interactive notebook demonstrating each preprocessing step with visualizations using a subset of the data.

---

## Known Limitations & Considerations

1. **CV prediction bias**: `cross_val_predict` uses the same inner CV splitter as `BayesSearchCV`. The CV predictions are mildly optimistically biased because the same folds were used for hyperparameter selection. Nested cross-validation would provide truly unbiased estimates.

2. **Augmented test set imbalance**: The augmented test set has far more controls than transition cases (unmatched controls are appended). ROC-AUC is robust to this, but accuracy/F1 should not be directly compared to the balanced test set.

3. **LIBRA score limitations**: The ADNI dataset does not capture all LIBRA factors. Alcohol use is proxied by `MH14AALCH` (which measures consumption during abuse episodes, not regular moderate use). Physical inactivity and diet are not available.

4. **Single outer split**: The current pipeline uses a single stratified group split rather than full K-fold cross-validation on the outer loop. Results may vary with different `seed_split` values.

---

## Dependencies

- **Data processing**: `pandas`, `numpy`, `scikit-learn`
- **Model training**: `catboost`, `xgboost`, `scikit-optimize`
- **WoE binning**: `optbinning`
- **Interpretability**: `shap`
- **Configuration**: `hydra-core`, `pyyaml`
- **Visualization**: `matplotlib`, `seaborn`
- **Utilities**: `rich`, `tqdm`, `joblib`
