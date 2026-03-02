"""
Extract a small representative subset from the full ADNI data
for use in the preprocessing demo notebook.

Uses ADNIPreprocess to load data and identify subject groups,
then samples from each group to create a compact demo dataset.
"""

import logging
import numpy as np
from src.data_preprocessing import ADNIPreprocess

logging.basicConfig(level=logging.INFO)

# --- Config ---
N_TRANSITION = 10     # CN subjects with multiple diagnoses
N_NO_TRANSITION = 30  # CN subjects with single diagnosis (need extras for matching pool)
SEED = 42

np.random.seed(SEED)

# Use ADNIPreprocess to load data and classify CN subjects
preprocessor = ADNIPreprocess(
    data_path="data/All_Subjects_My_Table_03Jul2025.csv",
    mri_path="data/UCSFFSX7_20Jun2025.csv",
)
preprocessor.load_data()
preprocessor.encode_multihot_variables()
preprocessor.coerce_numeric_columns()
preprocessor.compute_bmi()
preprocessor.filter_cn_subjects()

# The class has now classified subjects into:
#   preprocessor._subjects_multiple_dx  (CN who changed diagnosis)
#   preprocessor._subjects_one_dx       (CN who stayed CN)

# Build qualified transition list using the same logic as build_transition_cohort
preprocessor.build_transition_cohort()
qualified_transition = preprocessor.subjects_transition_df["subject_id"].unique()

print(f"Qualified transition subjects: {len(qualified_transition)}")
print(f"Single-diagnosis CN subjects:  {len(preprocessor._subjects_one_dx)}")

# Sample
sampled_transition = np.random.choice(
    qualified_transition,
    size=min(N_TRANSITION, len(qualified_transition)),
    replace=False,
)
sampled_no_transition = np.random.choice(
    preprocessor._subjects_one_dx,
    size=min(N_NO_TRANSITION, len(preprocessor._subjects_one_dx)),
    replace=False,
)

all_sampled = list(sampled_transition) + list(sampled_no_transition)

# Extract all rows for sampled subjects from the original loaded data
subset_df = preprocessor.data_df[preprocessor.data_df["subject_id"].isin(all_sampled)]
mri_subset_df = preprocessor.mri_df[preprocessor.mri_df["PTID"].isin(all_sampled)]

print(f"\nSubset: {subset_df.shape[0]} rows, {len(all_sampled)} subjects")
print(f"  - Transition:    {len(sampled_transition)} subjects")
print(f"  - No transition: {len(sampled_no_transition)} subjects")
print(f"MRI subset: {mri_subset_df.shape[0]} rows")

# Save
subset_df.to_csv("data/demo_subjects.csv", index=False)
mri_subset_df.to_csv("data/demo_mri.csv", index=False)
print("\nSaved: data/demo_subjects.csv, data/demo_mri.csv")
