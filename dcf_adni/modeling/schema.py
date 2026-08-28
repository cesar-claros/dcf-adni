"""Shared column schema for the matched-cohort feature tables.

Promoted unchanged from ``scripts/model_strate_cv_evaluation.py`` (phase 3 of
the code reorganization; see ``Documentation/experiment_harness_design.md``).
"""

from __future__ import annotations

import pandas as pd

# Columns that are never features: identifiers, cohort provenance, labels, and
# outcome-derived variables. first_conversion_month directly encodes the label
# (NaN for stable-CN, non-NaN for transition subjects); baseline_diagnosis and
# n_followup_visits_ge12_with_diag are cohort-selection and study-participation
# variables, not baseline risk factors.
METADATA_COLS = frozenset({
    "subject_id", "pair_id", "group", "transition", "transition_label",
    "matched_cohort", "analysis_set", "evaluation_eligible",
    "abs_age_gap", "split", "split_group_source",
    "first_conversion_month", "baseline_diagnosis", "n_followup_visits_ge12_with_diag",
})

# Two label columns exist in the exports. ``transition`` is defined for every
# subject (primary and augmentation); ``transition_label`` is the primary-pair
# label. In the current feature exports the two agree wherever both are
# present, but harness callers must still choose one explicitly.
TRANSITION_COL = "transition"
TRANSITION_LABEL_COL = "transition_label"
GROUP_COL = "group"


def feature_cols(df: pd.DataFrame, audit_path: str | None = None) -> list[str]:
    """Return the feature columns of ``df``, optionally filtered by a column audit.

    The audit CSV must have ``column`` and ``keep_for_modeling`` columns; only
    features with ``keep_for_modeling == 1`` survive.
    """
    all_feats = [c for c in df.columns if c not in METADATA_COLS]
    if audit_path is None:
        return all_feats
    audit = pd.read_csv(audit_path)
    keep = set(audit.loc[audit["keep_for_modeling"] == 1, "column"])
    return [c for c in all_feats if c in keep]


def load_combined(path: str, label_col: str = TRANSITION_COL) -> pd.DataFrame:
    """Load a combined matched-cohort CSV, coercing label and group to numeric."""
    df = pd.read_csv(path)
    df[label_col] = pd.to_numeric(df[label_col], errors="coerce")
    df[GROUP_COL] = pd.to_numeric(df[GROUP_COL], errors="coerce")
    return df


def evaluation_eligible(df: pd.DataFrame) -> pd.DataFrame:
    """Return the rows that count for primary-task evaluation."""
    return df[df["evaluation_eligible"] == 1].copy()


def primary_mask(df: pd.DataFrame):
    """Boolean array marking primary-pair rows (validation-scoring mask)."""
    return (df["analysis_set"] == "primary").values
