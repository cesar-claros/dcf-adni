"""
End-to-end export of LIBRA, MRF, and BMCA subject-level tables, plus matched
cohort manifests and feature tables restricted to those matched subjects.

This module orchestrates the existing feature builders:

- `data_preprocessing_libra.py`
- `data_preprocessing_mrf.py`
- `data_preprocessing_bmca.py`
- `data_preprocessing_matched_cohorts.py`

It is intentionally a thin wrapper: all feature derivation and cohort logic
remain in the original modules, while this file standardizes the CSV export
workflow from the current ADNI wide table.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Optional, TypeVar

import pandas as pd

try:
    from data_preprocessing_bmca import BMCAConfig, build_adni_bmca_features_from_wide
    from data_preprocessing_libra import LibraConfig, build_adni_libra_like_from_wide
    from data_preprocessing_matched_cohorts import (
        CohortMatchConfig,
        attach_matches_to_feature_table,
        build_subject_level_cohorts,
        match_remaining_stable_to_ci_dementia,
        match_transition_to_stable_cn,
    )
    from data_preprocessing_mrf import MRFConfig, build_adni_mrf_features_from_wide
except ModuleNotFoundError:
    from Code.data_preprocessing_bmca import BMCAConfig, build_adni_bmca_features_from_wide
    from Code.data_preprocessing_libra import (
        LibraConfig,
        build_adni_libra_like_from_wide,
    )
    from Code.data_preprocessing_matched_cohorts import (
        CohortMatchConfig,
        attach_matches_to_feature_table,
        build_subject_level_cohorts,
        match_remaining_stable_to_ci_dementia,
        match_transition_to_stable_cn,
    )
    from Code.data_preprocessing_mrf import MRFConfig, build_adni_mrf_features_from_wide


ConfigT = TypeVar("ConfigT")


@dataclass
class FeatureExportConfig(CohortMatchConfig):
    input_csv: str = "data/All_Subjects_My_Table_11Mar2026.csv"
    output_dir: str = "data"
    libra_output_name: str = "adni_libra.csv"
    mrf_output_name: str = "adni_mrf_features.csv"
    bmca_output_name: str = "adni_bmca_features.csv"
    write_primary_attached_features: bool = True
    write_combined_attached_features: bool = True


def _project_config(config: FeatureExportConfig, cls: type[ConfigT]) -> ConfigT:
    values = {
        field.name: getattr(config, field.name)
        for field in fields(cls)
        if hasattr(config, field.name)
    }
    return cls(**values)


def _append_suffix(filename: str, suffix: str) -> str:
    path = Path(filename)
    return f"{path.stem}{suffix}{path.suffix or '.csv'}"


def _write_matching_outputs(
    results: dict[str, pd.DataFrame], out_dir: Path, prefix: str
) -> None:
    output_map = {
        "cohort_df": f"{prefix}_subject_level_cohort.csv",
        "transition_cohort_df": f"{prefix}_transition_cohort.csv",
        "stable_cn_cohort_df": f"{prefix}_stable_cn_cohort.csv",
        "matched_pairs_df": f"{prefix}_matched_pairs.csv",
        "matched_subjects_df": f"{prefix}_matched_subjects.csv",
        "matched_dataset_df": f"{prefix}_matched_dataset.csv",
        "unmatched_stable_cn_df": f"{prefix}_unmatched_stable_cn.csv",
        "stratum_counts_df": f"{prefix}_stratum_counts.csv",
        "ci_to_dementia_cohort_df": f"{prefix}_ci_to_dementia_cohort.csv",
        "augmentation_pairs_df": f"{prefix}_augmentation_pairs.csv",
        "augmentation_subjects_df": f"{prefix}_augmentation_subjects.csv",
        "augmentation_dataset_df": f"{prefix}_augmentation_dataset.csv",
        "augmentation_stratum_counts_df": f"{prefix}_augmentation_stratum_counts.csv",
        "combined_matched_subjects_df": f"{prefix}_combined_matched_subjects.csv",
        "combined_matched_dataset_df": f"{prefix}_combined_matched_dataset.csv",
    }
    for key, filename in output_map.items():
        results[key].to_csv(out_dir / filename, index=False)


def build_feature_tables_from_wide(
    df: pd.DataFrame, config: Optional[FeatureExportConfig] = None
) -> dict[str, pd.DataFrame]:
    """
    Build the three subject-level feature tables from one ADNI wide table.
    """
    if config is None:
        config = FeatureExportConfig()

    libra_config = _project_config(config, LibraConfig)
    mrf_config = _project_config(config, MRFConfig)
    bmca_config = _project_config(config, BMCAConfig)

    libra_df = build_adni_libra_like_from_wide(df, config=libra_config)
    mrf_df = build_adni_mrf_features_from_wide(df, config=mrf_config)
    bmca_df = build_adni_bmca_features_from_wide(df, config=bmca_config)

    return {
        "libra_df": libra_df,
        "mrf_df": mrf_df,
        "bmca_df": bmca_df,
    }


def build_feature_tables_and_matches_from_wide(
    df: pd.DataFrame, config: Optional[FeatureExportConfig] = None
) -> dict[str, pd.DataFrame]:
    """
    Build subject-level feature tables, matched cohort manifests, and matched
    feature tables from one ADNI wide table already loaded in memory.
    """
    if config is None:
        config = FeatureExportConfig()

    match_config = _project_config(config, CohortMatchConfig)
    feature_results = build_feature_tables_from_wide(df, config=config)

    cohort_df = build_subject_level_cohorts(df, config=match_config)
    match_results = match_transition_to_stable_cn(cohort_df, config=match_config)
    augmentation_results = match_remaining_stable_to_ci_dementia(
        cohort_df, match_results, config=match_config
    )
    match_results.update(augmentation_results)

    results = {**feature_results, **match_results}
    subject_id_col = match_config.subject_id_col

    if config.write_primary_attached_features:
        results["libra_matched_df"] = attach_matches_to_feature_table(
            feature_results["libra_df"],
            match_results["matched_subjects_df"],
            subject_id_col=subject_id_col,
        )
        results["mrf_matched_df"] = attach_matches_to_feature_table(
            feature_results["mrf_df"],
            match_results["matched_subjects_df"],
            subject_id_col=subject_id_col,
        )
        results["bmca_matched_df"] = attach_matches_to_feature_table(
            feature_results["bmca_df"],
            match_results["matched_subjects_df"],
            subject_id_col=subject_id_col,
        )

    if config.write_combined_attached_features:
        results["libra_combined_matched_df"] = attach_matches_to_feature_table(
            feature_results["libra_df"],
            match_results["combined_matched_subjects_df"],
            subject_id_col=subject_id_col,
        )
        results["mrf_combined_matched_df"] = attach_matches_to_feature_table(
            feature_results["mrf_df"],
            match_results["combined_matched_subjects_df"],
            subject_id_col=subject_id_col,
        )
        results["bmca_combined_matched_df"] = attach_matches_to_feature_table(
            feature_results["bmca_df"],
            match_results["combined_matched_subjects_df"],
            subject_id_col=subject_id_col,
        )

    return results


def build_feature_tables_and_matches_from_csv(
    input_csv: Optional[str] = None,
    output_dir: Optional[str] = None,
    config: Optional[FeatureExportConfig] = None,
) -> dict[str, pd.DataFrame]:
    """
    End-to-end CSV wrapper:
    1. load the ADNI wide table,
    2. build LIBRA / MRF / BMCA subject-level tables,
    3. build matched cohort manifests,
    4. attach the manifests back to each feature table,
    5. write all outputs to disk.
    """
    if config is None:
        config = FeatureExportConfig()

    resolved_input_csv = input_csv or config.input_csv
    resolved_output_dir = output_dir or config.output_dir

    df = pd.read_csv(resolved_input_csv, low_memory=False)
    results = build_feature_tables_and_matches_from_wide(df, config=config)

    out_dir = Path(resolved_output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results["libra_df"].to_csv(out_dir / config.libra_output_name, index=False)
    results["mrf_df"].to_csv(out_dir / config.mrf_output_name, index=False)
    results["bmca_df"].to_csv(out_dir / config.bmca_output_name, index=False)

    _write_matching_outputs(results, out_dir, config.output_prefix)

    if config.write_primary_attached_features:
        results["libra_matched_df"].to_csv(
            out_dir / _append_suffix(config.libra_output_name, "_matched"),
            index=False,
        )
        results["mrf_matched_df"].to_csv(
            out_dir / _append_suffix(config.mrf_output_name, "_matched"),
            index=False,
        )
        results["bmca_matched_df"].to_csv(
            out_dir / _append_suffix(config.bmca_output_name, "_matched"),
            index=False,
        )

    if config.write_combined_attached_features:
        results["libra_combined_matched_df"].to_csv(
            out_dir / _append_suffix(config.libra_output_name, "_combined_matched"),
            index=False,
        )
        results["mrf_combined_matched_df"].to_csv(
            out_dir / _append_suffix(config.mrf_output_name, "_combined_matched"),
            index=False,
        )
        results["bmca_combined_matched_df"].to_csv(
            out_dir / _append_suffix(config.bmca_output_name, "_combined_matched"),
            index=False,
        )

    return results


EXAMPLE = r"""
from data_preprocessing_feature_exports import (
    FeatureExportConfig,
    build_feature_tables_and_matches_from_csv,
)

cfg = FeatureExportConfig(
    input_csv="data/All_Subjects_My_Table_11Mar2026.csv",
    output_dir="data",
    subject_id_col="subject_id",
    visit_col="visit",
    diagnosis_col="DIAGNOSIS",
    age_col="entry_age",
    sex_col="PTGENDER",
    genotype_col="GENOTYPE",
    entry_group_col="entry_research_group",
)

results = build_feature_tables_and_matches_from_csv(config=cfg)

print(results["libra_df"].head())
print(results["matched_pairs_df"].head())
print(results["mrf_combined_matched_df"].head())
"""
