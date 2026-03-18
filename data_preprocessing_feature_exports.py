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

import numpy as np
import pandas as pd

try:
    from sklearn.model_selection import StratifiedGroupKFold
except ModuleNotFoundError:
    StratifiedGroupKFold = None

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
    write_split_ready_features: bool = True
    primary_test_fraction: float = 0.2
    split_random_state: int = 0
    split_stratify_col: str = "transition_label"
    max_missing_fraction: Optional[float] = 0.8
    max_mode_fraction: Optional[float] = None
    min_numeric_variance: Optional[float] = None


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


def _metadata_columns(subject_id_col: str) -> set[str]:
    return {
        subject_id_col,
        "pair_id",
        "group",
        "transition",
        "transition_label",
        "matched_cohort",
        "analysis_set",
        "evaluation_eligible",
        "abs_age_gap",
        "split",
        "split_group_source",
    }


def _resolve_split_stratify_column(
    matched_subjects_df: pd.DataFrame,
    cohort_df: pd.DataFrame,
    config: FeatureExportConfig,
) -> tuple[pd.DataFrame, str]:
    split_df = matched_subjects_df.copy()
    stratify_col = config.split_stratify_col

    if stratify_col not in split_df.columns:
        if stratify_col in cohort_df.columns:
            split_df = split_df.merge(
                cohort_df[[config.subject_id_col, stratify_col]],
                on=config.subject_id_col,
                how="left",
            )
        elif "transition" in split_df.columns:
            stratify_col = "transition"
        else:
            raise ValueError(
                "No split stratification label is available. "
                f"Missing '{config.split_stratify_col}' and fallback 'transition'."
            )

    if split_df[stratify_col].isna().any():
        if stratify_col != "transition" and "transition" in split_df.columns:
            split_df[stratify_col] = split_df[stratify_col].fillna(split_df["transition"])
        if split_df[stratify_col].isna().any():
            raise ValueError(
                f"Split stratification column '{stratify_col}' has missing values."
            )

    return split_df, stratify_col


def _build_split_subjects_df(
    matched_subjects_df: pd.DataFrame,
    cohort_df: pd.DataFrame,
    config: FeatureExportConfig,
) -> pd.DataFrame:
    split_df, stratify_col = _resolve_split_stratify_column(
        matched_subjects_df, cohort_df, config
    )
    primary_df = split_df.loc[split_df["analysis_set"] == "primary"].copy()
    augmentation_df = split_df.loc[split_df["analysis_set"] == "augmentation"].copy()

    if primary_df["group"].nunique() < 2:
        raise ValueError(
            "At least two primary matched groups are required to build a train/test split."
        )

    n_primary_groups = primary_df["group"].nunique()
    target_n_splits = int(round(1.0 / config.primary_test_fraction))
    label_counts = primary_df[stratify_col].value_counts(dropna=False)
    min_label_count = int(label_counts.min())

    if StratifiedGroupKFold is not None and min_label_count >= 2:
        n_splits = min(max(target_n_splits, 2), n_primary_groups, min_label_count)
        splitter = StratifiedGroupKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=config.split_random_state,
        )
        split_iter = splitter.split(
            primary_df,
            y=primary_df[stratify_col],
            groups=primary_df["group"],
        )
        train_index, test_index = next(split_iter)
        train_groups = set(primary_df.iloc[train_index]["group"].unique().tolist())
        test_groups = set(primary_df.iloc[test_index]["group"].unique().tolist())
    else:
        train_groups, test_groups = _greedy_stratified_group_holdout(
            primary_df, stratify_col, config
        )

    primary_df["split"] = np.where(
        primary_df["group"].isin(test_groups), "test", "train"
    )
    primary_df["split_group_source"] = "primary"

    augmentation_df["split"] = "train"
    augmentation_df["split_group_source"] = "augmentation"

    split_subjects_df = pd.concat([primary_df, augmentation_df], ignore_index=True)
    if not train_groups:
        raise ValueError("Primary split produced no training groups.")
    if not test_groups:
        raise ValueError("Primary split produced no test groups.")
    return split_subjects_df


def _greedy_stratified_group_holdout(
    primary_df: pd.DataFrame,
    stratify_col: str,
    config: FeatureExportConfig,
) -> tuple[set[int], set[int]]:
    group_label_counts = (
        primary_df.groupby(["group", stratify_col], dropna=False)
        .size()
        .unstack(fill_value=0)
        .sort_index()
    )
    if len(group_label_counts) < 2:
        raise ValueError(
            "At least two primary matched groups are required to build a train/test split."
        )

    total_groups = len(group_label_counts)
    target_test_groups = int(round(total_groups * config.primary_test_fraction))
    target_test_groups = min(max(target_test_groups, 1), total_groups - 1)

    target_test_counts = (
        group_label_counts.sum(axis=0).to_numpy(dtype=float)
        * (target_test_groups / total_groups)
    )
    current_test_counts = np.zeros(len(group_label_counts.columns), dtype=float)

    rng = np.random.default_rng(config.split_random_state)
    group_order = list(group_label_counts.index)
    rng.shuffle(group_order)
    group_order.sort(
        key=lambda group: (
            -float(group_label_counts.loc[group].max()),
            -float(group_label_counts.loc[group].sum()),
        )
    )

    test_groups: list[int] = []
    for i, group in enumerate(group_order):
        groups_left = total_groups - i
        slots_left = target_test_groups - len(test_groups)
        if slots_left <= 0:
            break
        if slots_left >= groups_left:
            test_groups.extend(group_order[i:])
            break

        candidate_counts = group_label_counts.loc[group].to_numpy(dtype=float)
        keep_loss = np.square(current_test_counts - target_test_counts).sum()
        take_loss = np.square(
            current_test_counts + candidate_counts - target_test_counts
        ).sum()
        if take_loss <= keep_loss:
            test_groups.append(group)
            current_test_counts = current_test_counts + candidate_counts

    if len(test_groups) < target_test_groups:
        remaining_groups = [
            group for group in group_order if group not in set(test_groups)
        ]
        remaining_groups.sort(
            key=lambda group: float(
                np.square(
                    current_test_counts
                    + group_label_counts.loc[group].to_numpy(dtype=float)
                    - target_test_counts
                ).sum()
            )
        )
        needed = target_test_groups - len(test_groups)
        test_groups.extend(remaining_groups[:needed])

    test_group_set = set(test_groups)
    train_group_set = set(group_label_counts.index) - test_group_set
    return train_group_set, test_group_set


def _build_cohort_count_audits(
    cohort_df: pd.DataFrame,
    split_subjects_df: pd.DataFrame,
    config: FeatureExportConfig,
) -> dict[str, pd.DataFrame]:
    cohort_counts_df = (
        cohort_df.groupby(["cohort", "eligible_for_matching"], dropna=False)
        .size()
        .rename("n_subjects")
        .reset_index()
    )

    matched_counts_df = (
        split_subjects_df.groupby(
            ["analysis_set", "split", "matched_cohort", "transition"], dropna=False
        )
        .agg(
            n_subjects=(config.subject_id_col, "nunique"),
            n_groups=("group", "nunique"),
        )
        .reset_index()
    )

    split_summary_df = pd.DataFrame(
        [
            {
                "analysis_set": analysis_set,
                "split": split,
                "n_subjects": group[config.subject_id_col].nunique(),
                "n_groups": group["group"].nunique(),
                "n_transition": group.loc[
                    group["transition"] == 1, config.subject_id_col
                ].nunique(),
                "n_controls": group.loc[group["transition"] == 0, config.subject_id_col].nunique(),
            }
            for (analysis_set, split), group in split_subjects_df.groupby(
                ["analysis_set", "split"], dropna=False
            )
        ]
    )

    return {
        "cohort_counts_df": cohort_counts_df,
        "matched_counts_df": matched_counts_df,
        "split_summary_df": split_summary_df,
    }


def _build_column_audit(
    feature_df: pd.DataFrame,
    split_subjects_df: pd.DataFrame,
    config: FeatureExportConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metadata_cols = _metadata_columns(config.subject_id_col)
    feature_only_df = feature_df.drop(
        columns=[
            col
            for col in feature_df.columns
            if col != config.subject_id_col and col in metadata_cols
        ],
        errors="ignore",
    )
    merged_df = feature_only_df.merge(
        split_subjects_df,
        on=config.subject_id_col,
        how="inner",
    )

    train_df = merged_df.loc[merged_df["split"] == "train"].copy()
    test_df = merged_df.loc[merged_df["split"] == "test"].copy()

    candidate_feature_cols = [c for c in merged_df.columns if c not in metadata_cols]

    audit_rows = []
    kept_feature_cols = []
    for col in candidate_feature_cols:
        series = train_df[col]
        observed = int(series.notna().sum())
        total = int(len(series))
        missing = total - observed
        missing_fraction = float(missing / total) if total else np.nan
        nonmissing = series.dropna()
        unique_non_missing = int(nonmissing.nunique(dropna=True))

        mode_value = pd.NA
        mode_count = 0
        mode_fraction = np.nan
        if observed:
            value_counts = nonmissing.value_counts(dropna=False)
            mode_value = value_counts.index[0]
            mode_count = int(value_counts.iloc[0])
            mode_fraction = float(mode_count / observed)

        numeric_series = pd.to_numeric(series, errors="coerce")
        numeric_observed = int(numeric_series.notna().sum())
        variance = (
            float(numeric_series.var(ddof=0))
            if numeric_observed > 1
            else np.nan
        )

        drop_reason = ""
        if observed == 0:
            drop_reason = "all_missing_in_train"
        elif (
            config.max_missing_fraction is not None
            and missing_fraction > config.max_missing_fraction
        ):
            drop_reason = "high_missingness"
        elif unique_non_missing <= 1:
            drop_reason = "constant_in_train"
        elif (
            config.max_mode_fraction is not None
            and pd.notna(mode_fraction)
            and mode_fraction >= config.max_mode_fraction
        ):
            drop_reason = "mode_dominance"
        elif (
            config.min_numeric_variance is not None
            and pd.notna(variance)
            and variance <= config.min_numeric_variance
        ):
            drop_reason = "low_numeric_variance"

        keep = drop_reason == ""
        if keep:
            kept_feature_cols.append(col)

        audit_rows.append(
            {
                "column": col,
                "train_non_missing_n": observed,
                "train_missing_n": missing,
                "train_missing_fraction": missing_fraction,
                "train_unique_non_missing_n": unique_non_missing,
                "train_mode_value": mode_value,
                "train_mode_count": mode_count,
                "train_mode_fraction": mode_fraction,
                "train_numeric_variance": variance,
                "keep_for_modeling": int(keep),
                "drop_reason": drop_reason or pd.NA,
            }
        )

    column_audit_df = pd.DataFrame(audit_rows).sort_values(
        ["keep_for_modeling", "train_missing_fraction", "column"],
        ascending=[True, False, True],
        kind="stable",
    )

    metadata_cols_present = [config.subject_id_col] + sorted(
        col
        for col in metadata_cols - {config.subject_id_col}
        if col in merged_df.columns
    )
    model_cols = metadata_cols_present.copy()
    model_cols.extend([c for c in kept_feature_cols if c not in model_cols])
    model_ready_df = merged_df[model_cols].copy()
    train_ready_df = train_df[model_cols].copy()
    test_ready_df = test_df[model_cols].copy()

    carryout_summary_df = pd.DataFrame(
        [
            {
                "n_total_rows": len(merged_df),
                "n_train_rows": len(train_ready_df),
                "n_test_rows": len(test_ready_df),
                "n_train_primary_rows": int(
                    train_ready_df["analysis_set"].eq("primary").sum()
                ),
                "n_train_augmentation_rows": int(
                    train_ready_df["analysis_set"].eq("augmentation").sum()
                ),
                "n_test_primary_rows": int(
                    test_ready_df["analysis_set"].eq("primary").sum()
                ),
                "n_candidate_feature_columns": len(candidate_feature_cols),
                "n_retained_feature_columns": len(kept_feature_cols),
                "n_dropped_feature_columns": len(candidate_feature_cols) - len(kept_feature_cols),
                "n_dropped_all_missing": int(
                    column_audit_df["drop_reason"].eq("all_missing_in_train").sum()
                ),
                "n_dropped_high_missingness": int(
                    column_audit_df["drop_reason"].eq("high_missingness").sum()
                ),
                "n_dropped_constant": int(
                    column_audit_df["drop_reason"].eq("constant_in_train").sum()
                ),
                "n_dropped_mode_dominance": int(
                    column_audit_df["drop_reason"].eq("mode_dominance").sum()
                ),
                "n_dropped_low_numeric_variance": int(
                    column_audit_df["drop_reason"].eq("low_numeric_variance").sum()
                ),
            }
        ]
    )

    return model_ready_df, train_ready_df, test_ready_df, column_audit_df, carryout_summary_df


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


def _write_split_ready_outputs(
    results: dict[str, pd.DataFrame],
    out_dir: Path,
    config: FeatureExportConfig,
) -> None:
    if "split_subjects_df" in results:
        results["split_subjects_df"].to_csv(
            out_dir / f"{config.output_prefix}_split_subjects.csv", index=False
        )
    if "cohort_counts_df" in results:
        results["cohort_counts_df"].to_csv(
            out_dir / f"{config.output_prefix}_cohort_counts.csv", index=False
        )
    if "matched_counts_df" in results:
        results["matched_counts_df"].to_csv(
            out_dir / f"{config.output_prefix}_matched_counts.csv", index=False
        )
    if "split_summary_df" in results:
        results["split_summary_df"].to_csv(
            out_dir / f"{config.output_prefix}_split_summary.csv", index=False
        )

    feature_output_names = {
        "libra": config.libra_output_name,
        "mrf": config.mrf_output_name,
        "bmca": config.bmca_output_name,
    }
    for prefix, output_name in feature_output_names.items():
        model_ready_key = f"{prefix}_model_ready_df"
        train_key = f"{prefix}_train_df"
        test_key = f"{prefix}_test_df"
        column_audit_key = f"{prefix}_column_audit_df"
        carryout_key = f"{prefix}_carryout_summary_df"

        if model_ready_key in results:
            results[model_ready_key].to_csv(
                out_dir / _append_suffix(output_name, "_split_ready"),
                index=False,
            )
        if train_key in results:
            results[train_key].to_csv(
                out_dir / _append_suffix(output_name, "_train"),
                index=False,
            )
        if test_key in results:
            results[test_key].to_csv(
                out_dir / _append_suffix(output_name, "_test"),
                index=False,
            )
        if column_audit_key in results:
            results[column_audit_key].to_csv(
                out_dir / _append_suffix(output_name, "_column_audit"),
                index=False,
            )
        if carryout_key in results:
            results[carryout_key].to_csv(
                out_dir / _append_suffix(output_name, "_carryout_summary"),
                index=False,
            )


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

    if config.write_split_ready_features:
        split_subjects_df = _build_split_subjects_df(
            match_results["combined_matched_subjects_df"],
            cohort_df,
            config,
        )
        results["split_subjects_df"] = split_subjects_df
        results.update(_build_cohort_count_audits(cohort_df, split_subjects_df, config))

        for prefix in ["libra", "mrf", "bmca"]:
            model_ready_df, train_df, test_df, column_audit_df, carryout_summary_df = (
                _build_column_audit(feature_results[f"{prefix}_df"], split_subjects_df, config)
            )
            results[f"{prefix}_model_ready_df"] = model_ready_df
            results[f"{prefix}_train_df"] = train_df
            results[f"{prefix}_test_df"] = test_df
            results[f"{prefix}_column_audit_df"] = column_audit_df
            results[f"{prefix}_carryout_summary_df"] = carryout_summary_df

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

    if config.write_split_ready_features:
        _write_split_ready_outputs(results, out_dir, config)

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
