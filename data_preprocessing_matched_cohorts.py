"""
ADNI cohort construction and exact matching for the CN -> MCI/dementia task.

This module builds subject-level cohorts from the raw ADNI wide table using the
same baseline-row and transition-label logic as `data_preprocessing_libra.py`,
then solves an age-minimizing exact match problem under:

- exact sex match
- exact genotype match
- matching without replacement

The optimization is formulated as a binary linear program.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Optional

import pandas as pd
import pulp

try:
    from data_preprocessing_libra import (
        LibraConfig,
        _to_numeric,
        _visit_to_months,
        build_baseline_with_screening_fallback,
        build_transition_labels,
    )
except ModuleNotFoundError:
    from Code.data_preprocessing_libra import (
        LibraConfig,
        _to_numeric,
        _visit_to_months,
        build_baseline_with_screening_fallback,
        build_transition_labels,
    )


log = logging.getLogger(__name__)


@dataclass
class CohortMatchConfig(LibraConfig):
    age_col: str = "entry_age"
    sex_col: str = "PTGENDER"
    genotype_col: str = "GENOTYPE"
    entry_group_col: str = "entry_research_group"
    transition_baseline_diag: int = 1
    dementia_diag_code: int = 3
    impairment_entry_groups: tuple[str, ...] = ("MCI", "EMCI", "SMC", "LMCI")
    exclude_baseline_dementia_from_impairment: bool = True
    max_age_gap_years: Optional[float] = None
    require_full_matching: bool = True
    require_full_augmentation_matching: bool = False
    solver_msg: bool = False
    output_prefix: str = "cn_progression"


def build_impairment_to_dementia_labels(
    df: pd.DataFrame, config: Optional[CohortMatchConfig] = None
) -> pd.DataFrame:
    """
    Subject-level label for participants who enter the study in an impairment
    research group and later show dementia diagnosis.
    """
    if config is None:
        config = CohortMatchConfig()

    required = {config.subject_id_col, config.visit_col, config.diagnosis_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns for impairment labeling: {sorted(missing)}")

    work = df[[config.subject_id_col, config.visit_col, config.diagnosis_col]].copy()
    work[config.visit_col] = work[config.visit_col].astype(str).str.lower().str.strip()
    work["_visit_months"] = _visit_to_months(work[config.visit_col])
    work[config.diagnosis_col] = _to_numeric(work[config.diagnosis_col])

    work["_diag_notna"] = work[config.diagnosis_col].notna().astype(int)
    work = work.sort_values(
        [config.subject_id_col, "_visit_months", "_diag_notna"],
        ascending=[True, True, False],
        kind="stable",
    )
    work = work.drop_duplicates([config.subject_id_col, config.visit_col], keep="first")

    rows = []
    for sid, g in work.groupby(config.subject_id_col, sort=False):
        g = g.sort_values("_visit_months", kind="stable")
        bl = g.loc[g[config.visit_col] == config.baseline_visit, config.diagnosis_col]
        baseline_diag = bl.iloc[0] if len(bl) else pd.NA

        follow = g.loc[g["_visit_months"] >= 12, [config.diagnosis_col, "_visit_months"]].copy()
        follow_nonmissing = follow.loc[follow[config.diagnosis_col].notna()].copy()

        ci_to_dementia_label = 0.0
        if len(follow_nonmissing):
            if follow_nonmissing[config.diagnosis_col].eq(config.dementia_diag_code).any():
                ci_to_dementia_label = 1.0

        first_dementia_month = pd.NA
        if len(follow_nonmissing):
            dementia_months = follow_nonmissing.loc[
                follow_nonmissing[config.diagnosis_col].eq(config.dementia_diag_code),
                "_visit_months",
            ]
            if len(dementia_months):
                first_dementia_month = float(dementia_months.min())

        rows.append(
            {
                config.subject_id_col: sid,
                "ci_to_dementia_label": ci_to_dementia_label,
                "first_dementia_month": first_dementia_month,
                "baseline_diagnosis_lp": baseline_diag,
            }
        )

    return pd.DataFrame(rows)


def build_subject_level_cohorts(
    df: pd.DataFrame, config: Optional[CohortMatchConfig] = None
) -> pd.DataFrame:
    """
    Build one baseline row per subject with cohort labels for matching.

    Cohort definitions:
    - transition: baseline diagnosis is CN and later transitions to MCI/dementia
    - stable_cn: baseline diagnosis is CN and remains CN through the last
      observed follow-up diagnosis
    """
    if config is None:
        config = CohortMatchConfig()

    baseline_df = build_baseline_with_screening_fallback(df, config)
    labels_df = build_transition_labels(df, config)
    ci_labels_df = build_impairment_to_dementia_labels(df, config)
    cohort_df = baseline_df.merge(labels_df, on=config.subject_id_col, how="left")
    cohort_df = cohort_df.merge(ci_labels_df, on=config.subject_id_col, how="left")

    cohort_df[config.age_col] = _to_numeric(cohort_df[config.age_col])
    cohort_df["baseline_diagnosis_resolved"] = _to_numeric(cohort_df[config.diagnosis_col])
    if config.sex_col in cohort_df.columns:
        cohort_df[config.sex_col] = pd.to_numeric(
            cohort_df[config.sex_col], errors="coerce"
        ).astype("Int64")
    if config.genotype_col in cohort_df.columns:
        cohort_df[config.genotype_col] = (
            cohort_df[config.genotype_col].astype(str).str.strip().replace({"": pd.NA, "nan": pd.NA})
        )
    if config.entry_group_col in cohort_df.columns:
        cohort_df[config.entry_group_col] = (
            cohort_df[config.entry_group_col].astype(str).str.strip().replace({"": pd.NA, "nan": pd.NA})
        )

    cohort_df["cohort"] = pd.NA
    transition_mask = (
        cohort_df["baseline_diagnosis_resolved"].eq(config.transition_baseline_diag)
        & cohort_df["transition_label"].eq(1)
    )
    stable_mask = (
        cohort_df["baseline_diagnosis_resolved"].eq(config.transition_baseline_diag)
        & cohort_df["transition_label"].eq(0)
    )
    cohort_df.loc[transition_mask, "cohort"] = "transition"
    cohort_df.loc[stable_mask, "cohort"] = "stable_cn"

    eligible_impairment_groups = cohort_df[config.entry_group_col].isin(
        list(config.impairment_entry_groups)
    )
    baseline_not_dementia = ~cohort_df["baseline_diagnosis_resolved"].eq(config.dementia_diag_code)
    if not config.exclude_baseline_dementia_from_impairment:
        baseline_not_dementia = pd.Series(True, index=cohort_df.index)
    ci_mask = eligible_impairment_groups & baseline_not_dementia & cohort_df["ci_to_dementia_label"].eq(1)
    cohort_df.loc[ci_mask, "cohort"] = "ci_to_dementia"

    eligible_mask = (
        cohort_df["cohort"].isin(["transition", "stable_cn", "ci_to_dementia"])
        & cohort_df[config.age_col].notna()
        & cohort_df[config.sex_col].notna()
        & cohort_df[config.genotype_col].notna()
    )
    cohort_df["eligible_for_matching"] = eligible_mask.astype(int)

    return cohort_df


def _solve_stratum_lp(
    transition_df: pd.DataFrame,
    control_df: pd.DataFrame,
    config: CohortMatchConfig,
) -> pd.DataFrame:
    """
    Solve a minimum-cost exact matching problem inside one (genotype, sex) stratum.
    """
    if len(transition_df) == 0:
        return pd.DataFrame()
    if len(control_df) == 0:
        raise ValueError("No compatible controls available in a required matching stratum.")
    if len(transition_df) > len(control_df) and config.require_full_matching:
        raise ValueError(
            "Infeasible exact matching: fewer compatible controls than transition subjects "
            f"for genotype={transition_df[config.genotype_col].iloc[0]!r}, "
            f"sex={transition_df[config.sex_col].iloc[0]!r}."
        )

    t = transition_df.reset_index(drop=True).copy()
    c = control_df.reset_index(drop=True).copy()

    feasible_pairs: list[tuple[int, int, float]] = []
    for i in range(len(t)):
        for j in range(len(c)):
            age_gap = abs(float(t.at[i, config.age_col]) - float(c.at[j, config.age_col]))
            if config.max_age_gap_years is not None and age_gap > config.max_age_gap_years:
                continue
            feasible_pairs.append((i, j, age_gap))

    feasible_by_transition = {i: 0 for i in range(len(t))}
    for i, _, _ in feasible_pairs:
        feasible_by_transition[i] += 1
    no_match = [i for i, n in feasible_by_transition.items() if n == 0]
    if no_match:
        raise ValueError(
            "Infeasible exact matching: at least one transition subject has no feasible "
            "control under the current sex/genotype constraints"
            + (
                f" and max_age_gap_years={config.max_age_gap_years}."
                if config.max_age_gap_years is not None
                else "."
            )
        )

    problem = pulp.LpProblem("cn_progression_matching", pulp.LpMinimize)
    x = {
        (i, j): pulp.LpVariable(f"x_{i}_{j}", cat="Binary")
        for i, j, _ in feasible_pairs
    }

    problem += pulp.lpSum(cost * x[(i, j)] for i, j, cost in feasible_pairs)

    for i in range(len(t)):
        problem += pulp.lpSum(x[(ii, jj)] for (ii, jj) in x if ii == i) == 1

    for j in range(len(c)):
        problem += pulp.lpSum(x[(ii, jj)] for (ii, jj) in x if jj == j) <= 1

    status = problem.solve(pulp.PULP_CBC_CMD(msg=config.solver_msg))
    if pulp.LpStatus[status] != "Optimal":
        raise RuntimeError(f"Matching LP failed with status {pulp.LpStatus[status]!r}.")

    matched_rows = []
    for (i, j), var in x.items():
        if pulp.value(var) and pulp.value(var) > 0.5:
            matched_rows.append(
                {
                    "transition_subject_id": t.at[i, config.subject_id_col],
                    "control_subject_id": c.at[j, config.subject_id_col],
                    config.genotype_col: t.at[i, config.genotype_col],
                    config.sex_col: t.at[i, config.sex_col],
                    f"transition_{config.age_col}": t.at[i, config.age_col],
                    f"control_{config.age_col}": c.at[j, config.age_col],
                    "abs_age_gap": abs(
                        float(t.at[i, config.age_col]) - float(c.at[j, config.age_col])
                    ),
                    "first_conversion_month": t.at[i, "first_conversion_month"],
                }
            )

    return pd.DataFrame(matched_rows)


def match_transition_to_stable_cn(
    cohort_df: pd.DataFrame, config: Optional[CohortMatchConfig] = None
) -> dict[str, pd.DataFrame]:
    """
    Match each transition subject to one stable-CN control using exact sex and
    genotype matching with age-gap minimization.
    """
    if config is None:
        config = CohortMatchConfig()

    eligible_df = cohort_df.loc[cohort_df["eligible_for_matching"] == 1].copy()
    transition_df = eligible_df.loc[eligible_df["cohort"] == "transition"].copy()
    stable_df = eligible_df.loc[eligible_df["cohort"] == "stable_cn"].copy()

    stratum_cols = [config.genotype_col, config.sex_col]

    stratum_counts = (
        transition_df.groupby(stratum_cols, dropna=False)
        .size()
        .rename("n_transition")
        .to_frame()
        .join(
            stable_df.groupby(stratum_cols, dropna=False)
            .size()
            .rename("n_stable"),
            how="outer",
        )
        .fillna(0)
        .reset_index()
    )
    stratum_counts["n_transition"] = stratum_counts["n_transition"].astype(int)
    stratum_counts["n_stable"] = stratum_counts["n_stable"].astype(int)

    infeasible = stratum_counts.loc[
        stratum_counts["n_transition"] > stratum_counts["n_stable"]
    ]
    if len(infeasible) and config.require_full_matching:
        raise ValueError(
            "Exact matching is infeasible in at least one (GENOTYPE, PTGENDER) stratum:\n"
            + infeasible.to_string(index=False)
        )

    match_frames = []
    for (genotype, sex), trans_stratum in transition_df.groupby(stratum_cols, dropna=False):
        ctrl_stratum = stable_df.loc[
            (stable_df[config.genotype_col] == genotype)
            & (stable_df[config.sex_col] == sex)
        ].copy()
        match_frames.append(_solve_stratum_lp(trans_stratum, ctrl_stratum, config))

    matched_pairs_df = pd.concat(match_frames, ignore_index=True)
    matched_pairs_df["pair_id"] = range(1, len(matched_pairs_df) + 1)
    matched_pairs_df["group"] = matched_pairs_df["pair_id"].astype(int)
    matched_pairs_df["analysis_set"] = "primary"
    matched_pairs_df["evaluation_eligible"] = 1

    pair_map_transition = matched_pairs_df[
        ["pair_id", "group", "analysis_set", "evaluation_eligible", "transition_subject_id", "abs_age_gap"]
    ].rename(columns={"transition_subject_id": config.subject_id_col})
    pair_map_transition["transition"] = 1
    pair_map_transition["matched_cohort"] = "transition"

    pair_map_control = matched_pairs_df[
        ["pair_id", "group", "analysis_set", "evaluation_eligible", "control_subject_id", "abs_age_gap"]
    ].rename(columns={"control_subject_id": config.subject_id_col})
    pair_map_control["transition"] = 0
    pair_map_control["matched_cohort"] = "stable_cn_matched"

    matched_subjects_df = pd.concat(
        [pair_map_transition, pair_map_control], ignore_index=True
    )

    matched_dataset_df = cohort_df.merge(
        matched_subjects_df, on=config.subject_id_col, how="inner"
    )

    unmatched_stable_df = stable_df.loc[
        ~stable_df[config.subject_id_col].isin(
            matched_pairs_df["control_subject_id"]
        )
    ].copy()

    return {
        "cohort_df": cohort_df,
        "transition_cohort_df": transition_df,
        "stable_cn_cohort_df": stable_df,
        "matched_pairs_df": matched_pairs_df,
        "matched_subjects_df": matched_subjects_df,
        "matched_dataset_df": matched_dataset_df,
        "unmatched_stable_cn_df": unmatched_stable_df,
        "stratum_counts_df": stratum_counts,
    }


def match_remaining_stable_to_ci_dementia(
    cohort_df: pd.DataFrame,
    primary_match_results: dict[str, pd.DataFrame],
    config: Optional[CohortMatchConfig] = None,
) -> dict[str, pd.DataFrame]:
    """
    Match unmatched stable-CN controls to the impairment-entry cohort that later
    reaches dementia, using the same exact sex/genotype constraints and minimum
    age-gap objective.
    """
    if config is None:
        config = CohortMatchConfig()

    remaining_stable_df = primary_match_results["unmatched_stable_cn_df"].copy()
    ci_df = cohort_df.loc[
        (cohort_df["eligible_for_matching"] == 1)
        & (cohort_df["cohort"] == "ci_to_dementia")
    ].copy()

    stratum_cols = [config.genotype_col, config.sex_col]
    stratum_counts = (
        remaining_stable_df.groupby(stratum_cols, dropna=False)
        .size()
        .rename("n_remaining_stable")
        .to_frame()
        .join(
            ci_df.groupby(stratum_cols, dropna=False)
            .size()
            .rename("n_ci_dementia"),
            how="outer",
        )
        .fillna(0)
        .reset_index()
    )
    stratum_counts["n_remaining_stable"] = stratum_counts["n_remaining_stable"].astype(int)
    stratum_counts["n_ci_dementia"] = stratum_counts["n_ci_dementia"].astype(int)

    infeasible = stratum_counts.loc[
        stratum_counts["n_remaining_stable"] > stratum_counts["n_ci_dementia"]
    ]
    if len(infeasible) and config.require_full_augmentation_matching:
        raise ValueError(
            "Augmentation matching is infeasible in at least one (GENOTYPE, PTGENDER) stratum:\n"
            + infeasible.to_string(index=False)
        )
    if len(infeasible) and not config.require_full_augmentation_matching:
        log.warning(
            "Partial augmentation matching: some (GENOTYPE, PTGENDER) strata have fewer "
            "ci_to_dementia subjects than remaining stable-CN subjects.\n%s",
            infeasible.to_string(index=False),
        )

    match_frames = []
    for (genotype, sex), stable_stratum in remaining_stable_df.groupby(stratum_cols, dropna=False):
        ci_stratum = ci_df.loc[
            (ci_df[config.genotype_col] == genotype)
            & (ci_df[config.sex_col] == sex)
        ].copy()
        if len(stable_stratum) == 0:
            continue
        if len(ci_stratum) == 0:
            if config.require_full_augmentation_matching:
                raise ValueError(
                    "No compatible cognitive-impairment subjects available in augmentation stratum "
                    f"GENOTYPE={genotype!r}, PTGENDER={sex!r}."
                )
            log.warning(
                "Skipping augmentation stratum with no compatible ci_to_dementia subjects: "
                "GENOTYPE=%r, PTGENDER=%r.",
                genotype,
                sex,
            )
            continue

        stable_stratum = stable_stratum.reset_index(drop=True)
        ci_stratum = ci_stratum.reset_index(drop=True)
        feasible_pairs: list[tuple[int, int, float]] = []
        for i in range(len(stable_stratum)):
            for j in range(len(ci_stratum)):
                age_gap = abs(
                    float(stable_stratum.at[i, config.age_col])
                    - float(ci_stratum.at[j, config.age_col])
                )
                if config.max_age_gap_years is not None and age_gap > config.max_age_gap_years:
                    continue
                feasible_pairs.append((i, j, age_gap))

        feasible_by_stable = {i: 0 for i in range(len(stable_stratum))}
        for i, _, _ in feasible_pairs:
            feasible_by_stable[i] += 1
        no_match = [i for i, n in feasible_by_stable.items() if n == 0]
        if no_match and config.require_full_augmentation_matching:
            raise ValueError(
                "Augmentation matching is infeasible: at least one remaining stable-CN subject "
                "has no feasible cognitive-impairment match under the current constraints."
            )
        if not feasible_pairs:
            if config.require_full_augmentation_matching:
                raise ValueError(
                    "Augmentation matching is infeasible: no feasible pairs remain under the "
                    "current constraints."
                )
            log.warning(
                "Skipping augmentation stratum with no feasible exact matches after constraints: "
                "GENOTYPE=%r, PTGENDER=%r.",
                genotype,
                sex,
            )
            continue

        problem = pulp.LpProblem("augmentation_matching", pulp.LpMinimize)
        x = {
            (i, j): pulp.LpVariable(f"aug_x_{i}_{j}", cat="Binary")
            for i, j, _ in feasible_pairs
        }
        if config.require_full_augmentation_matching:
            problem += pulp.lpSum(cost * x[(i, j)] for i, j, cost in feasible_pairs)
        else:
            reward = sum(cost for _, _, cost in feasible_pairs) + 1.0
            problem.sense = pulp.LpMaximize
            problem += (
                reward * pulp.lpSum(x.values())
                - pulp.lpSum(cost * x[(i, j)] for i, j, cost in feasible_pairs)
            )

        for i in range(len(stable_stratum)):
            if config.require_full_augmentation_matching:
                problem += pulp.lpSum(x[(ii, jj)] for (ii, jj) in x if ii == i) == 1
            else:
                problem += pulp.lpSum(x[(ii, jj)] for (ii, jj) in x if ii == i) <= 1
        for j in range(len(ci_stratum)):
            problem += pulp.lpSum(x[(ii, jj)] for (ii, jj) in x if jj == j) <= 1

        status = problem.solve(pulp.PULP_CBC_CMD(msg=config.solver_msg))
        if pulp.LpStatus[status] != "Optimal":
            raise RuntimeError(
                f"Augmentation matching LP failed with status {pulp.LpStatus[status]!r}."
            )

        rows = []
        for (i, j), var in x.items():
            if pulp.value(var) and pulp.value(var) > 0.5:
                rows.append(
                    {
                        "stable_subject_id": stable_stratum.at[i, config.subject_id_col],
                        "ci_subject_id": ci_stratum.at[j, config.subject_id_col],
                        config.genotype_col: stable_stratum.at[i, config.genotype_col],
                        config.sex_col: stable_stratum.at[i, config.sex_col],
                        f"stable_{config.age_col}": stable_stratum.at[i, config.age_col],
                        f"ci_{config.age_col}": ci_stratum.at[j, config.age_col],
                        "abs_age_gap": abs(
                            float(stable_stratum.at[i, config.age_col])
                            - float(ci_stratum.at[j, config.age_col])
                        ),
                        "first_dementia_month": ci_stratum.at[j, "first_dementia_month"],
                        config.entry_group_col: ci_stratum.at[j, config.entry_group_col],
                    }
                )
        match_frames.append(pd.DataFrame(rows))

    augmentation_pair_columns = [
        "stable_subject_id",
        "ci_subject_id",
        config.genotype_col,
        config.sex_col,
        f"stable_{config.age_col}",
        f"ci_{config.age_col}",
        "abs_age_gap",
        "first_dementia_month",
        config.entry_group_col,
    ]
    augmentation_pairs_df = (
        pd.concat(match_frames, ignore_index=True)
        if match_frames
        else pd.DataFrame(columns=augmentation_pair_columns)
    )

    matched_counts = (
        augmentation_pairs_df.groupby(stratum_cols, dropna=False)
        .size()
        .rename("n_augmentation_matches")
        .reset_index()
    )
    stratum_counts = stratum_counts.merge(matched_counts, on=stratum_cols, how="left")
    stratum_counts["n_augmentation_matches"] = (
        stratum_counts["n_augmentation_matches"].fillna(0).astype(int)
    )
    stratum_counts["n_remaining_stable_unmatched"] = (
        stratum_counts["n_remaining_stable"] - stratum_counts["n_augmentation_matches"]
    )

    group_start = 0
    if len(primary_match_results["matched_pairs_df"]):
        group_start = int(primary_match_results["matched_pairs_df"]["group"].max())
    augmentation_pairs_df["pair_id"] = range(
        group_start + 1, group_start + 1 + len(augmentation_pairs_df)
    )
    augmentation_pairs_df["group"] = augmentation_pairs_df["pair_id"].astype(int)
    augmentation_pairs_df["analysis_set"] = "augmentation"
    augmentation_pairs_df["evaluation_eligible"] = 0

    augmentation_stable_df = augmentation_pairs_df[
        ["pair_id", "group", "analysis_set", "evaluation_eligible", "stable_subject_id", "abs_age_gap"]
    ].rename(columns={"stable_subject_id": config.subject_id_col})
    augmentation_stable_df["transition"] = 0
    augmentation_stable_df["matched_cohort"] = "stable_cn_augmentation"

    augmentation_case_df = augmentation_pairs_df[
        ["pair_id", "group", "analysis_set", "evaluation_eligible", "ci_subject_id", "abs_age_gap"]
    ].rename(columns={"ci_subject_id": config.subject_id_col})
    augmentation_case_df["transition"] = 1
    augmentation_case_df["matched_cohort"] = "ci_to_dementia"

    augmentation_subjects_df = pd.concat(
        [augmentation_stable_df, augmentation_case_df], ignore_index=True
    )
    augmentation_dataset_df = cohort_df.merge(
        augmentation_subjects_df, on=config.subject_id_col, how="inner"
    )

    combined_matched_subjects_df = pd.concat(
        [primary_match_results["matched_subjects_df"], augmentation_subjects_df],
        ignore_index=True,
    )
    combined_matched_dataset_df = cohort_df.merge(
        combined_matched_subjects_df, on=config.subject_id_col, how="inner"
    )

    return {
        "augmentation_pairs_df": augmentation_pairs_df,
        "augmentation_subjects_df": augmentation_subjects_df,
        "augmentation_dataset_df": augmentation_dataset_df,
        "ci_to_dementia_cohort_df": ci_df,
        "augmentation_stratum_counts_df": stratum_counts,
        "combined_matched_subjects_df": combined_matched_subjects_df,
        "combined_matched_dataset_df": combined_matched_dataset_df,
    }


def attach_matches_to_feature_table(
    feature_df: pd.DataFrame,
    matched_subjects_df: pd.DataFrame,
    subject_id_col: str = "subject_id",
) -> pd.DataFrame:
    """
    Restrict any subject-level feature table (LIBRA / MRF / BMCA) to the matched
    cohort and append pair metadata.
    """
    keep_cols = [
        subject_id_col,
        "pair_id",
        "group",
        "transition",
        "matched_cohort",
        "analysis_set",
        "evaluation_eligible",
        "abs_age_gap",
    ]
    existing = [c for c in keep_cols if c in matched_subjects_df.columns]
    return feature_df.merge(matched_subjects_df[existing], on=subject_id_col, how="inner")


def build_and_match_from_csv(
    input_csv: str, output_dir: str, config: Optional[CohortMatchConfig] = None
) -> dict[str, pd.DataFrame]:
    """
    End-to-end CSV wrapper: build cohorts, solve exact matches, and write outputs.
    """
    if config is None:
        config = CohortMatchConfig()

    df = pd.read_csv(input_csv, low_memory=False)
    cohort_df = build_subject_level_cohorts(df, config=config)
    results = match_transition_to_stable_cn(cohort_df, config=config)
    augmentation_results = match_remaining_stable_to_ci_dementia(
        cohort_df, results, config=config
    )
    results.update(augmentation_results)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = config.output_prefix

    results["cohort_df"].to_csv(out_dir / f"{prefix}_subject_level_cohort.csv", index=False)
    results["transition_cohort_df"].to_csv(out_dir / f"{prefix}_transition_cohort.csv", index=False)
    results["stable_cn_cohort_df"].to_csv(out_dir / f"{prefix}_stable_cn_cohort.csv", index=False)
    results["matched_pairs_df"].to_csv(out_dir / f"{prefix}_matched_pairs.csv", index=False)
    results["matched_subjects_df"].to_csv(out_dir / f"{prefix}_matched_subjects.csv", index=False)
    results["matched_dataset_df"].to_csv(out_dir / f"{prefix}_matched_dataset.csv", index=False)
    results["unmatched_stable_cn_df"].to_csv(out_dir / f"{prefix}_unmatched_stable_cn.csv", index=False)
    results["stratum_counts_df"].to_csv(out_dir / f"{prefix}_stratum_counts.csv", index=False)
    results["ci_to_dementia_cohort_df"].to_csv(out_dir / f"{prefix}_ci_to_dementia_cohort.csv", index=False)
    results["augmentation_pairs_df"].to_csv(out_dir / f"{prefix}_augmentation_pairs.csv", index=False)
    results["augmentation_subjects_df"].to_csv(out_dir / f"{prefix}_augmentation_subjects.csv", index=False)
    results["augmentation_dataset_df"].to_csv(out_dir / f"{prefix}_augmentation_dataset.csv", index=False)
    results["augmentation_stratum_counts_df"].to_csv(out_dir / f"{prefix}_augmentation_stratum_counts.csv", index=False)
    results["combined_matched_subjects_df"].to_csv(out_dir / f"{prefix}_combined_matched_subjects.csv", index=False)
    results["combined_matched_dataset_df"].to_csv(out_dir / f"{prefix}_combined_matched_dataset.csv", index=False)

    return results


EXAMPLE = r"""
import pandas as pd
from data_preprocessing_matched_cohorts import (
    CohortMatchConfig,
    build_and_match_from_csv,
    attach_matches_to_feature_table,
)

cfg = CohortMatchConfig(
    subject_id_col="subject_id",
    visit_col="visit",
    diagnosis_col="DIAGNOSIS",
    age_col="entry_age",
    sex_col="PTGENDER",
    genotype_col="GENOTYPE",
    max_age_gap_years=None,
)

results = build_and_match_from_csv(
    input_csv="data/All_Subjects_My_Table_11Mar2026.csv",
    output_dir="data/",
    config=cfg,
)

mrf_df = pd.read_csv("data/adni_mrf_features.csv")
mrf_matched = attach_matches_to_feature_table(
    mrf_df,
    results["matched_subjects_df"],
    subject_id_col="subject_id",
)
mrf_matched.to_csv("data/adni_mrf_features_matched.csv", index=False)
"""
