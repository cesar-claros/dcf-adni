"""
ADNI modifiable risk factor (MRF) feature construction from a single wide
patient table, plus diagnosis-transition labeling.

This module intentionally reuses the same:
- baseline visit construction,
- screening fallback rule, and
- subject-level transition labels

as `data_preprocessing_libra.py`, so that LIBRA-like scores and the broader
MRF feature set are directly comparable in downstream modeling.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

try:
    from data_preprocessing_libra import (
        ANTIHYPERTENSIVES,
        ANTIDEPRESSANTS,
        ANTIDIABETICS,
        LIBRA_WEIGHTS_2024,
        LATE_LIFE_SUPPORTED_ADNI,
        STATINS_AND_LIPID_LOWERING,
        SUPPORTED_ADNI_CANONICAL,
        LibraConfig,
        _ckd_epi_2021_egfr,
        _coerce_datetime,
        _contains_delimited_code,
        _contains_any,
        _convert_height_to_m,
        _convert_weight_to_kg,
        _ensure_binary,
        _first_existing_column,
        _normalize_text,
        _rescale_observed_weighted_sum,
        _to_numeric,
        build_baseline_with_screening_fallback,
        build_transition_labels,
    )
except ModuleNotFoundError:
    from Code.data_preprocessing_libra import (
        ANTIHYPERTENSIVES,
        ANTIDEPRESSANTS,
        ANTIDIABETICS,
        LIBRA_WEIGHTS_2024,
        LATE_LIFE_SUPPORTED_ADNI,
        STATINS_AND_LIPID_LOWERING,
        SUPPORTED_ADNI_CANONICAL,
        LibraConfig,
        _ckd_epi_2021_egfr,
        _coerce_datetime,
        _contains_delimited_code,
        _contains_any,
        _convert_height_to_m,
        _convert_weight_to_kg,
        _ensure_binary,
        _first_existing_column,
        _normalize_text,
        _rescale_observed_weighted_sum,
        _to_numeric,
        build_baseline_with_screening_fallback,
        build_transition_labels,
    )


@dataclass
class MRFConfig(LibraConfig):
    include_social_context: bool = True
    include_social_interactions: bool = True
    min_features_for_rescale: int = 5


def _zscore(series: pd.Series) -> pd.Series:
    s = _to_numeric(series)
    if s.notna().sum() <= 1:
        return pd.Series(np.nan, index=s.index)
    std = s.std(ddof=0)
    if pd.isna(std) or std == 0:
        return pd.Series(0.0, index=s.index).where(s.notna(), np.nan)
    return (s - s.mean()) / std


def _mean_from_parts(*parts: pd.Series) -> pd.Series:
    stack = np.vstack([_to_numeric(p).values for p in parts]).astype(float)
    valid_counts = np.sum(~np.isnan(stack), axis=0)
    means = np.divide(
        np.nansum(stack, axis=0),
        valid_counts,
        out=np.full(valid_counts.shape, np.nan, dtype=float),
        where=valid_counts > 0,
    )
    return pd.Series(means, index=parts[0].index)


def _sum_with_nan(*parts: pd.Series) -> pd.Series:
    df = pd.concat([_to_numeric(p) for p in parts], axis=1)
    return df.sum(axis=1, min_count=1)


def build_adni_mrf_features_from_wide(
    df: pd.DataFrame, config: Optional[MRFConfig] = None
) -> pd.DataFrame:
    """
    Build one baseline modifiable-risk-factor feature row per subject.
    """
    if config is None:
        config = MRFConfig()

    out = build_baseline_with_screening_fallback(df, config)
    labels = build_transition_labels(df, config)
    out = out.merge(labels, on=config.subject_id_col, how="left")

    gthr = (
        config.resolved_glucose_threshold()
        if config.glucose_threshold is None
        else config.glucose_threshold
    )
    cthr = (
        config.resolved_cholesterol_threshold()
        if config.cholesterol_threshold is None
        else config.cholesterol_threshold
    )

    if "CMMED" in out.columns:
        med_text = _normalize_text(out["CMMED"])
        out["has_antihypertensive_med"] = _contains_any(
            med_text, ANTIHYPERTENSIVES
        ).astype(float)
        out["has_lipid_med"] = _contains_any(
            med_text, STATINS_AND_LIPID_LOWERING
        ).astype(float)
        out["has_diabetes_med"] = _contains_any(med_text, ANTIDIABETICS).astype(float)
        out["has_antidepressant_med"] = _contains_any(
            med_text, ANTIDEPRESSANTS
        ).astype(float)
    else:
        out["has_antihypertensive_med"] = np.nan
        out["has_lipid_med"] = np.nan
        out["has_diabetes_med"] = np.nan
        out["has_antidepressant_med"] = np.nan

    wt_unit = out["VSWTUNIT"] if "VSWTUNIT" in out.columns else pd.Series(np.nan, index=out.index)
    ht_unit = out["VSHTUNIT"] if "VSHTUNIT" in out.columns else pd.Series(np.nan, index=out.index)
    out["weight_kg"] = (
        _convert_weight_to_kg(out["VSWEIGHT"], wt_unit)
        if "VSWEIGHT" in out.columns
        else np.nan
    )
    out["height_m"] = (
        _convert_height_to_m(out["VSHEIGHT"], ht_unit)
        if "VSHEIGHT" in out.columns
        else np.nan
    )
    out["BMI"] = out["weight_kg"] / (out["height_m"] ** 2)

    age_col = _first_existing_column(out, ["entry_age", "AGE", "age", "PTAGE"])
    out["_age_for_egfr"] = _to_numeric(out[age_col]) if age_col else np.nan
    out["serum_creatinine"] = _to_numeric(out["RCT392"]) if "RCT392" in out.columns else np.nan
    if {"serum_creatinine", "_age_for_egfr", "PTGENDER"}.issubset(out.columns):
        out["eGFR"] = _ckd_epi_2021_egfr(
            out["serum_creatinine"],
            out["_age_for_egfr"],
            out["PTGENDER"],
            creatinine_unit=config.creatinine_unit,
        )
    else:
        out["eGFR"] = np.nan

    # Smoking
    # MH16CSMOK = "If no longer smoking, years since quit." Intentionally blank for current
    # smokers; filled only for ex-smokers. MH16CSMOK == 0 does not occur in ADNI.
    #
    # Look-ahead: MH16CSMOK is frequently missing at screening even for confirmed ex-smokers
    # who report a quit year at later follow-up visits. Empirically, 100 of 374 ever-smokers
    # with NaN MH16CSMOK at screening have MH16CSMOK > 0 at some later visit. We recover
    # these by taking the earliest non-NaN value across all visits and back-filling it into
    # the baseline row (updating out["MH16CSMOK"] in place so tobacco_burden also benefits).
    smoke_hist = _ensure_binary(out["MH16SMOK"]) if "MH16SMOK" in out.columns else pd.Series(np.nan, index=out.index)

    if "MH16CSMOK" in df.columns and "MH16CSMOK" in out.columns:
        earliest_csmok = (
            df[df["MH16CSMOK"].notna()]
            .sort_values(config.visit_col)
            .groupby(config.subject_id_col)["MH16CSMOK"]
            .first()
        )
        missing_mask = out["MH16CSMOK"].isna()
        out.loc[missing_mask, "MH16CSMOK"] = (
            out.loc[missing_mask, config.subject_id_col].map(earliest_csmok)
        )

    yrs_quit = _to_numeric(out["MH16CSMOK"]) if "MH16CSMOK" in out.columns else pd.Series(np.nan, index=out.index)

    out["smoking_history"] = smoke_hist
    # current_smoking = 0: never smoked OR confirmed ex-smoker (yrs_quit > 0 after look-ahead)
    # current_smoking = 1: ever-smoked with no quit year found at any visit (assume still smoking)
    # current_smoking = NaN: MH16SMOK not recorded
    out["current_smoking"] = pd.Series(np.where(
        smoke_hist == 0, 0.0,
        np.where(
            (smoke_hist == 1) & (yrs_quit > 0), 0.0,
            np.where(smoke_hist == 1, 1.0, np.nan),
        )
    ), index=out.index)
    out["smoking"] = out["current_smoking"]
    out["smoking_packs_per_day"] = (
        _to_numeric(out["MH16ASMOK"]) if "MH16ASMOK" in out.columns else np.nan
    )
    out["smoking_duration_years"] = (
        _to_numeric(out["MH16BSMOK"]) if "MH16BSMOK" in out.columns else np.nan
    )
    out["smoking_years_since_quit"] = yrs_quit

    tobacco_parts = [
        _zscore(out["smoking_packs_per_day"]),
        _zscore(out["smoking_duration_years"]),
        _zscore(-out["smoking_years_since_quit"]),
    ]
    out["tobacco_burden"] = _mean_from_parts(*tobacco_parts)
    out.loc[out["smoking_history"] == 0, "tobacco_burden"] = 0.0

    # Alcohol abuse
    # MH14CALCH = "If not currently having alcohol abuse, years since end of problem."
    # Unlike MH16CSMOK (smoking), MH14CALCH == 0 does occur in ADNI (2 subjects), meaning
    # the abuse ended in the same calendar year as the visit. MH14CALCH == 0 is therefore
    # treated as current — not as a synonym for missing.
    #
    # Look-ahead: MH14CALCH is missing at screening for some confirmed ex-abusers who
    # report an end year at a later visit. Empirically, 5 of 33 abusers with NaN MH14CALCH
    # at screening have MH14CALCH at some later visit. We recover these by taking the
    # earliest non-NaN value across all visits and back-filling it into the baseline row
    # (updating out["MH14CALCH"] in place so alcohol_burden also benefits).
    #
    # Classification after look-ahead:
    #   current_alcohol_abuse = 0 : no history (MH14ALCH == 0)
    #                         = 0 : confirmed ex-abuser (MH14CALCH > 0)
    #                         = 1 : ended same year (MH14CALCH == 0) or no end year found
    #                               at any visit — assumed still current (parallel to
    #                               smoking convention)
    #                         = NaN: MH14ALCH not recorded
    alcohol_hist = _ensure_binary(out["MH14ALCH"]) if "MH14ALCH" in out.columns else pd.Series(np.nan, index=out.index)

    if "MH14CALCH" in df.columns and "MH14CALCH" in out.columns:
        earliest_calch = (
            df[df["MH14CALCH"].notna()]
            .sort_values(config.visit_col)
            .groupby(config.subject_id_col)["MH14CALCH"]
            .first()
        )
        missing_mask = out["MH14CALCH"].isna()
        out.loc[missing_mask, "MH14CALCH"] = (
            out.loc[missing_mask, config.subject_id_col].map(earliest_calch)
        )

    yrs_since_end = _to_numeric(out["MH14CALCH"]) if "MH14CALCH" in out.columns else pd.Series(np.nan, index=out.index)

    out["alcohol_abuse_history"] = alcohol_hist
    out["current_alcohol_abuse"] = pd.Series(np.where(
        alcohol_hist == 0, 0.0,
        np.where(
            (alcohol_hist == 1) & (yrs_since_end > 0), 0.0,
            np.where(alcohol_hist == 1, 1.0, np.nan),
        )
    ), index=out.index)
    out["alcohol_drinks_per_day"] = (
        _to_numeric(out["MH14AALCH"]) if "MH14AALCH" in out.columns else np.nan
    )
    out["alcohol_abuse_duration_years"] = (
        _to_numeric(out["MH14BALCH"]) if "MH14BALCH" in out.columns else np.nan
    )
    out["alcohol_years_since_end"] = yrs_since_end

    alcohol_parts = [
        _zscore(out["alcohol_drinks_per_day"]),
        _zscore(out["alcohol_abuse_duration_years"]),
        _zscore(-out["alcohol_years_since_end"]),
    ]
    out["alcohol_burden"] = _mean_from_parts(*alcohol_parts)
    out.loc[out["alcohol_abuse_history"] == 0, "alcohol_burden"] = 0.0

    out["systolic_bp"] = _to_numeric(out["VSBPSYS"]) if "VSBPSYS" in out.columns else np.nan
    out["diastolic_bp"] = _to_numeric(out["VSBPDIA"]) if "VSBPDIA" in out.columns else np.nan
    out["pulse"] = _to_numeric(out["VSPULSE"]) if "VSPULSE" in out.columns else np.nan
    htn_med = out["has_antihypertensive_med"]
    out["hypertension"] = (
        (out["systolic_bp"] >= 140)
        | (out["diastolic_bp"] >= 90)
        | (htn_med == 1)
    ).astype(float)
    out.loc[
        out["systolic_bp"].isna() & out["diastolic_bp"].isna() & htn_med.isna(),
        "hypertension",
    ] = np.nan

    out["obesity"] = np.where(out["BMI"].notna(), (out["BMI"] >= 30).astype(float), np.nan)

    glucose_col = _first_existing_column(out, ["RCT11", "GLUCOSE"])
    out["serum_glucose"] = _to_numeric(out[glucose_col]) if glucose_col else np.nan
    dm_med = out["has_diabetes_med"]
    diabetes_text = (
        _contains_any(
            out["MHDESC"],
            {"diabetes", "diabetes mellitus", "type 1 diabetes", "type 2 diabetes", "dm"},
        )
        if "MHDESC" in out.columns
        else pd.Series(False, index=out.index)
    )
    out["hyperglycemia_status"] = np.where(
        pd.Series(out["serum_glucose"]).notna(),
        (_to_numeric(out["serum_glucose"]) >= gthr).astype(float),
        np.nan,
    )
    out["diabetes"] = (
        (_to_numeric(out["serum_glucose"]) >= gthr)
        | (dm_med == 1)
        | diabetes_text
    ).astype(float)
    out.loc[
        pd.Series(out["serum_glucose"]).isna() & dm_med.isna() & (~diabetes_text),
        "diabetes",
    ] = np.nan

    out["serum_cholesterol"] = _to_numeric(out["RCT20"]) if "RCT20" in out.columns else np.nan
    lipid_med = out["has_lipid_med"]
    out["high_cholesterol"] = (
        (_to_numeric(out["serum_cholesterol"]) >= cthr) | (lipid_med == 1)
    ).astype(float)
    out.loc[pd.Series(out["serum_cholesterol"]).isna() & lipid_med.isna(), "high_cholesterol"] = np.nan

    # Heart disease
    # The LIBRA "heart disease" component (Neuffer et al. 2024, Table 1) covers:
    # "history of myocardial infarction, hospitalized stroke, coronary surgery/
    # angioplasty, history of leg artery surgery if arteritis of the lower limbs."
    # Stroke and TIA are therefore part of the canonical definition, not a broadening
    # of it. ADNI's MH4CARD captures cardiac history but does not explicitly include
    # stroke, so text matching on MHDESC is used to recover stroke and TIA cases that
    # MH4CARD would otherwise miss. "heart failure" and "ischemic heart disease" are
    # MRF additions beyond the canonical LIBRA set; they do not appear in the LIBRA
    # file but are plausibly prevention-relevant cardiovascular events.
    heart_history = (
        _ensure_binary(out["MH4CARD"])
        if "MH4CARD" in out.columns
        else pd.Series(np.nan, index=out.index)
    )
    heart_text = (
        _contains_any(
            out["MHDESC"],
            {
                "myocardial infarction",
                "mi",
                "coronary artery disease",
                "cad",
                "angina",
                "cabg",
                "bypass",
                "stent",
                "angioplasty",
                "coronary surgery",
                "stroke",
                "tia",
                "transient ischemic attack",
                "peripheral vascular disease",
                "pvd",
                "heart failure",
                "ischemic heart disease",
            },
        )
        if "MHDESC" in out.columns
        else pd.Series(False, index=out.index)
    )
    out["heart_disease"] = ((heart_history == 1) | heart_text).astype(float)
    out.loc[heart_history.isna() & (~heart_text), "heart_disease"] = np.nan

    renal_history = (
        _ensure_binary(out["MH12RENA"])
        if "MH12RENA" in out.columns
        else pd.Series(np.nan, index=out.index)
    )
    renal_text = (
        _contains_any(
            out["MHDESC"],
            {
                "chronic kidney disease",
                "ckd",
                "renal insufficiency",
                "renal failure",
                "kidney disease",
                "kidney failure",
                "renal disease",
            },
        )
        if "MHDESC" in out.columns
        else pd.Series(False, index=out.index)
    )
    out["renal_dysfunction"] = (
        (out["eGFR"] < 60) | (renal_history == 1) | renal_text
    ).astype(float)
    out.loc[out["eGFR"].isna() & renal_history.isna() & (~renal_text), "renal_dysfunction"] = np.nan

    gd = _to_numeric(out["GDTOTAL"]) if "GDTOTAL" in out.columns else pd.Series(np.nan, index=out.index)
    dxdep = _ensure_binary(out["DXDEP"]) if "DXDEP" in out.columns else pd.Series(np.nan, index=out.index)
    bcdep = _ensure_binary(out["BCDEPRES"]) if "BCDEPRES" in out.columns else pd.Series(np.nan, index=out.index)
    keymed_antidep = pd.Series(np.nan, index=out.index)
    if "KEYMED" in out.columns:
        keymed_antidep = _contains_delimited_code(out["KEYMED"], 6)
    rxdep = out["has_antidepressant_med"]
    out["depression"] = (
        (gd >= config.gds_threshold)
        | (dxdep == 1)
        | (bcdep == 1)
        | (keymed_antidep == 1)
        | (rxdep == 1)
    ).astype(float)
    out.loc[
        gd.isna() & dxdep.isna() & bcdep.isna() & keymed_antidep.isna() & rxdep.isna(),
        "depression",
    ] = np.nan
    out["depression_severity"] = gd

    out["education_years"] = _to_numeric(out["PTEDUCAT"]) if "PTEDUCAT" in out.columns else np.nan

    partnered = pd.Series(np.nan, index=out.index)
    if "PTMARRY" in out.columns:
        m = _to_numeric(out["PTMARRY"])
        partnered = pd.Series(
            np.where(m.isin([1, 6]), 1.0, np.where(m.notna(), 0.0, np.nan)),
            index=out.index,
        )
    out["partnered"] = partnered

    homeowner = pd.Series(np.nan, index=out.index)
    community_dwelling = pd.Series(np.nan, index=out.index)
    lives_alone = pd.Series(np.nan, index=out.index)
    if "PTHOME" in out.columns:
        h = _to_numeric(out["PTHOME"])
        homeowner = pd.Series(
            np.where(h.isin([1, 2, 10]), 1.0, np.where(h.notna(), 0.0, np.nan)),
            index=out.index,
        )
        community_dwelling = pd.Series(
            np.where(h.isin([1, 2, 3, 4, 9, 10]), 1.0, np.where(h.notna(), 0.0, np.nan)),
            index=out.index,
        )
        lives_alone = pd.Series(
            np.where(h.isin([3, 4]), 1.0, np.where(h.notna(), 0.0, np.nan)),
            index=out.index,
        )
    out["homeowner"] = homeowner
    out["community_dwelling"] = community_dwelling
    out["lives_alone"] = lives_alone

    retired = pd.Series(np.nan, index=out.index)
    if "PTNOTRT" in out.columns:
        r = _ensure_binary(out["PTNOTRT"])
        retired = pd.Series(
            np.where(r == 1, 1.0, np.where(r == 0, 0.0, np.nan)),
            index=out.index,
        )
    out["retired"] = retired

    out["work_history_sufficient"] = (
        _ensure_binary(out["PTWORKHS"])
        if "PTWORKHS" in out.columns
        else np.nan
    )

    out["social_isolation_score"] = _sum_with_nan(
        1 - out["partnered"], out["lives_alone"], out["retired"]
    )
    out["social_structural_engagement"] = _mean_from_parts(
        out["partnered"], out["community_dwelling"], 1 - out["retired"]
    )

    entry_year = pd.Series(np.nan, index=out.index)
    if "entry_date" in out.columns:
        entry_year = _coerce_datetime(out["entry_date"]).dt.year
    retirement_year = (
        _to_numeric(out["PTRTYR"])
        if "PTRTYR" in out.columns
        else pd.Series(np.nan, index=out.index)
    )
    out["years_retired"] = np.where(
        (out["retired"] == 1) & entry_year.notna() & retirement_year.notna(),
        entry_year - retirement_year,
        np.nan,
    )

    if config.include_social_interactions:
        out["education_retired"] = out["education_years"] * out["retired"]
        out["married_homeowner"] = out["partnered"] * out["homeowner"]
        out["retired_lives_alone"] = out["retired"] * out["lives_alone"]
        out["ses_score"] = _mean_from_parts(
            _zscore(out["education_years"]), out["homeowner"], out["partnered"]
        )

    out["vascular_treatment_gap"] = (
        np.where(
            ((out["systolic_bp"] >= 140) | (out["diastolic_bp"] >= 90))
            & ~(out["has_antihypertensive_med"] == 1),
            1.0,
            0.0,
        )
        + np.where(
            (out["serum_cholesterol"] >= cthr) & ~(out["has_lipid_med"] == 1),
            1.0,
            0.0,
        )
        + np.where(
            (out["serum_glucose"] >= gthr) & ~(out["has_diabetes_med"] == 1),
            1.0,
            0.0,
        )
    )
    # Set to NaN when all three measurement sources are missing; the sum of np.where
    # calls above would otherwise return 0.0 for subjects with no BP, cholesterol,
    # or glucose data, which is indistinguishable from a gap-free profile.
    sys = out["systolic_bp"]
    dia = out["diastolic_bp"]
    chol = out["serum_cholesterol"]
    glu = out["serum_glucose"]
    out.loc[sys.isna() & dia.isna() & chol.isna() & glu.isna(), "vascular_treatment_gap"] = np.nan

    # -------------------------------------------------------------------------
    # libra_supported_raw: fillna(0) bias warning
    # -------------------------------------------------------------------------
    # Missing components are filled with 0 before summing, which treats absence
    # of data as absence of risk. Empirically, 57.9% of subjects at screening are
    # missing at least one of the 8 supported components, with a median suppressed
    # weight-sum of 6.3 points. Only 42.1% of subjects have all 8 observed.
    # Prefer libra_supported_rescaled_0_100 for downstream modelling.
    # -------------------------------------------------------------------------
    out["libra_supported_raw"] = sum(
        LIBRA_WEIGHTS_2024[f] * out[f].fillna(0) for f in SUPPORTED_ADNI_CANONICAL
    )
    out["libra_supported_rescaled_0_100"] = out.apply(
        lambda r: _rescale_observed_weighted_sum(
            r,
            SUPPORTED_ADNI_CANONICAL,
            LIBRA_WEIGHTS_2024,
            min_n=config.min_features_for_rescale,
        ),
        axis=1,
    )
    out["libra_supported_late_life_raw"] = sum(
        LIBRA_WEIGHTS_2024[f] * out[f].fillna(0) for f in LATE_LIFE_SUPPORTED_ADNI
    )

    out["modifiable_risk_core_count"] = out[SUPPORTED_ADNI_CANONICAL].sum(axis=1, min_count=1)
    out["metabolic_risk_count"] = _sum_with_nan(
        out["diabetes"], out["high_cholesterol"], out["obesity"], out["hypertension"]
    )
    out["vascular_risk_count"] = _sum_with_nan(
        out["current_smoking"],
        out["heart_disease"],
        out["diabetes"],
        out["high_cholesterol"],
        out["hypertension"],
        out["renal_dysfunction"],
    )
    out["lifestyle_risk_count"] = _sum_with_nan(
        out["current_smoking"],
        out["current_alcohol_abuse"],
        out["depression"],
        1 - out["social_structural_engagement"],
    )

    preferred_cols = [
        config.subject_id_col,
        "has_baseline_row",
        "has_screening_row",
        "screening_fallback_allowed",
        "baseline_diagnosis",
        "transition_label",
        "first_conversion_month",
        "n_followup_visits_ge12_with_diag",
        "education_years",
        "partnered",
        "homeowner",
        "community_dwelling",
        "lives_alone",
        "retired",
        "work_history_sufficient",
        "social_isolation_score",
        "social_structural_engagement",
        "years_retired",
        "education_retired",
        "married_homeowner",
        "retired_lives_alone",
        "ses_score",
        "smoking_history",
        "current_smoking",
        "smoking_packs_per_day",
        "smoking_duration_years",
        "smoking_years_since_quit",
        "tobacco_burden",
        "alcohol_abuse_history",
        "current_alcohol_abuse",
        "alcohol_drinks_per_day",
        "alcohol_abuse_duration_years",
        "alcohol_years_since_end",
        "alcohol_burden",
        "weight_kg",
        "height_m",
        "BMI",
        "obesity",
        "systolic_bp",
        "diastolic_bp",
        "pulse",
        "hypertension",
        "serum_glucose",
        "hyperglycemia_status",
        "diabetes",
        "serum_cholesterol",
        "high_cholesterol",
        "serum_creatinine",
        "eGFR",
        "renal_dysfunction",
        "heart_disease",
        "depression",
        "depression_severity",
        "has_antihypertensive_med",
        "has_lipid_med",
        "has_diabetes_med",
        "has_antidepressant_med",
        "vascular_treatment_gap",
        "modifiable_risk_core_count",
        "metabolic_risk_count",
        "vascular_risk_count",
        "lifestyle_risk_count",
        "libra_supported_raw",
        "libra_supported_rescaled_0_100",
        "libra_supported_late_life_raw",
    ]
    existing = [c for c in preferred_cols if c in out.columns]
    return out[existing].copy()


def score_csv(
    input_csv: str, output_csv: str, config: Optional[MRFConfig] = None
) -> pd.DataFrame:
    df = pd.read_csv(input_csv)
    scored = build_adni_mrf_features_from_wide(df, config=config)
    scored.to_csv(output_csv, index=False)
    return scored


EXAMPLE = r"""
import pandas as pd
from data_preprocessing_mrf import MRFConfig, build_adni_mrf_features_from_wide

df = pd.read_csv("data/All_Subjects_My_Table_11Mar2026.csv")
cfg = MRFConfig(subject_id_col="subject_id", visit_col="visit", diagnosis_col="DIAGNOSIS")

mrf_df = build_adni_mrf_features_from_wide(df, config=cfg)
mrf_df.to_csv("data/adni_mrf_features.csv", index=False)
print(mrf_df.head())
"""
