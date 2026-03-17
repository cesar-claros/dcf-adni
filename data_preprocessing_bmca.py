"""
ADNI biomarker / medical / cognitive assessment (BMCA) feature construction
from a single wide patient table, plus diagnosis-transition labeling.

This module uses the same baseline-row construction and transition-label logic
as `data_preprocessing_libra.py` so BMCA features are cohort-aligned with the
LIBRA-like and MRF pipelines.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

try:
    from data_preprocessing_libra import (
        LibraConfig,
        _replace_or_add_columns,
        _to_numeric,
        build_baseline_with_screening_fallback,
        build_transition_labels,
    )
except ModuleNotFoundError:
    from Code.data_preprocessing_libra import (
        LibraConfig,
        _replace_or_add_columns,
        _to_numeric,
        build_baseline_with_screening_fallback,
        build_transition_labels,
    )


@dataclass
class BMCAConfig(LibraConfig):
    include_item_level_faq: bool = True
    include_item_level_npi: bool = True


def _map_binary_codes(
    series: pd.Series, present_values: set[int], absent_values: set[int]
) -> pd.Series:
    s = _to_numeric(series)
    return pd.Series(
        np.where(
            s.isin(list(present_values)),
            1.0,
            np.where(s.isin(list(absent_values)), 0.0, np.nan),
        ),
        index=series.index,
    )


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    num = _to_numeric(numerator)
    den = _to_numeric(denominator)
    out = num / den
    out = out.where(den.notna() & den.ne(0))
    return out


def _sum_with_nan(frame: pd.DataFrame, columns: list[str]) -> pd.Series:
    existing = [c for c in columns if c in frame.columns]
    if not existing:
        return pd.Series(np.nan, index=frame.index)
    return frame[existing].apply(_to_numeric).sum(axis=1, min_count=1)


def build_adni_bmca_features_from_wide(
    df: pd.DataFrame, config: Optional[BMCAConfig] = None
) -> pd.DataFrame:
    """
    Build one baseline biomarker/medical/cognitive feature row per subject.
    """
    if config is None:
        config = BMCAConfig()

    out = build_baseline_with_screening_fallback(df, config)
    labels = build_transition_labels(df, config)
    out = out.merge(labels, on=config.subject_id_col, how="left")

    biomarker_cols = [
        "ABETA40",
        "ABETA42",
        "TAU",
        "PTAU",
        "pT217_F",
        "AB42_F",
        "AB42_AB40_F",
        "pT217_AB42_F",
        "BAT126",
        "HMT40",
    ]
    out = _replace_or_add_columns(
        out,
        {
            col: (
                _to_numeric(out[col])
                if col in out.columns
                else pd.Series(np.nan, index=out.index)
            )
            for col in biomarker_cols
        },
    )

    biomarker_features = {
        "csf_abeta40": out["ABETA40"],
        "csf_abeta42": out["ABETA42"],
        "csf_tau": out["TAU"],
        "csf_ptau": out["PTAU"],
        "plasma_ptau217": out["pT217_F"],
        "plasma_abeta42": out["AB42_F"],
        "plasma_abeta42_abeta40_ratio": out["AB42_AB40_F"],
        "plasma_ptau217_abeta42_ratio": out["pT217_AB42_F"],
        "vitamin_b12": out["BAT126"],
        "hemoglobin": out["HMT40"],
    }
    biomarker_features.update(
        {
            "csf_tau_abeta42_ratio": _safe_divide(
                biomarker_features["csf_tau"], biomarker_features["csf_abeta42"]
            ),
            "csf_ptau_abeta42_ratio": _safe_divide(
                biomarker_features["csf_ptau"], biomarker_features["csf_abeta42"]
            ),
            "csf_ptau_tau_ratio": _safe_divide(
                biomarker_features["csf_ptau"], biomarker_features["csf_tau"]
            ),
            "csf_abeta42_abeta40_ratio": _safe_divide(
                biomarker_features["csf_abeta42"], biomarker_features["csf_abeta40"]
            ),
        }
    )
    out = _replace_or_add_columns(out, biomarker_features)

    cognitive_raw = {
        "mmse_total": "MMSCORE",
        "logical_memory_immediate": "LIMMTOTAL",
        "trail_a_time": "TRAASCOR",
        "logical_memory_delayed": "LDELTOTAL",
        "delayed_cue_used": "LDELCUE",
        "adas11_total": "TOTSCORE",
        "adas13_total": "TOTAL13",
        "cdrsb": "CDRSB",
        "faq_total": "FAQTOTAL",
    }
    faq_item_map = {
        "faq_finances": "FAQFINAN",
        "faq_forms": "FAQFORM",
        "faq_shopping": "FAQSHOP",
        "faq_games": "FAQGAME",
        "faq_beverages": "FAQBEVG",
        "faq_meal_prep": "FAQMEAL",
        "faq_events": "FAQEVENT",
        "faq_tv": "FAQTV",
        "faq_reminders": "FAQREM",
        "faq_travel": "FAQTRAVL",
    }
    cognitive_features = {
        new_col: (
            _to_numeric(out[raw_col])
            if raw_col in out.columns
            else pd.Series(np.nan, index=out.index)
        )
        for new_col, raw_col in {**cognitive_raw, **faq_item_map}.items()
    }
    cognitive_features.update(
        {
            "adas_q4_q14_delta": (
                cognitive_features["adas13_total"] - cognitive_features["adas11_total"]
            ),
            "logical_memory_retention_ratio": _safe_divide(
                cognitive_features["logical_memory_delayed"],
                cognitive_features["logical_memory_immediate"],
            ),
            "faq_any_impairment": np.where(
                cognitive_features["faq_total"].notna(),
                (cognitive_features["faq_total"] > 0).astype(float),
                np.nan,
            ),
        }
    )
    out = _replace_or_add_columns(out, cognitive_features)

    medical_features = {
        "somatic_complaints": (
            _map_binary_codes(out["HMSOMATC"], {1}, {0})
            if "HMSOMATC" in out.columns
            else pd.Series(np.nan, index=out.index)
        ),
        "hachinski_hypertension_history": (
            _map_binary_codes(out["HMHYPERT"], {1}, {0})
            if "HMHYPERT" in out.columns
            else pd.Series(np.nan, index=out.index)
        ),
        "hachinski_stroke_history": (
            _map_binary_codes(out["HMSTROKE"], {2}, {0})
            if "HMSTROKE" in out.columns
            else pd.Series(np.nan, index=out.index)
        ),
        "hachinski_total_score": (
            _to_numeric(out["HMSCORE"])
            if "HMSCORE" in out.columns
            else pd.Series(np.nan, index=out.index)
        ),
    }

    present_absent_12 = {
        "auditory_impairment": "NXAUDITO",
        "gait_abnormality": "NXGAIT",
        "physical_heart_exam_abnormal": "PXHEART",
        "peripheral_vascular_exam_abnormal": "PXPERIPH",
        "blurred_vision_symptom": "AXVISION",
        "headache_symptom": "AXHDACHE",
        "chest_pain_symptom": "AXCHEST",
        "musculoskeletal_pain_symptom": "AXMUSCLE",
        "insomnia_symptom": "AXINSOMN",
        "depressed_mood_symptom": "AXDPMOOD",
        "fall_symptom": "AXFALL",
        "baseline_vomiting": "BCVOMIT",
        "baseline_blurred_vision": "BCVISION",
        "baseline_headache": "BCHDACHE",
        "baseline_chest_pain": "BCCHEST",
        "baseline_musculoskeletal_pain": "BCMUSCLE",
        "baseline_insomnia": "BCINSOMN",
        "baseline_depressed_mood": "BCDPMOOD",
        "baseline_fall": "BCFALL",
    }
    medical_features.update(
        {
            new_col: (
                _map_binary_codes(out[raw_col], {2}, {1})
                if raw_col in out.columns
                else pd.Series(np.nan, index=out.index)
            )
            for new_col, raw_col in present_absent_12.items()
        }
    )

    medical_features["baseline_stroke"] = (
        _map_binary_codes(out["BCSTROKE"], {1}, {0})
        if "BCSTROKE" in out.columns
        else pd.Series(np.nan, index=out.index)
    )
    out = _replace_or_add_columns(out, medical_features)

    medical_frame = pd.DataFrame(medical_features, index=out.index)
    out = _replace_or_add_columns(
        out,
        {
            "neurovascular_burden": _sum_with_nan(
                medical_frame,
                [
                    "somatic_complaints",
                    "hachinski_hypertension_history",
                    "hachinski_stroke_history",
                    "physical_heart_exam_abnormal",
                    "peripheral_vascular_exam_abnormal",
                ],
            ),
            "symptom_burden_current": _sum_with_nan(
                medical_frame,
                [
                    "blurred_vision_symptom",
                    "headache_symptom",
                    "chest_pain_symptom",
                    "musculoskeletal_pain_symptom",
                    "insomnia_symptom",
                    "depressed_mood_symptom",
                    "fall_symptom",
                ],
            ),
            "symptom_burden_baseline": _sum_with_nan(
                medical_frame,
                [
                    "baseline_vomiting",
                    "baseline_blurred_vision",
                    "baseline_headache",
                    "baseline_chest_pain",
                    "baseline_musculoskeletal_pain",
                    "baseline_insomnia",
                    "baseline_depressed_mood",
                    "baseline_fall",
                ],
            ),
        },
    )

    depression_npi_cols = [
        "NPID1",
        "NPID2",
        "NPID3",
        "NPID4",
        "NPID5",
        "NPID6",
        "NPID7",
        "NPID8",
        "NPID9A",
        "NPID9B",
        "NPID9C",
    ]
    sleep_npi_cols = [
        "NPIK1",
        "NPIK2",
        "NPIK3",
        "NPIK4",
        "NPIK5",
        "NPIK6",
        "NPIK7",
        "NPIK8",
        "NPIK9A",
        "NPIK9B",
        "NPIK9C",
    ]
    npi_raw_cols = depression_npi_cols + sleep_npi_cols + ["NPIDTOT", "NPIKTOT", "NPIKSEV"]
    out = _replace_or_add_columns(
        out,
        {
            col: (
                _to_numeric(out[col])
                if col in out.columns
                else pd.Series(np.nan, index=out.index)
            )
            for col in npi_raw_cols
        },
    )
    out = _replace_or_add_columns(
        out,
        {
            "npi_depression_item_score": out["NPIDTOT"],
            "npi_sleep_item_score": out["NPIKTOT"],
            "npi_sleep_severity": out["NPIKSEV"],
            "npi_depression_domain_sum": _sum_with_nan(out, depression_npi_cols),
            "npi_sleep_domain_sum": _sum_with_nan(out, sleep_npi_cols),
        },
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
        "csf_abeta40",
        "csf_abeta42",
        "csf_tau",
        "csf_ptau",
        "csf_tau_abeta42_ratio",
        "csf_ptau_abeta42_ratio",
        "csf_ptau_tau_ratio",
        "csf_abeta42_abeta40_ratio",
        "plasma_ptau217",
        "plasma_abeta42",
        "plasma_abeta42_abeta40_ratio",
        "plasma_ptau217_abeta42_ratio",
        "vitamin_b12",
        "hemoglobin",
        "mmse_total",
        "logical_memory_immediate",
        "logical_memory_delayed",
        "logical_memory_retention_ratio",
        "trail_a_time",
        "delayed_cue_used",
        "adas11_total",
        "adas13_total",
        "adas_q4_q14_delta",
        "cdrsb",
        "faq_total",
        "faq_any_impairment",
        "faq_finances",
        "faq_forms",
        "faq_shopping",
        "faq_games",
        "faq_beverages",
        "faq_meal_prep",
        "faq_events",
        "faq_tv",
        "faq_reminders",
        "faq_travel",
        "npi_depression_item_score",
        "npi_sleep_item_score",
        "npi_sleep_severity",
        "npi_depression_domain_sum",
        "npi_sleep_domain_sum",
        "somatic_complaints",
        "hachinski_hypertension_history",
        "hachinski_stroke_history",
        "hachinski_total_score",
        "auditory_impairment",
        "gait_abnormality",
        "physical_heart_exam_abnormal",
        "peripheral_vascular_exam_abnormal",
        "blurred_vision_symptom",
        "headache_symptom",
        "chest_pain_symptom",
        "musculoskeletal_pain_symptom",
        "insomnia_symptom",
        "depressed_mood_symptom",
        "fall_symptom",
        "baseline_vomiting",
        "baseline_blurred_vision",
        "baseline_headache",
        "baseline_chest_pain",
        "baseline_musculoskeletal_pain",
        "baseline_insomnia",
        "baseline_depressed_mood",
        "baseline_fall",
        "baseline_stroke",
        "neurovascular_burden",
        "symptom_burden_current",
        "symptom_burden_baseline",
    ]

    if config.include_item_level_faq:
        preferred_cols.extend([c for c in faq_item_map if c not in preferred_cols])

    if config.include_item_level_npi:
        preferred_cols.extend(
            [c for c in depression_npi_cols + sleep_npi_cols if c not in preferred_cols]
        )

    existing = [c for c in preferred_cols if c in out.columns]
    return out[existing].copy()


def score_csv(
    input_csv: str, output_csv: str, config: Optional[BMCAConfig] = None
) -> pd.DataFrame:
    df = pd.read_csv(input_csv)
    scored = build_adni_bmca_features_from_wide(df, config=config)
    scored.to_csv(output_csv, index=False)
    return scored


EXAMPLE = r"""
import pandas as pd
from data_preprocessing_bmca import BMCAConfig, build_adni_bmca_features_from_wide

df = pd.read_csv("All_Subjects_My_Table_11Mar2026.csv")
cfg = BMCAConfig(subject_id_col="subject_id", visit_col="visit", diagnosis_col="DIAGNOSIS")

bmca_df = build_adni_bmca_features_from_wide(df, config=cfg)
bmca_df.to_csv("data/adni_bmca_features.csv", index=False)
print(bmca_df.head())
"""
