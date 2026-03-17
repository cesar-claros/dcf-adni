
"""
ADNI LIBRA-like score construction from a single wide patient table,
plus diagnosis-transition labeling.

Core assumptions
----------------
- One row per subject/visit
- subject identifier column: subject_id
- visit column: visit
- baseline measurements come from visit == "bl"
- missing baseline fields are backfilled from visit == "sc"

Outcome labeling
----------------
A subject receives:
- label = 1 if baseline DIAGNOSIS == 1 and any follow-up visit at >= 12 months
  has DIAGNOSIS in {2, 3}
- label = 0 if baseline DIAGNOSIS == 1 and all available follow-up diagnoses
  through the last visit remain 1
- label = NaN otherwise

Important implementation note
-----------------------------
For screening fallback, this module assumes the intended rule is:
- allow screening -> baseline backfill only when the baseline-screening gap is
  not more than 12 months.

Why: using screening as fallback when it is farther than 12 months from
baseline would usually be hard to justify epidemiologically. If you truly want
the opposite rule, set `fallback_requires_gap_at_most_days=None` and customize
`_screening_eligible_mask`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Dict, Iterable
import re
import numpy as np
import pandas as pd


LIBRA_WEIGHTS_2024 = {
    "healthy_diet": -1.7,
    "physical_inactivity": 1.1,
    "cognitive_activity_low": -3.2,
    "low_to_moderate_alcohol": -1.0,
    "smoking": 1.5,
    "heart_disease": 1.0,
    "diabetes": 1.3,
    "high_cholesterol": 1.4,
    "obesity": 1.6,
    "hypertension": 1.6,
    "renal_dysfunction": 1.1,
    "depression": 2.1,
}

SUPPORTED_ADNI_CANONICAL = [
    "smoking",
    "heart_disease",
    "diabetes",
    "high_cholesterol",
    "obesity",
    "hypertension",
    "renal_dysfunction",
    "depression",
]

LATE_LIFE_SUPPORTED_ADNI = [
    "smoking",
    "heart_disease",
    "diabetes",
    "high_cholesterol",
    "renal_dysfunction",
    "depression",
]


@dataclass
class LibraConfig:
    subject_id_col: str = "subject_id"
    visit_col: str = "visit"
    baseline_visit: str = "bl"
    fallback_visit: str = "sc"
    diagnosis_col: str = "DIAGNOSIS"

    # Optional visit date columns used to determine whether screening can
    # backfill baseline. The code will try these in order.
    visit_date_candidates: tuple[str, ...] = (
        "EXAMDATE", "VISDATE", "SCANDATE", "COLDATE", "LBDATE",
        "USERDATE", "entry_date", "date", "visit_date"
    )
    fallback_requires_gap_at_most_days: Optional[int] = 365

    glucose_unit: str = "mg/dL"
    cholesterol_unit: str = "mg/dL"
    creatinine_unit: str = "mg/dL"

    glucose_threshold: Optional[float] = None
    cholesterol_threshold: Optional[float] = None
    gds_threshold: int = 6

    use_medication_evidence_for_htn: bool = True
    use_medication_evidence_for_diabetes: bool = True
    use_medication_evidence_for_cholesterol: bool = True
    use_antidepressant_as_depression_evidence: bool = True
    use_current_smoking_proxy: bool = True
    min_supported_components_for_rescale: int = 4

    def resolved_glucose_threshold(self) -> float:
        return 126.0 if self.glucose_unit.lower() == "mg/dl" else 7.0

    def resolved_cholesterol_threshold(self) -> float:
        return 240.0 if self.cholesterol_unit.lower() == "mg/dl" else 6.2


ANTIHYPERTENSIVES = {
    "lisinopril", "enalapril", "ramipril", "benazepril",
    "losartan", "valsartan", "irbesartan", "olmesartan",
    "amlodipine", "nifedipine", "diltiazem", "verapamil",
    "metoprolol", "atenolol", "carvedilol", "propranolol",
    "hydrochlorothiazide", "chlorthalidone", "furosemide", "spironolactone",
    "clonidine", "hydralazine", "aliskiren", "telmisartan", "candesartan",
}

STATINS_AND_LIPID_LOWERING = {
    "atorvastatin", "simvastatin", "rosuvastatin", "pravastatin", "lovastatin",
    "fluvastatin", "pitavastatin", "ezetimibe", "fenofibrate", "gemfibrozil",
    "niacin", "colesevelam", "cholestyramine", "alirocumab", "evolocumab",
}

ANTIDIABETICS = {
    "metformin", "insulin", "glipizide", "glyburide", "glimepiride",
    "pioglitazone", "rosiglitazone", "sitagliptin", "linagliptin",
    "liraglutide", "semaglutide", "empagliflozin", "canagliflozin",
    "dapagliflozin", "acarbose", "repaglinide", "nateglinide",
}

ANTIDEPRESSANTS = {
    "sertraline", "fluoxetine", "paroxetine", "citalopram", "escitalopram",
    "venlafaxine", "desvenlafaxine", "duloxetine", "bupropion",
    "mirtazapine", "amitriptyline", "nortriptyline", "trazodone",
    "clomipramine", "imipramine", "phenelzine", "tranylcypromine",
}


def _to_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _normalize_text(s: pd.Series) -> pd.Series:
    return (
        s.astype(str)
         .str.lower()
         .str.replace(r"[^a-z0-9\s\-\/\.]", " ", regex=True)
         .str.replace(r"\s+", " ", regex=True)
         .str.strip()
    )


def _contains_any(text: pd.Series, terms: Iterable[str]) -> pd.Series:
    terms = sorted(set(t.lower() for t in terms))
    if not terms:
        return pd.Series(False, index=text.index)
    pattern = r"\b(?:%s)\b" % "|".join(re.escape(t) for t in terms)
    return _normalize_text(text).str.contains(pattern, regex=True, na=False)


def _ensure_binary(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    return s.where(s.isin([0, 1]), np.nan)


def _coerce_datetime(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce")


def _visit_to_months(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.lower().str.strip()
    out = pd.Series(np.nan, index=series.index, dtype=float)
    out.loc[s == "bl"] = 0.0
    out.loc[s == "sc"] = -1.0
    mask = s.str.match(r"^m\d+$", na=False)
    out.loc[mask] = s.loc[mask].str.replace("m", "", regex=False).astype(float)
    return out


def _first_existing_column(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _replace_or_add_columns(
    frame: pd.DataFrame, columns: Dict[str, pd.Series | np.ndarray | float | int]
) -> pd.DataFrame:
    if not columns:
        return frame

    updates = pd.DataFrame(columns, index=frame.index)
    preserved = frame.drop(columns=list(updates.columns), errors="ignore")
    return pd.concat([preserved, updates], axis=1)


def _nanmean_rows(*parts: pd.Series) -> pd.Series:
    if not parts:
        return pd.Series(dtype=float)

    stack = np.vstack([_to_numeric(p).values for p in parts]).astype(float)
    valid_counts = np.sum(~np.isnan(stack), axis=0)
    means = np.divide(
        np.nansum(stack, axis=0),
        valid_counts,
        out=np.full(valid_counts.shape, np.nan, dtype=float),
        where=valid_counts > 0,
    )
    return pd.Series(means, index=parts[0].index)


def _convert_weight_to_kg(weight: pd.Series, unit: pd.Series) -> pd.Series:
    w = _to_numeric(weight)
    u = _to_numeric(unit)
    return pd.Series(np.where(u == 1, w * 0.45359237, w), index=weight.index)


def _convert_height_to_m(height: pd.Series, unit: pd.Series) -> pd.Series:
    h = _to_numeric(height)
    u = _to_numeric(unit)
    out = np.where(u == 1, h * 0.0254, h / 100.0)
    out = np.where(u.isna() & (h <= 3), h, out)
    return pd.Series(out, index=height.index)


def _ckd_epi_2021_egfr(creatinine: pd.Series, age: pd.Series, sex_code: pd.Series, creatinine_unit: str = "mg/dL") -> pd.Series:
    scr = _to_numeric(creatinine).copy()
    if creatinine_unit.lower() in {"umol/l", "µmol/l", "umol"}:
        scr = scr / 88.4

    age = _to_numeric(age)
    sex_code = _to_numeric(sex_code)

    female = (sex_code == 2).astype(float)
    k = np.where(female == 1, 0.7, 0.9)
    a = np.where(female == 1, -0.241, -0.302)

    min_part = np.minimum(scr / k, 1.0) ** a
    max_part = np.maximum(scr / k, 1.0) ** -1.2
    sex_mult = np.where(female == 1, 1.012, 1.0)

    egfr = 142 * min_part * max_part * (0.9938 ** age) * sex_mult
    return pd.Series(egfr, index=creatinine.index)


def _rescale_observed_weighted_sum(row: pd.Series, factors: Sequence[str], weights: Dict[str, float], min_n: int = 4) -> float:
    weighted = []
    present_weights = []
    for f in factors:
        v = row.get(f, np.nan)
        if pd.notna(v):
            weighted.append(weights[f] * float(v))
            present_weights.append(weights[f])

    if len(present_weights) < min_n:
        return np.nan

    observed_min = sum(w for w in present_weights if w < 0)
    observed_max = sum(w for w in present_weights if w > 0)
    score = sum(weighted)

    if observed_max == observed_min:
        return np.nan
    return 100.0 * (score - observed_min) / (observed_max - observed_min)


def _deduplicate_visit_rows(df: pd.DataFrame, subject_id_col: str, visit_col: str) -> pd.DataFrame:
    """
    If multiple rows exist for the same subject/visit, keep the row with the
    highest non-missing count; tie-break by first occurrence.
    """
    tmp = df.copy()
    tmp["_nonnulls"] = tmp.notna().sum(axis=1)
    tmp["_row_order"] = np.arange(len(tmp))
    tmp = tmp.sort_values(
        [subject_id_col, visit_col, "_nonnulls", "_row_order"],
        ascending=[True, True, False, True],
        kind="stable",
    )
    tmp = tmp.drop_duplicates([subject_id_col, visit_col], keep="first")
    return tmp.drop(columns=["_nonnulls", "_row_order"])


def _screening_eligible_mask(bl: pd.DataFrame, sc: pd.DataFrame, config: LibraConfig) -> pd.Series:
    """
    Returns a boolean Series indexed by subject_id indicating whether screening
    is allowed to backfill baseline for that subject.
    """
    eligible = pd.Series(True, index=bl.index.union(sc.index))
    if config.fallback_requires_gap_at_most_days is None:
        return eligible

    date_col = _first_existing_column(pd.concat([bl, sc], axis=0), config.visit_date_candidates)
    if date_col is None or date_col not in bl.columns or date_col not in sc.columns:
        # No usable dates -> allow fallback, but caller should be aware.
        return eligible

    bl_dates = _coerce_datetime(bl[date_col]).reindex(eligible.index)
    sc_dates = _coerce_datetime(sc[date_col]).reindex(eligible.index)
    diff_days = (bl_dates - sc_dates).dt.days.abs()
    # If one of the dates is missing, default to allowing fallback.
    ok = diff_days.le(config.fallback_requires_gap_at_most_days) | diff_days.isna()
    return ok.fillna(True)


def build_baseline_with_screening_fallback(df: pd.DataFrame, config: Optional[LibraConfig] = None) -> pd.DataFrame:
    """
    Build one row per subject:
    - primary source: baseline visit
    - field-level fallback: screening visit for values missing at baseline,
      subject to the screening eligibility rule
    """
    if config is None:
        config = LibraConfig()

    required = {config.subject_id_col, config.visit_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    work = df.copy()
    work[config.visit_col] = work[config.visit_col].astype(str).str.lower().str.strip()
    work = _deduplicate_visit_rows(work, config.subject_id_col, config.visit_col)

    bl = work.loc[work[config.visit_col] == config.baseline_visit].copy()
    sc = work.loc[work[config.visit_col] == config.fallback_visit].copy()

    bl = bl.drop_duplicates(config.subject_id_col, keep="first").set_index(config.subject_id_col)
    sc = sc.drop_duplicates(config.subject_id_col, keep="first").set_index(config.subject_id_col)

    all_subjects = bl.index.union(sc.index)
    bl = bl.reindex(all_subjects)
    sc = sc.reindex(all_subjects)

    screening_allowed = _screening_eligible_mask(bl, sc, config)
    sc_allowed = sc.copy()
    sc_allowed.loc[~screening_allowed, :] = np.nan

    combined = bl.combine_first(sc_allowed)
    source_columns = {}
    for c in combined.columns:
        if c not in bl.columns and c not in sc_allowed.columns:
            continue
        blc = bl[c] if c in bl.columns else pd.Series(index=combined.index, dtype=object)
        scc = (
            sc_allowed[c]
            if c in sc_allowed.columns
            else pd.Series(index=combined.index, dtype=object)
        )
        src_col = pd.Series([None] * len(combined), index=combined.index, dtype=object)
        src_col.loc[scc.notna()] = config.fallback_visit
        src_col.loc[blc.notna()] = config.baseline_visit
        source_columns[f"__src__{c}"] = src_col

    combined = _replace_or_add_columns(
        combined,
        {
            config.subject_id_col: pd.Series(combined.index, index=combined.index),
            "has_baseline_row": combined.index.isin(bl.dropna(how="all").index).astype(int),
            "has_screening_row": combined.index.isin(sc.dropna(how="all").index).astype(int),
            "screening_fallback_allowed": screening_allowed.astype(int)
            .reindex(combined.index)
            .fillna(1)
            .astype(int),
            **source_columns,
        },
    )

    return combined.reset_index(drop=True)


def build_transition_labels(df: pd.DataFrame, config: Optional[LibraConfig] = None) -> pd.DataFrame:
    """
    Build subject-level transition labels from DIAGNOSIS.

    Rule:
    - label 1:
        baseline diagnosis == 1 and any follow-up visit at >=12 months has
        diagnosis in {2, 3}
    - label 0:
        baseline diagnosis == 1 and all observed follow-up diagnoses through the
        last visit remain 1
    - NaN otherwise
    """
    if config is None:
        config = LibraConfig()

    required = {config.subject_id_col, config.visit_col, config.diagnosis_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns for labeling: {sorted(missing)}")

    work = df[[config.subject_id_col, config.visit_col, config.diagnosis_col]].copy()
    work[config.visit_col] = work[config.visit_col].astype(str).str.lower().str.strip()
    work["_visit_months"] = _visit_to_months(work[config.visit_col])
    work[config.diagnosis_col] = _to_numeric(work[config.diagnosis_col])

    # Deduplicate by subject/visit using first non-missing diagnosis.
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
        baseline_diag = bl.iloc[0] if len(bl) else np.nan

        follow = g.loc[g["_visit_months"] >= 12, [config.visit_col, config.diagnosis_col, "_visit_months"]].copy()
        follow_nonmissing = follow.loc[follow[config.diagnosis_col].notna()].copy()

        if pd.isna(baseline_diag) or baseline_diag != 1:
            label = np.nan
        elif len(follow_nonmissing) == 0:
            label = np.nan
        elif follow_nonmissing[config.diagnosis_col].isin([2, 3]).any():
            label = 1.0
        elif follow_nonmissing[config.diagnosis_col].eq(1).all():
            label = 0.0
        else:
            label = np.nan

        first_conversion_month = np.nan
        if len(follow_nonmissing):
            conv = follow_nonmissing.loc[follow_nonmissing[config.diagnosis_col].isin([2, 3]), "_visit_months"]
            if len(conv):
                first_conversion_month = float(conv.min())

        rows.append(
            {
                config.subject_id_col: sid,
                "baseline_diagnosis": baseline_diag,
                "transition_label": label,
                "first_conversion_month": first_conversion_month,
                "n_followup_visits_ge12_with_diag": int(len(follow_nonmissing)),
            }
        )

    return pd.DataFrame(rows)


def build_adni_libra_like_from_wide(df: pd.DataFrame, config: Optional[LibraConfig] = None) -> pd.DataFrame:
    """
    Build a baseline LIBRA-like score from a single wide subject-by-visit table.
    """
    if config is None:
        config = LibraConfig()

    out = build_baseline_with_screening_fallback(df, config)
    labels = build_transition_labels(df, config)
    out = out.merge(labels, on=config.subject_id_col, how="left")

    gthr = config.resolved_glucose_threshold() if config.glucose_threshold is None else config.glucose_threshold
    cthr = config.resolved_cholesterol_threshold() if config.cholesterol_threshold is None else config.cholesterol_threshold

    # Medications
    if "CMMED" in out.columns:
        med_text = _normalize_text(out["CMMED"])
        out["has_antihypertensive_med"] = _contains_any(med_text, ANTIHYPERTENSIVES).astype(float)
        out["has_lipid_med"] = _contains_any(med_text, STATINS_AND_LIPID_LOWERING).astype(float)
        out["has_diabetes_med"] = _contains_any(med_text, ANTIDIABETICS).astype(float)
        out["has_antidepressant_med"] = _contains_any(med_text, ANTIDEPRESSANTS).astype(float)
    else:
        out["has_antihypertensive_med"] = np.nan
        out["has_lipid_med"] = np.nan
        out["has_diabetes_med"] = np.nan
        out["has_antidepressant_med"] = np.nan

    # BMI
    if {"VSWEIGHT", "VSHEIGHT"}.issubset(out.columns):
        wt_unit = out["VSWTUNIT"] if "VSWTUNIT" in out.columns else pd.Series(np.nan, index=out.index)
        ht_unit = out["VSHTUNIT"] if "VSHTUNIT" in out.columns else pd.Series(np.nan, index=out.index)
        weight_kg = _convert_weight_to_kg(out["VSWEIGHT"], wt_unit)
        height_m = _convert_height_to_m(out["VSHEIGHT"], ht_unit)
        out["BMI"] = weight_kg / (height_m ** 2)
    else:
        out["BMI"] = np.nan

    # Age and sex for eGFR
    age_col = _first_existing_column(out, ["entry_age", "AGE", "age", "PTAGE"])
    out["_age_for_egfr"] = _to_numeric(out[age_col]) if age_col else np.nan

    if {"RCT392", "_age_for_egfr", "PTGENDER"}.issubset(out.columns):
        out["eGFR"] = _ckd_epi_2021_egfr(
            out["RCT392"],
            out["_age_for_egfr"],
            out["PTGENDER"],
            creatinine_unit=config.creatinine_unit,
        )
    else:
        out["eGFR"] = np.nan

    # Smoking
    smoke_hist = _ensure_binary(out["MH16SMOK"]) if "MH16SMOK" in out.columns else pd.Series(np.nan, index=out.index)
    yrs_quit = _to_numeric(out["MH16CSMOK"]) if "MH16CSMOK" in out.columns else pd.Series(np.nan, index=out.index)
    if config.use_current_smoking_proxy:
        out["smoking"] = np.where(
            (smoke_hist == 1) & (yrs_quit.fillna(0) <= 0),
            1.0,
            np.where(smoke_hist == 0, 0.0, np.nan),
        )
    else:
        out["smoking"] = smoke_hist

    # Tobacco burden
    if {"MH16ASMOK", "MH16BSMOK", "MH16CSMOK"}.issubset(out.columns):
        ppd = _to_numeric(out["MH16ASMOK"])
        dur = _to_numeric(out["MH16BSMOK"])
        quit_inv = -_to_numeric(out["MH16CSMOK"])
        parts = []
        for s in [ppd, dur, quit_inv]:
            if s.notna().sum() > 1:
                parts.append((s - s.mean()) / s.std(ddof=0))
            else:
                parts.append(pd.Series(np.nan, index=out.index))
        tb = _nanmean_rows(*parts)
        out["tobacco_burden"] = np.where(
            smoke_hist == 1,
            tb.values,
            np.where(smoke_hist == 0, 0.0, np.nan),
        )
    else:
        out["tobacco_burden"] = np.nan

    # Hypertension
    sys = _to_numeric(out["VSBPSYS"]) if "VSBPSYS" in out.columns else pd.Series(np.nan, index=out.index)
    dia = _to_numeric(out["VSBPDIA"]) if "VSBPDIA" in out.columns else pd.Series(np.nan, index=out.index)
    htn_med = out["has_antihypertensive_med"] if config.use_medication_evidence_for_htn else pd.Series(np.nan, index=out.index)
    out["hypertension"] = ((sys >= 140) | (dia >= 90) | (htn_med == 1)).astype(float)
    out.loc[sys.isna() & dia.isna() & htn_med.isna(), "hypertension"] = np.nan

    # Obesity
    bmi = _to_numeric(out["BMI"])
    out["obesity"] = np.where(bmi.notna(), (bmi >= 30).astype(float), np.nan)

    # Cholesterol
    chol_col = _first_existing_column(out, ["RCT20", "CHOLESTEROL"])
    chol = _to_numeric(out[chol_col]) if chol_col else pd.Series(np.nan, index=out.index)
    lipid_med = out["has_lipid_med"] if config.use_medication_evidence_for_cholesterol else pd.Series(np.nan, index=out.index)
    out["high_cholesterol"] = ((chol >= cthr) | (lipid_med == 1)).astype(float)
    out.loc[chol.isna() & lipid_med.isna(), "high_cholesterol"] = np.nan

    # Diabetes
    glucose_col = _first_existing_column(out, ["RCT11", "GLUCOSE"])
    glu = _to_numeric(out[glucose_col]) if glucose_col else pd.Series(np.nan, index=out.index)
    dm_med = out["has_diabetes_med"] if config.use_medication_evidence_for_diabetes else pd.Series(np.nan, index=out.index)

    diabetes_text = pd.Series(False, index=out.index)
    if "MHDESC" in out.columns:
        diabetes_text = _contains_any(out["MHDESC"], {"diabetes", "diabetes mellitus", "type 1 diabetes", "type 2 diabetes", "dm"})
    out["diabetes"] = ((glu >= gthr) | (dm_med == 1) | diabetes_text).astype(float)
    out.loc[glu.isna() & dm_med.isna() & (~diabetes_text), "diabetes"] = np.nan
    out["hyperglycemia_status"] = np.where(glu.notna(), (glu >= gthr).astype(float), np.nan)

    # Heart disease
    mh4 = _ensure_binary(out["MH4CARD"]) if "MH4CARD" in out.columns else pd.Series(np.nan, index=out.index)
    heart_text = pd.Series(False, index=out.index)
    if "MHDESC" in out.columns:
        heart_text = _contains_any(
            out["MHDESC"],
            {
                "myocardial infarction", "mi", "coronary artery disease", "cad", "angina",
                "cabg", "bypass", "stent", "angioplasty", "stroke", "tia",
                "transient ischemic attack", "peripheral vascular disease", "pvd",
                "coronary surgery",
            },
        )
    out["heart_disease"] = ((mh4 == 1) | heart_text).astype(float)
    out.loc[mh4.isna() & (~heart_text), "heart_disease"] = np.nan
    out["cardiovascular_event_burden"] = out["heart_disease"]

    # Renal dysfunction
    mh12 = _ensure_binary(out["MH12RENA"]) if "MH12RENA" in out.columns else pd.Series(np.nan, index=out.index)
    renal_text = pd.Series(False, index=out.index)
    if "MHDESC" in out.columns:
        renal_text = _contains_any(
            out["MHDESC"],
            {"chronic kidney disease", "ckd", "renal insufficiency", "renal failure", "kidney disease", "kidney failure", "renal disease"},
        )
    egfr_low = np.where(out["eGFR"].notna(), (out["eGFR"] < 60).astype(float), np.nan)
    out["renal_dysfunction"] = ((pd.Series(egfr_low, index=out.index) == 1) | (mh12 == 1) | renal_text).astype(float)
    out.loc[pd.Series(egfr_low, index=out.index).isna() & mh12.isna() & (~renal_text), "renal_dysfunction"] = np.nan

    # Depression
    gd = _to_numeric(out["GDTOTAL"]) if "GDTOTAL" in out.columns else pd.Series(np.nan, index=out.index)
    dxdep = _ensure_binary(out["DXDEP"]) if "DXDEP" in out.columns else pd.Series(np.nan, index=out.index)
    bcdep = _ensure_binary(out["BCDEPRES"]) if "BCDEPRES" in out.columns else pd.Series(np.nan, index=out.index)

    keymed_antidep = pd.Series(np.nan, index=out.index)
    if "KEYMED" in out.columns:
        keymed_num = _to_numeric(out["KEYMED"])
        keymed_antidep = pd.Series(
            np.where(keymed_num == 6, 1.0, np.where(keymed_num.notna(), 0.0, np.nan)),
            index=out.index,
        )

    rxdep = out["has_antidepressant_med"] if config.use_antidepressant_as_depression_evidence else pd.Series(np.nan, index=out.index)

    out["depression"] = ((gd >= config.gds_threshold) | (dxdep == 1) | (bcdep == 1) | (keymed_antidep == 1) | (rxdep == 1)).astype(float)
    out.loc[gd.isna() & dxdep.isna() & bcdep.isna() & pd.Series(keymed_antidep).isna() & rxdep.isna(), "depression"] = np.nan
    out["depression_severity"] = gd

    # Alcohol risk proxy
    out["alcohol_risk_status"] = _ensure_binary(out["MH14ALCH"]) if "MH14ALCH" in out.columns else np.nan

    # Social structural engagement
    partnered = pd.Series(np.nan, index=out.index)
    if "PTMARRY" in out.columns:
        m = _to_numeric(out["PTMARRY"])
        partnered = pd.Series(np.where(m.isin([1, 6]), 1.0, np.where(m.notna(), 0.0, np.nan)), index=out.index)

    community = pd.Series(np.nan, index=out.index)
    if "PTHOME" in out.columns:
        h = _to_numeric(out["PTHOME"])
        community = pd.Series(np.where(h.isin([1, 2, 3, 4, 9, 10]), 1.0, np.where(h.notna(), 0.0, np.nan)), index=out.index)

    not_retired = pd.Series(np.nan, index=out.index)
    if "PTNOTRT" in out.columns:
        r = _to_numeric(out["PTNOTRT"])
        not_retired = pd.Series(np.where(r == 0, 1.0, np.where(r.notna(), 0.0, np.nan)), index=out.index)

    out["social_structural_engagement"] = _nanmean_rows(
        partnered, community, not_retired
    ).values

    # Treatment gap
    out["vascular_treatment_gap"] = (
        np.where(((sys >= 140) | (dia >= 90)) & ~(htn_med == 1), 1.0, 0.0) +
        np.where((chol >= cthr) & ~(lipid_med == 1), 1.0, 0.0) +
        np.where((glu >= gthr) & ~(dm_med == 1), 1.0, 0.0)
    )

    # Summary scores
    out["libra_supported_raw"] = sum(LIBRA_WEIGHTS_2024[f] * out[f].fillna(0) for f in SUPPORTED_ADNI_CANONICAL)
    out["libra_supported_n_observed"] = out[SUPPORTED_ADNI_CANONICAL].notna().sum(axis=1)
    out["libra_supported_rescaled_0_100"] = out.apply(
        lambda r: _rescale_observed_weighted_sum(
            r, SUPPORTED_ADNI_CANONICAL, LIBRA_WEIGHTS_2024, min_n=config.min_supported_components_for_rescale
        ),
        axis=1,
    )

    out["libra_supported_late_life_raw"] = sum(LIBRA_WEIGHTS_2024[f] * out[f].fillna(0) for f in LATE_LIFE_SUPPORTED_ADNI)
    out["libra_supported_late_life_n_observed"] = out[LATE_LIFE_SUPPORTED_ADNI].notna().sum(axis=1)
    out["libra_supported_late_life_rescaled_0_100"] = out.apply(
        lambda r: _rescale_observed_weighted_sum(
            r, LATE_LIFE_SUPPORTED_ADNI, LIBRA_WEIGHTS_2024, min_n=max(3, config.min_supported_components_for_rescale - 1)
        ),
        axis=1,
    )

    out["modifiable_risk_core8_count"] = out[SUPPORTED_ADNI_CANONICAL].sum(axis=1, min_count=1)
    out["libra_missing_canonical_diet"] = 1
    out["libra_missing_canonical_physical_activity"] = 1
    out["libra_missing_canonical_cognitive_activity"] = 1
    out["libra_missing_canonical_protective_alcohol"] = 1

    preferred_cols = [
        config.subject_id_col,
        "has_baseline_row",
        "has_screening_row",
        "screening_fallback_allowed",
        "baseline_diagnosis",
        "transition_label",
        "first_conversion_month",
        "n_followup_visits_ge12_with_diag",
        "smoking", "heart_disease", "diabetes", "high_cholesterol",
        "obesity", "hypertension", "renal_dysfunction", "depression",
        "BMI", "eGFR", "tobacco_burden", "vascular_treatment_gap",
        "social_structural_engagement", "alcohol_risk_status",
        "libra_supported_raw", "libra_supported_rescaled_0_100",
        "libra_supported_n_observed",
        "libra_supported_late_life_raw", "libra_supported_late_life_rescaled_0_100",
        "libra_supported_late_life_n_observed",
        "modifiable_risk_core8_count",
    ]
    existing = [c for c in preferred_cols if c in out.columns]
    remainder = [c for c in out.columns if c not in existing]
    return out[existing + remainder]


def score_csv(input_csv: str, output_csv: str, config: Optional[LibraConfig] = None) -> pd.DataFrame:
    df = pd.read_csv(input_csv)
    scored = build_adni_libra_like_from_wide(df, config=config)
    scored.to_csv(output_csv, index=False)
    return scored


EXAMPLE = r"""
import pandas as pd
from libra_adni_wide_v2 import LibraConfig, build_adni_libra_like_from_wide

df = pd.read_csv("sample_table.csv")

cfg = LibraConfig(
    subject_id_col="subject_id",
    visit_col="visit",
    diagnosis_col="DIAGNOSIS",
    baseline_visit="bl",
    fallback_visit="sc",
    fallback_requires_gap_at_most_days=365,
    glucose_unit="mg/dL",
    cholesterol_unit="mg/dL",
    creatinine_unit="mg/dL",
    gds_threshold=6,
)

score_df = build_adni_libra_like_from_wide(df, config=cfg)
score_df.to_csv("adni_libra_like_scores_with_labels.csv", index=False)
print(score_df.head())
"""


# %%
