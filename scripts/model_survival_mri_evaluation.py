"""Survival + MRI: does volumetric MRI add signal under the time-to-event framing?

The survival feasibility probe established C-index anchors on the 886 labeled
CN subjects: BMCA 0.794, MRF 0.532, BMCA+MRF 0.797 (results/survival/). The
cross-sectional FreeSurfer 7 table (UCSFFSX7_20Jun2025.csv) covers 834 of
those subjects at baseline yet is used by no experiment in the record. This
script derives 13 literature-standard structural features and asks four
questions, each answered with a paired bootstrap C-index difference over
subject-level resamples:

  Q1  BMCA+MRI vs BMCA      does structure add beyond CSF/plasma/cognition?
  Q2  MRI vs BMCA           is MRI alone competitive with the biomarker set?
  Q3  MRF+MRI vs MRI        do modifiable risk factors add on top of structure?
  Q4  full vs BMCA+MRI      does MRF add once biomarkers AND structure are in?

MRI features (ST codes verified against DATADIC_11Mar2026.csv for UCSFFSX7):
bilateral volume / ICV for hippocampus (ST29SV+ST88SV), amygdala
(ST12SV+ST71SV), entorhinal (ST24CV+ST83CV), fusiform (ST26CV+ST85CV),
middle temporal (ST40CV+ST99CV), inferior lateral ventricle (ST30SV+ST89SV),
lateral ventricle (ST37SV+ST96SV), and WM hypointensities (ST128SV); plus ICV
itself (ST10CV) and mean left/right cortical thickness of entorhinal
(ST24TA/ST83TA), fusiform (ST26TA/ST85TA), and middle temporal
(ST40TA/ST99TA). WM hypointensities carry the vascular-burden signal closest
to the MRF story.

Baseline scan = the earliest month-0 visit (sc/scmri/bl/f/nv/v01) not marked
OVERALLQC Fail. The QC field is unpopulated for 95% of FSX7 rows, so only
explicit failures are excluded. Subjects without a baseline scan (52/886) keep
NaN features; CatBoost's native NaN handling matches how the record treats
CSF and plasma missingness. The estimator and its fixed hyperparameters are
identical to the survival probe, so every C-index is directly comparable and
the BMCA arm doubles as a replication check (expected 0.7944 at seed 0).

Usage::

    python scripts/model_survival_mri_evaluation.py
    python scripts/model_survival_mri_evaluation.py --n_boot 2000 --seed 0
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dcf_adni.paths import RESULTS_DIR

from model_survival_feasibility import (
    _METADATA_COLS,
    _catboost_survival_label,
    _concordance,
    build_survival_frame,
    evaluate_feature_set,
)

logging.basicConfig(level=logging.INFO, format="%(name)s — %(message)s")
logger = logging.getLogger(__name__)

MONTH0_VISITS = {"sc", "scmri", "bl", "f", "nv", "v01"}

# name -> (left ST code, right ST code); summed, then divided by ICV (ST10CV).
BILATERAL_VOLUMES = {
    "mri_hippocampus_icv": ("ST29SV", "ST88SV"),
    "mri_amygdala_icv": ("ST12SV", "ST71SV"),
    "mri_entorhinal_icv": ("ST24CV", "ST83CV"),
    "mri_fusiform_icv": ("ST26CV", "ST85CV"),
    "mri_middle_temporal_icv": ("ST40CV", "ST99CV"),
    "mri_inf_lat_ventricle_icv": ("ST30SV", "ST89SV"),
    "mri_lateral_ventricle_icv": ("ST37SV", "ST96SV"),
}
# name -> (left ST code, right ST code); mean of the two thickness averages.
BILATERAL_THICKNESS = {
    "mri_entorhinal_thickness": ("ST24TA", "ST83TA"),
    "mri_fusiform_thickness": ("ST26TA", "ST85TA"),
    "mri_middle_temporal_thickness": ("ST40TA", "ST99TA"),
}
ICV_COL = "ST10CV"
WMH_COL = "ST128SV"


def build_mri_features(fsx7_path: str) -> pd.DataFrame:
    """Return one baseline structural-MRI feature row per subject.

    Selects each subject's earliest month-0 scan that is not an explicit QC
    failure (ties broken by IMAGEUID for determinism), then derives
    ICV-normalized bilateral volumes and mean bilateral thicknesses.
    """
    fs = pd.read_csv(fsx7_path, low_memory=False)
    base = fs[fs["VISCODE2"].isin(MONTH0_VISITS) & (fs["OVERALLQC"] != "Fail")]
    sel = base.sort_values(["PTID", "EXAMDATE", "IMAGEUID"]).drop_duplicates("PTID")

    out = pd.DataFrame({"subject_id": sel["PTID"].values})
    icv = pd.to_numeric(sel[ICV_COL], errors="coerce").values
    out["mri_icv"] = icv
    for name, (left, right) in BILATERAL_VOLUMES.items():
        vol = (
            pd.to_numeric(sel[left], errors="coerce").values
            + pd.to_numeric(sel[right], errors="coerce").values
        )
        out[name] = vol / icv
    out["mri_wm_hypointensities_icv"] = (
        pd.to_numeric(sel[WMH_COL], errors="coerce").values / icv
    )
    for name, (left, right) in BILATERAL_THICKNESS.items():
        out[name] = (
            pd.to_numeric(sel[left], errors="coerce").values
            + pd.to_numeric(sel[right], errors="coerce").values
        ) / 2.0
    logger.info(
        "MRI features: %d subjects with a baseline scan, %d features",
        len(out),
        out.shape[1] - 1,
    )
    return out


def paired_bootstrap_c_diff(
    time: np.ndarray,
    event: np.ndarray,
    risk_a: np.ndarray,
    risk_b: np.ndarray,
    n_boot: int = 1000,
    seed: int = 0,
) -> dict:
    """Paired bootstrap test for C(A) - C(B), resampling subjects."""
    rng = np.random.default_rng(seed)
    observed = _concordance(time, event, risk_a) - _concordance(time, event, risk_b)
    diffs = []
    for _ in range(n_boot):
        idx = rng.choice(len(time), size=len(time), replace=True)
        if event[idx].sum() < 2:
            continue
        c_a = _concordance(time[idx], event[idx], risk_a[idx])
        c_b = _concordance(time[idx], event[idx], risk_b[idx])
        if not (np.isnan(c_a) or np.isnan(c_b)):
            diffs.append(c_a - c_b)
    diffs = np.array(diffs)
    return {
        "observed_diff": observed,
        "ci_low": float(np.percentile(diffs, 2.5)),
        "ci_high": float(np.percentile(diffs, 97.5)),
        "p_value": float(np.mean(diffs <= 0)),
        "n_boot": len(diffs),
    }


def full_data_importance(
    df: pd.DataFrame, features: list[str], seed: int
) -> pd.DataFrame:
    """CatBoost-Cox feature importance from one fit on all subjects.

    Hyperparameters mirror ``evaluate_feature_set`` so the importances describe
    the same model family the C-indices come from.
    """
    model = CatBoostRegressor(
        loss_function="Cox",
        iterations=500,
        learning_rate=0.03,
        depth=4,
        l2_leaf_reg=3.0,
        random_seed=seed,
        verbose=0,
        allow_writing_files=False,
        nan_mode="Min",
    )
    y = _catboost_survival_label(
        df["time_months"].to_numpy(float), df["event"].to_numpy(int)
    )
    model.fit(Pool(df[features], y))
    return (
        pd.DataFrame({"feature": features, "importance": model.get_feature_importance()})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def run(
    data_dir: str = "data",
    fsx7_path: str = "data/UCSFFSX7_20Jun2025.csv",
    output_dir: str = str(RESULTS_DIR / "survival_mri"),
    seed: int = 0,
    n_boot: int = 1000,
) -> pd.DataFrame:
    """Compare BMCA, MRF, and MRI feature sets under the time-to-event framing."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    surv = build_survival_frame(data_dir)

    mrf = pd.read_csv(f"{data_dir}/adni_mrf_features.csv", low_memory=False)
    merged = surv.merge(
        mrf.drop(columns=[c for c in mrf.columns if c in _METADATA_COLS and c != "subject_id"]),
        on="subject_id",
        how="left",
        suffixes=("", "_mrf_dup"),
    )
    merged = merged[[c for c in merged.columns if not c.endswith("_mrf_dup")]]

    mri = build_mri_features(fsx7_path)
    merged = merged.merge(mri, on="subject_id", how="left")
    mri_features = [c for c in mri.columns if c != "subject_id"]
    n_covered = int(merged["mri_icv"].notna().sum())
    logger.info(
        "MRI coverage in survival frame: %d/%d subjects (%.1f%%)",
        n_covered,
        len(merged),
        100.0 * n_covered / len(merged),
    )

    bmca_audit = pd.read_csv(f"{data_dir}/adni_bmca_features_column_audit.csv")
    mrf_audit = pd.read_csv(f"{data_dir}/adni_mrf_features_column_audit.csv")
    bmca_keep = set(bmca_audit.loc[bmca_audit["keep_for_modeling"] == 1, "column"])
    mrf_keep = set(mrf_audit.loc[mrf_audit["keep_for_modeling"] == 1, "column"])

    drop = _METADATA_COLS | {"event", "time_months", "last_diag_month"}
    numeric = set(merged.select_dtypes(include=[np.number]).columns)
    bmca_features = [c for c in merged.columns if c in bmca_keep and c not in drop and c in numeric]
    mrf_features = [c for c in merged.columns if c in mrf_keep and c not in drop and c in numeric]

    sets = {
        "BMCA": bmca_features,
        "MRF": mrf_features,
        "MRI": mri_features,
        "BMCA+MRI": sorted(set(bmca_features) | set(mri_features)),
        "MRF+MRI": sorted(set(mrf_features) | set(mri_features)),
        "BMCA+MRF+MRI": sorted(set(bmca_features) | set(mrf_features) | set(mri_features)),
    }
    results = {
        name: evaluate_feature_set(merged, feats, name, seed=seed, n_boot=n_boot)
        for name, feats in sets.items()
    }

    summary = pd.DataFrame(
        [
            {
                "model": name,
                "n_subjects": len(merged),
                "n_events": int(merged["event"].sum()),
                "n_features": len(sets[name]),
                "c_index": round(r["c_index"], 4),
                "ci_low_95": round(r["ci_low"], 4),
                "ci_high_95": round(r["ci_high"], 4),
            }
            for name, r in results.items()
        ]
    )
    summary.to_csv(f"{output_dir}/survival_mri_summary.csv", index=False)
    logger.info("\n%s", summary.to_string(index=False))

    time = merged["time_months"].to_numpy(float)
    event = merged["event"].to_numpy(int)
    comparisons = [
        ("Q1", "BMCA+MRI", "BMCA"),
        ("Q2", "MRI", "BMCA"),
        ("Q3", "MRF+MRI", "MRI"),
        ("Q4", "BMCA+MRF+MRI", "BMCA+MRI"),
    ]
    rows = []
    for label, name_a, name_b in comparisons:
        d = paired_bootstrap_c_diff(
            time, event, results[name_a]["risk"], results[name_b]["risk"],
            n_boot=n_boot, seed=seed,
        )
        rows.append(
            {
                "question": label,
                "comparison": f"{name_a} vs {name_b}",
                "observed_diff": round(d["observed_diff"], 4),
                "ci_low_95": round(d["ci_low"], 4),
                "ci_high_95": round(d["ci_high"], 4),
                "p_value": round(d["p_value"], 4),
                "n_boot": d["n_boot"],
            }
        )
        logger.info(
            "  [%s] %s vs %s: dC = %+.4f  95%% CI [%+.4f, %+.4f]  p(dC<=0) = %.4f",
            label, name_a, name_b,
            d["observed_diff"], d["ci_low"], d["ci_high"], d["p_value"],
        )
    diff_df = pd.DataFrame(rows)
    diff_df.to_csv(f"{output_dir}/paired_c_index_diff.csv", index=False)

    for name in ("MRI", "BMCA+MRI"):
        imp = full_data_importance(merged, sets[name], seed=seed)
        stem = name.lower().replace("+", "_")
        imp.to_csv(f"{output_dir}/{stem}_importance.csv", index=False)
        logger.info(
            "\n[%s] top 10 features:\n%s", name, imp.head(10).to_string(index=False)
        )

    return summary


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Survival framing with structural MRI feature sets"
    )
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--fsx7", default="data/UCSFFSX7_20Jun2025.csv")
    parser.add_argument("--output_dir", default=str(RESULTS_DIR / "survival_mri"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n_boot", type=int, default=1000)
    args = parser.parse_args()
    run(
        data_dir=args.data_dir,
        fsx7_path=args.fsx7,
        output_dir=args.output_dir,
        seed=args.seed,
        n_boot=args.n_boot,
    )


if __name__ == "__main__":
    main()
