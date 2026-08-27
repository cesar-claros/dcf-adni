"""Time-to-event feasibility probe for the CN-to-impairment task.

Experiment 18 names "temporal distance" as the leading explanation for why MRF
features fail on the primary task, yet no experiment in the record models time.
This script tests that explanation directly by replacing the matched-pair binary
classifier with a time-to-event model over the *unmatched* CN cohort.

Why this uses more data than the matched design:

* The matched design keeps 1 control per transition subject (179 pairs at most),
  discarding roughly 517 of the 707 stable-CN subjects that the pipeline already
  derives features for.
* Survival analysis needs no control matching. Every labeled CN subject
  contributes: converters as events, stable CN as right-censored observations.
* Stable CN are followed for a median of 3 follow-up visits versus 5 for
  converters. Binary labelling calls the short-follow-up subjects "stable" when
  many were simply not observed long enough to convert. The existing pipeline can
  only respond by *deleting* them (``min_stable_followup_months``), which shrinks
  the cohort; censoring keeps their partial information instead.

The estimator stays CatBoost so that NaN handling and tuning behaviour match the
existing record; only the loss changes (``Cox`` instead of ``Logloss``). This
isolates the effect of the *framing* from the effect of the *model*.

Censoring times are not currently persisted by the preprocessing pipeline
(``first_conversion_month`` is populated for converters only), so they are
recovered here from the raw longitudinal table as the last visit carrying a
non-null diagnosis. If this probe is promising, that derivation belongs in
``data_preprocessing_libra.py``.

Usage::

    uv run --python 3.11 --with catboost --with pandas --with scikit-learn \\
        python model_survival_feasibility.py
"""

from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import StratifiedKFold

logging.basicConfig(level=logging.INFO, format="%(name)s — %(message)s")
logger = logging.getLogger(__name__)

LIBRA_COL = "libra_supported_rescaled_0_100"

_METADATA_COLS = {
    "subject_id",
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
    "first_conversion_month",
    "first_dementia_month",
    "baseline_diagnosis",
    "n_followup_visits_ge12_with_diag",
    "has_baseline_row",
    "has_screening_row",
    "screening_fallback_allowed",
}


def _visit_month(visit: object) -> float:
    """Map an ADNI visit code to months since baseline."""
    if not isinstance(visit, str):
        return np.nan
    code = visit.strip().lower()
    if code in {"sc", "scmri", "bl", "f", "nv", "v01"}:
        return 0.0
    match = re.fullmatch(r"m(\d+)", code)
    return float(match.group(1)) if match else np.nan


def build_survival_frame(data_dir: str = "data") -> pd.DataFrame:
    """Return one row per labeled CN subject with event indicator and time."""
    raw = pd.read_csv(f"{data_dir}/data_11Mar2026.csv", low_memory=False)
    raw["month"] = raw["visit"].map(_visit_month)

    diagnosed = raw[raw["DIAGNOSIS"].notna() & raw["month"].notna()]
    last_seen = diagnosed.groupby("subject_id")["month"].max().rename("last_diag_month")

    bmca = pd.read_csv(f"{data_dir}/adni_bmca_features.csv", low_memory=False)
    labeled = bmca[bmca["transition_label"].notna()].copy()
    labeled = labeled.merge(last_seen, on="subject_id", how="left")

    labeled["event"] = labeled["transition_label"].astype(int)
    labeled["time_months"] = np.where(
        labeled["event"] == 1,
        labeled["first_conversion_month"],
        labeled["last_diag_month"],
    )

    before = len(labeled)
    labeled = labeled[labeled["time_months"].notna() & (labeled["time_months"] > 0)]
    logger.info(
        "survival frame: %d subjects (%d dropped for missing/zero time), %d events, %d censored",
        len(labeled),
        before - len(labeled),
        int(labeled["event"].sum()),
        int((1 - labeled["event"]).sum()),
    )
    return labeled


def _catboost_survival_label(time_months: np.ndarray, event: np.ndarray) -> np.ndarray:
    """Encode survival targets for CatBoost's Cox loss (censored times are negative)."""
    return np.where(event == 1, time_months, -time_months)


def _concordance(time: np.ndarray, event: np.ndarray, risk: np.ndarray) -> float:
    """Harrell's C-index. ``risk`` is higher-is-sooner-event."""
    concordant = 0.0
    total = 0.0
    for i in range(len(time)):
        if event[i] != 1:
            continue
        comparable = time > time[i]
        n_comp = int(comparable.sum())
        if n_comp == 0:
            continue
        total += n_comp
        concordant += float((risk[i] > risk[comparable]).sum())
        concordant += 0.5 * float((risk[i] == risk[comparable]).sum())
    return concordant / total if total else float("nan")


def _bootstrap_c_index(
    time: np.ndarray, event: np.ndarray, risk: np.ndarray, n_boot: int, seed: int
) -> tuple[float, float]:
    """Percentile bootstrap CI for the C-index, resampling subjects."""
    rng = np.random.default_rng(seed)
    stats = []
    for _ in range(n_boot):
        idx = rng.choice(len(time), size=len(time), replace=True)
        if event[idx].sum() < 2:
            continue
        c = _concordance(time[idx], event[idx], risk[idx])
        if not np.isnan(c):
            stats.append(c)
    return float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def evaluate_feature_set(
    df: pd.DataFrame,
    feature_cols: list[str],
    name: str,
    n_folds: int = 5,
    seed: int = 0,
    n_boot: int = 1000,
) -> dict:
    """Cross-validated CatBoost-Cox C-index for one feature set."""
    time = df["time_months"].to_numpy(float)
    event = df["event"].to_numpy(int)
    x = df[feature_cols]

    oof_risk = np.full(len(df), np.nan)
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for train_idx, test_idx in cv.split(x, event):
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
        y_train = _catboost_survival_label(time[train_idx], event[train_idx])
        model.fit(Pool(x.iloc[train_idx], y_train))
        oof_risk[test_idx] = model.predict(x.iloc[test_idx])

    c_index = _concordance(time, event, oof_risk)
    lo, hi = _bootstrap_c_index(time, event, oof_risk, n_boot=n_boot, seed=seed)
    logger.info(
        "  [%s] OOF C-index = %.3f  95%% CI [%.3f, %.3f]  (%d features, n=%d, events=%d)",
        name,
        c_index,
        lo,
        hi,
        len(feature_cols),
        len(df),
        int(event.sum()),
    )
    return {"name": name, "c_index": c_index, "ci_low": lo, "ci_high": hi, "risk": oof_risk}


def run(data_dir: str = "data", output_dir: str = "results_survival", seed: int = 0) -> pd.DataFrame:
    """Compare BMCA, MRF, BMCA+MRF and LIBRA under a time-to-event framing."""
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

    bmca_audit = pd.read_csv(f"{data_dir}/adni_bmca_features_column_audit.csv")
    mrf_audit = pd.read_csv(f"{data_dir}/adni_mrf_features_column_audit.csv")
    bmca_keep = set(bmca_audit.loc[bmca_audit["keep_for_modeling"] == 1, "column"])
    mrf_keep = set(mrf_audit.loc[mrf_audit["keep_for_modeling"] == 1, "column"])

    drop = _METADATA_COLS | {"event", "time_months", "last_diag_month"}
    bmca_features = [c for c in merged.columns if c in bmca_keep and c not in drop]
    mrf_features = [c for c in merged.columns if c in mrf_keep and c not in drop]
    numeric = set(merged.select_dtypes(include=[np.number]).columns)
    bmca_features = [c for c in bmca_features if c in numeric]
    mrf_features = [c for c in mrf_features if c in numeric]

    results = [
        evaluate_feature_set(merged, bmca_features, "BMCA", seed=seed),
        evaluate_feature_set(merged, mrf_features, "MRF", seed=seed),
        evaluate_feature_set(
            merged, sorted(set(bmca_features) | set(mrf_features)), "BMCA+MRF", seed=seed
        ),
    ]

    if LIBRA_COL in merged.columns:
        valid = merged[LIBRA_COL].notna()
        c = _concordance(
            merged.loc[valid, "time_months"].to_numpy(float),
            merged.loc[valid, "event"].to_numpy(int),
            merged.loc[valid, LIBRA_COL].to_numpy(float),
        )
        logger.info("  [LIBRA] raw-score C-index = %.3f  (n=%d)", c, int(valid.sum()))
        results.append(
            {"name": "LIBRA", "c_index": c, "ci_low": np.nan, "ci_high": np.nan, "risk": None}
        )

    summary = pd.DataFrame(
        [
            {
                "model": r["name"],
                "n_subjects": len(merged),
                "n_events": int(merged["event"].sum()),
                "c_index": round(r["c_index"], 4),
                "ci_low_95": round(r["ci_low"], 4) if not np.isnan(r["ci_low"]) else None,
                "ci_high_95": round(r["ci_high"], 4) if not np.isnan(r["ci_high"]) else None,
            }
            for r in results
        ]
    )
    summary.to_csv(f"{output_dir}/survival_summary.csv", index=False)
    logger.info("\n%s", summary.to_string(index=False))
    return summary


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Time-to-event feasibility probe")
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--output_dir", default="results_survival")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    run(data_dir=args.data_dir, output_dir=args.output_dir, seed=args.seed)


if __name__ == "__main__":
    main()
