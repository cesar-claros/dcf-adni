"""TabPFN tabular-foundation-model evaluation, mirroring the CatBoost nested-CV harness.

This script answers a single question: does a pretrained tabular foundation model
(TabPFN) change the established negative MRF finding, or does it reproduce it?

To keep the comparison honest, the fold structure, grouping, bootstrap CIs and
paired bootstrap tests are imported directly from
:mod:`model_strate_cv_evaluation` rather than reimplemented. The only thing that
changes is the estimator: CatBoost + Optuna is replaced by TabPFN, which performs
in-context learning and therefore needs no hyperparameter search (there is no
inner CV loop and no Optuna study).

TabPFN handles NaN natively, so -- unlike the imputation-based neural approach of
Experiment 16, which lost significantly to CatBoost (0.749 vs 0.823) partly
because median imputation destroyed missingness structure -- no imputation is
required here.

Requires ``tabpfn==2.0.9`` on Python 3.11. Both pins matter:

* The version should not be floated. TabPFN v2 is pretrained purely on synthetic
  data, whereas the TabPFN-2.5 default classifier checkpoint is fine-tuned on 43
  real datasets, which would put an ADNI/TADPOLE leakage caveat on any
  no-leakage claim made in the paper. Versions from 8.x additionally gate weight
  download behind an interactive license acceptance and a ``TABPFN_TOKEN``.
* On Python 3.14 this segfaults (SIGSEGV) partway through the first fit. Python
  3.11 is known good.

Usage::

    uv run --python 3.11 --with "tabpfn==2.0.9" --with catboost --with optuna \\
        python scripts/model_tabpfn_cv_evaluation.py --cohort all
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from model_strate_cv_evaluation import (
    _METADATA_COLS,
    GROUP_COL,
    LABEL_COL,
    _bootstrap_auc,
    _bootstrap_paired_auc_diff,
    _feature_cols,
    _load_combined,
)

logging.basicConfig(level=logging.INFO, format="%(name)s — %(message)s")
logger = logging.getLogger(__name__)

LIBRA_COL = "libra_supported_rescaled_0_100"

#: Cohort tag -> labeling strategy described in paper_experimentation_pipeline.md.
COHORTS: dict[str, str] = {
    "default": "L1 default (any_impairment, no cleaning)",
    "clean": "L4 clean labels (no reverters + confirmed progression)",
    "stratm": "L6 mci_only",
    "stratg": "strategy G",
    "strate": "L5 dementia_only",
}


def _resolve_audit(data_dir: str, block: str, tag: str) -> str | None:
    """Return the audit CSV for a feature block, preferring relaxed thresholds.

    ``paper_experimentation_pipeline.md`` specifies ``max_missing_fraction=0.9``
    and ``max_mode_fraction=0.98`` for every experiment, so that plasma
    biomarkers (~70% missing) survive to reach CatBoost's NaN-aware splits. Those
    thresholds live in the ``*_column_audit_relaxed.csv`` files. Only the clean
    cohort currently has one; the other cohorts ship the stricter default audit,
    which keeps 25-29 BMCA features instead of 54. Preferring the relaxed file
    where it exists keeps this comparable to the CatBoost record.
    """
    relaxed = Path(f"{data_dir}/adni_{block}_features{tag}_column_audit_relaxed.csv")
    if relaxed.exists():
        return str(relaxed)
    strict = Path(f"{data_dir}/adni_{block}_features{tag}_column_audit.csv")
    if strict.exists():
        logger.warning("no relaxed audit for %s%s, falling back to %s", block, tag, strict.name)
        return str(strict)
    logger.warning("no audit file for %s%s, using all features", block, tag)
    return None


def _resolve_device(requested: str) -> str:
    """Return a torch device string, falling back to cpu when mps/cuda is absent."""
    import torch

    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _build_estimator(seed: int, device: str):
    """Construct a TabPFN classifier.

    ``ignore_pretraining_limits`` is enabled because the BMCA+MRF union exceeds
    TabPFN's soft feature-count guidance; sample counts stay far below the limit.
    """
    from tabpfn import TabPFNClassifier

    return TabPFNClassifier(
        device=device,
        random_state=seed,
        ignore_pretraining_limits=True,
    )


def run_cv_for_feature_set(
    df: pd.DataFrame,
    feature_cols: list[str],
    name: str,
    n_outer: int = 5,
    seed: int = 0,
    device: str = "cpu",
    training_mode: str = "combined",
) -> dict:
    """Run outer CV with TabPFN, returning OOF scores and AUC with bootstrap CI.

    Mirrors :func:`model_strate_cv_evaluation.run_cv_for_feature_set` but without
    the inner Optuna loop, since TabPFN is not tuned.
    """
    if training_mode == "augmentation_only":
        cv_df = df[df["analysis_set"] == "augmentation"].copy()
        extra_train_df = pd.DataFrame(columns=df.columns)
    else:
        cv_df = df[df["analysis_set"] == "primary"].copy()
        if training_mode == "combined":
            extra_train_df = df[df["analysis_set"] == "augmentation"].copy()
        else:
            extra_train_df = pd.DataFrame(columns=df.columns)

    outer_cv = StratifiedGroupKFold(n_splits=n_outer, shuffle=True, random_state=seed)

    oof_scores = np.full(len(cv_df), np.nan)
    fold_aucs = []

    for fold_idx, (train_idx, test_idx) in enumerate(
        outer_cv.split(cv_df, y=cv_df[LABEL_COL], groups=cv_df[GROUP_COL])
    ):
        cv_train = cv_df.iloc[train_idx]
        cv_test = cv_df.iloc[test_idx]
        fold_train = pd.concat([cv_train, extra_train_df], ignore_index=True)

        x_train = fold_train[feature_cols].to_numpy(dtype=np.float64)
        y_train = fold_train[LABEL_COL].astype(float).to_numpy()
        x_test = cv_test[feature_cols].to_numpy(dtype=np.float64)
        y_test = cv_test[LABEL_COL].astype(float).to_numpy()

        model = _build_estimator(seed=seed, device=device)
        model.fit(x_train, y_train)
        fold_preds = model.predict_proba(x_test)[:, 1]
        oof_scores[test_idx] = fold_preds

        fold_auc = float("nan")
        if len(np.unique(y_test)) >= 2:
            fold_auc = roc_auc_score(y_test, fold_preds)
            fold_aucs.append(fold_auc)

        logger.info(
            "  [%s] Fold %d/%d: n_train=%d, test pairs=%d, fold AUC=%.3f",
            name,
            fold_idx + 1,
            n_outer,
            len(fold_train),
            cv_test[GROUP_COL].nunique(),
            fold_auc,
        )

    y_all = cv_df[LABEL_COL].astype(float).to_numpy()
    groups_all = cv_df[GROUP_COL].to_numpy()
    oof_auc = roc_auc_score(y_all, oof_scores)
    ci_low, ci_high = _bootstrap_auc(y_all, oof_scores, groups_all, n_boot=2000, seed=seed)

    logger.info(
        "  [%s] OOF AUC = %.3f  95%% CI [%.3f, %.3f]  (n=%d pairs, mode=%s)",
        name,
        oof_auc,
        ci_low,
        ci_high,
        len(cv_df) // 2,
        training_mode,
    )

    return {
        "name": name,
        "oof_auc": oof_auc,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "fold_aucs": fold_aucs,
        "oof_scores": oof_scores,
        "y_true": y_all,
        "groups": groups_all,
    }


def run_cohort(
    cohort: str,
    data_dir: str = "data",
    output_dir: str | None = None,
    n_outer: int = 5,
    seed: int = 0,
    device: str = "auto",
    training_mode: str = "combined",
    use_audit: bool = True,
) -> pd.DataFrame:
    """Evaluate BMCA, MRF, BMCA+MRF and LIBRA for one labeling-strategy cohort."""
    tag = "" if cohort == "default" else f"_{cohort}"
    bmca_path = f"{data_dir}/adni_bmca_features{tag}_combined_matched.csv"
    mrf_path = f"{data_dir}/adni_mrf_features{tag}_combined_matched.csv"
    bmca_audit = _resolve_audit(data_dir, "bmca", tag) if use_audit else None
    mrf_audit = _resolve_audit(data_dir, "mrf", tag) if use_audit else None

    output_dir = output_dir or f"results_tabpfn/{cohort}"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    resolved_device = _resolve_device(device)
    bmca_df = _load_combined(bmca_path)
    mrf_df = _load_combined(mrf_path)

    bmca_features = _feature_cols(bmca_df, bmca_audit)
    mrf_features = _feature_cols(mrf_df, mrf_audit)

    meta_cols = [c for c in bmca_df.columns if c in _METADATA_COLS]
    bmca_mrf_df = bmca_df.merge(
        mrf_df.drop(columns=[c for c in meta_cols if c != "subject_id"], errors="ignore"),
        on="subject_id",
        how="inner",
        suffixes=("", "_mrf_dup"),
    )
    bmca_mrf_df = bmca_mrf_df[[c for c in bmca_mrf_df.columns if not c.endswith("_mrf_dup")]]
    bmca_mrf_features = sorted(set(bmca_features) | set(mrf_features))

    n_cv_pairs = bmca_df[bmca_df["analysis_set"] == "primary"][GROUP_COL].nunique()
    logger.info(
        "\n%s\nCohort %s (%s): %d primary pairs, device=%s\n%s",
        "=" * 70,
        cohort,
        COHORTS.get(cohort, "?"),
        n_cv_pairs,
        resolved_device,
        "=" * 70,
    )

    results: dict[str, dict] = {}
    for name, frame, feats in [
        ("BMCA", bmca_df, bmca_features),
        ("MRF", mrf_df, mrf_features),
        ("BMCA+MRF", bmca_mrf_df, bmca_mrf_features),
    ]:
        logger.info("%s (%d features)", name, len(feats))
        results[name] = run_cv_for_feature_set(
            frame,
            feats,
            name,
            n_outer=n_outer,
            seed=seed,
            device=resolved_device,
            training_mode=training_mode,
        )

    # LIBRA raw-score baseline (no model training).
    y_cv = results["BMCA"]["y_true"]
    groups_cv = results["BMCA"]["groups"]
    if LIBRA_COL in mrf_df.columns:
        libra_pop = mrf_df[mrf_df["analysis_set"] == "primary"]
        libra_scores = libra_pop[LIBRA_COL].to_numpy()
        valid = ~np.isnan(libra_scores) & ~np.isnan(y_cv)
        if valid.sum() > 0 and len(np.unique(y_cv[valid])) >= 2:
            libra_auc = roc_auc_score(y_cv[valid], libra_scores[valid])
            lo, hi = _bootstrap_auc(
                y_cv[valid], libra_scores[valid], groups_cv[valid], n_boot=2000, seed=seed
            )
            results["LIBRA"] = {
                "name": "LIBRA",
                "oof_auc": libra_auc,
                "ci_low": lo,
                "ci_high": hi,
                "fold_aucs": [],
                "oof_scores": libra_scores,
                "y_true": y_cv,
                "groups": groups_cv,
            }
            logger.info("  [LIBRA] Raw-score AUC = %.3f  95%% CI [%.3f, %.3f]", libra_auc, lo, hi)

    oof_df = pd.DataFrame({"group": groups_cv, "y_true": y_cv})
    for name, res in results.items():
        oof_df[f"oof_{name.lower().replace('+', '_')}"] = res["oof_scores"]
    oof_df.to_csv(f"{output_dir}/oof_predictions.csv", index=False)

    summary_df = pd.DataFrame(
        [
            {
                "cohort": cohort,
                "strategy": COHORTS.get(cohort, "?"),
                "model": name,
                "estimator": "LIBRA raw score" if name == "LIBRA" else "TabPFN",
                "n_primary_pairs": n_cv_pairs,
                "bmca_audit": Path(bmca_audit).name if bmca_audit else "none",
                "mrf_audit": Path(mrf_audit).name if mrf_audit else "none",
                "oof_auc": round(res["oof_auc"], 4),
                "ci_low_95": round(res["ci_low"], 4),
                "ci_high_95": round(res["ci_high"], 4),
                "fold_aucs": str([round(a, 3) for a in res["fold_aucs"]]),
            }
            for name, res in results.items()
        ]
    )
    summary_df.to_csv(f"{output_dir}/tabpfn_cv_summary.csv", index=False)
    logger.info("\n%s", summary_df.drop(columns=["strategy"]).to_string(index=False))

    bootstrap_rows = []
    for name_a, name_b in [("BMCA+MRF", "BMCA"), ("MRF", "BMCA"), ("LIBRA", "BMCA")]:
        if name_a in results and name_b in results:
            bt = _bootstrap_paired_auc_diff(
                y_cv,
                results[name_a]["oof_scores"],
                results[name_b]["oof_scores"],
                groups_cv,
                n_boot=10000,
                seed=seed,
            )
            bootstrap_rows.append(
                {
                    "cohort": cohort,
                    "comparison": f"{name_a} vs {name_b}",
                    "observed_diff": round(bt["observed_diff"], 4),
                    "ci_low_95": round(bt["ci_low"], 4),
                    "ci_high_95": round(bt["ci_high"], 4),
                    "p_value": round(bt["p_value"], 4),
                    "n_boot": bt["n_boot"],
                }
            )
            logger.info(
                "  %s vs %s: delta = %+.4f  95%% CI [%+.4f, %+.4f]  p(delta<=0) = %.4f",
                name_a,
                name_b,
                bt["observed_diff"],
                bt["ci_low"],
                bt["ci_high"],
                bt["p_value"],
            )
    if bootstrap_rows:
        pd.DataFrame(bootstrap_rows).to_csv(f"{output_dir}/bootstrap_paired_diff.csv", index=False)

    return summary_df


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="TabPFN nested-CV evaluation")
    parser.add_argument(
        "--cohort",
        default="strate",
        choices=[*COHORTS, "all"],
        help="Labeling-strategy cohort to evaluate, or 'all'.",
    )
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--output_root", default="results_tabpfn")
    parser.add_argument("--n_outer", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto", help="auto | cpu | mps | cuda")
    parser.add_argument(
        "--training_mode",
        default="combined",
        choices=["combined", "primary_only", "augmentation_only"],
    )
    parser.add_argument("--no_audit", action="store_true", help="Ignore column audit files.")
    args = parser.parse_args()

    cohorts = list(COHORTS) if args.cohort == "all" else [args.cohort]
    summaries = []
    for cohort in cohorts:
        summaries.append(
            run_cohort(
                cohort=cohort,
                data_dir=args.data_dir,
                output_dir=f"{args.output_root}/{cohort}",
                n_outer=args.n_outer,
                seed=args.seed,
                device=args.device,
                training_mode=args.training_mode,
                use_audit=not args.no_audit,
            )
        )

    combined = pd.concat(summaries, ignore_index=True)
    Path(args.output_root).mkdir(parents=True, exist_ok=True)
    combined.to_csv(f"{args.output_root}/tabpfn_all_cohorts.csv", index=False)
    logger.info("\n%s\nALL COHORTS\n%s", "=" * 70, "=" * 70)
    logger.info("\n%s", combined.drop(columns=["strategy"]).to_string(index=False))


if __name__ == "__main__":
    main()
