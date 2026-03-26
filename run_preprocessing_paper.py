"""
Preprocess ADNI data for all labeling strategies used in the paper.

Loads the raw ADNI wide table once, then runs the full preprocessing pipeline
for each of the 6 labeling strategies (L1-L6) with relaxed audit thresholds.

Usage::

    python run_preprocessing_paper.py
    python run_preprocessing_paper.py --strategies L1,L4
    python run_preprocessing_paper.py --input_csv data/custom.csv --output_dir data
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from data_preprocessing_feature_exports import (
    FeatureExportConfig,
    _append_suffix,
    _write_matching_outputs,
    _write_split_ready_outputs,
    build_feature_tables_and_matches_from_wide,
)

logging.basicConfig(level=logging.INFO, format="%(name)s — %(message)s")
logger = logging.getLogger(__name__)

STRATEGIES: dict[str, dict] = {
    "L1": {
        "desc": "default",
    },
    "L2": {
        "desc": "no_reverters",
        "exclude_mci_reverters": True,
        "exclude_dementia_reverters": True,
    },
    "L3": {
        "desc": "confirmed",
        "min_consecutive_impaired_visits": 2,
    },
    "L4": {
        "desc": "clean",
        "exclude_mci_reverters": True,
        "exclude_dementia_reverters": True,
        "min_consecutive_impaired_visits": 2,
    },
    "L5": {
        "desc": "dementia_only",
        "transition_target": "dementia_only",
    },
    "L6": {
        "desc": "mci_only",
        "transition_target": "mci_only",
    },
}


def _build_config(strategy_id: str, strategy: dict, output_dir: str) -> FeatureExportConfig:
    desc = strategy["desc"]
    labeling_kwargs = {k: v for k, v in strategy.items() if k != "desc"}
    return FeatureExportConfig(
        output_dir=output_dir,
        bmca_output_name=f"adni_bmca_features_{strategy_id}.csv",
        mrf_output_name=f"adni_mrf_features_{strategy_id}.csv",
        libra_output_name=f"adni_libra_{strategy_id}.csv",
        output_prefix=f"{strategy_id}_{desc}",
        # Relaxed audit thresholds for CatBoost's native NaN handling
        max_missing_fraction=0.9,
        max_mode_fraction=0.98,
        # Write all output types
        write_primary_attached_features=True,
        write_combined_attached_features=True,
        write_split_ready_features=True,
        **labeling_kwargs,
    )


def _write_results(
    results: dict[str, pd.DataFrame], out_dir: Path, config: FeatureExportConfig
) -> None:
    """Write all output CSVs, mirroring build_feature_tables_and_matches_from_csv."""
    out_dir.mkdir(parents=True, exist_ok=True)

    results["libra_df"].to_csv(out_dir / config.libra_output_name, index=False)
    results["mrf_df"].to_csv(out_dir / config.mrf_output_name, index=False)
    results["bmca_df"].to_csv(out_dir / config.bmca_output_name, index=False)

    _write_matching_outputs(results, out_dir, config.output_prefix)

    if config.write_primary_attached_features:
        for prefix, output_name in [
            ("libra", config.libra_output_name),
            ("mrf", config.mrf_output_name),
            ("bmca", config.bmca_output_name),
        ]:
            key = f"{prefix}_matched_df"
            if key in results:
                results[key].to_csv(
                    out_dir / _append_suffix(output_name, "_matched"), index=False
                )

    if config.write_combined_attached_features:
        for prefix, output_name in [
            ("libra", config.libra_output_name),
            ("mrf", config.mrf_output_name),
            ("bmca", config.bmca_output_name),
        ]:
            key = f"{prefix}_combined_matched_df"
            if key in results:
                results[key].to_csv(
                    out_dir / _append_suffix(output_name, "_combined_matched"), index=False
                )

    if config.write_split_ready_features:
        _write_split_ready_outputs(results, out_dir, config)


def run(
    input_csv: str = "data/All_Subjects_My_Table_11Mar2026.csv",
    output_dir: str = "data",
    strategies: list[str] | None = None,
) -> dict[str, dict]:
    """Run preprocessing for selected labeling strategies.

    Returns a dict mapping strategy_id -> {config, n_primary_pairs, n_aug_pairs}.
    """
    if strategies is None:
        strategies = list(STRATEGIES.keys())

    logger.info(f"Loading raw ADNI table from {input_csv} ...")
    raw_df = pd.read_csv(input_csv, low_memory=False)
    logger.info(f"  {len(raw_df)} rows, {len(raw_df.columns)} columns")

    out_dir = Path(output_dir)
    summary = {}

    for strategy_id in strategies:
        strategy = STRATEGIES[strategy_id]
        desc = strategy["desc"]
        config = _build_config(strategy_id, strategy, output_dir)

        logger.info(f"\n{'='*60}")
        logger.info(f"Strategy {strategy_id} ({desc})")
        logger.info(f"{'='*60}")

        results = build_feature_tables_and_matches_from_wide(raw_df.copy(), config=config)
        _write_results(results, out_dir, config)

        # Log cohort sizes
        combined_df = results.get("bmca_combined_matched_df")
        if combined_df is not None:
            n_primary = combined_df[combined_df["analysis_set"] == "primary"]["group"].nunique()
            n_aug = combined_df[combined_df["analysis_set"] == "augmentation"]["group"].nunique()
        else:
            n_primary = n_aug = 0

        logger.info(f"  Primary pairs: {n_primary}")
        logger.info(f"  Augmentation pairs: {n_aug}")
        logger.info(
            f"  Output: {_append_suffix(config.bmca_output_name, '_combined_matched')}, "
            f"{_append_suffix(config.mrf_output_name, '_combined_matched')}"
        )

        summary[strategy_id] = {
            "desc": desc,
            "config": config,
            "n_primary_pairs": n_primary,
            "n_augmentation_pairs": n_aug,
        }

    # Write cohort summary
    summary_rows = []
    for sid, info in summary.items():
        summary_rows.append({
            "strategy": sid,
            "description": info["desc"],
            "n_primary_pairs": info["n_primary_pairs"],
            "n_augmentation_pairs": info["n_augmentation_pairs"],
        })
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out_dir / "paper_preprocessing_summary.csv", index=False)
    logger.info(f"\n{summary_df.to_string(index=False)}")

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess ADNI data for paper experiments")
    parser.add_argument(
        "--input_csv",
        default="data/All_Subjects_My_Table_11Mar2026.csv",
    )
    parser.add_argument("--output_dir", default="data")
    parser.add_argument(
        "--strategies",
        default=None,
        help="Comma-separated strategy IDs (e.g. L1,L4). Default: all.",
    )
    args = parser.parse_args()

    strategy_list = args.strategies.split(",") if args.strategies else None
    run(input_csv=args.input_csv, output_dir=args.output_dir, strategies=strategy_list)
