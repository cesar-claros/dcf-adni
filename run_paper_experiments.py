"""
Master orchestrator for all paper experiments.

Runs experiment blocks and collects results into summary tables.

Usage::

    # Run individual blocks
    python run_paper_experiments.py --block labeling
    python run_paper_experiments.py --block training --strategy L4
    python run_paper_experiments.py --block transfer --strategy L4
    python run_paper_experiments.py --block bmca_comparison --strategy L4
    python run_paper_experiments.py --block summary

    # Run everything (labeling uses seed 0; training/transfer use seeds 0-4)
    python run_paper_experiments.py --block all --strategy L4
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(name)s — %(message)s")
logger = logging.getLogger(__name__)

RESULTS_BASE = "results_paper"
DATA_DIR = "data"

LABELING_STRATEGIES = ["L1", "L2", "L3", "L4", "L5", "L6"]
TRAINING_MODES = ["combined", "primary_only", "augmentation_only"]
SEEDS = list(range(5))


def _data_paths(strategy: str, data_dir: str = DATA_DIR) -> dict[str, str]:
    """Return data file paths for a given strategy."""
    return {
        "bmca": f"{data_dir}/adni_bmca_features_{strategy}_combined_matched.csv",
        "mrf": f"{data_dir}/adni_mrf_features_{strategy}_combined_matched.csv",
        "bmca_audit": f"{data_dir}/adni_bmca_features_{strategy}_column_audit.csv",
        "mrf_audit": f"{data_dir}/adni_mrf_features_{strategy}_column_audit.csv",
    }


def run_labeling(
    strategies: list[str] | None = None,
    n_iter: int = 50,
    n_jobs: int = 1,
    seed: int = 0,
) -> None:
    """Block 1: Compare labeling strategies (single seed)."""
    from run_preprocessing_paper import STRATEGIES

    from model_strate_cv_evaluation import run as run_cv

    if strategies is None:
        strategies = LABELING_STRATEGIES

    for strategy_id in strategies:
        desc = STRATEGIES[strategy_id]["desc"]
        output_dir = f"{RESULTS_BASE}/labeling/{strategy_id}_{desc}"
        paths = _data_paths(strategy_id)

        # Check data exists
        if not Path(paths["bmca"]).exists():
            logger.warning(
                f"Skipping {strategy_id}: {paths['bmca']} not found. "
                f"Run run_preprocessing_paper.py first."
            )
            continue

        audit_bmca = paths["bmca_audit"] if Path(paths["bmca_audit"]).exists() else None
        audit_mrf = paths["mrf_audit"] if Path(paths["mrf_audit"]).exists() else None

        logger.info(f"\n{'#'*60}")
        logger.info(f"# Block 1: Labeling strategy {strategy_id} ({desc})")
        logger.info(f"{'#'*60}")

        run_cv(
            bmca_path=paths["bmca"],
            mrf_path=paths["mrf"],
            output_dir=output_dir,
            n_iter=n_iter,
            seed=seed,
            n_jobs=n_jobs,
            bmca_audit=audit_bmca,
            mrf_audit=audit_mrf,
            training_mode="combined",
        )


def run_training(
    strategy: str = "L4",
    seeds: list[int] | None = None,
    n_iter: int = 50,
    n_jobs: int = 1,
) -> None:
    """Block 2: Training composition comparison (multi-seed)."""
    from model_strate_cv_evaluation import run as run_cv

    if seeds is None:
        seeds = SEEDS

    paths = _data_paths(strategy)
    audit_bmca = paths["bmca_audit"] if Path(paths["bmca_audit"]).exists() else None
    audit_mrf = paths["mrf_audit"] if Path(paths["mrf_audit"]).exists() else None

    for seed in seeds:
        for mode in TRAINING_MODES:
            output_dir = f"{RESULTS_BASE}/training/{mode}_seed{seed}"

            logger.info(f"\n{'#'*60}")
            logger.info(f"# Block 2: {mode}, seed {seed}")
            logger.info(f"{'#'*60}")

            run_cv(
                bmca_path=paths["bmca"],
                mrf_path=paths["mrf"],
                output_dir=output_dir,
                n_iter=n_iter,
                seed=seed,
                n_jobs=n_jobs,
                bmca_audit=audit_bmca,
                mrf_audit=audit_mrf,
                training_mode=mode,
            )


def run_transfer(
    strategy: str = "L4",
    seeds: list[int] | None = None,
    n_iter: int = 50,
    n_jobs: int = 1,
) -> None:
    """Block 2b: Cross-task transfer (multi-seed)."""
    from analysis_cross_task_transfer import run as run_xfer

    if seeds is None:
        seeds = SEEDS

    paths = _data_paths(strategy)
    audit_bmca = paths["bmca_audit"] if Path(paths["bmca_audit"]).exists() else None
    audit_mrf = paths["mrf_audit"] if Path(paths["mrf_audit"]).exists() else None

    for seed in seeds:
        output_dir = f"{RESULTS_BASE}/training/aug_to_primary_seed{seed}"

        logger.info(f"\n{'#'*60}")
        logger.info(f"# Block 2b: Cross-task transfer, seed {seed}")
        logger.info(f"{'#'*60}")

        run_xfer(
            bmca_path=paths["bmca"],
            mrf_path=paths["mrf"],
            output_dir=output_dir,
            n_iter=n_iter,
            seed=seed,
            n_jobs=n_jobs,
            bmca_audit=audit_bmca,
            mrf_audit=audit_mrf,
        )


def run_bmca_comparison(
    strategy: str = "L4",
    n_iter: int = 50,
    n_jobs: int = 1,
    seed: int = 0,
) -> None:
    """Block 3: BMCA feature comparison between pathways."""
    from analysis_bmca_feature_comparison import run as run_comp

    paths = _data_paths(strategy)
    audit_bmca = paths["bmca_audit"] if Path(paths["bmca_audit"]).exists() else None

    logger.info(f"\n{'#'*60}")
    logger.info(f"# Block 3: BMCA feature comparison")
    logger.info(f"{'#'*60}")

    run_comp(
        bmca_path=paths["bmca"],
        output_dir=f"{RESULTS_BASE}/bmca_comparison",
        bmca_audit=audit_bmca,
        n_iter=n_iter,
        seed=seed,
        n_jobs=n_jobs,
    )


def collect_summary() -> None:
    """Block 4: Collect results into summary tables."""
    results_dir = Path(RESULTS_BASE)
    summary_dir = results_dir / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)

    # --- Labeling comparison ---
    labeling_rows = []
    labeling_dir = results_dir / "labeling"
    if labeling_dir.exists():
        for subdir in sorted(labeling_dir.iterdir()):
            summary_file = subdir / "strate_cv_summary.csv"
            if summary_file.exists():
                df = pd.read_csv(summary_file)
                strategy_id = subdir.name.split("_")[0]
                desc = "_".join(subdir.name.split("_")[1:])
                for _, row in df.iterrows():
                    labeling_rows.append({
                        "strategy": strategy_id,
                        "description": desc,
                        "model": row["model"],
                        "oof_auc": row["oof_auc"],
                        "ci_low_95": row["ci_low_95"],
                        "ci_high_95": row["ci_high_95"],
                    })

            # Add bootstrap diff
            bt_file = subdir / "bootstrap_paired_diff.csv"
            if bt_file.exists():
                bt = pd.read_csv(bt_file)
                for _, brow in bt.iterrows():
                    if brow["comparison"] == "BMCA+MRF vs BMCA":
                        # Find matching row and add delta info
                        for lr in labeling_rows:
                            if lr["strategy"] == strategy_id and lr["model"] == "BMCA+MRF":
                                lr["delta_vs_bmca"] = brow["observed_diff"]
                                lr["delta_p_value"] = brow["p_value"]

    if labeling_rows:
        labeling_df = pd.DataFrame(labeling_rows)
        labeling_df.to_csv(summary_dir / "labeling_comparison.csv", index=False)
        logger.info(f"Labeling comparison:\n{labeling_df.to_string(index=False)}")

    # --- Training composition comparison ---
    training_rows = []
    training_dir = results_dir / "training"
    if training_dir.exists():
        for subdir in sorted(training_dir.iterdir()):
            if not subdir.name.startswith("aug_to_primary"):
                summary_file = subdir / "strate_cv_summary.csv"
            else:
                summary_file = subdir / "transfer_summary.csv"

            if not summary_file.exists():
                continue

            parts = subdir.name.rsplit("_seed", 1)
            if len(parts) != 2:
                continue
            mode, seed_str = parts
            seed = int(seed_str)

            df = pd.read_csv(summary_file)
            auc_col = "oof_auc" if "oof_auc" in df.columns else "transfer_auc"
            for _, row in df.iterrows():
                training_rows.append({
                    "mode": mode,
                    "seed": seed,
                    "model": row["model"],
                    "auc": row[auc_col],
                    "ci_low_95": row.get("ci_low_95", None),
                    "ci_high_95": row.get("ci_high_95", None),
                })

    if training_rows:
        training_df = pd.DataFrame(training_rows)
        training_df.to_csv(summary_dir / "training_comparison.csv", index=False)

        # Compute mean/std across seeds
        agg = (
            training_df.groupby(["mode", "model"])["auc"]
            .agg(["mean", "std", "count"])
            .round(4)
            .reset_index()
        )
        agg.to_csv(summary_dir / "training_comparison_aggregated.csv", index=False)
        logger.info(f"\nTraining composition (aggregated):\n{agg.to_string(index=False)}")

    # --- BMCA comparison ---
    comp_file = results_dir / "bmca_comparison" / "comparison_summary.csv"
    if comp_file.exists():
        comp = pd.read_csv(comp_file)
        logger.info(f"\nBMCA comparison:\n{comp.to_string(index=False)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Paper experiments orchestrator")
    parser.add_argument(
        "--block",
        required=True,
        choices=["labeling", "training", "transfer", "bmca_comparison", "summary", "all"],
    )
    parser.add_argument("--strategy", default="L4", help="Strategy for training/transfer/bmca blocks")
    parser.add_argument("--n_iter", type=int, default=50)
    parser.add_argument("--n_jobs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0, help="Seed for labeling block (single seed)")
    parser.add_argument(
        "--seeds",
        default=None,
        help="Comma-separated seeds for training/transfer blocks (default: 0,1,2,3,4)",
    )
    args = parser.parse_args()

    seed_list = [int(s) for s in args.seeds.split(",")] if args.seeds else None

    if args.block in ("labeling", "all"):
        run_labeling(n_iter=args.n_iter, n_jobs=args.n_jobs, seed=args.seed)

    if args.block in ("training", "all"):
        run_training(
            strategy=args.strategy, seeds=seed_list, n_iter=args.n_iter, n_jobs=args.n_jobs
        )

    if args.block in ("transfer", "all"):
        run_transfer(
            strategy=args.strategy, seeds=seed_list, n_iter=args.n_iter, n_jobs=args.n_jobs
        )

    if args.block in ("bmca_comparison", "all"):
        run_bmca_comparison(
            strategy=args.strategy, n_iter=args.n_iter, n_jobs=args.n_jobs, seed=args.seed
        )

    if args.block in ("summary", "all"):
        collect_summary()
