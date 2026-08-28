"""
BMCA+MRF Combined Model Training and Evaluation
================================================

Trains a CatBoost classifier on the union of the BMCA (Biomarker / Medical /
Cognitive Assessment) and MRF (Modifiable Risk Factor) feature sets produced
by dcf_adni/preprocessing/feature_exports.py, using the shared single-split
protocol (see dcf_adni.modeling.protocols.run_single_split).

The two feature tables are joined on ``subject_id``. Because the BMCA and MRF
preprocessing pipelines share the same cohort and subject-ID definitions, every
subject present in one table is present in the other. There are no overlapping
feature column names between the two sets, so no renaming or deduplication is
needed (MatchedDataset.merge enforces this).

Usage::

    python scripts/model_bmca_mrf_evaluation.py
    python scripts/model_bmca_mrf_evaluation.py --n_iter 100 --seed 42 --n_jobs 4
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dcf_adni.modeling.datasets import MatchedDataset
from dcf_adni.modeling.protocols import run_single_split

logging.basicConfig(level=logging.INFO, format="%(name)s — %(message)s")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train and evaluate a CatBoost model on combined BMCA+MRF features"
    )
    parser.add_argument("--bmca_train", default="data/adni_bmca_features_train.csv")
    parser.add_argument("--bmca_test", default="data/adni_bmca_features_test.csv")
    parser.add_argument("--mrf_train", default="data/adni_mrf_features_train.csv")
    parser.add_argument("--mrf_test", default="data/adni_mrf_features_test.csv")
    parser.add_argument("--output_dir", default="results")
    parser.add_argument("--plots_dir", default="plots")
    parser.add_argument("--n_iter", type=int, default=50,
                        help="Number of Optuna trials (default: 50)")
    parser.add_argument("--n_splits", type=int, default=5,
                        help="Number of inner CV folds (default: 5)")
    parser.add_argument("--n_boot", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n_jobs", type=int, default=-1)
    parser.add_argument("--gpu", action="store_true", default=False)
    args = parser.parse_args()

    bmca = MatchedDataset.from_csv(args.bmca_train, args.bmca_test)
    mrf = MatchedDataset.from_csv(args.mrf_train, args.mrf_test)
    dataset = bmca.merge(mrf)

    result = run_single_split(
        dataset,
        name="bmca_mrf",
        display_name="BMCA+MRF",
        n_iter=args.n_iter,
        n_splits=args.n_splits,
        n_boot=args.n_boot,
        seed=args.seed,
        n_jobs=args.n_jobs,
        gpu=args.gpu,
    )
    result.save_bundle(args.output_dir, args.plots_dir, color="forestgreen")


if __name__ == "__main__":
    main()
