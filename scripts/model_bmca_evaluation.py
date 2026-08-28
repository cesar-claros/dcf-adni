"""
BMCA Model Training and Evaluation
====================================

Trains a CatBoost classifier on the BMCA (Biomarker / Medical / Cognitive
Assessment) feature set produced by dcf_adni/preprocessing/feature_exports.py,
using the shared single-split protocol: Optuna TPE tuning with
StratifiedGroupKFold(5) inner CV (groups = matched pair IDs), evaluation on
the primary held-out test set (evaluation_eligible == 1), and a matched-pair
bootstrap 95% CI. See dcf_adni.modeling.protocols.run_single_split.

CatBoost is used without WoE pre-transformation: it handles continuous,
binary, and ordinal features natively and tolerates missing values internally
via its default NaN treatment.

One caveat: the column audit retained ``baseline_diagnosis`` despite a large
train/test mode shift (67% → 100%). This variable equals 1 for all primary
subjects (CN at baseline) but varies for augmentation subjects in train.
CatBoost may use it as a proxy for analysis_set membership. Inspect feature
importances after training to assess its influence.

Usage::

    python scripts/model_bmca_evaluation.py
    python scripts/model_bmca_evaluation.py --n_iter 100 --seed 42 --n_jobs 4
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dcf_adni.modeling.cli import single_split_argparser
from dcf_adni.modeling.datasets import MatchedDataset
from dcf_adni.modeling.protocols import run_single_split

logging.basicConfig(level=logging.INFO, format="%(name)s — %(message)s")


def main() -> None:
    args = single_split_argparser(
        description="Train and evaluate a CatBoost model on BMCA features",
        train_default="data/adni_bmca_features_train.csv",
        test_default="data/adni_bmca_features_test.csv",
    ).parse_args()

    dataset = MatchedDataset.from_csv(args.train, args.test)
    result = run_single_split(
        dataset,
        name="bmca",
        display_name="BMCA",
        n_iter=args.n_iter,
        n_splits=args.n_splits,
        n_boot=args.n_boot,
        seed=args.seed,
        n_jobs=args.n_jobs,
        gpu=args.gpu,
    )
    result.save_bundle(args.output_dir, args.plots_dir, color="steelblue")


if __name__ == "__main__":
    main()
