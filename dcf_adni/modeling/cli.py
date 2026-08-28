"""Shared command-line interface for single-split experiment scripts.

Flag names, types, and defaults replicate the pre-harness scripts so existing
invocations and documentation keep working.
"""

from __future__ import annotations

import argparse


def single_split_argparser(
    *,
    description: str,
    train_default: str,
    test_default: str,
    output_dir_default: str = "results",
    plots_dir_default: str = "plots",
) -> argparse.ArgumentParser:
    """Build the standard argument parser for a single-split experiment."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--train", default=train_default)
    parser.add_argument("--test", default=test_default)
    parser.add_argument("--output_dir", default=output_dir_default)
    parser.add_argument("--plots_dir", default=plots_dir_default)
    parser.add_argument("--n_iter", type=int, default=50,
                        help="Number of Optuna trials (default: 50)")
    parser.add_argument("--n_splits", type=int, default=5,
                        help="Number of inner CV folds (default: 5)")
    parser.add_argument("--n_boot", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n_jobs", type=int, default=-1)
    parser.add_argument("--gpu", action="store_true", default=False)
    return parser
