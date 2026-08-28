"""Matched-cohort dataset loading for the experiment harness."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from dcf_adni.modeling.schema import feature_cols


@dataclass(frozen=True)
class MatchedDataset:
    """Train/test feature tables plus the resolved feature columns.

    Feature columns are resolved from the training table (all columns minus
    metadata, optionally filtered by a column audit) and shared by both splits.
    """

    train: pd.DataFrame
    test: pd.DataFrame
    feature_cols: tuple[str, ...]

    @classmethod
    def from_csv(
        cls, train_path: str, test_path: str, *, audit_path: str | None = None
    ) -> "MatchedDataset":
        """Load a train/test CSV pair as exported by the preprocessing pipeline."""
        train = pd.read_csv(train_path)
        test = pd.read_csv(test_path)
        return cls(train=train, test=test, feature_cols=tuple(feature_cols(train, audit_path)))
