"""Matched-cohort dataset loading for the experiment harness."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

from dcf_adni.modeling.schema import feature_cols

logger = logging.getLogger(__name__)


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

    def merge(self, other: "MatchedDataset") -> "MatchedDataset":
        """Join two datasets on ``subject_id``, keeping metadata from ``self``.

        Feature names must be disjoint (the BMCA and MRF pipelines guarantee
        this by construction); the merged feature order is ``self`` features
        followed by ``other`` features.
        """
        overlap = set(self.feature_cols) & set(other.feature_cols)
        if overlap:
            raise ValueError(
                f"Overlapping feature columns between the two datasets: {sorted(overlap)}. "
                "Resolve duplicates before merging."
            )

        def _merge_split(a: pd.DataFrame, b: pd.DataFrame, label: str) -> pd.DataFrame:
            merged = a.merge(
                b[["subject_id"] + list(other.feature_cols)],
                on="subject_id",
                how="inner",
                validate="1:1",
            )
            n_dropped = len(a) - len(merged)
            if n_dropped:
                logger.warning(
                    f"{n_dropped} {label} subjects dropped during merge "
                    "(present in one table but not the other)."
                )
            return merged

        train = _merge_split(self.train, other.train, "train")
        test = _merge_split(self.test, other.test, "test")
        return MatchedDataset(train=train, test=test, feature_cols=tuple(feature_cols(train)))
