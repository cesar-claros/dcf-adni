"""Result container and export bundle for single-split experiments.

The output files and their contents replicate the pre-harness experiment
scripts exactly (phase 3 acceptance gate): ``<name>_evaluation.csv``,
``<name>_feature_importance.csv``, ``<name>_roc.pdf``, ``<name>_model.joblib``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import RocCurveDisplay

logger = logging.getLogger(__name__)


@dataclass
class ExperimentResult:
    """One trained-and-evaluated experiment, ready to save."""

    name: str                 # output file stem, e.g. "bmca"
    display_name: str         # label used in logs and plot text, e.g. "BMCA"
    auc: float
    ci_low: float
    ci_high: float
    y_true: np.ndarray
    y_score: np.ndarray
    groups: np.ndarray
    importance: pd.DataFrame
    study: object             # optuna.Study from the tuning run
    model: object             # refitted best estimator
    feature_cols: list[str]

    def save_bundle(
        self, output_dir: str, plots_dir: str, *, color: str = "steelblue"
    ) -> None:
        """Write the metrics CSV, importance CSV, ROC plot, and model joblib."""
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        Path(plots_dir).mkdir(parents=True, exist_ok=True)

        metrics_df = pd.DataFrame(
            [
                {
                    "model": f"{self.name}_catboost",
                    "auc": round(self.auc, 4),
                    "auc_ci_low_95": round(self.ci_low, 4),
                    "auc_ci_high_95": round(self.ci_high, 4),
                    "best_inner_cv_auc": round(self.study.best_value, 4),
                    **{f"param_{k}": v for k, v in self.study.best_params.items()},
                }
            ]
        )
        metrics_df.to_csv(f"{output_dir}/{self.name}_evaluation.csv", index=False)
        self.importance.to_csv(
            f"{output_dir}/{self.name}_feature_importance.csv", index=False
        )
        logger.info(f"Results saved to {output_dir}/")

        plot_roc(self, f"{plots_dir}/{self.name}_roc.pdf", color=color)

        joblib.dump(
            {
                "model": self.model,
                "study": self.study,
                "feature_cols": self.feature_cols,
                "result": {
                    "auc": self.auc,
                    "ci_low": self.ci_low,
                    "ci_high": self.ci_high,
                    "y_true": self.y_true,
                    "y_score": self.y_score,
                    "groups": self.groups,
                },
            },
            f"{output_dir}/{self.name}_model.joblib",
        )
        logger.info(f"Model saved to {output_dir}/{self.name}_model.joblib")


def plot_roc(result: ExperimentResult, output_path: str, *, color: str) -> None:
    """ROC curve on the primary test set, in the house style."""
    fig, ax = plt.subplots(figsize=(6, 6))
    n_pairs = int(result.y_true.sum())
    RocCurveDisplay.from_predictions(
        result.y_true,
        result.y_score,
        ax=ax,
        name=f"{result.display_name} CatBoost (AUC = {result.auc:.3f})",
        color=color,
        plot_chance_level=True,
    )
    ax.set_title(
        f"{result.display_name} — ROC Curve  (primary test set, n = {n_pairs} pairs)"
    )
    ax.set_xlabel("1 − Specificity (FPR)")
    ax.set_ylabel("Sensitivity (TPR)")
    ax.grid(True, which="both", alpha=0.3)
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"ROC plot saved to {output_path}")
