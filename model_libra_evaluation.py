"""
LIBRA Evaluation
================

Evaluates the LIBRA-like score as a predictor of CN → MCI/dementia transition
using the matched cohort produced by data_preprocessing_feature_exports.py.

Two estimators:

1. Raw-score AUC — the LIBRA score is used directly as a decision function
   (no fitting). This matches how LIBRA is reported in the validation
   literature and provides the primary reference point.

2. Conditional logistic regression — fits a single-predictor matched-pair
   model on the primary training pairs, then evaluates on the primary test
   pairs. For 1:1 matched pairs, the conditional likelihood reduces to a
   standard logistic regression of ones on the within-pair score differences
   (Δx = score_transitioner − score_control), so no external solver is needed.
   The conditional model gives a calibrated within-pair log-odds estimate and
   a p-value for the LIBRA coefficient.

Note on AUC equivalence: for a single predictor, the raw-score AUC and the
conditional logit AUC are identical when the fitted coefficient is positive
(AUC is invariant to monotone transformations of the score). The value of the
conditional logit is the within-pair log-odds ratio estimate and its inference,
not an improvement in ranking.

Bootstrap 95% CIs (1 000 resamples by default) are reported for the AUC.
Resampling is at the matched-pair (group) level to respect the pair structure.

Usage::

    python model_libra_evaluation.py
    python model_libra_evaluation.py --train data/adni_libra_train.csv \\
        --test data/adni_libra_test.csv --n_boot 2000 --seed 42
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import norm
from sklearn.metrics import RocCurveDisplay, roc_auc_score
from statsmodels.discrete.conditional_models import ConditionalLogit

logging.basicConfig(level=logging.INFO, format="%(name)s — %(message)s")
logger = logging.getLogger(__name__)

SCORE_COL = "libra_supported_rescaled_0_100"
LABEL_COL = "transition_label"
GROUP_COL = "group"
SUBJECT_ID_COL = "subject_id"


# =============================================================================
# Data helpers
# =============================================================================


def _primary_pairs(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["analysis_set"] == "primary"].copy()


def _evaluation_eligible(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["evaluation_eligible"] == 1].copy()


# =============================================================================
# Bootstrap AUC CI
# =============================================================================


def _bootstrap_auc(
    y_true: np.ndarray,
    y_score: np.ndarray,
    groups: np.ndarray,
    n_boot: int = 1000,
    seed: int = 0,
) -> tuple[float, float]:
    """
    Bootstrap 95% CI for AUC by resampling matched pairs (groups) with
    replacement. Returns (lower, upper).
    """
    rng = np.random.default_rng(seed)
    unique_groups = np.unique(groups)
    boot_aucs = []
    for _ in range(n_boot):
        sampled = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        idx = np.concatenate([np.where(groups == g)[0] for g in sampled])
        y_b, s_b = y_true[idx], y_score[idx]
        if len(np.unique(y_b)) < 2:
            continue
        boot_aucs.append(roc_auc_score(y_b, s_b))
    boot_aucs = np.array(boot_aucs)
    return float(np.percentile(boot_aucs, 2.5)), float(np.percentile(boot_aucs, 97.5))


# =============================================================================
# Raw-score evaluation (no fitting)
# =============================================================================


def evaluate_raw_score(
    test_df: pd.DataFrame,
    n_boot: int = 1000,
    seed: int = 0,
) -> dict:
    """
    Evaluate the raw LIBRA score as a decision function on the primary test
    set. No model is fitted; the score itself is the ranking function.
    """
    eligible = _evaluation_eligible(test_df)
    y = eligible[LABEL_COL].values.astype(float)
    score = eligible[SCORE_COL].values.astype(float)
    groups = eligible[GROUP_COL].values

    auc = roc_auc_score(y, score)
    ci_low, ci_high = _bootstrap_auc(y, score, groups, n_boot=n_boot, seed=seed)

    logger.info(
        f"Raw LIBRA score  AUC = {auc:.3f}  "
        f"95% CI [{ci_low:.3f}, {ci_high:.3f}]"
    )
    return {
        "estimator": "raw_score",
        "auc": auc,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "coef": np.nan,
        "se": np.nan,
        "or_": np.nan,
        "or_ci_low": np.nan,
        "or_ci_high": np.nan,
        "pvalue": np.nan,
        "y_true": y,
        "y_score": score,
        "groups": groups,
    }


# =============================================================================
# Conditional logistic regression (1:1 matched pairs)
# =============================================================================


def fit_conditional_logit(train_df: pd.DataFrame) -> object:
    """
    Fit a conditional logistic regression on primary matched training pairs.

    For 1:1 matched pairs, all pairs are discordant by construction (every
    pair contains exactly one transitioner and one stable-CN subject), so all
    143 primary training pairs contribute to the likelihood.

    Uses statsmodels ConditionalLogit, which conditions on the sum of outcomes
    within each matched group and is the standard estimator for matched
    case-control data.
    """
    primary = _primary_pairs(train_df).dropna(subset=[SCORE_COL, LABEL_COL])

    model = ConditionalLogit(
        endog=primary[LABEL_COL].astype(float),
        exog=primary[[SCORE_COL]].astype(float),
        groups=primary[GROUP_COL],
    )
    result = model.fit(disp=False)
    logger.info(f"\nConditional logistic regression summary:\n{result.summary()}")
    return result


def evaluate_conditional_logit(
    result: object,
    test_df: pd.DataFrame,
    n_boot: int = 1000,
    seed: int = 0,
) -> dict:
    """
    Score the primary test set using the fitted conditional logit and compute
    AUC with bootstrap CI.

    Because conditional logit conditions out the baseline odds, absolute
    probabilities are not identified. The linear predictor (score × coefficient)
    is used as the ranking function for AUC — this is a monotone transformation
    of the raw score, so the AUC equals the raw-score AUC when the coefficient
    is positive (as expected for a risk score).
    """
    eligible = _evaluation_eligible(test_df)
    y = eligible[LABEL_COL].values.astype(float)
    score = eligible[SCORE_COL].values.astype(float)
    groups = eligible[GROUP_COL].values

    coef = float(result.params[SCORE_COL])
    se = float(result.bse[SCORE_COL])
    pvalue = float(result.pvalues[SCORE_COL])
    or_ = float(np.exp(coef))
    z975 = norm.ppf(0.975)
    or_ci_low = float(np.exp(coef - z975 * se))
    or_ci_high = float(np.exp(coef + z975 * se))

    linear_predictor = score * coef
    auc = roc_auc_score(y, linear_predictor)
    ci_low, ci_high = _bootstrap_auc(y, linear_predictor, groups, n_boot=n_boot, seed=seed)

    logger.info(
        f"Conditional logit  AUC = {auc:.3f}  "
        f"95% CI [{ci_low:.3f}, {ci_high:.3f}]"
    )
    logger.info(
        f"  Coefficient = {coef:.4f}  SE = {se:.4f}  "
        f"OR = {or_:.3f}  95% CI [{or_ci_low:.3f}, {or_ci_high:.3f}]  "
        f"p = {pvalue:.4f}"
    )
    return {
        "estimator": "conditional_logit",
        "auc": auc,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "coef": coef,
        "se": se,
        "or_": or_,
        "or_ci_low": or_ci_low,
        "or_ci_high": or_ci_high,
        "pvalue": pvalue,
        "y_true": y,
        "y_score": linear_predictor,
        "groups": groups,
    }


# =============================================================================
# Output
# =============================================================================


def plot_roc(
    raw_result: dict,
    clogit_result: dict,
    output_path: str,
) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))

    RocCurveDisplay.from_predictions(
        raw_result["y_true"],
        raw_result["y_score"],
        ax=ax,
        name=f"Raw LIBRA score (AUC = {raw_result['auc']:.3f})",
        color="steelblue",
    )
    RocCurveDisplay.from_predictions(
        clogit_result["y_true"],
        clogit_result["y_score"],
        ax=ax,
        name=f"Conditional logit (AUC = {clogit_result['auc']:.3f})",
        color="darkorange",
        plot_chance_level=True,
    )

    n_pairs = int(raw_result["y_true"].sum())
    ax.set_title(f"LIBRA — ROC Curve  (primary test set, n = {n_pairs} pairs)")
    ax.set_xlabel("1 − Specificity (FPR)")
    ax.set_ylabel("Sensitivity (TPR)")
    ax.grid(True, which="both", alpha=0.3)
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"ROC plot saved to {output_path}")


def save_results(
    raw_result: dict,
    clogit_result: dict,
    output_path: str,
) -> None:
    rows = []
    for r in [raw_result, clogit_result]:
        rows.append(
            {
                "estimator": r["estimator"],
                "auc": round(r["auc"], 4),
                "auc_ci_low_95": round(r["ci_low"], 4),
                "auc_ci_high_95": round(r["ci_high"], 4),
                "coef": r["coef"],
                "se": r["se"],
                "odds_ratio": r["or_"],
                "or_ci_low_95": r["or_ci_low"],
                "or_ci_high_95": r["or_ci_high"],
                "pvalue": r["pvalue"],
            }
        )
    pd.DataFrame(rows).to_csv(output_path, index=False)
    logger.info(f"Results saved to {output_path}")


# =============================================================================
# Entry point
# =============================================================================


def run(
    train_path: str = "data/adni_libra_train.csv",
    test_path: str = "data/adni_libra_test.csv",
    output_dir: str = "results",
    plots_dir: str = "plots",
    n_boot: int = 1000,
    seed: int = 0,
) -> dict:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    Path(plots_dir).mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    raw_result = evaluate_raw_score(test_df, n_boot=n_boot, seed=seed)
    clogit_model = fit_conditional_logit(train_df)
    clogit_result = evaluate_conditional_logit(clogit_model, test_df, n_boot=n_boot, seed=seed)

    plot_roc(raw_result, clogit_result, output_path=f"{plots_dir}/libra_roc.pdf")
    save_results(raw_result, clogit_result, output_path=f"{output_dir}/libra_evaluation.csv")

    return {
        "raw": raw_result,
        "conditional_logit": clogit_result,
        "model": clogit_model,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate LIBRA score on the matched ADNI cohort"
    )
    parser.add_argument("--train", default="data/adni_libra_train.csv")
    parser.add_argument("--test", default="data/adni_libra_test.csv")
    parser.add_argument("--output_dir", default="results")
    parser.add_argument("--plots_dir", default="plots")
    parser.add_argument("--n_boot", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    run(
        train_path=args.train,
        test_path=args.test,
        output_dir=args.output_dir,
        plots_dir=args.plots_dir,
        n_boot=args.n_boot,
        seed=args.seed,
    )
