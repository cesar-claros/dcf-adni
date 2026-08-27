"""Canonical repository locations for inputs and outputs.

Output paths are anchored at the repository root so scripts write to the same
place regardless of the current working directory. Input CSVs under ``data/``
stay CWD-relative in the scripts (a wrong CWD fails loudly on read, whereas an
unanchored output directory would scatter silently).

Layout rule from the 2026-08 consolidation: the former flat directory
``results_<name>`` is now ``results/<name>`` (same for ``plots_<name>``), and
legacy model artifacts live in ``results/models``.
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
RESULTS_DIR = REPO_ROOT / "results"
PLOTS_DIR = REPO_ROOT / "plots"
MODELS_DIR = RESULTS_DIR / "models"


def results_dir(name: str = "") -> Path:
    """Return ``results/<name>`` (or ``results/`` itself), created if absent."""
    d = RESULTS_DIR / name if name else RESULTS_DIR
    d.mkdir(parents=True, exist_ok=True)
    return d


def plots_dir(name: str = "") -> Path:
    """Return ``plots/<name>`` (or ``plots/`` itself), created if absent."""
    d = PLOTS_DIR / name if name else PLOTS_DIR
    d.mkdir(parents=True, exist_ok=True)
    return d
