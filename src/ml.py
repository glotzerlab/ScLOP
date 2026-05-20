"""Stage 4: cross-validated random-forest classification."""

from typing import Callable

import numpy as np
import pandas as pd


def build_feature_matrix(
    df: pd.DataFrame,
    label_column: str,
    feature_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Drop label-missing rows, prune all-NaN columns, return ``(X, y)``."""
    raise NotImplementedError


def cross_validate(
    X: pd.DataFrame,
    y: np.ndarray,
    n_splits: int = 5,
    random_state: int = 42,
) -> dict:
    """Stratified k-fold CV with a random-forest classifier.

    Returns out-of-fold predictions, per-fold metrics (AUROC, accuracy,
    balanced accuracy, F1), and averaged impurity-based feature importances.
    """
    raise NotImplementedError


def evaluate(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None = None,
) -> dict[str, float]:
    """Compute AUROC, accuracy, balanced accuracy, F1, and confusion matrix."""
    raise NotImplementedError


def bootstrap_ci(
    metric: Callable,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None = None,
    n_bootstraps: int = 1000,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Percentile bootstrap confidence interval for ``metric``."""
    raise NotImplementedError
