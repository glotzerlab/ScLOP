"""Stage 4: cross-validated random-forest classification."""

from typing import Callable

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold


def build_feature_matrix(
    df: pd.DataFrame,
    label_column: str,
    feature_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Drop label-missing rows, prune all-NaN columns, return ``(X, y)``."""
    df = df.dropna(subset=[label_column])

    if feature_columns is None:
        feature_columns = [
            c for c in df.columns
            if c != label_column and pd.api.types.is_numeric_dtype(df[c])
        ]

    X = df[feature_columns].copy()
    X = X.dropna(axis=1, how="all")
    y = df[label_column].to_numpy()
    return X, y


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
    X_arr = X.to_numpy()
    n = len(y)
    classes = np.unique(y)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    oof_pred = np.empty(n, dtype=y.dtype)
    oof_proba = np.zeros((n, len(classes)), dtype=float)
    fold_metrics: list[dict] = []
    importances = np.zeros(X.shape[1], dtype=float)

    for fold, (train_idx, test_idx) in enumerate(skf.split(X_arr, y)):
        X_train = pd.DataFrame(X_arr[train_idx], columns=X.columns)
        X_test = pd.DataFrame(X_arr[test_idx], columns=X.columns)

        # RF can't handle NaN — impute with train-fold column means.
        col_means = X_train.mean(axis=0)
        X_train = X_train.fillna(col_means).fillna(0.0)
        X_test = X_test.fillna(col_means).fillna(0.0)

        clf = RandomForestClassifier(random_state=random_state, n_jobs=-1)
        clf.fit(X_train.to_numpy(), y[train_idx])

        pred = clf.predict(X_test.to_numpy())
        proba = clf.predict_proba(X_test.to_numpy())

        oof_pred[test_idx] = pred
        # Align proba columns to global class order.
        for j, c in enumerate(classes):
            if c in clf.classes_:
                k = list(clf.classes_).index(c)
                oof_proba[test_idx, j] = proba[:, k]

        fold_metrics.append({
            "fold": fold,
            **evaluate(y[test_idx], pred,
                       y_proba=proba[:, 1] if len(clf.classes_) == 2 else proba),
        })
        importances += clf.feature_importances_

    importances /= n_splits
    feature_importances = pd.Series(importances, index=X.columns).sort_values(
        ascending=False,
    )

    overall_proba = oof_proba[:, 1] if len(classes) == 2 else oof_proba
    overall = evaluate(y, oof_pred, y_proba=overall_proba)

    return {
        "classes": classes,
        "oof_pred": oof_pred,
        "oof_proba": oof_proba,
        "fold_metrics": pd.DataFrame(fold_metrics),
        "overall": overall,
        "feature_importances": feature_importances,
    }


def evaluate(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None = None,
) -> dict[str, float]:
    """Compute AUROC, accuracy, balanced accuracy, F1, and confusion matrix."""
    classes = np.unique(y_true)
    binary = len(classes) == 2

    auroc: float = float("nan")
    if y_proba is not None:
        try:
            if binary:
                auroc = float(roc_auc_score(y_true, y_proba))
            else:
                auroc = float(roc_auc_score(
                    y_true, y_proba, multi_class="ovr", average="macro",
                ))
        except ValueError:
            auroc = float("nan")

    return {
        "auroc": auroc,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(
            y_true, y_pred,
            average="binary" if binary else "macro",
        )),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=classes),
    }


def bootstrap_ci(
    metric: Callable,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None = None,
    n_bootstraps: int = 1000,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Percentile bootstrap confidence interval for ``metric``.

    ``metric`` is called as ``metric(y_true, y_pred)`` when ``y_proba`` is None,
    otherwise as ``metric(y_true, y_proba)`` (e.g. for AUROC).
    """
    rng = np.random.default_rng(42)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = len(y_true)
    scores: list[float] = []

    score_input = y_proba if y_proba is not None else y_pred
    score_input = np.asarray(score_input)

    for _ in range(n_bootstraps):
        idx = rng.integers(0, n, size=n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        try:
            s = float(metric(y_true[idx], score_input[idx]))
        except ValueError:
            continue
        scores.append(s)

    if not scores:
        return float("nan"), float("nan")

    lo = float(np.quantile(scores, alpha / 2))
    hi = float(np.quantile(scores, 1 - alpha / 2))
    return lo, hi
