"""Stage 4: cross-validated random-forest classification."""

import re
from typing import Callable, Iterable

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


def cross_validate_random_forest(
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
    per_fold_importances: list[np.ndarray] = []

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
            **classification_metrics(y[test_idx], pred,
                       y_proba=proba[:, 1] if len(clf.classes_) == 2 else proba),
        })
        per_fold_importances.append(clf.feature_importances_)

    imp_matrix = np.vstack(per_fold_importances)
    feature_importances = pd.Series(
        imp_matrix.mean(axis=0), index=X.columns,
    ).sort_values(ascending=False)
    feature_importances_table = pd.DataFrame({
        "feature": X.columns,
        "mean_importance": imp_matrix.mean(axis=0),
        "std_importance": imp_matrix.std(axis=0, ddof=1) if n_splits > 1 else 0.0,
    }).sort_values("mean_importance", ascending=False).reset_index(drop=True)

    overall_proba = oof_proba[:, 1] if len(classes) == 2 else oof_proba
    overall = classification_metrics(y, oof_pred, y_proba=overall_proba)

    return {
        "classes": classes,
        "oof_pred": oof_pred,
        "oof_proba": oof_proba,
        "fold_metrics": pd.DataFrame(fold_metrics),
        "overall": overall,
        "feature_importances": feature_importances,
        "feature_importances_table": feature_importances_table,
    }


def classification_metrics(
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

    f1_kwargs = {"average": "binary", "pos_label": classes[-1]} if binary \
                else {"average": "macro"}
    return {
        "auroc": auroc,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, **f1_kwargs)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=classes),
    }


def bootstrap_confidence_interval(
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


def cross_validate_random_forest_with_ci(
    X: pd.DataFrame,
    y: np.ndarray,
    n_splits: int = 5,
    random_state: int = 42,
    n_bootstraps: int = 1000,
    alpha: float = 0.05,
) -> dict:
    """One-call CV + bootstrap CIs.

    Runs stratified k-fold RF, then computes percentile bootstrap CIs on the
    pooled out-of-fold predictions for accuracy, balanced accuracy, F1, and
    (binary only) AUROC. Returns a flat dict suitable for writing to a single
    CSV row, alongside the full ``cross_validate_random_forest`` result.
    """
    cv = cross_validate_random_forest(X, y, n_splits=n_splits, random_state=random_state)

    y_pred = cv["oof_pred"]
    classes = cv["classes"]
    binary = len(classes) == 2
    y_proba = cv["oof_proba"][:, 1] if binary else cv["oof_proba"]

    overall = {k: v for k, v in cv["overall"].items() if k != "confusion_matrix"}

    pos_label = classes[-1] if binary else None
    metric_specs: list[tuple[str, Callable, bool]] = [
        ("accuracy", accuracy_score, False),
        ("balanced_accuracy", balanced_accuracy_score, False),
        ("f1", (lambda yt, yp: f1_score(yt, yp,
                                        average="binary", pos_label=pos_label,
                                        zero_division=0))
                if binary else
                (lambda yt, yp: f1_score(yt, yp, average="macro",
                                         zero_division=0)),
         False),
    ]
    if binary:
        metric_specs.append(("auroc", roc_auc_score, True))

    cis: dict[str, float] = {}
    for name, func, use_proba in metric_specs:
        lo, hi = bootstrap_confidence_interval(
            func, y, y_pred,
            y_proba=y_proba if use_proba else None,
            n_bootstraps=n_bootstraps, alpha=alpha,
        )
        cis[f"{name}_ci_lower"] = lo
        cis[f"{name}_ci_upper"] = hi

    return {
        "metrics": overall,
        "cis": cis,
        "summary": {**overall, **cis},
        "cv": cv,
    }


# ── feature-subset selectors ──────────────────────────────────────────────────


def extract_psi_k(column: str) -> int | None:
    """Pull the integer ``k`` out of a ``psi_k_*`` column name, or None."""
    m = re.search(r"psi_(\d+)", column.lower())
    return int(m.group(1)) if m else None


def select_psi_columns(columns: Iterable[str]) -> list[str]:
    """All ``psi_*`` columns."""
    return [c for c in columns if "psi" in c.lower()]


def select_density_columns(columns: Iterable[str]) -> list[str]:
    """All ``local_density*`` columns."""
    return [c for c in columns if "local_density" in c.lower()]


def select_by_cutoff(columns: Iterable[str], cutoff: float) -> list[str]:
    """Columns suffixed with ``_r{cutoff:g}`` (multi-radius sweep convention)."""
    suffix = f"_r{cutoff:g}"
    return [c for c in columns if c.endswith(suffix)]


def select_by_cell_type(columns: Iterable[str], cell_type: str) -> list[str]:
    """Columns naming a specific cell type (matches ``_<cell_type>`` as a token)."""
    pat = re.compile(rf"(?:^|_){re.escape(cell_type)}(?:_|$)")
    return [c for c in columns if pat.search(c)]


def select_by_psi_k(columns: Iterable[str], k: int) -> list[str]:
    """Columns for a single psi order ``k``."""
    return [c for c in columns if extract_psi_k(c) == k]


def build_feature_groups(
    columns: Iterable[str],
    *,
    psi_k_range: tuple[int, int] | None = (1, 10),
    cutoff: float | None = None,
    cell_types: Iterable[str] | None = None,
) -> dict[str, list[str]]:
    """Build named feature subsets for per-group RF experiments.

    Always returns: ``all``, ``density``, ``psi``, ``psi_<lo>_<hi>``, and one
    ``psi_<k>`` per k seen in ``columns``. Restrict to a single cutoff with
    ``cutoff=60.0``. Pass ``cell_types=[...]`` to also emit ``<celltype>``,
    ``<celltype>_density``, ``<celltype>_psi`` groups.
    """
    cols = list(columns)
    if cutoff is not None:
        cols = select_by_cutoff(cols, cutoff)

    psi = select_psi_columns(cols)
    den = select_density_columns(cols)

    groups: dict[str, list[str]] = {
        "all": psi + den,
        "density": den,
        "psi": psi,
    }

    if psi_k_range is not None:
        lo, hi = psi_k_range
        in_range = [c for c in psi
                    if (k := extract_psi_k(c)) is not None and lo <= k <= hi]
        if in_range:
            groups[f"psi_{lo}_{hi}"] = in_range

    ks = sorted({k for c in psi if (k := extract_psi_k(c)) is not None})
    for k in ks:
        cols_k = select_by_psi_k(psi, k)
        if cols_k:
            groups[f"psi_{k}"] = cols_k

    if cell_types:
        for ct in cell_types:
            ct_cols = select_by_cell_type(cols, ct)
            if not ct_cols:
                continue
            groups[ct] = ct_cols
            ct_den = [c for c in ct_cols if c in den]
            ct_psi = [c for c in ct_cols if c in psi]
            if ct_den:
                groups[f"{ct}_density"] = ct_den
            if ct_psi:
                groups[f"{ct}_psi"] = ct_psi

    return {name: feats for name, feats in groups.items() if feats}


def cross_validate_feature_groups(
    df: pd.DataFrame,
    label_column: str,
    groups: dict[str, list[str]],
    *,
    n_splits: int = 5,
    random_state: int = 42,
    n_bootstraps: int = 1000,
    alpha: float = 0.05,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Run :func:`cross_validate_random_forest_with_ci` on each named feature subset.

    Returns ``(summary_df, importances_by_group)``:
      * ``summary_df`` — one row per group with metrics + bootstrap CIs.
      * ``importances_by_group`` — ``group -> feature_importances_table``
        (mean + std impurity importance per feature) from each CV run.
    """
    rows: list[dict] = []
    importances: dict[str, pd.DataFrame] = {}

    for name, feats in groups.items():
        if not feats:
            continue
        X, y = build_feature_matrix(df, label_column, feature_columns=feats)
        if X.shape[1] == 0 or len(np.unique(y)) < 2:
            continue
        out = cross_validate_random_forest_with_ci(
            X, y,
            n_splits=n_splits, random_state=random_state,
            n_bootstraps=n_bootstraps, alpha=alpha,
        )
        rows.append({"group": name, "n_features": X.shape[1], **out["summary"]})
        importances[name] = out["cv"]["feature_importances_table"]

    summary_df = pd.DataFrame(rows)
    return summary_df, importances
