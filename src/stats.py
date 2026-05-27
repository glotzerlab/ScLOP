"""Stage 3: two-sample Kolmogorov-Smirnov test with multiple-comparisons correction."""

from typing import Iterable

import numpy as np
import pandas as pd
from scipy import stats as sps


def _clean_pair(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Drop NaN/inf entries from each sample independently."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    return a, b


def ks_test(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Two-sample KS test. Returns ``(statistic, p_value)``, or NaNs if too few finite values."""
    a, b = _clean_pair(a, b)
    if len(a) < 2 or len(b) < 2:
        return float("nan"), float("nan")
    r = sps.ks_2samp(a, b)
    return float(r.statistic), float(r.pvalue)


def ks_test_per_feature(
    df: pd.DataFrame,
    feature_columns: Iterable[str],
    group_column: str,
) -> pd.DataFrame:
    """Run a two-sample KS test on every feature column.

    Splits ``df`` into the two groups defined by ``group_column`` (must contain
    exactly two distinct values) and applies the KS test to every feature.

    Returns a long-format dataframe: ``feature``, ``statistic``, ``p_value``.
    """
    feature_columns = list(feature_columns)

    groups = sorted(df[group_column].dropna().unique())
    if len(groups) != 2:
        raise ValueError(
            f"group_column {group_column!r} must have exactly 2 values, found {groups}"
        )
    g1, g2 = groups
    df1 = df[df[group_column] == g1]
    df2 = df[df[group_column] == g2]

    rows = []
    for feature in feature_columns:
        stat, p = ks_test(df1[feature].values, df2[feature].values)
        rows.append({"feature": feature, "statistic": stat, "p_value": p})
    return pd.DataFrame(rows, columns=["feature", "statistic", "p_value"])


def adjust_pvalues(
    df: pd.DataFrame,
    pvalue_column: str = "p_value",
    methods: tuple[str, ...] = ("bh", "by"),
) -> pd.DataFrame:
    """Append Benjamini-Hochberg (bh) and/or Benjamini-Yekutieli (by) corrected columns.

    NaN p-values are preserved as NaN.
    """
    df = df.copy()
    unknown = set(methods) - {"bh", "by"}
    if unknown:
        raise ValueError(f"unknown correction method(s): {unknown}")

    pvals = df[pvalue_column]
    mask = pvals.notna()
    for method in methods:
        col = f"{pvalue_column}_{method}"
        out = pd.Series(np.nan, index=pvals.index, dtype=float)
        if mask.any():
            out.loc[mask] = sps.false_discovery_control(
                pvals[mask].values, method=method,
            )
        df[col] = out
    return df
