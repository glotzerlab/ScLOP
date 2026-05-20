"""Stage 3: univariate two-sample tests and multiple-comparisons correction."""

from typing import Iterable

import numpy as np
import pandas as pd
from scipy import stats as sps


# Supported test names.
TESTS = ("ks", "t", "mannwhitney", "cramervonmises", "median", "levene", "wasserstein")


def _clean_pair(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Drop NaN/inf entries from each sample independently."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    return a, b


def compare_distributions(
    a: np.ndarray,
    b: np.ndarray,
    test: str,
) -> tuple[float, float | None]:
    """Run a single two-sample test.

    Returns ``(statistic, p_value)``. For ``"wasserstein"`` the p-value is ``None``.
    Returns ``(nan, nan)`` (or ``(nan, None)`` for wasserstein) when either
    sample is empty or the test cannot run.
    """
    if test not in TESTS:
        raise ValueError(f"unknown test {test!r}; choose from {TESTS}")

    a, b = _clean_pair(a, b)
    if test == "wasserstein":
        if len(a) == 0 or len(b) == 0:
            return float("nan"), None
        return float(sps.wasserstein_distance(a, b)), None

    if len(a) < 2 or len(b) < 2:
        return float("nan"), float("nan")

    if test == "ks":
        r = sps.ks_2samp(a, b)
        return float(r.statistic), float(r.pvalue)
    if test == "t":
        r = sps.ttest_ind(a, b, equal_var=False, nan_policy="omit")
        return float(r.statistic), float(r.pvalue)
    if test == "mannwhitney":
        r = sps.mannwhitneyu(a, b)
        return float(r.statistic), float(r.pvalue)
    if test == "cramervonmises":
        r = sps.cramervonmises_2samp(a, b)
        return float(r.statistic), float(r.pvalue)
    if test == "median":
        stat, p, _, _ = sps.median_test(a, b)
        return float(stat), float(p)
    if test == "levene":
        r = sps.levene(a, b)
        return float(r.statistic), float(r.pvalue)
    raise AssertionError(f"unhandled test {test!r}")  # unreachable


def run_battery(
    df: pd.DataFrame,
    feature_columns: Iterable[str],
    group_column: str,
    tests: Iterable[str] = ("ks", "t", "mannwhitney"),
) -> pd.DataFrame:
    """Run every test in ``tests`` across every feature column.

    Splits ``df`` into the two groups defined by ``group_column`` (must contain
    exactly two distinct values) and applies each test to every feature.

    Returns a long-format dataframe: ``feature``, ``test``, ``statistic``, ``p_value``.
    """
    feature_columns = list(feature_columns)
    tests = list(tests)

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
        a = df1[feature].values
        b = df2[feature].values
        for test in tests:
            stat, p = compare_distributions(a, b, test)
            rows.append({
                "feature": feature,
                "test": test,
                "statistic": stat,
                "p_value": p,
            })
    return pd.DataFrame(rows, columns=["feature", "test", "statistic", "p_value"])


def adjust_pvalues(
    df: pd.DataFrame,
    pvalue_column: str = "p_value",
    methods: tuple[str, ...] = ("bh", "by"),
) -> pd.DataFrame:
    """Append Benjamini-Hochberg (bh) and/or Benjamini-Yekutieli (by) corrected columns.

    Correction is applied per ``test`` group when a ``test`` column is present
    (so each test's p-values are corrected against its own family), otherwise
    against the whole column. NaN p-values are preserved as NaN.
    """
    df = df.copy()
    unknown = set(methods) - {"bh", "by"}
    if unknown:
        raise ValueError(f"unknown correction method(s): {unknown}")

    def _adjust(pvals: pd.Series, method: str) -> pd.Series:
        out = pd.Series(np.nan, index=pvals.index, dtype=float)
        mask = pvals.notna()
        if mask.any():
            out.loc[mask] = sps.false_discovery_control(
                pvals[mask].values, method=method,
            )
        return out

    group_keys = ["test"] if "test" in df.columns else None
    for method in methods:
        col = f"{pvalue_column}_{method}"
        if group_keys is None:
            df[col] = _adjust(df[pvalue_column], method)
        else:
            df[col] = (
                df.groupby(group_keys, group_keys=False)[pvalue_column]
                .apply(lambda s, m=method: _adjust(s, m))
            )
    return df
