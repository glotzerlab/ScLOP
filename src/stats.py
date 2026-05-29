"""Stage 3: two-sample Kolmogorov-Smirnov test with multiple-comparisons correction."""

from typing import Iterable, Sequence

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


# ── visualisation ─────────────────────────────────────────────────────────────

def _feature_axis_labels(feat: str, cutoff: int) -> tuple[str, str | None]:
    """Return ``(xlabel, title)`` for a feature name in the project's schema.

    Recognises psi_k and local_density features named like
    ``psi_<k>_<types>_..._<ref_ct>`` and
    ``local_density_<types>_<cutoff>_<ref_ct>`` (with optional ``normtot`` /
    ``normbase`` / ``normcomb`` suffix). Title format: "<ref_ct> with <A>".
    Falls back to ``(feat, None)`` for unrecognised names.
    """
    parts = feat.split("_")

    if "psi" in parts:
        subsc = parts[1]
        ref_ct = parts[-1]
        if "all" in parts:
            end_idx = parts.index("all")
        elif str(cutoff) in parts:
            end_idx = parts.index(str(cutoff))
        else:
            return rf"$\psi_{{{subsc}}}$", None
        a_parts = parts[2:end_idx]
        a = a_parts[0] if len(a_parts) == 1 else "{" + ", ".join(a_parts) + "}"
        return rf"$\psi_{{{subsc}}}$", f"{ref_ct} with {a}"

    if "density" in parts:
        ref_ct = parts[-1]
        start_idx = parts.index("density") + 1
        if "normbase" in parts:
            end_idx = parts.index("normbase")
        elif "normtot" in parts:
            end_idx = parts.index("normtot")
        elif "normcomb" in parts:
            end_idx = parts.index("normcomb")
        else:
            end_idx = parts.index(str(cutoff))
        a_parts = parts[start_idx:end_idx]
        a = a_parts[0] if len(a_parts) == 1 else "{" + ", ".join(a_parts) + "}"
        return r"$\mathrm{cells} / \mathrm{px^2}$", f"{ref_ct} with {a}"

    return feat, None


def plot_ks_kde_grid(
    feature_dfs: dict[str, pd.DataFrame],
    features: dict[str, Sequence[str]],
    palettes: dict[str, tuple[Sequence[str], Sequence[str]]],
    cutoff: int,
    group_column: str = "pathology",
    nrows: int = 2,
    ncols: int = 2,
):
    """Filled-KDE comparison plot of selected features across two groups per dataset.

    Mirrors the paper-figure style: filled + outlined KDEs with count-annotated
    legend, and feature-name-aware axis labels / titles. One panel per
    ``(dataset, feature)`` pair, flattened row-major into an ``nrows x ncols`` grid.

    Parameters
    ----------
    feature_dfs :
        ``dataset_name -> per-image feature dataframe`` (must contain ``group_column``).
    features :
        ``dataset_name -> sequence of feature column names`` to plot.
    palettes :
        ``dataset_name -> (colors, group_order)``. ``group_order`` is the
        two values of ``group_column`` in legend order.
    cutoff :
        Bond radius encoded in the feature names (used for label parsing).
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.ticker import ScalarFormatter

    pairs = [(d, f) for d, fs in features.items() for f in fs]
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5 * ncols, 3 * nrows))
    axes = np.asarray(axes).flatten()

    for idx, (dataset, feat) in enumerate(pairs):
        colors, order = palettes[dataset]
        df = feature_dfs[dataset]
        long_df = pd.DataFrame({"value": df[feat].values, "label": df[group_column].values})

        ax = axes[idx]
        sns.kdeplot(data=long_df, x="value", hue="label", hue_order=list(order),
                    linewidth=2, cut=0, fill=True, legend=False,
                    palette=list(colors), ax=ax)
        sns.kdeplot(data=long_df, x="value", hue="label", hue_order=list(order),
                    linewidth=2, cut=0, legend=True,
                    palette=list(colors), ax=ax)

        counts = long_df.dropna(subset=["value"])["label"].value_counts()
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            new_labels = [f"{lab} (n={int(counts.get(lab, 0))})" for lab in labels]
            ax.legend(handles=handles, labels=new_labels)

        ax.set_ylabel("probability density")
        xlabel, title = _feature_axis_labels(feat, cutoff)
        ax.set_xlabel(xlabel)
        if title:
            ax.set_title(title)
        else:
            ax.set_title(f"{dataset}: {feat}")

        if "psi" in feat.split("_"):
            ax.set_xlim(-0.1, 1.1)
        elif "density" in feat.split("_"):
            ax.set_xticks(ax.get_xticks()[::2])
            fmt = ScalarFormatter(useMathText=True)
            fmt.set_scientific(True)
            fmt.set_powerlimits((-2, 2))
            ax.xaxis.set_major_formatter(fmt)
            ax.set_yticks([t for t in ax.get_yticks() if not np.isclose(t, 0)])

    for j in range(len(pairs), nrows * ncols):
        fig.delaxes(axes[j])

    fig.tight_layout()
    return fig, axes
