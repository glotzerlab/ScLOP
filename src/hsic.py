"""HSIC test for independence between psi (k-atic order) and density feature blocks.

Public API
----------
``prepare_psi_density_blocks``      — pull psi/density columns, impute, z-score, return arrays.
``hsic``                — biased empirical HSIC with RBF kernels.
``hsic_test``           — HSIC + permutation p-value + null Z-score (fast path).
``hsic_sweep``          — run the test over (split x cutoff); returns tidy DataFrame.
``plot_hsic_heatmap``   — pivot a sweep result and render a cutoff x split heatmap.
"""

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import pairwise_kernels


PSI_KEY = "psi"
DENSITY_KEY = "local_density"


# ── preprocessing ─────────────────────────────────────────────────────────────

def prepare_psi_density_blocks(
    df: pd.DataFrame,
    *,
    psi_key: str = PSI_KEY,
    density_key: str = DENSITY_KEY,
    columns: Iterable[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    """Select psi and density columns, impute NaNs with column mean, drop zero-
    variance columns, z-score, and return ``(X_psi, X_dens, psi_cols, dens_cols)``.

    Pass ``columns`` to restrict the search (e.g. to a top-20 subset).
    """
    pool = list(columns) if columns is not None else list(df.columns)
    psi_cols = [c for c in pool if psi_key in c.lower()]
    dens_cols = [c for c in pool if density_key in c.lower()]
    feats = [c for c in psi_cols + dens_cols if not df[c].isna().all()]

    X = df[feats].astype(float).copy()
    X = X.fillna(X.mean())
    keep = X.std(ddof=1) > 0
    X = X.loc[:, keep]
    X = (X - X.mean()) / X.std(ddof=1)

    psi_cols = [c for c in X.columns if psi_key in c.lower()]
    dens_cols = [c for c in X.columns if density_key in c.lower()]
    return X[psi_cols].to_numpy(), X[dens_cols].to_numpy(), psi_cols, dens_cols


# ── kernel bandwidth ──────────────────────────────────────────────────────────

def _median_heuristic_gamma(X: np.ndarray) -> float:
    """gamma = 1/(2 sigma^2), sigma = median of off-diagonal pairwise Euclidean distances."""
    d = pairwise_distances(X, metric="euclidean")
    iu = np.triu_indices_from(d, k=1)
    sigma = float(np.median(d[iu]))
    return 1.0 if sigma == 0.0 else 1.0 / (2.0 * sigma * sigma)


def _resolve_gamma(X: np.ndarray, bandwidth) -> float | None:
    """Resolve a bandwidth spec into a gamma value (or None for sklearn default)."""
    if isinstance(bandwidth, (int, float)):
        return float(bandwidth)
    if bandwidth == "median":
        return _median_heuristic_gamma(X)
    if bandwidth == "sklearn":
        return None  # sklearn picks gamma = 1 / n_features
    raise ValueError(f"unknown bandwidth: {bandwidth!r}")


def _rbf(X: np.ndarray, gamma: float | None) -> np.ndarray:
    if gamma is None:
        return pairwise_kernels(X, metric="rbf")
    return pairwise_kernels(X, metric="rbf", gamma=gamma)


# ── HSIC ──────────────────────────────────────────────────────────────────────

def hsic(
    X: np.ndarray,
    Y: np.ndarray,
    *,
    bandwidth: str | float = "median",
    gamma_x: float | None = None,
    gamma_y: float | None = None,
) -> float:
    """Biased empirical HSIC: ``tr(K H L H) / (n-1)^2`` with RBF kernels.

    ``bandwidth`` chooses how the RBF gamma is set:
      * ``"median"`` (default) — median-heuristic gamma computed per block.
      * ``"sklearn"`` — sklearn's default ``gamma = 1/n_features``.
      * a float — use this gamma for both blocks.

    ``gamma_x`` / ``gamma_y`` override ``bandwidth`` per block when supplied.
    """
    n = X.shape[0]
    gx = gamma_x if gamma_x is not None else _resolve_gamma(X, bandwidth)
    gy = gamma_y if gamma_y is not None else _resolve_gamma(Y, bandwidth)
    K = _rbf(X, gx)
    L = _rbf(Y, gy)
    H = np.eye(n) - np.ones((n, n)) / n
    return float(np.trace(K @ H @ L @ H) / (n - 1) ** 2)


def hsic_test(
    X: np.ndarray,
    Y: np.ndarray,
    *,
    bandwidth: str | float = "median",
    gamma_x: float | None = None,
    gamma_y: float | None = None,
    n_permutations: int = 5000,
    random_state: int = 42,
) -> dict:
    """Biased HSIC plus one-sided permutation p-value and null Z-score.

    Fast path: K and L are computed once, then for each permutation ``pi``
    HSIC_pi = sum(centered_K * L[pi, pi]) / (n-1)^2. RBF kernels depend only
    on pairwise data, so permuting rows of Y just reorders both rows and
    columns of L by the same permutation — this is bit-identical to
    rebuilding the Y-kernel from Y[pi] at the same seed.
    """
    n = X.shape[0]
    gx = gamma_x if gamma_x is not None else _resolve_gamma(X, bandwidth)
    gy = gamma_y if gamma_y is not None else _resolve_gamma(Y, bandwidth)
    K = _rbf(X, gx)
    L = _rbf(Y, gy)
    H = np.eye(n) - np.ones((n, n)) / n
    Kc = H @ K @ H
    denom = (n - 1) ** 2
    obs = float(np.sum(Kc * L) / denom)

    rng = np.random.default_rng(random_state)
    null = np.empty(n_permutations)
    for i in range(n_permutations):
        pi = rng.permutation(n)
        null[i] = np.sum(Kc * L[np.ix_(pi, pi)]) / denom

    null_mean = float(null.mean())
    null_std = float(null.std(ddof=1))
    p_value = float((np.sum(null >= obs) + 1) / (n_permutations + 1))
    z = (obs - null_mean) / null_std if null_std > 0 else float("nan")

    return {
        "hsic": obs,
        "p_value": p_value,
        "null_mean": null_mean,
        "null_std": null_std,
        "z_score": float(z),
        "gamma_x": float(gx) if gx is not None else None,
        "gamma_y": float(gy) if gy is not None else None,
        "n": int(n),
        "n_permutations": int(n_permutations),
    }


# ── sweep ─────────────────────────────────────────────────────────────────────

def _columns_for_cutoff(df: pd.DataFrame, cutoff: float | None) -> list[str]:
    """Columns belonging to ``cutoff`` (suffix ``_r{cutoff:g}``). ``None`` → all columns."""
    if cutoff is None:
        return list(df.columns)
    suffix = f"_r{cutoff:g}"
    return [c for c in df.columns if c.endswith(suffix)]


def hsic_sweep(
    df: pd.DataFrame,
    *,
    splits_by: str | None = None,
    cutoffs: Iterable[float] | None = None,
    include_combined: bool = True,
    bandwidth: str | float = "median",
    n_permutations: int = 5000,
    random_state: int = 42,
    psi_key: str = PSI_KEY,
    density_key: str = DENSITY_KEY,
) -> pd.DataFrame:
    """Run :func:`hsic_test` over ``(split, cutoff)`` and return a tidy DataFrame.

    Rows: one per ``(split, cutoff)``. Columns: split, cutoff, n, hsic, p_value,
    null_mean, null_std, z_score, gamma_x, gamma_y.

    ``splits_by`` — column to subset rows by (e.g. ``"pathology"``). ``None`` runs
    a single "combined" row.

    ``cutoffs`` — list of cutoff values; columns are filtered by ``_r{cutoff:g}``
    suffix. ``None`` uses all psi/density columns regardless of suffix.
    """
    cutoff_list: list[float | None] = (
        list(cutoffs) if cutoffs is not None else [None]
    )

    if splits_by is None:
        split_groups: list[tuple[str, pd.DataFrame]] = [("combined", df)]
    else:
        split_groups = []
        if include_combined:
            split_groups.append(("combined", df))
        for value, sub in df.groupby(splits_by):
            split_groups.append((str(value), sub))

    rows: list[dict] = []
    for split_name, sub in split_groups:
        for cutoff in cutoff_list:
            cols = _columns_for_cutoff(sub, cutoff)
            X, Y, psi_cols, dens_cols = prepare_psi_density_blocks(
                sub, psi_key=psi_key, density_key=density_key, columns=cols,
            )
            if X.size == 0 or Y.size == 0 or X.shape[0] < 3:
                continue
            res = hsic_test(
                X, Y,
                bandwidth=bandwidth,
                n_permutations=n_permutations,
                random_state=random_state,
            )
            rows.append({
                "split": split_name,
                "cutoff": cutoff,
                "n_psi": len(psi_cols),
                "n_dens": len(dens_cols),
                **res,
            })

    return pd.DataFrame(rows)


# ── plotting ──────────────────────────────────────────────────────────────────

def plot_hsic_heatmap(
    results: pd.DataFrame,
    *,
    value: str = "z_score",
    index: str = "cutoff",
    columns: str = "split",
    cutoff_order: Iterable[float] | None = None,
    split_order: Iterable[str] | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    annotate: bool = True,
    fmt: str = ".2f",
    title: str | None = None,
    cmap: str = "viridis",
    out_path: str | Path | None = None,
    ax: plt.Axes | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Pivot ``results`` as ``index x columns`` and draw an annotated heatmap.

    Defaults reproduce the cutoff x split heatmap of ``z_score`` from the
    scratcher notebook. Save with ``out_path``.
    """
    mat = results.pivot(index=index, columns=columns, values=value)
    if cutoff_order is not None:
        mat = mat.reindex(list(cutoff_order))
    if split_order is not None:
        mat = mat.reindex(list(split_order), axis=1)

    if ax is None:
        fig, ax = plt.subplots(figsize=(1.2 * mat.shape[1] + 2, 0.6 * mat.shape[0] + 2))
    else:
        fig = ax.figure

    im = ax.imshow(mat.values, aspect="auto", vmin=vmin, vmax=vmax, cmap=cmap)
    ax.set_xticks(np.arange(mat.shape[1]))
    ax.set_xticklabels(mat.columns, rotation=45, ha="right")
    ax.set_yticks(np.arange(mat.shape[0]))
    ax.set_yticklabels(mat.index)
    ax.set_xlabel(columns)
    ax.set_ylabel(index)

    if annotate:
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                v = mat.iat[i, j]
                if pd.notna(v):
                    ax.text(j, i, format(v, fmt), ha="center", va="center", fontsize=7)

    if title:
        ax.set_title(title)
    fig.colorbar(im, ax=ax, label=value)
    fig.tight_layout()
    if out_path is not None:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=300)
    return fig, ax
