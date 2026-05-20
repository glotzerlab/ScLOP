"""Stage 2: local density and k-atic bond-orientational order (psi_k).

All computations use freud neighbor-list queries on the per-image system
returned by :func:`preprocessing.build_system`.  The bond radius (``cutoff``)
is a required parameter on every public function so callers can sweep it.
"""

from itertools import combinations
from typing import Iterable

import freud
import numpy as np
import pandas as pd

from .preprocessing import build_system


# ── neighbor-list helpers ─────────────────────────────────────────────────────

def _radius_nlist(box, points, cutoff: float):
    """All neighbors within ``cutoff`` (exclude self-pairs)."""
    aq = freud.locality.AABBQuery(box, points)
    return aq.query(
        points, {"r_max": cutoff, "exclude_ii": True}
    ).toNeighborList()


def _katic_nlists(box, points, cutoff: float, k: int):
    """Return the three neighbor-list variants used for psi_k.

    - ``nlist_all``: every neighbor within ``cutoff``.
    - ``nlist_ngek``: drop query cells with fewer than ``k`` neighbors.
    - ``nlist_nek``: keep only the ``k`` nearest neighbors per query cell.

    Variants that contain no qualifying cells come back as ``None`` so
    callers can substitute NaN columns instead of computing psi_k.
    """
    nlist_all = _radius_nlist(box, points, cutoff)

    nlist_ngek = nlist_all.copy()
    keep = ~np.isin(
        nlist_all.query_point_indices,
        np.where(nlist_all.neighbor_counts < k)[0],
    )
    nlist_ngek.filter(keep)
    nlist_ngek.sort(by_distance=True)

    if len(nlist_ngek[:]) == 0:
        return nlist_all, None, None

    q_idx, p_idx, vecs = [], [], []
    for i in np.unique(nlist_ngek.query_point_indices):
        start = nlist_ngek.find_first_index(i)
        end = start + k
        top_p = nlist_ngek.point_indices[start:end]
        top_v = nlist_ngek.vectors[start:end]
        if len(top_p) == k:
            q_idx.extend([i] * k)
            p_idx.extend(top_p)
            vecs.extend(top_v)

    if not q_idx:
        return nlist_all, nlist_ngek, None

    nlist_nek = freud.locality.NeighborList.from_arrays(
        num_query_points=int(np.max(nlist_ngek.query_point_indices)) + 1,
        num_points=int(np.max(nlist_ngek.point_indices)) + 1,
        query_point_indices=q_idx,
        point_indices=p_idx,
        vectors=np.array(vecs),
    )
    return nlist_all, nlist_ngek, nlist_nek


def _psi_k(box, points, nlist, k: int) -> np.ndarray:
    """|psi_k| per particle, or NaN-filled vector if ``nlist`` is None."""
    if nlist is None:
        return np.full(len(points), np.nan)
    hex_order = freud.order.Hexatic(k=k)
    hex_order.compute(system=(box, points), neighbors=nlist)
    return np.abs(hex_order.particle_order)


def _enumerate_combos(cell_types: list[str]) -> list[tuple[str, ...]]:
    """Singletons, pairs, and the full set — matches the original COMBOS list."""
    combos = [(t,) for t in cell_types]
    combos.extend(combinations(cell_types, 2))
    if len(cell_types) > 2:
        combos.append(tuple(cell_types))
    return combos


# ── local density ─────────────────────────────────────────────────────────────

def compute_local_density(
    df: pd.DataFrame,
    cell_types: list[str],
    cutoff: float,
) -> pd.DataFrame:
    """Compute per-cell local density and per-cell-type neighbor counts.

    Adds columns ``local_density``, ``num_neighbors``, and
    ``num_<celltype>_neighbors`` for each entry in ``cell_types``.
    """
    df = df.reset_index(drop=True).copy()

    if len(df) == 0:
        df["local_density"] = np.nan
        df["num_neighbors"] = np.nan
        for ct in cell_types:
            df[f"num_{ct}_neighbors"] = 0
        return df

    box, points = build_system(df)
    nlist = _radius_nlist(box, points, cutoff)

    ld = freud.density.LocalDensity(r_max=cutoff, diameter=0.0)
    ld.compute(system=(box, points), neighbors=nlist)
    df["local_density"] = ld.density
    df["num_neighbors"] = ld.num_neighbors

    neighbor_types = df["cell_type"].values[nlist.point_indices]
    counts = (
        pd.DataFrame({
            "q": nlist.query_point_indices,
            "t": neighbor_types,
        })
        .groupby(["q", "t"])
        .size()
        .unstack(fill_value=0)
    )
    for ct in cell_types:
        col = f"num_{ct}_neighbors"
        if ct in counts.columns:
            df[col] = counts[ct].reindex(df.index, fill_value=0).astype(int)
        else:
            df[col] = 0

    return df


def compute_density_combos(
    df: pd.DataFrame,
    cell_types: list[str],
    cutoff: float,
) -> pd.DataFrame:
    """Compute local density for every cell-type combination (singletons, pairs, full set).

    Applies three normalizations per combination: by total cells (normtot),
    by cells in the combination (normcomb), and by cells of the query type (normbase).
    """
    df = df.reset_index(drop=True).copy()
    n_total = len(df)

    for combo in _enumerate_combos(cell_types):
        name = "_".join(combo)
        density_col = f"local_density_{name}"
        neighbors_col = f"num_neighbors_{name}"
        cols = [
            density_col, neighbors_col,
            f"{density_col}_normtot", f"{neighbors_col}_normtot",
            f"{density_col}_normcomb", f"{neighbors_col}_normcomb",
            f"{density_col}_normbase", f"{neighbors_col}_normbase",
        ]
        for c in cols:
            df[c] = np.nan

        sub = df[df["cell_type"].isin(combo)]
        n_comb = len(sub)
        if n_comb == 0:
            continue

        box, points = build_system(sub)
        ld = freud.density.LocalDensity(r_max=cutoff, diameter=0.0)
        ld.compute(system=(box, points))
        density = np.asarray(ld.density)
        neighbors = np.asarray(ld.num_neighbors)

        type_counts = sub["cell_type"].value_counts()
        base_n = sub["cell_type"].map(type_counts).values.astype(float)

        idxs = sub.index
        df.loc[idxs, density_col] = density
        df.loc[idxs, neighbors_col] = neighbors
        df.loc[idxs, f"{density_col}_normtot"] = density / n_total
        df.loc[idxs, f"{neighbors_col}_normtot"] = neighbors / n_total
        df.loc[idxs, f"{density_col}_normcomb"] = density / n_comb
        df.loc[idxs, f"{neighbors_col}_normcomb"] = neighbors / n_comb
        df.loc[idxs, f"{density_col}_normbase"] = density / base_n
        df.loc[idxs, f"{neighbors_col}_normbase"] = neighbors / base_n

    return df


# ── k-atic order ──────────────────────────────────────────────────────────────

def compute_katic_order(
    df: pd.DataFrame,
    cutoff: float,
    k_values: Iterable[int],
) -> pd.DataFrame:
    """Compute psi_k for each k over the full cell population.

    Three neighbor-list variants per k: all neighbors within ``cutoff``,
    only cells with >= k neighbors, and exactly k nearest neighbors.
    """
    df = df.reset_index(drop=True).copy()
    k_values = list(k_values)

    if len(df) == 0:
        for k in k_values:
            df[f"psi_{k}_all"] = np.nan
            df[f"psi_{k}_ngek"] = np.nan
            df[f"psi_{k}_nek"] = np.nan
        return df

    box, points = build_system(df)
    for k in k_values:
        nlist_all, nlist_ngek, nlist_nek = _katic_nlists(box, points, cutoff, k)
        df[f"psi_{k}_all"] = _psi_k(box, points, nlist_all, k)
        df[f"psi_{k}_ngek"] = _psi_k(box, points, nlist_ngek, k)
        df[f"psi_{k}_nek"] = _psi_k(box, points, nlist_nek, k)

    return df


def compute_katic_combos(
    df: pd.DataFrame,
    cell_types: list[str],
    cutoff: float,
    k_values: Iterable[int],
) -> pd.DataFrame:
    """Compute psi_k for every cell-type combination, including cross-type pairs."""
    df = df.reset_index(drop=True).copy()
    k_values = list(k_values)

    for combo in _enumerate_combos(cell_types):
        name = "_".join(combo)
        for k in k_values:
            df[f"psi_{k}_{name}_all"] = np.nan
            df[f"psi_{k}_{name}_ngek"] = np.nan
            df[f"psi_{k}_{name}_nek"] = np.nan

        sub = df[df["cell_type"].isin(combo)]
        if len(sub) == 0:
            continue

        box, points = build_system(sub)
        idxs = sub.index
        for k in k_values:
            nlist_all, nlist_ngek, nlist_nek = _katic_nlists(box, points, cutoff, k)
            df.loc[idxs, f"psi_{k}_{name}_all"] = _psi_k(box, points, nlist_all, k)
            df.loc[idxs, f"psi_{k}_{name}_ngek"] = _psi_k(box, points, nlist_ngek, k)
            df.loc[idxs, f"psi_{k}_{name}_nek"] = _psi_k(box, points, nlist_nek, k)

    return df
