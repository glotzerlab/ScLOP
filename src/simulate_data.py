"""Generate a small synthetic dataset in the unified pipeline schema.

The real study CSVs cannot be shared, so this module fabricates a tiny,
biologically-flavoured stand-in so the pipeline can be run end-to-end for
demonstration. The output uses the exact unified schema consumed by
:func:`src.preprocessing.load_dataset`::

    cell_id, x, y, cell_type, image_id, pathology, patient

There is one tumour cell type and three immune cell types. Two synthetic
conditions are generated, chosen to be visibly different to the downstream
feature / stats / ML / HSIC stages:

``immune_cold``
    Tumour-dominated. Tumour cells sit on a lightly-jittered triangular
    lattice (high local order -> high psi_k, fairly even density) and the
    few immune cells are *excluded* from tumour regions.

``immune_hot``
    More immune cells, which *infiltrate* the tumour. Tumour cells are
    scattered in loose Gaussian blobs (disordered -> low psi_k), so both
    tumour order and tumour/immune co-localisation differ from the cold case.

These differences are what let the demo pipeline produce non-trivial KS,
random-forest, and HSIC output. The numbers are illustrative, not real data.

CLI
---
    python -m src.simulate_data data/raw/simulated_data.csv --seed 42
"""

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


UNIFIED_COLUMNS = [
    "cell_id", "x", "y", "cell_type", "image_id", "pathology", "patient",
]

# ── field / population sizes ──────────────────────────────────────────────────
FIELD_W, FIELD_H = 1400.0, 1050.0     # matches the CP_PDAC plot extents
MARGIN = 60.0                          # keep cells away from the very edge

TUMOR_TYPE = "Tumor"
IMMUNE_TYPES = ["CD4_T", "CD8_T", "Macrophage"]

N_IMAGES_PER_DISEASE = 10
CELLS_PER_IMAGE = 300
CELL_COUNT_JITTER = 15                  # per-image variation in cell count
IMAGES_PER_PATIENT = 2

# ── spatial-structure parameters ──────────────────────────────────────────────
LATTICE_SPACING = 55.0                  # < bond_cutoff so lattice cells are neighbours
LATTICE_JITTER = 6.0                    # small vs spacing -> stays ordered
N_TUMOR_BLOBS = 4
TUMOR_BLOB_SIGMA = 110.0
EXCLUSION_RADIUS = 70.0                 # immune kept this far from tumour (cold)
INFILTRATION_SIGMA = 35.0               # immune scatter around tumour (hot)


@dataclass
class DiseaseProfile:
    """Per-condition knobs controlling composition and spatial layout."""
    name: str
    immune_frac: float                  # fraction of cells that are immune
    immune_mix: tuple[float, float, float]  # split across IMMUNE_TYPES
    tumor_ordered: bool                 # lattice (True) vs Gaussian blobs (False)
    immune_mode: str                    # "excluded" | "infiltrated"


DISEASES = [
    DiseaseProfile("immune_cold", 0.25, (0.3, 0.2, 0.5), True, "excluded"),
    DiseaseProfile("immune_hot", 0.55, (0.3, 0.5, 0.2), False, "infiltrated"),
]


# ── tumour placement ──────────────────────────────────────────────────────────

def _triangular_lattice(rng: np.random.Generator) -> np.ndarray:
    """Jittered triangular lattice filling the field (high bond-orientational order)."""
    dy = LATTICE_SPACING * math.sqrt(3) / 2
    pts: list[tuple[float, float]] = []
    y, row = MARGIN, 0
    while y <= FIELD_H - MARGIN:
        x = MARGIN + (LATTICE_SPACING / 2 if row % 2 else 0.0)
        while x <= FIELD_W - MARGIN:
            pts.append((x, y))
            x += LATTICE_SPACING
        y += dy
        row += 1
    lattice = np.array(pts, dtype=float)
    lattice += rng.normal(0.0, LATTICE_JITTER, lattice.shape)
    return lattice


def _sample_ordered_tumor(rng: np.random.Generator, n: int) -> np.ndarray:
    lattice = _triangular_lattice(rng)
    n = min(n, len(lattice))
    idx = rng.choice(len(lattice), size=n, replace=False)
    return lattice[idx]


def _sample_blob_tumor(rng: np.random.Generator, n: int) -> np.ndarray:
    lo = np.array([MARGIN, MARGIN])
    hi = np.array([FIELD_W - MARGIN, FIELD_H - MARGIN])
    centers = rng.uniform(lo, hi, size=(N_TUMOR_BLOBS, 2))
    assign = rng.integers(0, N_TUMOR_BLOBS, size=n)
    return centers[assign] + rng.normal(0.0, TUMOR_BLOB_SIGMA, size=(n, 2))


# ── immune placement (relative to tumour) ─────────────────────────────────────

def _sample_excluded_immune(
    rng: np.random.Generator, n: int, tumor_xy: np.ndarray
) -> np.ndarray:
    """Uniform positions kept at least EXCLUSION_RADIUS from any tumour cell."""
    lo = np.array([MARGIN, MARGIN])
    hi = np.array([FIELD_W - MARGIN, FIELD_H - MARGIN])
    out: list[np.ndarray] = []
    for _ in range(n * 50):
        if len(out) >= n:
            break
        cand = rng.uniform(lo, hi, size=2)
        if tumor_xy.size == 0:
            out.append(cand)
            continue
        d = np.hypot(tumor_xy[:, 0] - cand[0], tumor_xy[:, 1] - cand[1])
        if d.min() > EXCLUSION_RADIUS:
            out.append(cand)
    while len(out) < n:  # top up if rejection sampling fell short
        out.append(rng.uniform(lo, hi, size=2))
    return np.array(out)


def _sample_infiltrated_immune(
    rng: np.random.Generator, n: int, tumor_xy: np.ndarray
) -> np.ndarray:
    """Positions scattered tightly around randomly chosen tumour cells."""
    if tumor_xy.size == 0:
        lo = np.array([MARGIN, MARGIN])
        hi = np.array([FIELD_W - MARGIN, FIELD_H - MARGIN])
        return rng.uniform(lo, hi, size=(n, 2))
    anchors = tumor_xy[rng.integers(0, len(tumor_xy), size=n)]
    return anchors + rng.normal(0.0, INFILTRATION_SIGMA, size=(n, 2))


# ── assembly ──────────────────────────────────────────────────────────────────

def simulate_image(
    rng: np.random.Generator,
    profile: DiseaseProfile,
    image_id: str,
    patient: str,
    n_cells: int = CELLS_PER_IMAGE,
) -> pd.DataFrame:
    """Build one synthetic image as a per-cell dataframe (no cell_id yet)."""
    n_cells = max(50, int(rng.normal(n_cells, CELL_COUNT_JITTER)))
    n_immune = int(round(n_cells * profile.immune_frac))
    n_tumor = n_cells - n_immune

    if profile.tumor_ordered:
        tumor_xy = _sample_ordered_tumor(rng, n_tumor)
    else:
        tumor_xy = _sample_blob_tumor(rng, n_tumor)
    n_tumor = len(tumor_xy)

    mix = np.array(profile.immune_mix, dtype=float)
    counts = rng.multinomial(n_immune, mix / mix.sum())  # sums to n_immune

    if profile.immune_mode == "excluded":
        immune_xy = _sample_excluded_immune(rng, n_immune, tumor_xy)
    else:
        immune_xy = _sample_infiltrated_immune(rng, n_immune, tumor_xy)

    types = [TUMOR_TYPE] * n_tumor
    for cell_type, count in zip(IMMUNE_TYPES, counts):
        types.extend([cell_type] * int(count))

    xy = np.vstack([tumor_xy, immune_xy]) if n_immune else tumor_xy
    xy[:, 0] = np.clip(xy[:, 0], 0.0, FIELD_W)
    xy[:, 1] = np.clip(xy[:, 1], 0.0, FIELD_H)

    df = pd.DataFrame({"x": xy[:, 0], "y": xy[:, 1], "cell_type": types})
    df = df.iloc[rng.permutation(len(df))].reset_index(drop=True)  # mix type order
    df["image_id"] = image_id
    df["pathology"] = profile.name
    df["patient"] = patient
    return df


def simulate_dataset(seed: int = 42) -> pd.DataFrame:
    """Generate the full multi-image dataset in the unified schema."""
    rng = np.random.default_rng(seed)
    frames: list[pd.DataFrame] = []
    for profile in DISEASES:
        for i in range(N_IMAGES_PER_DISEASE):
            image_id = f"{profile.name}_img{i:02d}"
            patient = f"{profile.name}_p{i // IMAGES_PER_PATIENT:02d}"
            frames.append(simulate_image(rng, profile, image_id, patient))

    df = pd.concat(frames, ignore_index=True)
    df.insert(0, "cell_id", np.arange(len(df), dtype=np.int64))
    return df[UNIFIED_COLUMNS]


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "output", type=Path, nargs="?",
        default=Path("data/raw/simulated_data.csv"),
        help="destination CSV (default: data/raw/simulated_data.csv)",
    )
    parser.add_argument("--seed", type=int, default=42, help="RNG seed")
    args = parser.parse_args(argv)

    df = simulate_dataset(seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(
        f"wrote {args.output}  ({len(df)} cells, "
        f"{df['image_id'].nunique()} images, "
        f"{df['pathology'].nunique()} conditions, "
        f"{df['cell_type'].nunique()} cell types)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
