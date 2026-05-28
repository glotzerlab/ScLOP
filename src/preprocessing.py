"""Stage 1: load raw cell data and build per-image dataframes.

Reads a single CSV covering all images, splits by image, deduplicates
overlapping positions, encodes cell types, and optionally writes
per-image CSVs to a processed directory.

Expected input CSV columns (unified schema)
-------------------------------------------

    cell_id    — integer cell identifier (unique within the CSV)
    x, y       — float cell-centre coordinates
    cell_type  — string cell-type label (NaN → "undefined")
    image_id   — string image identifier (rows are grouped by this)
    pathology  — string disease/condition label
    patient    — patient identifier (may be NaN if not available)

Raw CSVs with study-specific column names (e.g. CRC's
``pathology, patient, img, X, Y, type_orig, type`` or CP_PDAC's
``cell_id, x, y, cell_type, cell_type_1hot, dataset, job_id``) must be
converted to this schema first — see ``src/normalize_raw.py``.
"""

from pathlib import Path

import freud
import numpy as np
import pandas as pd


# ── cell-type handling ────────────────────────────────────────────────────────

def build_cell_type_encoding(df: pd.DataFrame, type_col: str = "cell_type") -> dict[str, int]:
    """Derive a stable string→integer mapping from all types in ``df``.

    NaN entries are converted to ``"undefined"`` before sorting,
    matching the convention in the original dataset.
    """
    types = df[type_col].fillna("undefined").astype(str).values
    unique_types = np.unique(types)
    return {t: i for i, t in enumerate(unique_types)}


def encode_cell_types(
    df: pd.DataFrame,
    encoding: dict[str, int],
    type_col: str = "cell_type",
) -> pd.DataFrame:
    """Add a ``cell_type_encoded`` integer column using ``encoding``."""
    df = df.copy()
    df["cell_type_encoded"] = df[type_col].map(encoding)
    return df


# ── cleaning ──────────────────────────────────────────────────────────────────

def drop_duplicate_positions(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows sharing identical (x, y) coordinates within an image."""
    return df.drop_duplicates(subset=["x", "y"]).reset_index(drop=True)


# ── loading ───────────────────────────────────────────────────────────────────

def load_dataset(
    csv_path: str | Path,
    drop_undefined: bool = False,
) -> tuple[dict[str, pd.DataFrame], dict[str, int]]:
    """Load all images from the raw CSV.

    Splits the full dataset by image ID, standardises column names,
    deduplicates positions, and encodes cell types.

    Parameters
    ----------
    csv_path :
        Path to a CSV in the unified schema (see module docstring).
    drop_undefined :
        If True, remove cells whose type resolved to ``"undefined"`` (i.e.
        originally NaN) before returning.

    Returns
    -------
    images : dict mapping image_id → cleaned per-image DataFrame with columns
        ``cell_id``, ``x``, ``y``, ``cell_type``, ``cell_type_encoded``,
        ``image_id``, ``pathology``, ``patient``.
    encoding : the cell-type string→integer mapping shared across all images.
    """
    raw = pd.read_csv(csv_path)

    # normalise cell-type strings (NaN → "undefined")
    raw["cell_type"] = raw["cell_type"].fillna("undefined").astype(str)

    encoding = build_cell_type_encoding(raw)

    images: dict[str, pd.DataFrame] = {}
    for image_id, group in raw.groupby("image_id"):
        df = group.reset_index(drop=True)
        df["cell_id"] = np.arange(len(df))

        df = drop_duplicate_positions(df)
        df = encode_cell_types(df, encoding)

        if drop_undefined:
            df = df[df["cell_type"] != "undefined"].reset_index(drop=True)
            df["cell_id"] = np.arange(len(df))

        df = df[["cell_id", "x", "y", "cell_type", "cell_type_encoded",
                  "image_id", "pathology", "patient"]]

        images[str(image_id)] = df

    return images, encoding


def load_image(
    csv_path: str | Path,
    image_id: str,
    drop_undefined: bool = False,
) -> pd.DataFrame:
    """Load and preprocess a single image by its ID.

    The encoding is derived from the full CSV so integer codes are consistent
    across images.
    """
    images, _ = load_dataset(csv_path, drop_undefined=drop_undefined)
    if image_id not in images:
        raise KeyError(
            f"image_id '{image_id}' not found. "
            f"Available: {sorted(images)[:5]} ..."
        )
    return images[image_id]


# ── freud system builder ──────────────────────────────────────────────────────

BOX_SCALE = 1000  # makes the box non-periodic: no cell can be near a boundary


def build_freud_system(df: pd.DataFrame) -> tuple[freud.box.Box, np.ndarray]:
    """Build a freud 2D simulation box and point array from a per-image dataframe.

    The box is scaled to 1000× the coordinate span so that freud treats it as
    effectively non-periodic — matching the convention in the original analysis.
    Points are kept at their raw coordinates (not shifted to the box centre).

    Returns
    -------
    box : ``freud.box.Box``
    points : (N, 3) float32 array with z = 0.
    """
    xs = df["x"].values.astype(np.float32)
    ys = df["y"].values.astype(np.float32)

    lx = float(xs.max() - xs.min()) * BOX_SCALE
    ly = float(ys.max() - ys.min()) * BOX_SCALE

    box = freud.box.Box(Lx=lx, Ly=ly, xy=0, is2D=True)
    points = np.column_stack([xs, ys, np.zeros(len(df), dtype=np.float32)])

    return box, points


# ── persistence ───────────────────────────────────────────────────────────────

def save_processed(
    images: dict[str, pd.DataFrame],
    output_dir: str | Path,
) -> None:
    """Write each per-image dataframe to ``<output_dir>/<image_id>.csv``."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for image_id, df in images.items():
        df.to_csv(output_dir / f"{image_id}.csv", index=False)


def load_processed(processed_dir: str | Path) -> dict[str, pd.DataFrame]:
    """Read all per-image CSVs written by :func:`save_processed`."""
    processed_dir = Path(processed_dir)
    return {
        p.stem: pd.read_csv(p)
        for p in sorted(processed_dir.glob("*.csv"))
    }
