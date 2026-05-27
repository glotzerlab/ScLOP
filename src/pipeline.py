"""Run the full SCSAP analysis pipeline end-to-end.

Stages
------
1. **Preprocess** the raw CSV into per-image dataframes (delegates to
   :mod:`preprocessing`) and write them to ``processed_data_dir``.
2. **Compute features** per image — local density and psi_k for every k in
   ``[k_min, k_max)`` — concatenate them into one cell-level table, and
   aggregate to a per-image summary (one row per image, mean of each feature).
3. **Run statistical tests** comparing the two pathology groups across every
   feature column, with BH and BY p-value correction.
4. **Train a cross-validated random-forest classifier** that predicts
   pathology from the per-image feature summary.

All intermediate and final tables are written to ``output_dir``.

CLI usage
---------
    python -m src.pipeline --config configs/default.yaml --output-dir results/
"""

import argparse
from pathlib import Path

import pandas as pd
import yaml

from .features import compute_katic_order, compute_local_density
from .ml import build_feature_matrix, cross_validate
from .preprocessing import load_dataset, save_processed
from .stats import adjust_pvalues, ks_test_per_feature


IDENTIFIER_COLS = {
    "cell_id", "x", "y", "cell_type", "cell_type_encoded",
    "image_id", "pathology", "patient",
}


def _resolve_raw_csv(config: dict) -> Path:
    """Pick the raw CSV: ``config['raw_csv']`` if set, else the first CSV in ``raw_data_dir``."""
    if "raw_csv" in config:
        return Path(config["raw_csv"])
    raw_dir = Path(config["raw_data_dir"])
    csvs = sorted(raw_dir.glob("*.csv"))
    if not csvs:
        raise FileNotFoundError(
            f"no raw CSV found; set 'raw_csv' in config or place a CSV in {raw_dir}",
        )
    return csvs[0]


def _resolve_cutoffs(config: dict) -> list[float]:
    """Read 'bond_cutoffs' (list) or fall back to scalar 'bond_cutoff'."""
    if "bond_cutoffs" in config:
        cutoffs = config["bond_cutoffs"]
        if not isinstance(cutoffs, (list, tuple)) or not cutoffs:
            raise ValueError("'bond_cutoffs' must be a non-empty list")
        return [float(c) for c in cutoffs]
    if "bond_cutoff" in config:
        return [float(config["bond_cutoff"])]
    raise KeyError("config must define 'bond_cutoffs' (list) or 'bond_cutoff' (scalar)")


def _feature_columns(df: pd.DataFrame) -> list[str]:
    """All numeric feature columns produced by stage 2 (everything except identifiers)."""
    return [c for c in df.columns if c not in IDENTIFIER_COLS]


def _features_at_radius(
    df: pd.DataFrame,
    cell_types: list[str],
    cutoff: float,
    k_values: list[int],
    include_neighbor_counts: bool,
    suffix: str,
) -> pd.DataFrame:
    """Compute density + psi_k at one ``cutoff``; suffix the new columns with ``suffix``."""
    out = compute_local_density(
        df, cell_types, cutoff,
        include_neighbor_counts=include_neighbor_counts,
    )
    out = compute_katic_order(out, cutoff, k_values)
    if suffix:
        rename = {c: f"{c}{suffix}" for c in out.columns if c not in IDENTIFIER_COLS}
        out = out.rename(columns=rename)
    return out


def run(config: dict, output_dir: Path) -> None:
    """Execute all four stages in sequence, writing outputs to ``output_dir``."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Stage 1: preprocessing ────────────────────────────────────────────────
    raw_csv = _resolve_raw_csv(config)
    print(f"[1/4] loading raw data from {raw_csv}")
    images, encoding = load_dataset(raw_csv, drop_undefined=False)
    save_processed(images, config["processed_data_dir"])
    print(f"      loaded {len(images)} images, {len(encoding)} cell types")

    # ── Stage 2: features ─────────────────────────────────────────────────────
    cutoffs = _resolve_cutoffs(config)
    k_values = list(range(int(config["k_min"]), int(config["k_max"])))
    cell_types = sorted(encoding)
    include_counts = bool(config.get("include_neighbor_counts", True))
    print(f"[2/4] computing features for {len(images)} images "
          f"(cutoffs={cutoffs}, k={k_values[0]}..{k_values[-1]}, "
          f"neighbor_counts={include_counts})")

    per_image_feats: list[pd.DataFrame] = []
    for image_id, df in images.items():
        merged = None
        for cutoff in cutoffs:
            suffix = f"_r{cutoff:g}" if len(cutoffs) > 1 else ""
            feats = _features_at_radius(
                df, cell_types, cutoff, k_values, include_counts, suffix,
            )
            if merged is None:
                merged = feats
            else:
                new_cols = [c for c in feats.columns if c not in merged.columns]
                merged = pd.concat([merged, feats[new_cols]], axis=1)
        per_image_feats.append(merged)

    cells = pd.concat(per_image_feats, ignore_index=True)
    cells.to_csv(output_dir / "cells_features.csv", index=False)

    feat_cols = _feature_columns(cells)
    image_summary = (
        cells.groupby(["image_id", "pathology", "patient"], as_index=False)[feat_cols]
        .mean()
    )
    image_summary.to_csv(output_dir / "image_summary.csv", index=False)
    print(f"      wrote {len(feat_cols)} feature columns; "
          f"{len(image_summary)} per-image rows")

    # ── Stage 3: stats ────────────────────────────────────────────────────────
    print("[3/4] running KS tests (pathology 1 vs 2)")
    stats_df = ks_test_per_feature(
        image_summary,
        feature_columns=feat_cols,
        group_column="pathology",
    )
    stats_df = adjust_pvalues(stats_df)
    stats_df.to_csv(output_dir / "stats.csv", index=False)
    sig = (stats_df["p_value_bh"] < 0.05).sum()
    print(f"      {sig} features significant at BH q<0.05")

    # ── Stage 4: ML ───────────────────────────────────────────────────────────
    print("[4/4] cross-validated random forest")
    X, y = build_feature_matrix(image_summary, label_column="pathology",
                                 feature_columns=feat_cols)
    result = cross_validate(
        X, y,
        n_splits=5,
        random_state=int(config["random_state"]),
    )
    result["fold_metrics"].to_csv(output_dir / "ml_fold_metrics.csv", index=False)
    result["feature_importances"].to_csv(
        output_dir / "ml_feature_importances.csv", header=["importance"],
    )
    overall = {k: v for k, v in result["overall"].items() if k != "confusion_matrix"}
    pd.Series(overall).to_csv(output_dir / "ml_overall.csv", header=["value"])
    print(f"      overall AUROC={overall['auroc']:.3f}, "
          f"balanced accuracy={overall['balanced_accuracy']:.3f}")
    print(f"done. results in {output_dir}/")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    run(config, args.output_dir)


if __name__ == "__main__":
    main()
