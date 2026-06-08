# ScLOP - Single-cell Local Order Parameters

Code accompanying [paper citation]. Analyzes the spatial organization of cells
in tissue images to distinguish between tissue conditions using spatial
statistics and machine learning.

## Dependencies

Dependencies are managed with [pixi](https://pixi.sh). The spec lives in
`pixi.toml` and is pinned for linux-64, osx-64, osx-arm64, and win-64 in
`pixi.lock`.
Install pixi (`curl -fsSL https://pixi.sh/install.sh | bash`), then create the
environment with an exact, reproducible install:

```bash
pixi install
```

Run commands inside the environment with `pixi run`, or open a shell in it:

```bash
pixi run python -m src.pipeline --config configs/default.yaml --output-dir results/
pixi shell   # activate the environment interactively
```

After editing `pixi.toml`, refresh the lock with `pixi install` (it re-solves
and updates `pixi.lock` automatically).

## Data

The tissue datasets analyzed in the paper are **not publicly available** and
are not distributed with this repository. To make the pipeline runnable and to
document the expected input format, `src/simulate_data.py` generates a small
**synthetic** dataset with the same structure the pipeline consumes — it is an
illustrative stand-in, not real data.

The pipeline reads a single CSV in the following unified schema (one row per
cell), and `src/preprocessing.py` groups it into per-image tables:

| Column | Description |
|---|---|
| `cell_id` | Integer cell identifier |
| `x`, `y` | Float cell-centre coordinates |
| `cell_type` | Cell-type label (blank → `undefined`) |
| `image_id` | Image identifier; rows are grouped by this |
| `pathology` | Condition / disease label (the classification target) |
| `patient` | Patient identifier (may be blank) |

To use your own data, convert it to this schema and point `raw_csv` (or
`raw_data_dir`) in the config at it.

## Usage

The recommended starting point is the tutorial notebook `notebooks/pipeline.ipynb`,
which walks through the full analysis end-to-end — preprocessing, features, KS
tests, random-forest classification, and the HSIC independence test — with
plots inline.

To reproduce all result tables headlessly (e.g. on a cluster), run the CLI
orchestrator, which executes the same five stages (preprocess → features → KS
tests → random forest → HSIC) and writes its outputs to `results/`:

```bash
python -m src.pipeline --config configs/default.yaml --output-dir results/
```

### Try it on synthetic data

Since the study CSVs are not distributed (see [Data](#data)), you can run the
pipeline end-to-end on synthetic data instead. Generate a small dataset (two
conditions, one tumour + three immune cell types) and run the demo config:

```bash
python -m src.simulate_data data/raw/simulated_data.csv
python -m src.pipeline --config configs/demo.yaml --output-dir results/demo
```

Or import individual stages directly:

```python
from src.preprocessing import load_dataset, build_system
from src.features import compute_local_density, compute_katic_order
from src.stats import ks_test_per_feature, adjust_pvalues
from src.ml import cross_validate_random_forest
from src.hsic import hsic_test, hsic_sweep
```

## Code structure

| File | Stage |
|---|---|
| `src/preprocessing.py` | Load centroids, deduplicate, encode cell types, build simulation boxes |
| `src/features.py` | Local density and k-atic order parameters (psi_k) per cell and per cell-type combination |
| `src/stats.py` | Univariate two-sample tests (KS) with BH/BY correction and KDE-grid plots |
| `src/ml.py` | Cross-validated random-forest classification with bootstrap confidence intervals and feature-group selection |
| `src/hsic.py` | HSIC test for independence between the psi (k-atic order) and local-density feature blocks, with permutation p-values and a cutoff × split sweep |
| `src/pipeline.py` | End-to-end orchestrator; also the CLI entry point |
| `notebooks/pipeline.ipynb` | Tutorial notebook walking through the full analysis |
| `src/simulate_data.py` | Generate a small synthetic dataset in the unified schema for demos/testing |

## Configuration

Edit `configs/default.yaml` to set the dataset path, bond cutoff, k-range,
and cell-type definitions.

## License

Copyright The Regents of the University of Michigan.

Released under the BSD 3-Clause License — see [LICENSE](LICENSE).
