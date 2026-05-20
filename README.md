# SCSAP — Single-Cell Spatial Analysis Pipeline

Code accompanying [paper citation]. Analyzes the spatial organization of cells
in tissue images to distinguish between tissue conditions using spatial
statistics and machine learning.

## Dependencies

```bash
pip install -r requirements.txt
```

## Usage

```bash
python src/pipeline.py --config configs/default.yaml --output-dir results/
```

Or import individual stages directly:

```python
from src.preprocessing import load_dataset, build_system
from src.features import compute_local_density, compute_katic_order
from src.stats import run_battery, adjust_pvalues
from src.ml import cross_validate
```

## Code structure

| File | Stage |
|---|---|
| `src/preprocessing.py` | Load centroids, deduplicate, encode cell types, build simulation boxes |
| `src/features.py` | Local density and k-atic order parameters (psi_k) per cell and per cell-type combination |
| `src/stats.py` | Univariate two-sample tests (KS, t, Mann-Whitney, etc.) with BH/BY correction |
| `src/ml.py` | Cross-validated random-forest classification with bootstrap confidence intervals |
| `src/pipeline.py` | End-to-end orchestrator; also the CLI entry point |

## Configuration

Edit `configs/default.yaml` to set the dataset path, bond cutoff, k-range,
and cell-type definitions.
