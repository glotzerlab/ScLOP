"""Run the full analysis pipeline end-to-end.

Usage:
    python pipeline.py --config configs/default.yaml --output-dir results/

Or run stages individually by importing from the stage modules directly.
"""

import argparse
from pathlib import Path

import yaml


def run(config: dict, output_dir: Path) -> None:
    """Execute all four stages in sequence."""
    raise NotImplementedError


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    run(config, args.output_dir)


if __name__ == "__main__":
    main()
