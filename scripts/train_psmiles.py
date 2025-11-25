#!/usr/bin/env python
"""
Train PSMILES generator via configs/psmiles/train.yaml.
"""

import argparse
from pathlib import Path

from models.psmiles_generator.train import run as run_psmiles_train


def main():
    parser = argparse.ArgumentParser(description="Train PSMILES generator.")
    parser.add_argument("--config", type=Path, default=Path("configs/psmiles/train.yaml"))
    args = parser.parse_args()
    run_psmiles_train(_load_yaml(args.config))


def _load_yaml(path: Path):
    import yaml

    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


if __name__ == "__main__":
    main()
