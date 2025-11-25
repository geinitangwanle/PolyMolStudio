#!/usr/bin/env python
"""
Train GNN predictor using config/gnn/train.yaml.
"""

import argparse
from pathlib import Path

from models.gnn_predictor.train import run as run_gnn_train


def main():
    parser = argparse.ArgumentParser(description="Train GNN predictor.")
    parser.add_argument("--config", type=Path, default=Path("configs/gnn/train.yaml"))
    args = parser.parse_args()
    run_gnn_train(_load_yaml(args.config))


def _load_yaml(path: Path):
    import yaml

    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


if __name__ == "__main__":
    main()
