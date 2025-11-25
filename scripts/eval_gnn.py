#!/usr/bin/env python
"""
Evaluate/predict with GNN using configs/gnn/eval.yaml.
"""

import argparse
from pathlib import Path

from models.gnn_predictor.eval import run as run_gnn_eval


def main():
    parser = argparse.ArgumentParser(description="Evaluate GNN predictor.")
    parser.add_argument("--config", type=Path, default=Path("configs/gnn/eval.yaml"))
    args = parser.parse_args()
    run_gnn_eval(_load_yaml(args.config))


def _load_yaml(path: Path):
    import yaml

    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


if __name__ == "__main__":
    main()
