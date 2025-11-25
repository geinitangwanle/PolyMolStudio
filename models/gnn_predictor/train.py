"""
Wrapper trainer that delegates to the existing polyGeoGAT training script.

Reads a YAML config and forwards the options to `python -m models.polyGeoGAT.train`.
"""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict

import yaml


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def run(cfg: Dict[str, Any]):
    args = [
        sys.executable,
        "-m",
        "models.polyGeoGAT.train",
        "--data_path",
        str(cfg["data_path"]),
        "--root_dir",
        str(cfg.get("root_dir", ".")),
        "--epochs",
        str(cfg.get("epochs", 50)),
        "--batch_size",
        str(cfg.get("batch_size", 32)),
        "--device",
        str(cfg.get("device", "auto")),
        "--log_dir",
        str(cfg.get("log_dir", "./logs")),
        "--checkpoint_dir",
        str(cfg.get("checkpoint_dir", "./checkpoints")),
    ]
    if "lr" in cfg:
        args += ["--lr", str(cfg["lr"])]
    if "weight_decay" in cfg:
        args += ["--weight_decay", str(cfg["weight_decay"])]
    subprocess.run(args, check=True)


def main():
    parser = argparse.ArgumentParser(description="Train GNN predictor via config.")
    parser.add_argument("--config", type=Path, default=Path("configs/gnn/train.yaml"))
    parsed = parser.parse_args()
    cfg = load_config(parsed.config)
    run(cfg)


if __name__ == "__main__":
    main()
