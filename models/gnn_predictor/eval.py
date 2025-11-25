"""
Evaluation wrapper: dispatch to polyGeoGAT.predict for held-out CSVs.
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
        "models.polyGeoGAT.predict",
        "--ckpt_path",
        str(cfg["ckpt_path"]),
        "--csv_path",
        str(cfg["csv_path"]),
        "--psmiles_col",
        str(cfg.get("psmiles_col", "PSMILES")),
        "--save_dir",
        str(cfg.get("save_dir", "pred_graphs")),
    ]
    if "out_csv" in cfg:
        args += ["--out_csv", str(cfg["out_csv"])]
    if "device" in cfg:
        args += ["--device", str(cfg["device"])]
    subprocess.run(args, check=True)


def main():
    parser = argparse.ArgumentParser(description="Evaluate/predict with GNN via config.")
    parser.add_argument("--config", type=Path, default=Path("configs/gnn/eval.yaml"))
    parsed = parser.parse_args()
    cfg = load_config(parsed.config)
    run(cfg)


if __name__ == "__main__":
    main()
