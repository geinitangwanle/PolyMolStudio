"""
Wrapper that dispatches to models.PolySmith.unified_cli for training.
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
    version = cfg.get("version", "v4")
    mode = cfg.get("mode")
    extra_args = cfg.get("extra_args", [])
    args = [
        sys.executable,
        "-m",
        "models.PolySmith.unified_cli",
        "train",
        "--version",
        version,
    ]
    if mode:
        args += ["--mode", mode]
    args += ["--"] + [str(x) for x in extra_args]
    subprocess.run(args, check=True)


def main():
    parser = argparse.ArgumentParser(description="Train PSMILES generator via config.")
    parser.add_argument("--config", type=Path, default=Path("configs/psmiles/train.yaml"))
    parsed = parser.parse_args()
    cfg = load_config(parsed.config)
    run(cfg)


if __name__ == "__main__":
    main()
