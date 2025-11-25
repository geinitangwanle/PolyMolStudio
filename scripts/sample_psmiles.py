#!/usr/bin/env python
"""
Sample PSMILES using configs/psmiles/sample.yaml.
"""

import argparse
from pathlib import Path

from models.psmiles_generator.sample import run as run_psmiles_sample


def main():
    parser = argparse.ArgumentParser(description="Sample PSMILES.")
    parser.add_argument("--config", type=Path, default=Path("configs/psmiles/sample.yaml"))
    args = parser.parse_args()
    run_psmiles_sample(_load_yaml(args.config))


def _load_yaml(path: Path):
    import yaml

    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


if __name__ == "__main__":
    main()
