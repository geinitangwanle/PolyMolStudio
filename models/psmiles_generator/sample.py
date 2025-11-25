"""
Wrapper that dispatches to models.PolySmith.unified_cli for sampling.
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
    extra_args = list(cfg.get("extra_args", []))

    # Optional routing for output paths (works with PolySmith sampling scripts that accept these flags).
    output_path = cfg.get("output_path")
    output_dir = cfg.get("output_dir")
    samples_file = cfg.get("samples_file")
    if output_path:
        op = Path(output_path)
        output_dir = output_dir or op.parent
        samples_file = samples_file or op.name
    if output_dir and "--output-dir" not in extra_args:
        extra_args += ["--output-dir", str(output_dir)]
    if samples_file and "--samples-file" not in extra_args:
        extra_args += ["--samples-file", str(samples_file)]
    args = [
        sys.executable,
        "-m",
        "models.PolySmith.unified_cli",
        "sample",
        "--version",
        version,
    ]
    if mode:
        args += ["--mode", mode]
    args += ["--"] + [str(x) for x in extra_args]
    subprocess.run(args, check=True)


def main():
    parser = argparse.ArgumentParser(description="Sample PSMILES via config.")
    parser.add_argument("--config", type=Path, default=Path("configs/psmiles/sample.yaml"))
    parsed = parser.parse_args()
    cfg = load_config(parsed.config)
    run(cfg)


if __name__ == "__main__":
    main()
