#!/usr/bin/env python
"""
Pipeline: generate PSMILES -> normalize/filter (optional) -> predict property with GNN.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import yaml

from models.psmiles_generator.sample import run as run_psmiles_sample
from models.gnn_predictor.eval import run as run_gnn_eval
from models.psmiles_generator.filter import filter_and_dedup


def load_yaml(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def main():
    parser = argparse.ArgumentParser(description="Generate PSMILES then predict properties.")
    parser.add_argument("--config", type=Path, default=Path("configs/pipeline.yaml"))
    args = parser.parse_args()
    cfg = load_yaml(args.config)

    # 1) Run sampling if requested
    sample_cfg_path = Path(cfg.get("sample_config", "configs/psmiles/sample.yaml"))
    sample_cfg = load_yaml(sample_cfg_path) if sample_cfg_path.exists() else {}
    generated_csv = Path(cfg.get("generated_csv", "outputs/psmiles/samples.csv"))

    if not generated_csv.exists():
        generated_csv.parent.mkdir(parents=True, exist_ok=True)
        run_psmiles_sample(sample_cfg)
        if not generated_csv.exists():
            print(f"[pipeline] Sampling finished but {generated_csv} not found; please point sample config to this path.")

    # 2) Optional normalization/dedup (in-place)
    if generated_csv.exists():
        import pandas as pd

        df = pd.read_csv(generated_csv)
        col = cfg.get("psmiles_col", "PSMILES")
        if col in df.columns:
            filtered = filter_and_dedup(df[col].tolist())
            df = pd.DataFrame({col: filtered})
            df.to_csv(generated_csv, index=False)
            print(f"[pipeline] Filtered & deduped -> {generated_csv}")
        else:
            print(f"[pipeline] Column {col} not found in {generated_csv}, skipping filtering.")

    # 3) Predict with GNN
    gnn_cfg = {
        "ckpt_path": cfg["gnn_ckpt"],
        "csv_path": generated_csv,
        "psmiles_col": cfg.get("psmiles_col", "PSMILES"),
        "save_dir": cfg.get("save_dir", "outputs/gnn/pred_graphs"),
        "out_csv": cfg.get("pipeline_output", "outputs/pipeline/predicted.csv"),
        "device": cfg.get("device", "auto"),
    }
    run_gnn_eval(gnn_cfg)
    print(f"[pipeline] Predictions written to {gnn_cfg['out_csv']}")


if __name__ == "__main__":
    main()
