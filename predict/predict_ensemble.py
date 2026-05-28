#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import glob
import json
import random
import re
import sys
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from predict.predict import build_model_from_ckpt, predict_loader
from utils.GraphDataset import GraphDataset
from utils.PSMILES_to_graph import convert_csv_to_graphs


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def expand_ckpt_paths(patterns: List[str]) -> List[Path]:
    def natural_key(path: str) -> List:
        return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", path)]

    paths: List[Path] = []
    for pattern in patterns:
        matched = [Path(p) for p in sorted(glob.glob(pattern), key=natural_key)] if any(ch in pattern for ch in "*?[]") else [Path(pattern)]
        paths.extend(matched)

    unique_paths = []
    seen = set()
    for path in paths:
        resolved = path.resolve()
        if resolved in seen:
            continue
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        seen.add(resolved)
        unique_paths.append(path)
    return unique_paths


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    if y_true.size == 0:
        return {}

    diff = y_pred - y_true
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff**2)))
    ss_res = float(np.sum(diff**2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    return {"n": int(y_true.size), "mae": mae, "rmse": rmse, "r2": r2}


def predict_one_checkpoint(
    ckpt_path: Path,
    manifest_df: pd.DataFrame,
    graph_dir: Path,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> np.ndarray:
    checkpoint = torch.load(str(ckpt_path), map_location="cpu")
    model, tokenizer, cfg = build_model_from_ckpt(checkpoint, device)

    dataset = GraphDataset(
        manifest=manifest_df,
        root=graph_dir,
        separate_pos=True,
        feature_cols=(0, 1, 2, 3),
        coord_cols=(4, 5, 6),
        standardize_y=False,
        tokenizer=tokenizer,
        psmiles_col="psmiles",
        seq_max_length=cfg.get("seq_max_length", 256),
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    if isinstance(checkpoint, dict):
        state_dict = checkpoint.get("model_state") or checkpoint.get("state_dict")
    else:
        state_dict = None
    if state_dict is None:
        state_dict = checkpoint
    model.load_state_dict(state_dict, strict=False)
    model.to(device)

    y_mean = checkpoint.get("y_mean") if isinstance(checkpoint, dict) else None
    y_std = checkpoint.get("y_std") if isinstance(checkpoint, dict) else None
    y_mean_t = torch.tensor(y_mean, dtype=torch.float32, device=device) if y_mean is not None else None
    y_std_t = torch.tensor(y_std, dtype=torch.float32, device=device) if y_std is not None else None

    pred_rows = predict_loader(model, loader, device, y_mean_t, y_std_t)
    pred_by_id = {row["mol_id"]: row["pred"] for row in pred_rows}
    preds = np.full(len(manifest_df), np.nan, dtype=float)
    for i, row in manifest_df.reset_index(drop=True).iterrows():
        mol_id = int(row["mol_id"]) if "mol_id" in row else i
        if mol_id in pred_by_id:
            preds[i] = pred_by_id[mol_id]
    return preds


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict with multiple checkpoints and estimate ensemble uncertainty.")
    parser.add_argument(
        "--ckpt_paths",
        nargs="+",
        required=True,
        help="Checkpoint paths or glob patterns, e.g. 'checkpoints/*/best.pt'.",
    )
    parser.add_argument("--csv_path", type=str, required=True, help="Input CSV containing PSMILES.")
    parser.add_argument("--psmiles_col", type=str, required=True, help="Column name for PSMILES.")
    parser.add_argument("--label_col", type=str, default=None, help="Optional true Tg column for metrics.")
    parser.add_argument("--save_dir", type=str, default="ensemble_pred_graphs")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_csv", type=str, default=None)
    parser.add_argument("--summary_json", type=str, default=None)
    parser.add_argument(
        "--graph_timeout_seconds",
        type=float,
        default=120.0,
        help="Per-molecule graph construction timeout in seconds; <=0 disables it.",
    )
    args = parser.parse_args()

    set_seed(args.seed)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    input_df = pd.read_csv(args.csv_path)
    if args.psmiles_col not in input_df.columns:
        if "smiles" in input_df.columns:
            print(f"[warn] column '{args.psmiles_col}' not found, falling back to 'smiles'")
            args.psmiles_col = "smiles"
        else:
            raise ValueError(f"PSMILES column '{args.psmiles_col}' not found in CSV")

    ckpt_paths = expand_ckpt_paths(args.ckpt_paths)
    if len(ckpt_paths) < 2:
        raise ValueError("Ensemble uncertainty needs at least 2 checkpoints.")

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    _, manifest_df = convert_csv_to_graphs(
        csv_path=args.csv_path,
        label_col=None,
        PSMILES_col=args.psmiles_col,
        save_dir=str(save_dir),
        graph_timeout_seconds=args.graph_timeout_seconds,
    )

    all_preds = []
    for i, ckpt_path in enumerate(ckpt_paths, start=1):
        print(f"[{i}/{len(ckpt_paths)}] Predicting with {ckpt_path}")
        preds = predict_one_checkpoint(
            ckpt_path=ckpt_path,
            manifest_df=manifest_df,
            graph_dir=save_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=device,
        )
        all_preds.append(preds)

    out_df = input_df.copy()
    output_col = "csv_row" if "csv_row" in manifest_df.columns else "mol_id"
    output_indices = [int(i) for i in manifest_df[output_col]]

    pred_matrix_manifest = np.vstack(all_preds).T
    pred_matrix = np.full((len(out_df), len(ckpt_paths)), np.nan, dtype=float)
    for manifest_row, output_idx in enumerate(output_indices):
        if output_idx is None or output_idx < 0 or output_idx >= len(out_df):
            continue
        pred_matrix[output_idx, :] = pred_matrix_manifest[manifest_row, :]

    pred_mean = np.nanmean(pred_matrix, axis=1)
    pred_std = np.nanstd(pred_matrix, axis=1, ddof=1)
    pred_sem = pred_std / np.sqrt(len(ckpt_paths))

    for i in range(pred_matrix.shape[1]):
        out_df[f"pred_model_{i + 1:02d}"] = pred_matrix[:, i]
    out_df["pred_ensemble_mean"] = pred_mean
    out_df["pred_ensemble_std"] = pred_std
    out_df["pred_ensemble_sem"] = pred_sem
    out_df["pred_ensemble_ci95_low"] = pred_mean - 1.96 * pred_std
    out_df["pred_ensemble_ci95_high"] = pred_mean + 1.96 * pred_std

    out_csv = Path(args.out_csv) if args.out_csv else save_dir / "ensemble_predictions.csv"
    out_df.to_csv(out_csv, index=False)

    summary = {
        "n_models": len(ckpt_paths),
        "ckpt_paths": [str(path) for path in ckpt_paths],
        "n_input_rows": int(len(out_df)),
        "n_graph_rows": int(len(manifest_df)),
        "n_predicted_rows": int(np.isfinite(pred_mean).sum()),
        "n_unscored_rows": int((~np.isfinite(pred_mean)).sum()),
        "mean_ensemble_std": float(np.nanmean(pred_std)),
        "median_ensemble_std": float(np.nanmedian(pred_std)),
        "max_ensemble_std": float(np.nanmax(pred_std)),
        "out_csv": str(out_csv),
    }
    if args.label_col and args.label_col in out_df.columns:
        summary["metrics"] = metrics(
            out_df[args.label_col].to_numpy(dtype=float),
            out_df["pred_ensemble_mean"].to_numpy(dtype=float),
        )

    summary_json = Path(args.summary_json) if args.summary_json else save_dir / "ensemble_summary.json"
    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved ensemble predictions to: {out_csv.resolve()}")
    print(f"Saved ensemble summary to: {summary_json.resolve()}")
    print(
        "Ensemble std: "
        f"mean={summary['mean_ensemble_std']:.3f} K, "
        f"median={summary['median_ensemble_std']:.3f} K, "
        f"max={summary['max_ensemble_std']:.3f} K"
    )
    if "metrics" in summary:
        m = summary["metrics"]
        print(f"Metrics: MAE={m['mae']:.3f} K, RMSE={m['rmse']:.3f} K, R2={m['r2']:.4f}")


if __name__ == "__main__":
    main()
