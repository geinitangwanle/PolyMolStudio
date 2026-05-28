#!/usr/bin/env python3
"""
Quick analysis/visualization for prediction CSVs produced by GNN/Tg models.

Features:
- Load a CSV (required) and a reference CSV (optional) and extract prediction columns.
- Compute basic stats (count/mean/std/min/max/percentiles) and, if labels are provided,
  regression metrics (MAE/RMSE/R2).
- Save metrics to JSON and plot histograms (and scatter when labels exist).

Example:
  python demo/analyze_predictions.py \
    --csv outputs_pretrain/samples_pretrain_.csv \
    --pred-col pred \
    --label-col label \
    --out-dir demo/analysis_outputs \
    --prefix rl_run
# 仅预测分布
python demo/analyze_predictions.py \
  --csv outputs_pretrain/samples_pretrain_prediction.csv \
  --pred-col pred \
  --out-dir outputs_pretrain/analysis_outputs \
  --prefix pretrain

# 对比 RL 与非 RL 的预测分布
python demo/analyze_predictions.py \
  --csv outputs_pretrain2/samples_pretrain_prediction.csv \
  --pred-col pred \
  --ref-csv outputs_pretrain/samples_pretrain_prediction.csv \
  --out-dir demo/analysis_outputs \
  --prefix rl_vs_base

# 有标签时计算回归指标并画散点
python demo/analyze_predictions.py \
  --csv data/raw/LAMALAB_CURATED_Tg_structured.csv \
  --pred-col pred --label-col Tg \
  --out-dir demo/analysis_outputs \
  --prefix labeled_run

"""

import argparse
import json
from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def basic_stats(arr: np.ndarray) -> Dict[str, float]:
    percentiles = [5, 25, 50, 75, 95]
    stats = {
        "count": int(arr.size),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }
    for p in percentiles:
        stats[f"p{p}"] = float(np.percentile(arr, p))
    return stats


def plot_hist(values: np.ndarray, label: str, out_path: Path, overlay: Optional[np.ndarray] = None):
    plt.figure(figsize=(6, 4))
    plt.hist(values, bins=50, alpha=0.7, label=label, color="#1f77b4", density=True)
    if overlay is not None:
        plt.hist(overlay, bins=50, alpha=0.5, label="reference", color="#ff7f0e", density=True)
    plt.xlabel(label)
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_scatter(y_true: np.ndarray, y_pred: np.ndarray, out_path: Path, title: str):
    plt.figure(figsize=(5, 5))
    plt.scatter(y_true, y_pred, s=10, alpha=0.5, color="#1f77b4")
    min_v = min(y_true.min(), y_pred.min())
    max_v = max(y_true.max(), y_pred.max())
    plt.plot([min_v, max_v], [min_v, max_v], "r--", label="ideal")
    plt.xlabel("True")
    plt.ylabel("Pred")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()


def analyze(csv_path: Path, pred_col: str, label_col: Optional[str], out_dir: Path, prefix: str, ref_csv: Optional[Path], ref_pred_col: Optional[str]):
    df = pd.read_csv(csv_path)
    if pred_col not in df.columns:
        raise ValueError(f"Column {pred_col} not found in {csv_path}")
    preds = df[pred_col].dropna().astype(float).to_numpy()

    ref_preds = None
    if ref_csv is not None:
        ref_df = pd.read_csv(ref_csv)
        col = ref_pred_col or pred_col
        if col not in ref_df.columns:
            raise ValueError(f"Column {col} not found in reference CSV {ref_csv}")
        ref_preds = ref_df[col].dropna().astype(float).to_numpy()

    metrics = {
        "pred_stats": basic_stats(preds),
        "source": str(csv_path),
    }

    if label_col and label_col in df.columns:
        labels = df[label_col].dropna().astype(float).to_numpy()
        n = min(len(preds), len(labels))
        y_p = preds[:n]
        y_t = labels[:n]
        metrics["regression"] = {
            "mae": float(mean_absolute_error(y_t, y_p)),
            "rmse": float(mean_squared_error(y_t, y_p, squared=False)),
            "r2": float(r2_score(y_t, y_p)),
        }
        plot_scatter(y_t, y_p, out_dir / f"{prefix}_scatter.png", title=f"Pred vs True ({prefix})")

    plot_hist(preds, label=pred_col, overlay=ref_preds, out_path=out_dir / f"{prefix}_hist.png")

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{prefix}_metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description="Analyze prediction CSV: stats and plots.")
    parser.add_argument("--csv", type=Path, required=True, help="Input prediction CSV.")
    parser.add_argument("--pred-col", type=str, default="pred", help="Column name for predictions.")
    parser.add_argument("--label-col", type=str, default=None, help="Optional label column for regression metrics.")
    parser.add_argument("--out-dir", type=Path, default=Path("demo/analysis_outputs"), help="Directory to save outputs.")
    parser.add_argument("--prefix", type=str, default="run", help="Prefix for output files.")
    parser.add_argument("--ref-csv", type=Path, default=None, help="Optional reference CSV to overlay hist.")
    parser.add_argument("--ref-pred-col", type=str, default=None, help="Prediction column in reference CSV (defaults to pred-col).")
    args = parser.parse_args()

    analyze(
        csv_path=args.csv,
        pred_col=args.pred_col,
        label_col=args.label_col,
        out_dir=args.out_dir,
        prefix=args.prefix,
        ref_csv=args.ref_csv,
        ref_pred_col=args.ref_pred_col,
    )


if __name__ == "__main__":
    main()
