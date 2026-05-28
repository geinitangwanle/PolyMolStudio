#!/usr/bin/env python3
"""Nearest-neighbor Morgan/Tanimoto analysis for Tg test predictions.

Examples
--------
Use an explicit train CSV and labeled prediction CSV:

  python demo/nearest_neighbor_similarity_analysis.py \
    --train-csv data/raw/PSMILES_Tg_only.csv \
    --test-csv demo/analysis_outputs/test_predictions.csv \
    --train-smiles-col PSMILES \
    --test-smiles-col psmiles \
    --label-col label \
    --pred-col pred

Recreate the train/test split used by train_predictor.py, then attach a
prediction CSV by mol_id if available, or by test split order otherwise:

  python demo/nearest_neighbor_similarity_analysis.py \
    --manifest data/graphs_tg2/manifest.csv \
    --pred-csv demo/analysis_outputs/test_predictions.csv \
    --split-from-manifest \
    --out-dir demo/analysis_outputs/nn_similarity
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm

try:
    from rdkit import Chem, DataStructs, RDLogger
    from rdkit.Chem import AllChem
except Exception as exc:  # pragma: no cover - handled at runtime
    Chem = None
    DataStructs = None
    AllChem = None
    RDLogger = None
    RDKIT_IMPORT_ERROR = exc
else:
    RDKIT_IMPORT_ERROR = None


BIN_ORDER = ["high", "medium", "low"]
BIN_LABELS = {
    "high": "Tanimoto >= 0.8",
    "medium": "0.5 <= Tanimoto < 0.8",
    "low": "Tanimoto < 0.5",
}


def require_rdkit() -> None:
    if Chem is None or DataStructs is None or AllChem is None:
        raise RuntimeError(f"RDKit is required for this analysis: {RDKIT_IMPORT_ERROR}")
    RDLogger.DisableLog("rdApp.warning")
    RDLogger.DisableLog("rdApp.error")


def canonical_text(value) -> str:
    return str(value).strip()


def make_fp(smiles: str, radius: int, n_bits: int):
    mol = Chem.MolFromSmiles(canonical_text(smiles))
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)


def build_fingerprints(
    df: pd.DataFrame,
    smiles_col: str,
    radius: int,
    n_bits: int,
    desc: str,
) -> Tuple[List, List[int]]:
    fps = []
    valid_indices = []
    for idx, smiles in tqdm(df[smiles_col].items(), total=len(df), desc=desc):
        fp = make_fp(smiles, radius=radius, n_bits=n_bits)
        if fp is None:
            continue
        fps.append(fp)
        valid_indices.append(idx)
    return fps, valid_indices


def nearest_neighbor_similarity(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    train_smiles_col: str,
    test_smiles_col: str,
    radius: int,
    n_bits: int,
) -> pd.DataFrame:
    train_fps, train_valid_idx = build_fingerprints(
        train_df, train_smiles_col, radius, n_bits, "Train Morgan fingerprints"
    )
    if not train_fps:
        raise ValueError("No valid train fingerprints were generated.")

    rows = []
    for test_idx, test_smiles in tqdm(
        test_df[test_smiles_col].items(), total=len(test_df), desc="Nearest train neighbor"
    ):
        test_fp = make_fp(test_smiles, radius=radius, n_bits=n_bits)
        if test_fp is None:
            rows.append(
                {
                    "_test_index": test_idx,
                    "test_smiles": test_smiles,
                    "nn_similarity": np.nan,
                    "nn_train_index": np.nan,
                    "nn_train_smiles": None,
                }
            )
            continue

        sims = DataStructs.BulkTanimotoSimilarity(test_fp, train_fps)
        best_pos = int(np.argmax(sims))
        train_idx = train_valid_idx[best_pos]
        rows.append(
            {
                "_test_index": test_idx,
                "test_smiles": test_smiles,
                "nn_similarity": float(sims[best_pos]),
                "nn_train_index": train_idx,
                "nn_train_smiles": train_df.at[train_idx, train_smiles_col],
            }
        )
    return pd.DataFrame(rows)


def split_manifest(
    manifest: pd.DataFrame,
    seed: int,
    test_split: float,
    val_split: float,
    label_col: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    stratify_labels = None
    if label_col in manifest.columns:
        try:
            bins = pd.qcut(manifest[label_col], q=10, duplicates="drop")
            stratify_labels = bins.astype(str)
        except Exception:
            stratify_labels = None

    train_val_df, test_df = train_test_split(
        manifest,
        test_size=test_split,
        random_state=seed,
        stratify=stratify_labels,
    )

    stratify_trainval = None
    if stratify_labels is not None:
        try:
            train_val_bins = pd.qcut(train_val_df[label_col], q=10, duplicates="drop")
            stratify_trainval = train_val_bins.astype(str)
        except Exception:
            stratify_trainval = None

    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_split / (1 - test_split),
        random_state=seed,
        stratify=stratify_trainval,
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


def attach_predictions(
    test_df: pd.DataFrame,
    pred_csv: Optional[Path],
    pred_col: str,
    label_col: str,
    id_col: Optional[str],
) -> pd.DataFrame:
    if pred_csv is None:
        return test_df.copy()

    pred_df = pd.read_csv(pred_csv)
    pred_df.columns = pred_df.columns.str.strip()
    if pred_col not in pred_df.columns:
        raise ValueError(f"Prediction column {pred_col!r} not found in {pred_csv}")

    out = test_df.copy()
    if id_col and id_col in out.columns and id_col in pred_df.columns:
        keep_cols = [id_col, pred_col]
        if label_col in pred_df.columns and label_col not in out.columns:
            keep_cols.append(label_col)
        return out.merge(pred_df[keep_cols], on=id_col, how="left", suffixes=("", "_predcsv"))

    if len(pred_df) != len(out):
        raise ValueError(
            f"Cannot align predictions by order: test rows={len(out)}, pred rows={len(pred_df)}. "
            "Provide --id-col present in both files."
        )
    out[pred_col] = pred_df[pred_col].to_numpy()
    if label_col in pred_df.columns and label_col not in out.columns:
        out[label_col] = pred_df[label_col].to_numpy()
    return out


def similarity_bin(value: float) -> str:
    if pd.isna(value):
        return "invalid"
    if value >= 0.8:
        return "high"
    if value >= 0.5:
        return "medium"
    return "low"


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    metrics = {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(mean_squared_error(y_true, y_pred, squared=False)),
    }
    if len(y_true) >= 2:
        metrics["r2"] = float(r2_score(y_true, y_pred))
    else:
        metrics["r2"] = float("nan")
    return metrics


def summarize_by_bin(df: pd.DataFrame, label_col: str, pred_col: str) -> pd.DataFrame:
    rows = []
    valid_metric_df = df.dropna(subset=["nn_similarity", label_col, pred_col]).copy()
    if not valid_metric_df.empty:
        y_true = valid_metric_df[label_col].astype(float).to_numpy()
        y_pred = valid_metric_df[pred_col].astype(float).to_numpy()
        rows.append(
            {
                "bin": "overall",
                "range": "all valid",
                "n": len(valid_metric_df),
                "mean_similarity": float(valid_metric_df["nn_similarity"].mean()),
                **regression_metrics(y_true, y_pred),
            }
        )

    for bin_name in BIN_ORDER:
        sub = valid_metric_df[valid_metric_df["similarity_bin"] == bin_name]
        if sub.empty:
            rows.append(
                {
                    "bin": bin_name,
                    "range": BIN_LABELS[bin_name],
                    "n": 0,
                    "mean_similarity": np.nan,
                    "mae": np.nan,
                    "rmse": np.nan,
                    "r2": np.nan,
                }
            )
            continue
        y_true = sub[label_col].astype(float).to_numpy()
        y_pred = sub[pred_col].astype(float).to_numpy()
        rows.append(
            {
                "bin": bin_name,
                "range": BIN_LABELS[bin_name],
                "n": len(sub),
                "mean_similarity": float(sub["nn_similarity"].mean()),
                **regression_metrics(y_true, y_pred),
            }
        )
    return pd.DataFrame(rows)


def plot_error_vs_similarity(df: pd.DataFrame, out_path: Path, label_col: str, pred_col: str) -> None:
    plot_df = df.dropna(subset=["nn_similarity", label_col, pred_col]).copy()
    if plot_df.empty:
        return
    plot_df["abs_error"] = (plot_df[pred_col].astype(float) - plot_df[label_col].astype(float)).abs()

    colors = {"high": "#2ca02c", "medium": "#ff7f0e", "low": "#d62728"}
    plt.figure(figsize=(6.5, 4.5))
    for bin_name in BIN_ORDER:
        sub = plot_df[plot_df["similarity_bin"] == bin_name]
        if sub.empty:
            continue
        plt.scatter(
            sub["nn_similarity"],
            sub["abs_error"],
            s=18,
            alpha=0.7,
            color=colors[bin_name],
            label=f"{bin_name}: {BIN_LABELS[bin_name]}",
        )
    plt.axvline(0.5, color="#666666", linestyle="--", linewidth=1)
    plt.axvline(0.8, color="#666666", linestyle="--", linewidth=1)
    plt.xlabel("Nearest train Tanimoto similarity")
    plt.ylabel("Absolute Tg error / K")
    plt.legend(frameon=False, fontsize=8)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=250)
    plt.close()


def plot_metric_bars(summary: pd.DataFrame, out_path: Path) -> None:
    plot_df = summary[summary["bin"].isin(BIN_ORDER)].copy()
    if plot_df.empty:
        return

    colors = {"high": "#2ca02c", "medium": "#ff7f0e", "low": "#d62728"}
    labels = [f"{row.bin}\n(n={int(row.n)})" for row in plot_df.itertuples()]
    x = np.arange(len(plot_df))
    width = 0.36

    plt.figure(figsize=(6.2, 4.2))
    mae_bars = plt.bar(
        x - width / 2,
        plot_df["mae"],
        width,
        color=[colors[b] for b in plot_df["bin"]],
        alpha=0.85,
        label="MAE",
    )
    rmse_bars = plt.bar(
        x + width / 2,
        plot_df["rmse"],
        width,
        color=[colors[b] for b in plot_df["bin"]],
        alpha=0.38,
        hatch="//",
        label="RMSE",
    )
    for bars in (mae_bars, rmse_bars):
        for bar in bars:
            height = bar.get_height()
            if np.isfinite(height):
                plt.text(
                    bar.get_x() + bar.get_width() / 2,
                    height,
                    f"{height:.1f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

    plt.xticks(x, labels)
    plt.ylabel("Prediction error / K")
    plt.xlabel("Nearest-neighbor similarity group")
    plt.legend(frameon=False)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=250)
    plt.close()


def metrics_to_json(summary: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    out = {}
    for _, row in summary.iterrows():
        out[str(row["bin"])] = {
            "range": row["range"],
            "n": int(row["n"]),
            "mean_similarity": None if pd.isna(row["mean_similarity"]) else float(row["mean_similarity"]),
            "mae": None if pd.isna(row["mae"]) else float(row["mae"]),
            "rmse": None if pd.isna(row["rmse"]) else float(row["rmse"]),
            "r2": None if pd.isna(row["r2"]) else float(row["r2"]),
        }
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--manifest", type=Path, help="Manifest CSV used by train_predictor.py.")
    source.add_argument("--train-csv", type=Path, help="Explicit train CSV.")
    parser.add_argument("--test-csv", type=Path, help="Explicit labeled test/prediction CSV.")
    parser.add_argument("--pred-csv", type=Path, help="Prediction CSV to attach to manifest test split.")
    parser.add_argument("--split-from-manifest", action="store_true", help="Recreate train/val/test split from --manifest.")

    parser.add_argument("--train-smiles-col", default="psmiles")
    parser.add_argument("--test-smiles-col", default="psmiles")
    parser.add_argument("--label-col", default="label")
    parser.add_argument("--pred-col", default="pred")
    parser.add_argument("--id-col", default="mol_id", help="Optional id column for aligning pred-csv to manifest test split.")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-split", type=float, default=0.1)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--morgan-radius", type=int, default=2)
    parser.add_argument("--morgan-bits", type=int, default=2048)
    parser.add_argument("--out-dir", type=Path, default=Path("demo/analysis_outputs/nn_similarity"))
    parser.add_argument("--prefix", default="nn_similarity")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    require_rdkit()

    if args.manifest:
        manifest = pd.read_csv(args.manifest)
        manifest.columns = manifest.columns.str.strip()
        if not args.split_from_manifest:
            raise ValueError("--manifest requires --split-from-manifest so train/test are well defined.")
        train_df, _, test_df = split_manifest(
            manifest,
            seed=args.seed,
            test_split=args.test_split,
            val_split=args.val_split,
            label_col=args.label_col,
        )
        test_df = attach_predictions(test_df, args.pred_csv, args.pred_col, args.label_col, args.id_col)
    else:
        if args.test_csv is None:
            raise ValueError("--train-csv requires --test-csv.")
        train_df = pd.read_csv(args.train_csv)
        test_df = pd.read_csv(args.test_csv)
        train_df.columns = train_df.columns.str.strip()
        test_df.columns = test_df.columns.str.strip()

    for col, name, df in [
        (args.train_smiles_col, "train smiles", train_df),
        (args.test_smiles_col, "test smiles", test_df),
    ]:
        if col not in df.columns:
            raise ValueError(f"Missing {name} column {col!r}. Available columns: {list(df.columns)}")

    nn_df = nearest_neighbor_similarity(
        train_df=train_df,
        test_df=test_df,
        train_smiles_col=args.train_smiles_col,
        test_smiles_col=args.test_smiles_col,
        radius=args.morgan_radius,
        n_bits=args.morgan_bits,
    )
    result_df = test_df.reset_index(drop=True).copy()
    result_df = pd.concat([result_df, nn_df.drop(columns=["_test_index"]).reset_index(drop=True)], axis=1)
    result_df["similarity_bin"] = result_df["nn_similarity"].map(similarity_bin)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    detail_path = args.out_dir / f"{args.prefix}_per_sample.csv"
    result_df.to_csv(detail_path, index=False)

    has_metrics = args.label_col in result_df.columns and args.pred_col in result_df.columns
    if has_metrics:
        summary = summarize_by_bin(result_df, label_col=args.label_col, pred_col=args.pred_col)
        summary_path = args.out_dir / f"{args.prefix}_summary.csv"
        json_path = args.out_dir / f"{args.prefix}_summary.json"
        plot_path = args.out_dir / f"{args.prefix}_error_vs_similarity.png"
        bars_path = args.out_dir / f"{args.prefix}_mae_rmse_by_similarity.png"
        summary.to_csv(summary_path, index=False)
        json_path.write_text(json.dumps(metrics_to_json(summary), indent=2, ensure_ascii=False))
        plot_error_vs_similarity(result_df, plot_path, label_col=args.label_col, pred_col=args.pred_col)
        plot_metric_bars(summary, bars_path)
        print(summary.to_string(index=False))
        print(f"Wrote per-sample results: {detail_path}")
        print(f"Wrote summary: {summary_path}")
        print(f"Wrote plot: {plot_path}")
        print(f"Wrote metric bars: {bars_path}")
    else:
        counts = result_df["similarity_bin"].value_counts(dropna=False).to_dict()
        counts_path = args.out_dir / f"{args.prefix}_bin_counts.json"
        counts_path.write_text(json.dumps(counts, indent=2, ensure_ascii=False))
        print("Prediction/label columns not both present; wrote similarity distribution only.")
        print(json.dumps(counts, indent=2, ensure_ascii=False))
        print(f"Wrote per-sample results: {detail_path}")


if __name__ == "__main__":
    main()
