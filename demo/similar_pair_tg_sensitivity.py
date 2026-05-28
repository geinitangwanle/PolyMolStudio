#!/usr/bin/env python3
"""Pair-level sensitivity analysis for similar polymers.

The script finds highly similar polymer pairs by Morgan fingerprint Tanimoto
similarity and checks whether predicted Tg differences track experimental Tg
differences, especially the high/low Tg ordering within each pair.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem
from scipy.stats import pearsonr, spearmanr
from sklearn.model_selection import train_test_split
from torch_geometric.loader import DataLoader
from tqdm import tqdm
from transformers import AutoModel

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.generator.tokenizer import PolyBertTokenizer
from models.predictor.GeoGATModel import GeoGATModel
from utils.GraphDataset import GraphDataset


def split_manifest(
    manifest: pd.DataFrame,
    *,
    seed: int,
    test_split: float,
    val_split: float,
    label_col: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    stratify_labels = None
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


def build_model_from_checkpoint(ckpt_path: Path, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = ckpt["config"]
    polybert = None
    if cfg.get("use_polybert", False):
        polybert_name = cfg.get("polybert_name", str(REPO_ROOT / "polybert"))
        if polybert_name == "./polybert":
            polybert_name = str(REPO_ROOT / "polybert")
        polybert = AutoModel.from_pretrained(polybert_name)

    model = GeoGATModel(
        use_polybert=cfg.get("use_polybert", False),
        polybert=polybert,
        freeze_polybert=cfg.get("freeze_polybert", True),
        polybert_name=cfg.get("polybert_name", str(REPO_ROOT / "polybert")),
        seq_max_length=cfg.get("seq_max_length", 256),
        cross_attn_heads=cfg.get("cross_attn_heads", 4),
        cross_attn_dim=cfg.get("cross_attn_dim"),
        **{
            k: cfg[k]
            for k in cfg
            if k
            in {
                "layers_in_conv",
                "channels",
                "use_nodetype_coeffs",
                "num_node_types",
                "num_edge_types",
                "use_jumping_knowledge",
                "use_bias_for_update",
                "use_dropout",
                "num_convs",
                "num_fc_layers",
                "neighbors_aggr",
                "dropout_p",
                "num_targets",
                "geom_K",
                "geom_rmax",
                "concat_original_edge",
            }
        },
    )
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    return model, ckpt, cfg


def predict_split(
    df: pd.DataFrame,
    *,
    manifest_root: Path,
    ckpt_path: Path,
    device: torch.device,
    batch_size: int,
    psmiles_col: str,
) -> pd.DataFrame:
    model, ckpt, cfg = build_model_from_checkpoint(ckpt_path, device)
    polybert_name = cfg.get("polybert_name", str(REPO_ROOT / "polybert"))
    if polybert_name == "./polybert":
        polybert_name = str(REPO_ROOT / "polybert")
    tokenizer = PolyBertTokenizer(polybert_name) if cfg.get("use_polybert", False) else None
    y_mean = torch.tensor(float(ckpt["y_mean"]), dtype=torch.float32, device=device)
    y_std = torch.tensor(float(ckpt["y_std"]), dtype=torch.float32, device=device)

    dataset = GraphDataset(
        df,
        root=manifest_root,
        separate_pos=True,
        feature_cols=(0, 1, 2, 3),
        coord_cols=(4, 5, 6),
        standardize_y=True,
        tokenizer=tokenizer,
        psmiles_col=psmiles_col,
        seq_max_length=cfg.get("seq_max_length", 256),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    preds = []
    labels = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Predict Tg"):
            batch = batch.to(device)
            pred = model(batch).view(-1) * y_std + y_mean
            y = batch.y.view(-1).float().to(device) * y_std + y_mean
            preds.extend(pred.cpu().numpy().tolist())
            labels.extend(y.cpu().numpy().tolist())

    out = df.reset_index(drop=True).copy()
    out["pred"] = preds
    out["label_from_graph"] = labels
    return out


def attach_prediction_csv(
    df: pd.DataFrame,
    pred_csv: Path,
    *,
    pred_col: str,
    id_col: Optional[str],
) -> pd.DataFrame:
    pred_df = pd.read_csv(pred_csv)
    pred_df.columns = pred_df.columns.str.strip()
    if pred_col not in pred_df.columns:
        raise ValueError(f"Missing prediction column {pred_col!r} in {pred_csv}")

    out = df.copy()
    if id_col and id_col in out.columns and id_col in pred_df.columns:
        return out.merge(pred_df[[id_col, pred_col]], on=id_col, how="left")
    if len(out) != len(pred_df):
        raise ValueError(
            f"Cannot align prediction CSV by order: split rows={len(out)}, pred rows={len(pred_df)}."
        )
    out[pred_col] = pred_df[pred_col].to_numpy()
    return out


def morgan_fp(smiles: str, radius: int, n_bits: int):
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)


def find_similar_pairs(
    df: pd.DataFrame,
    *,
    smiles_col: str,
    label_col: str,
    pred_col: str,
    threshold: float,
    radius: int,
    n_bits: int,
    min_exp_delta: float,
    max_pairs: Optional[int],
) -> pd.DataFrame:
    fps = []
    valid_positions = []
    for pos, smiles in tqdm(list(enumerate(df[smiles_col])), desc="Morgan fingerprints"):
        fp = morgan_fp(smiles, radius=radius, n_bits=n_bits)
        if fp is None:
            continue
        fps.append(fp)
        valid_positions.append(pos)

    pair_rows = []
    for local_i, fp_i in tqdm(list(enumerate(fps)), desc="Find similar pairs"):
        if local_i + 1 >= len(fps):
            continue
        sims = DataStructs.BulkTanimotoSimilarity(fp_i, fps[local_i + 1 :])
        i = valid_positions[local_i]
        for offset, sim in enumerate(sims, start=local_i + 1):
            if sim < threshold:
                continue
            j = valid_positions[offset]
            exp_delta = float(df.at[i, label_col]) - float(df.at[j, label_col])
            pred_delta = float(df.at[i, pred_col]) - float(df.at[j, pred_col])
            if abs(exp_delta) < min_exp_delta:
                continue
            pair_rows.append(
                {
                    "i": i,
                    "j": j,
                    "mol_id_i": df.at[i, "mol_id"] if "mol_id" in df.columns else i,
                    "mol_id_j": df.at[j, "mol_id"] if "mol_id" in df.columns else j,
                    "similarity": float(sim),
                    "smiles_i": df.at[i, smiles_col],
                    "smiles_j": df.at[j, smiles_col],
                    "exp_tg_i": float(df.at[i, label_col]),
                    "exp_tg_j": float(df.at[j, label_col]),
                    "pred_tg_i": float(df.at[i, pred_col]),
                    "pred_tg_j": float(df.at[j, pred_col]),
                    "exp_delta_i_minus_j": exp_delta,
                    "pred_delta_i_minus_j": pred_delta,
                    "abs_exp_delta": abs(exp_delta),
                    "abs_pred_delta": abs(pred_delta),
                    "direction_consistent": bool(np.sign(exp_delta) == np.sign(pred_delta)),
                    "abs_delta_error": abs(pred_delta - exp_delta),
                }
            )

    pairs = pd.DataFrame(pair_rows)
    if pairs.empty:
        return pairs
    pairs = pairs.sort_values(
        ["similarity", "abs_exp_delta"], ascending=[False, False]
    ).reset_index(drop=True)
    if max_pairs is not None and max_pairs > 0:
        pairs = pairs.head(max_pairs).copy()
    return pairs


def summarize_pairs(pairs: pd.DataFrame) -> Dict[str, float]:
    if pairs.empty:
        return {"n_pairs": 0}
    exp_delta = pairs["exp_delta_i_minus_j"].to_numpy(dtype=float)
    pred_delta = pairs["pred_delta_i_minus_j"].to_numpy(dtype=float)
    abs_exp_delta = np.abs(exp_delta)
    abs_pred_delta = np.abs(pred_delta)
    direction = pairs["direction_consistent"].to_numpy(dtype=bool)

    summary = {
        "n_pairs": int(len(pairs)),
        "mean_similarity": float(pairs["similarity"].mean()),
        "median_similarity": float(pairs["similarity"].median()),
        "direction_accuracy": float(direction.mean()),
        "delta_mae": float(np.mean(np.abs(pred_delta - exp_delta))),
        "abs_delta_mae": float(np.mean(np.abs(abs_pred_delta - abs_exp_delta))),
        "mean_abs_exp_delta": float(abs_exp_delta.mean()),
        "mean_abs_pred_delta": float(abs_pred_delta.mean()),
    }
    if len(pairs) >= 2 and np.std(exp_delta) > 0 and np.std(pred_delta) > 0:
        summary["pearson_delta"] = float(pearsonr(exp_delta, pred_delta).statistic)
        summary["spearman_delta"] = float(spearmanr(exp_delta, pred_delta).statistic)
    else:
        summary["pearson_delta"] = None
        summary["spearman_delta"] = None
    return summary


def summarize_by_delta_bins(pairs: pd.DataFrame) -> pd.DataFrame:
    bins = [
        ("small", 0.0, 20.0),
        ("moderate", 20.0, 50.0),
        ("large", 50.0, np.inf),
    ]
    rows = []
    for name, lo, hi in bins:
        if np.isinf(hi):
            sub = pairs[pairs["abs_exp_delta"] >= lo]
            label = f"|experimental delta| >= {lo:g} K"
        else:
            sub = pairs[(pairs["abs_exp_delta"] >= lo) & (pairs["abs_exp_delta"] < hi)]
            label = f"{lo:g} <= |experimental delta| < {hi:g} K"
        if sub.empty:
            rows.append({"bin": name, "range": label, "n": 0, "direction_accuracy": np.nan, "delta_mae": np.nan})
        else:
            rows.append(
                {
                    "bin": name,
                    "range": label,
                    "n": len(sub),
                    "direction_accuracy": float(sub["direction_consistent"].mean()),
                    "delta_mae": float(sub["abs_delta_error"].mean()),
                }
            )
    return pd.DataFrame(rows)


def plot_delta_scatter(pairs: pd.DataFrame, out_path: Path) -> None:
    if pairs.empty:
        return
    plt.figure(figsize=(5.2, 5.0))
    colors = np.where(pairs["direction_consistent"], "#2ca02c", "#d62728")
    plt.scatter(
        pairs["exp_delta_i_minus_j"],
        pairs["pred_delta_i_minus_j"],
        c=colors,
        s=20,
        alpha=0.72,
        edgecolors="none",
    )
    lim = max(
        float(np.abs(pairs["exp_delta_i_minus_j"]).max()),
        float(np.abs(pairs["pred_delta_i_minus_j"]).max()),
        1.0,
    )
    plt.plot([-lim, lim], [-lim, lim], color="#555555", linestyle="--", linewidth=1)
    plt.axhline(0, color="#999999", linewidth=0.8)
    plt.axvline(0, color="#999999", linewidth=0.8)
    plt.xlim(-lim, lim)
    plt.ylim(-lim, lim)
    plt.xlabel("Experimental delta Tg / K")
    plt.ylabel("Predicted delta Tg / K")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=250)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("data/graphs_tg2/manifest.csv"))
    parser.add_argument("--root-dir", type=Path, default=Path("data/graphs_tg2"))
    parser.add_argument("--ckpt", type=Path, default=Path("checkpoints/best_rmse_35.089K_ep032.pt"))
    parser.add_argument("--pred-csv", type=Path, default=None, help="Optional precomputed prediction CSV.")
    parser.add_argument("--pred-col", default="pred")
    parser.add_argument("--id-col", default="mol_id")
    parser.add_argument("--split", choices=["train", "val", "test", "all"], default="test")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-split", type=float, default=0.1)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--smiles-col", default="psmiles")
    parser.add_argument("--label-col", default="label")
    parser.add_argument("--threshold", type=float, default=0.85)
    parser.add_argument("--min-exp-delta", type=float, default=0.0)
    parser.add_argument("--morgan-radius", type=int, default=2)
    parser.add_argument("--morgan-bits", type=int, default=2048)
    parser.add_argument("--max-pairs", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--out-dir", type=Path, default=Path("demo/analysis_outputs/similar_pairs"))
    parser.add_argument("--prefix", default="test_pairs_tanimoto085")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    RDLogger.DisableLog("rdApp.warning")
    RDLogger.DisableLog("rdApp.error")

    manifest = pd.read_csv(args.manifest)
    manifest.columns = manifest.columns.str.strip()
    train_df, val_df, test_df = split_manifest(
        manifest,
        seed=args.seed,
        test_split=args.test_split,
        val_split=args.val_split,
        label_col=args.label_col,
    )
    split_map = {"train": train_df, "val": val_df, "test": test_df, "all": manifest.reset_index(drop=True)}
    df = split_map[args.split].reset_index(drop=True)

    if args.pred_csv:
        df = attach_prediction_csv(df, args.pred_csv, pred_col=args.pred_col, id_col=args.id_col)
    else:
        if args.device == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(args.device)
        df = predict_split(
            df,
            manifest_root=args.root_dir,
            ckpt_path=args.ckpt,
            device=device,
            batch_size=args.batch_size,
            psmiles_col=args.smiles_col,
        )

    df = df.dropna(subset=[args.smiles_col, args.label_col, args.pred_col]).reset_index(drop=True)
    pairs = find_similar_pairs(
        df,
        smiles_col=args.smiles_col,
        label_col=args.label_col,
        pred_col=args.pred_col,
        threshold=args.threshold,
        radius=args.morgan_radius,
        n_bits=args.morgan_bits,
        min_exp_delta=args.min_exp_delta,
        max_pairs=args.max_pairs,
    )
    summary = summarize_pairs(pairs)
    by_delta = summarize_by_delta_bins(pairs) if not pairs.empty else pd.DataFrame()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = args.out_dir / f"{args.prefix}_predictions.csv"
    pairs_path = args.out_dir / f"{args.prefix}_pairs.csv"
    summary_path = args.out_dir / f"{args.prefix}_summary.json"
    delta_bins_path = args.out_dir / f"{args.prefix}_delta_bins.csv"
    plot_path = args.out_dir / f"{args.prefix}_delta_scatter.png"

    df.to_csv(pred_path, index=False)
    pairs.to_csv(pairs_path, index=False)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    if not by_delta.empty:
        by_delta.to_csv(delta_bins_path, index=False)
    plot_delta_scatter(pairs, plot_path)

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if not by_delta.empty:
        print(by_delta.to_string(index=False))
    print(f"Wrote predictions: {pred_path}")
    print(f"Wrote pairs: {pairs_path}")
    print(f"Wrote summary: {summary_path}")
    print(f"Wrote plot: {plot_path}")


if __name__ == "__main__":
    main()
