#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简易基线基准脚本：
 - 固定划分 train/val/test（与 train_predictor.py 一致的分桶 stratify）
 - 使用图 .npz 的统计特征训练传统模型（RF/SVR/MLP）
 - 输出 MAE / RMSE / R2，保存到 CSV

用法示例：
python demo/benchmark_tg.py \
  --manifest data/graphs_tg2/manifest.csv \
  --root_dir data/graphs_tg2 \
  --polybert_dir /Users/tangren/Documents/PolyMolStudio/polybert \
  --polybert_pooling cls \
  --polybert_max_len 256 \
  --device cpu \
  --seed 42 \
  --out_csv demo/benchmarks_tg.csv
"""

import argparse
from pathlib import Path
from typing import Dict, List, Optional
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    import torch
    from transformers import AutoModel
    from models.generator.tokenizer import PolyBertTokenizer
except Exception as e:
    print(f"[WARN] transformers/PolyBERT import failed: {e}")
    PolyBertTokenizer = None
    AutoModel = None
    torch = None

try:
    from rdkit import DataStructs
    from rdkit import Chem
    from rdkit.Chem import AllChem
except Exception as e:
    print(f"[WARN] RDKit import failed: {e}")
    Chem = None
    AllChem = None
    DataStructs = None


def resolve_path(fp: str, root: Path) -> Path:
    p = Path(fp)
    if p.is_absolute():
        return p
    # 避免重复拼接 root
    p_str, root_str = p.as_posix(), root.as_posix()
    if p_str.startswith(root_str):
        return p
    return root / p


def graph_summary(npz_path: Path) -> np.ndarray:
    with np.load(npz_path, allow_pickle=False) as npz:
        node_feats = npz["node_feats"]  # (N, F)
        edge_attr = npz["edge_attr"]    # (E, A)
    feats = []
    # 节点/边数量
    feats.extend([node_feats.shape[0], edge_attr.shape[0]])
    # 节点均值/方差
    feats.extend(node_feats.mean(axis=0).tolist())
    feats.extend(node_feats.std(axis=0).tolist())
    # 边均值/方差
    feats.extend(edge_attr.mean(axis=0).tolist())
    feats.extend(edge_attr.std(axis=0).tolist())
    return np.asarray(feats, dtype=np.float32)


def build_features(df: pd.DataFrame, root: Path) -> np.ndarray:
    rows = []
    for fp in df["file_path"]:
        path = resolve_path(fp, root)
        rows.append(graph_summary(path))
    return np.vstack(rows)


def evaluate_split(y_true, y_pred) -> Dict[str, float]:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    r2 = r2_score(y_true, y_pred)
    return {"mae": mae, "rmse": rmse, "r2": r2}


def morgan_fingerprint(smiles: str, radius: int = 2, n_bits: int = 2048) -> np.ndarray:
    if Chem is None or AllChem is None or DataStructs is None:
        raise RuntimeError("RDKit is not available.")
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        raise ValueError(f"Invalid SMILES/pSMILES for Morgan fingerprint: {smiles}")
    bitvect = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    arr = np.zeros((n_bits,), dtype=np.float32)
    DataStructs.ConvertToNumpyArray(bitvect, arr)
    return arr


def build_morgan_features(smiles: pd.Series, radius: int = 2, n_bits: int = 2048) -> np.ndarray:
    rows = [
        morgan_fingerprint(smi, radius=radius, n_bits=n_bits)
        for smi in tqdm(smiles, desc="Morgan fingerprint", leave=False)
    ]
    return np.vstack(rows)


def polybert_embeddings(psmiles: pd.Series, name: str, max_length: int, device: str = "cpu",
                        pooling: str = "cls") -> Optional[np.ndarray]:
    if PolyBertTokenizer is None or AutoModel is None or torch is None:
        print("transformers not available; skip polybert baseline.")
        return None
    tokenizer = PolyBertTokenizer(name)
    model = AutoModel.from_pretrained(name).to(device)
    model.eval()
    rows = []
    with torch.no_grad():
        for smi in tqdm(psmiles, desc="PolyBERT embedding", leave=False):
            tok = tokenizer.tokenizer(
                smi,
                padding="max_length",
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            tok = {k: v.to(device) for k, v in tok.items()}
            out = model(**tok)
            hidden = out.last_hidden_state  # [1, L, H]
            if pooling == "mean":
                mask = tok["attention_mask"].unsqueeze(-1)
                emb = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-6)
            else:
                emb = hidden[:, 0, :]
            rows.append(emb.cpu().numpy().reshape(-1))
    return np.stack(rows, axis=0)


def main():
    parser = argparse.ArgumentParser(description="Benchmark classical regressors on Tg manifest.")
    parser.add_argument("--manifest", type=str, required=True, help="CSV with file_path,label.")
    parser.add_argument("--root_dir", type=str, default=".", help="Root for npz files if file_path is relative.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test_split", type=float, default=0.1)
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--out_csv", type=str, default="benchmarks_tg.csv")
    parser.add_argument("--polybert_dir", type=str, default="./polybert", help="PolyBERT directory or HF name.")
    parser.add_argument("--polybert_pooling", type=str, default="cls", choices=["cls", "mean"], help="Pooling for PolyBERT baseline.")
    parser.add_argument("--polybert_max_len", type=int, default=256)
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--psmiles_col", type=str, default="psmiles", help="Manifest column name that holds pSMILES.")
    parser.add_argument("--psmiles_csv", type=str, default=None, help="Optional CSV to fetch pSMILES if manifest lacks it.")
    parser.add_argument("--psmiles_csv_col", type=str, default=None, help="Column name in psmiles_csv (defaults to psmiles_col).")
    parser.add_argument("--smiles_col", type=str, default="smiles", help="Manifest column name that holds SMILES for Morgan baseline.")
    parser.add_argument("--morgan_radius", type=int, default=2, help="Morgan fingerprint radius.")
    parser.add_argument("--morgan_bits", type=int, default=2048, help="Morgan fingerprint bit length.")
    args = parser.parse_args()

    root = Path(args.root_dir)
    df = pd.read_csv(args.manifest)
    df.columns = df.columns.str.strip()
    if "label" not in df:
        raise ValueError("manifest must contain 'label' column for Tg.")
    # 兼容大小写/去空格列名；若 manifest 缺 pSMILES，可从外部 CSV 补充
    col_map = {c.lower(): c for c in df.columns}
    psm_key = args.psmiles_col.strip().lower()
    psm_col = col_map.get(psm_key, None)
    smi_key = args.smiles_col.strip().lower()
    smi_col = col_map.get(smi_key, None)

    if psm_col is None and args.psmiles_csv:
        p_csv = Path(args.psmiles_csv)
        if not p_csv.exists():
            raise FileNotFoundError(f"psmiles_csv not found: {p_csv}")
        df_ps = pd.read_csv(p_csv)
        df_ps.columns = df_ps.columns.str.strip()
        psm_csv_key = (args.psmiles_csv_col or args.psmiles_col).strip().lower()
        psm_csv_map = {c.lower(): c for c in df_ps.columns}
        psm_csv_col = psm_csv_map.get(psm_csv_key, None)
        if psm_csv_col is None:
            raise ValueError(f"psmiles_csv missing column '{psm_csv_key}'. Available: {list(df_ps.columns)}")
        if "mol_id" not in df.columns or "mol_id" not in df_ps.columns:
            raise ValueError("Need mol_id in manifest and psmiles_csv to merge pSMILES.")
        df = df.merge(df_ps[["mol_id", psm_csv_col]], on="mol_id", how="left")
        psm_col = psm_csv_col
        col_map = {c.lower(): c for c in df.columns}
        smi_col = col_map.get(smi_key, None)

    # stratify bins 同 train_predictor.py
    stratify_labels = None
    try:
        bins = pd.qcut(df["label"], q=10, duplicates="drop")
        stratify_labels = bins.astype(str)
    except Exception:
        stratify_labels = None

    train_val_df, test_df = train_test_split(
        df, test_size=args.test_split, random_state=args.seed, stratify=stratify_labels
    )
    if stratify_labels is not None:
        train_val_bins = pd.qcut(train_val_df["label"], q=10, duplicates="drop")
        stratify_trainval = train_val_bins.astype(str)
    else:
        stratify_trainval = None

    train_df, val_df = train_test_split(
        train_val_df,
        test_size=args.val_split / (1 - args.test_split),
        random_state=args.seed,
        stratify=stratify_trainval if stratify_trainval is not None else None,
    )

    # 特征构建
    X_train = build_features(train_df, root)
    X_val = build_features(val_df, root)
    X_test = build_features(test_df, root)
    y_train = train_df["label"].to_numpy()
    y_val = val_df["label"].to_numpy()
    y_test = test_df["label"].to_numpy()

    base_models: Dict[str, Pipeline] = {
        "rf": Pipeline([
            ("rf", RandomForestRegressor(
                n_estimators=400,
                max_depth=None,
                random_state=args.seed,
                n_jobs=-1
            ))
        ]),
        "svr": Pipeline([
            ("scaler", StandardScaler()),
            ("svr", SVR(C=10.0, epsilon=0.1, kernel="rbf", gamma="scale")),
        ]),
        "mlp": Pipeline([
            ("scaler", StandardScaler()),
            ("mlp", MLPRegressor(
                hidden_layer_sizes=(256, 128),
                activation="relu",
                batch_size=256,
                max_iter=1000,
                learning_rate_init=1e-3,
                random_state=args.seed,
            )),
        ]),
    }

    # 可选 Morgan fingerprint baseline：优先使用 SMILES，若无则使用 pSMILES
    morgan_rows: List[Dict[str, str]] = []
    morgan_col = smi_col or psm_col
    if Chem is None or AllChem is None or DataStructs is None:
        print("RDKit not available; skipping Morgan fingerprint baseline.")
    elif morgan_col not in df.columns:
        print("SMILES/pSMILES column not found; skipping Morgan fingerprint baseline.")
    else:
        X_morgan_train = build_morgan_features(train_df[morgan_col], args.morgan_radius, args.morgan_bits)
        X_morgan_val = build_morgan_features(val_df[morgan_col], args.morgan_radius, args.morgan_bits)
        X_morgan_test = build_morgan_features(test_df[morgan_col], args.morgan_radius, args.morgan_bits)
        for name, model in base_models.items():
            model.fit(X_morgan_train, y_train)
            for split_name, X, y in [
                ("val", X_morgan_val, y_val),
                ("test", X_morgan_test, y_test),
            ]:
                metrics = evaluate_split(y, model.predict(X))
                morgan_rows.append({
                    "model": f"morgan_{name}",
                    "split": split_name,
                    "embedding": "morgan",
                    **metrics,
                })

    # 可选 PolyBERT baseline（仅当 ps... 列存在且 transformers 可用）
    polybert_rows: List[Dict[str, str]] = []
    if PolyBertTokenizer is None or AutoModel is None or torch is None:
        print("transformers not available; skipping PolyBERT baseline.")
    elif psm_col not in df.columns:
        print("pSMILES column not found; skipping PolyBERT baseline.")
    else:
        device = torch.device(args.device)
        emb_train = polybert_embeddings(train_df[psm_col], args.polybert_dir, args.polybert_max_len, device, args.polybert_pooling)
        emb_val = polybert_embeddings(val_df[psm_col], args.polybert_dir, args.polybert_max_len, device, args.polybert_pooling)
        emb_test = polybert_embeddings(test_df[psm_col], args.polybert_dir, args.polybert_max_len, device, args.polybert_pooling)
        poly_models = {
            "polybert_svr": Pipeline([
                ("scaler", StandardScaler()),
                ("svr", SVR(C=10.0, epsilon=0.1, kernel="rbf", gamma="scale")),
            ]),
            "polybert_mlp": Pipeline([
                ("scaler", StandardScaler()),
                ("mlp", MLPRegressor(
                    hidden_layer_sizes=(512, 256),
                    activation="relu",
                    batch_size=256,
                    max_iter=1000,
                    learning_rate_init=1e-3,
                    random_state=args.seed,
                )),
            ]),
        }
        for name, model in poly_models.items():
            model.fit(emb_train, y_train)
            for split_name, X, y in [
                ("val", emb_val, y_val),
                ("test", emb_test, y_test),
            ]:
                metrics = evaluate_split(y, model.predict(X))
                polybert_rows.append({
                    "model": name,
                    "split": split_name,
                    "embedding": "polybert",
                    **metrics,
                })

    rows: List[Dict[str, str]] = []
    for name, model in base_models.items():
        model.fit(X_train, y_train)
        for split_name, X, y in [
            ("val", X_val, y_val),
            ("test", X_test, y_test),
        ]:
            metrics = evaluate_split(y, model.predict(X))
            rows.append({
                "model": name,
                "split": split_name,
                "embedding": "graph_stats",
                **metrics,
            })
    rows.extend(morgan_rows)
    rows.extend(polybert_rows)

    out_csv = Path(args.out_csv)
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"Saved benchmarks to {out_csv.resolve()}")


if __name__ == "__main__":
    main()
