#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
from pathlib import Path
import datetime
import random
import sys
import numpy as np
import pandas as pd
import torch
from typing import Optional, Tuple, Dict, Any
from torch_geometric.loader import DataLoader
from transformers import AutoModel

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.predictor.GeoGATModel import GeoGATModel
from utils.GraphDataset import GraphDataset
from utils.PSMILES_to_graph import convert_csv_to_graphs
from models.generator.tokenizer import PolyBertTokenizer


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def denorm(t: torch.Tensor, y_mean: Optional[torch.Tensor], y_std: Optional[torch.Tensor]) -> torch.Tensor:
    if y_mean is None or y_std is None:
        return t
    return t * y_std + y_mean


def build_model_from_ckpt(checkpoint: dict, device: torch.device) -> Tuple[torch.nn.Module, Optional[Any], Dict[str, Any]]:
    # If checkpoint carries a saved config, prefer it; otherwise use training defaults in this repo
    cfg = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}
    use_polybert = cfg.get("use_polybert", False)
    tokenizer = None
    polybert_model = None
    if use_polybert:
        tokenizer_name = cfg.get("polybert_name") or checkpoint.get("tokenizer_name") or str(REPO_ROOT / "polybert")
        tokenizer = PolyBertTokenizer(tokenizer_name)
        polybert_model = AutoModel.from_pretrained(tokenizer_name)
        if cfg.get("freeze_polybert", True):
            for p in polybert_model.parameters():
                p.requires_grad = False
            polybert_model.eval()

    model = GeoGATModel(
        layers_in_conv=cfg.get("layers_in_conv", 3),
        channels=cfg.get("channels", 64),
        use_nodetype_coeffs=cfg.get("use_nodetype_coeffs", False),
        num_node_types=cfg.get("num_node_types", 0),
        num_edge_types=cfg.get("num_edge_types", 4),
        use_jumping_knowledge=cfg.get("use_jumping_knowledge", False),
        use_bias_for_update=cfg.get("use_bias_for_update", True),
        use_dropout=cfg.get("use_dropout", True),
        num_convs=cfg.get("num_convs", 3),
        num_fc_layers=cfg.get("num_fc_layers", 3),
        neighbors_aggr=cfg.get("neighbors_aggr", "add"),
        dropout_p=cfg.get("dropout_p", 0.1),
        num_targets=cfg.get("num_targets", 1),
        geom_K=cfg.get("geom_K", 16),
        geom_rmax=cfg.get("geom_rmax", 4.0),
        concat_original_edge=cfg.get("concat_original_edge", True),
        use_polybert=use_polybert,
        polybert=polybert_model,
        freeze_polybert=cfg.get("freeze_polybert", True),
        seq_max_length=cfg.get("seq_max_length", 256),
        cross_attn_heads=cfg.get("cross_attn_heads", 4),
        cross_attn_dim=cfg.get("cross_attn_dim", None),
    ).to(device)
    return model, tokenizer, cfg


@torch.no_grad()
def predict_loader(model: torch.nn.Module, loader: DataLoader, device: torch.device,
                   y_mean: Optional[torch.Tensor], y_std: Optional[torch.Tensor]):
    model.eval()
    preds = []
    for batch in loader:
        batch = batch.to(device)
        out = model(batch)  # [B, 1]
        out = out.view(-1, 1)
        out_denorm = denorm(out, y_mean, y_std)
        # attach ids for alignment back to CSV rows
        mol_ids = batch.mol_id if hasattr(batch, "mol_id") else None
        for i in range(out.size(0)):
            preds.append({
                "mol_id": int(mol_ids[i].item()) if mol_ids is not None else None,
                "pred_norm": float(out[i].item()),
                "pred": float(out_denorm[i].item()),
            })
    return preds


def main():
    parser = argparse.ArgumentParser(description="Predict from CSV of PSMILES without labels by building graphs, then running GeoGATModel.")

    # Inputs
    parser.add_argument("--ckpt_path", type=str, required=True, help="Path to model checkpoint (.pt) saved by train.")
    parser.add_argument("--csv_path", type=str, required=True, help="Input CSV containing a PSMILES column (no labels needed).")
    parser.add_argument("--psmiles_col", type=str, required=True, help="Column name for PSMILES in input CSV.")

    # Graph saving/loading
    parser.add_argument("--save_dir", type=str, default="pred_graphs", help="Directory to save generated .npz graphs.")

    # Inference
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--seed", type=int, default=42)

    # Outputs
    parser.add_argument("--out_csv", type=str, default=None, help="Path to write predictions CSV; default <save_dir>/predictions.csv")


    args = parser.parse_args()
    set_seed(args.seed)

    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    # Step 1) Convert CSV PSMILES -> graph .npz + manifest
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    _, manifest_df = convert_csv_to_graphs(
        csv_path=args.csv_path,
        label_col=None,               # test CSV has no labels
        PSMILES_col=args.psmiles_col,
        save_dir=str(save_dir),
    )

    # Step 2) Build model/tokenizer and load checkpoint
    ckpt_path = Path(args.ckpt_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    checkpoint = torch.load(str(ckpt_path), map_location="cpu")
    model, tokenizer, cfg = build_model_from_ckpt(checkpoint, device)
    # Step 3) Build dataset/loader from manifest (after knowing tokenizer config)
    dataset = GraphDataset(
        manifest=manifest_df,
        root=save_dir,
        separate_pos=True,
        feature_cols=(0, 1, 2, 3),
        coord_cols=(4, 5, 6),
        standardize_y=False,  # no labels
        tokenizer=tokenizer,
        psmiles_col="psmiles",
        seq_max_length=cfg.get("seq_max_length", 256),
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    # Step 4) Load checkpoint weights (tolerate multiple formats)
    state_dict = None
    if isinstance(checkpoint, dict):
        state_dict = checkpoint.get("model_state") or checkpoint.get("state_dict")
    if state_dict is None:
        state_dict = checkpoint  # assume raw state_dict
    model.load_state_dict(state_dict, strict=False)
    model.to(device)

    # Denorm stats if present
    y_mean = checkpoint.get("y_mean") if isinstance(checkpoint, dict) else None
    y_std = checkpoint.get("y_std") if isinstance(checkpoint, dict) else None
    y_mean_t = torch.tensor(y_mean, dtype=torch.float32, device=device) if y_mean is not None else None
    y_std_t = torch.tensor(y_std, dtype=torch.float32, device=device) if y_std is not None else None

    # Step 5) Predict
    pred_rows = predict_loader(model, loader, device, y_mean_t, y_std_t)

    # Step 6) Merge back to input CSV order and save
    in_df = pd.read_csv(args.csv_path)
    out_df = in_df.copy()
    out_df["pred"] = np.nan
    for r in pred_rows:
        mid = r["mol_id"]
        if mid is not None and 0 <= mid < len(out_df):
            out_df.at[mid, "pred"] = r["pred"]

    out_csv = Path(args.out_csv) if args.out_csv is not None else (save_dir / "predictions.csv")
    out_df.to_csv(out_csv, index=False)
    print(f"Saved predictions to: {out_csv.resolve()}")


if __name__ == "__main__":
    main()
