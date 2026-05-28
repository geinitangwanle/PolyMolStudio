#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import json
import datetime
import logging
from pathlib import Path
import random
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
from transformers import AutoModel
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.GraphDataset import GraphDataset
from models.predictor.GeoGATModel import GeoGATModel
from models.generator.tokenizer import PolyBertTokenizer

def configure_polybert_finetuning(polybert, train_last_n_layers: int = 0, unfreeze_embeddings: bool = False):
    """
    Freeze all polyBERT params first, then选择性解冻：
      - unfreeze_embeddings: 解冻 embeddings
      - train_last_n_layers: 解冻 encoder 的最后 N 层
    返回需要训练的参数列表。
    """
    if polybert is None:
        return []
    for p in polybert.parameters():
        p.requires_grad = False

    trainable = []
    if unfreeze_embeddings:
        embeds = getattr(polybert, "embeddings", None)
        if embeds is not None:
            for p in embeds.parameters():
                p.requires_grad = True
                trainable.append(p)

    encoder = getattr(polybert, "encoder", None)
    if train_last_n_layers > 0 and encoder is not None and hasattr(encoder, "layer"):
        layers = encoder.layer[-train_last_n_layers:]
        for layer in layers:
            for p in layer.parameters():
                p.requires_grad = True
                trainable.append(p)

    return trainable


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def denorm(t, y_mean, y_std):
    return t * y_std + y_mean



def main():
    parser = argparse.ArgumentParser(description="Train GatedGCN model for polymer property prediction.")

    # Dataset arguments
    parser.add_argument("--data_path", type=str, required=True, help="Path to manifest CSV.")
    parser.add_argument("--root_dir", type=str, default=".", help="Root directory for .npz files.")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--test_split", type=float, default=0.1)
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42, help="Legacy seed used for both split and model when split/model seeds are not set.")
    parser.add_argument("--split_seed", type=int, default=None, help="Seed for train/val/test split. Defaults to --seed.")
    parser.add_argument("--model_seed", type=int, default=None, help="Seed for model initialization, dropout, and dataloader shuffling. Defaults to --seed.")
    parser.add_argument("--seq_max_length", type=int, default=256, help="Max token length for pSMILES.")
    parser.add_argument("--use_polybert", action="store_true", help="Enable PolyBERT encoder + cross-attention.")
    parser.add_argument(
        "--polybert_dir", "--polybert-dir",
        dest="polybert_dir",
        type=str,
        default=str(REPO_ROOT / "polybert"),
        help="Path or HF name for PolyBERT."
    )
    parser.add_argument("--freeze_polybert", action="store_true", default=True, help="Freeze PolyBERT weights.")
    parser.add_argument("--no-freeze_polybert", dest="freeze_polybert", action="store_false", help="Unfreeze PolyBERT.")
    parser.add_argument("--polybert_lr", type=float, default=None, help="Optional LR for PolyBERT params.")
    parser.add_argument("--polybert_train_last_n", type=int, default=0, help="Unfreeze last N transformer layers of PolyBERT.")
    parser.add_argument("--unfreeze_polybert_emb", action="store_true", help="Unfreeze PolyBERT embeddings.")
    parser.add_argument("--cross_attn_heads", type=int, default=4, help="Heads for cross-attention.")
    parser.add_argument("--cross_attn_dim", type=int, default=None, help="Hidden dim for cross-attention projection.")

    # Training arguments
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--optimizer", type=str, default="adamw", choices=["adamw", "adam"])
    parser.add_argument("--loss", type=str, default="mse", choices=["mse", "smoothl1"])
    parser.add_argument("--scheduler", type=str, default="cosine", choices=["cosine", "none"])
    parser.add_argument("--clip_grad_norm", type=float, default=5.0)
    parser.add_argument("--early_stop_patience", type=int, default=None, help="Stop if val RMSE does not improve for N epochs.")

    # Device
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])

    # Logging
    parser.add_argument("--log_dir", type=str, default="./logs")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints")
    parser.add_argument("--run_name", type=str, default=None, help="Optional prefix for log/checkpoint run directory names.")

    args = parser.parse_args()
    if args.split_seed is None:
        args.split_seed = args.seed
    if args.model_seed is None:
        args.model_seed = args.seed
    set_seed(args.model_seed)

    # Device selection
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    # 只在 logs 下放置单个日志文件；在 checkpoints 下创建与该日志文件名（去掉 .log）相同的子目录保存 .pt
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_prefix = f"{args.run_name}_" if args.run_name else ""
    log_filename = f"{run_prefix}train_{timestamp}_split{args.split_seed}_model{args.model_seed}.log"

    # 确保 logs 目录存在，日志文件放到 ./logs/train_<timestamp>.log
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / log_filename

    # 在 checkpoints 下创建与日志文件名相同（去掉 .log）的子目录保存模型
    ckpt_run_dir = Path(args.checkpoint_dir) / log_path.stem
    ckpt_run_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("train")
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    fh = logging.FileHandler(log_path)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    ch.setFormatter(fmt); fh.setFormatter(fmt)
    logger.addHandler(ch); logger.addHandler(fh)

    logger.info(json.dumps(vars(args), indent=2))

    # ========== Load and split manifest ==========
    manifest = pd.read_csv(args.data_path)
    if "label" in manifest:
        try:
            bins = pd.qcut(manifest["label"], q=10, duplicates="drop")
            stratify_labels = bins.astype(str)
        except Exception:
            stratify_labels = None
    else:
        stratify_labels = None

    train_val_df, test_df = train_test_split(
        manifest, test_size=args.test_split, random_state=args.split_seed, stratify=stratify_labels
    )

    if stratify_labels is not None:
        train_val_bins = pd.qcut(train_val_df["label"], q=10, duplicates="drop")
        stratify_trainval = train_val_bins.astype(str)
    else:
        stratify_trainval = None

    train_df, val_df = train_test_split(
        train_val_df,
        test_size=args.val_split / (1 - args.test_split),
        random_state=args.split_seed,
        stratify=stratify_trainval if stratify_trainval is not None else None,
    )

    logger.info(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    # ========== Build tokenizer / PolyBERT ==========
    tokenizer = None
    polybert_model = None
    trainable_polybert_params = []
    if args.use_polybert:
        tokenizer = PolyBertTokenizer(str(args.polybert_dir))
        polybert_model = AutoModel.from_pretrained(str(args.polybert_dir))
        if args.freeze_polybert:
            for p in polybert_model.parameters():
                p.requires_grad = False
            polybert_model.eval()
        else:
            # 默认全解冻；若指定 last_n/emb 则仅部分解冻
            if args.polybert_train_last_n > 0 or args.unfreeze_polybert_emb:
                trainable_polybert_params = configure_polybert_finetuning(
                    polybert_model,
                    train_last_n_layers=args.polybert_train_last_n,
                    unfreeze_embeddings=args.unfreeze_polybert_emb,
                )
            else:
                for p in polybert_model.parameters():
                    p.requires_grad = True
                trainable_polybert_params = [p for p in polybert_model.parameters() if p.requires_grad]
            polybert_model.train()

    # ========== Build datasets ==========
    root_dir = Path(args.root_dir)
    train_dataset = GraphDataset(
        manifest=train_df, root=root_dir,
        separate_pos=True, feature_cols=(0,1,2,3), coord_cols=(4,5,6), standardize_y=True,
        tokenizer=tokenizer, psmiles_col="psmiles", seq_max_length=args.seq_max_length,
    )
    val_dataset = GraphDataset(
        manifest=val_df, root=root_dir,
        separate_pos=True, feature_cols=(0,1,2,3), coord_cols=(4,5,6), standardize_y=True,
        tokenizer=tokenizer, psmiles_col="psmiles", seq_max_length=args.seq_max_length,
    )
    val_dataset._y_mean, val_dataset._y_std = train_dataset.y_mean, train_dataset.y_std
    test_dataset = GraphDataset(
        manifest=test_df, root=root_dir,
        separate_pos=True, feature_cols=(0,1,2,3), coord_cols=(4,5,6), standardize_y=True,
        tokenizer=tokenizer, psmiles_col="psmiles", seq_max_length=args.seq_max_length,
    )
    test_dataset._y_mean, test_dataset._y_std = train_dataset.y_mean, train_dataset.y_std

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True)

    logger.info(f"y_mean(train)={train_dataset.y_mean:.4f}, y_std(train)={train_dataset.y_std:.4f}")

    # ========== Model ==========
    model = GeoGATModel(
        layers_in_conv=3, channels=64, use_nodetype_coeffs=False, num_node_types=0,
        num_edge_types=4, use_jumping_knowledge=False, use_bias_for_update=True,
        use_dropout=True, num_convs=3, num_fc_layers=3, neighbors_aggr='add',
        dropout_p=0.1, num_targets=1, geom_K=16, geom_rmax=4.0, concat_original_edge=True,
        use_polybert=args.use_polybert, polybert=polybert_model, freeze_polybert=args.freeze_polybert,
        polybert_name=str(args.polybert_dir),
        seq_max_length=args.seq_max_length, cross_attn_heads=args.cross_attn_heads, cross_attn_dim=args.cross_attn_dim,
    ).to(device)

    if args.loss == "mse":
        criterion = nn.MSELoss()
    else:
        criterion = nn.SmoothL1Loss()

    def build_param_groups():
        if not args.use_polybert or args.polybert_lr is None:
            return [{"params": [p for p in model.parameters() if p.requires_grad], "lr": args.lr, "weight_decay": args.weight_decay}]
        polybert_params, other_params = [], []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if name.startswith("polybert."):
                polybert_params.append(param)
            else:
                other_params.append(param)
        groups = [{"params": other_params, "lr": args.lr, "weight_decay": args.weight_decay}]
        if polybert_params:
            groups.append({"params": polybert_params, "lr": args.polybert_lr, "weight_decay": args.weight_decay})
        return groups

    param_groups = build_param_groups()
    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(param_groups)
    else:
        optimizer = torch.optim.Adam(param_groups)

    if args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    else:
        scheduler = None

    y_mean = torch.tensor(train_dataset.y_mean, dtype=torch.float32, device=device)
    y_std  = torch.tensor(train_dataset.y_std, dtype=torch.float32, device=device)

    best_val_rmse = float("inf")
    best_epoch = -1
    best_ckpt_path = None
    patience_ctr = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, n_tr = 0.0, 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch:03d} [train]", leave=False):
            batch = batch.to(device)
            pred = model(batch)
            y = batch.y.view(-1, 1).float().to(device)
            loss = criterion(pred, y)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
            optimizer.step()

            total_loss += loss.item() * batch.num_graphs
            n_tr += batch.num_graphs

        if scheduler:
            scheduler.step()
        train_loss = total_loss / max(n_tr, 1)

        # Validation
        model.eval()
        mae, rmse, n_val = 0.0, 0.0, 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch:03d} [val]", leave=False):
                batch = batch.to(device)
                pred_norm = model(batch)
                y_norm = batch.y.view(-1, 1).float().to(device)
                pred_k = denorm(pred_norm, y_mean, y_std)
                y_k = denorm(y_norm, y_mean, y_std)
                diff = pred_k - y_k
                mae += diff.abs().sum().item()
                rmse += (diff ** 2).sum().item()
                n_val += batch.num_graphs
        mae = mae / max(n_val, 1)
        rmse = (rmse / max(n_val, 1)) ** 0.5

        lr_now = optimizer.param_groups[0]["lr"]
        logger.info(f"Epoch {epoch:03d} | TrainLoss(norm) {train_loss:.4f} | Val MAE(K) {mae:.3f} | Val RMSE(K) {rmse:.3f} | LR {lr_now:.6f}")

        if rmse < best_val_rmse:
            best_val_rmse = rmse
            # 只保留当前 run 的最佳权重，避免每次提升都堆积一个 checkpoint。
            ckpt_path = ckpt_run_dir / "best.pt"
            torch.save({
                "epoch": epoch,
                "best_val_rmse": float(best_val_rmse),
                "model_state": model.state_dict(),
                "y_mean": float(y_mean.item()),
                "y_std": float(y_std.item()),
                "config": model.export_config(),
            }, ckpt_path)
            logger.info(f"Saved best checkpoint: {ckpt_path} | Val RMSE(K) {best_val_rmse:.3f}")
            best_epoch = epoch
            best_ckpt_path = ckpt_path
            patience_ctr = 0
        else:
            patience_ctr += 1

        if args.early_stop_patience is not None and patience_ctr >= args.early_stop_patience:
            logger.info(f"Early stopping at epoch {epoch}: no val RMSE improvement for {patience_ctr} epochs.")
            break

    # Final test using best validation checkpoint (Scheme B)
    # Reload best checkpoint before evaluating the test set
    if best_ckpt_path is not None and best_ckpt_path.exists():
        logger.info(f"Loading best checkpoint (epoch {best_epoch}) for final test: {best_ckpt_path}")
        ckpt = torch.load(best_ckpt_path, map_location="cpu")
        state_dict = ckpt.get("model_state", ckpt)
        model.load_state_dict(state_dict)
        model.to(device)
        # Use normalization stats saved with the best checkpoint if present
        y_mean = torch.tensor(ckpt.get("y_mean", float(train_dataset.y_mean)), dtype=torch.float32, device=device)
        y_std  = torch.tensor(ckpt.get("y_std",  float(train_dataset.y_std)), dtype=torch.float32, device=device)
    else:
        if best_ckpt_path is None:
            logger.warning("No best checkpoint recorded; evaluating last-epoch model on test set.")
        else:
            logger.warning(f"Best checkpoint not found at {best_ckpt_path}; evaluating last-epoch model.")

    model.eval()
    mae_t, rmse_t, n_te = 0.0, 0.0, 0
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            pred_k = denorm(model(batch), y_mean, y_std)
            y_k = denorm(batch.y.view(-1,1).float().to(device), y_mean, y_std)
            diff = pred_k - y_k
            mae_t += diff.abs().sum().item()
            rmse_t += (diff ** 2).sum().item()
            n_te += batch.num_graphs
    mae_t /= max(n_te, 1)
    rmse_t = (rmse_t / max(n_te, 1)) ** 0.5
    if best_epoch >= 1:
        logger.info(f"[TEST-BEST ep{best_epoch:03d}] MAE(K) {mae_t:.3f} | RMSE(K) {rmse_t:.3f}")
    else:
        logger.info(f"[TEST] MAE(K) {mae_t:.3f} | RMSE(K) {rmse_t:.3f}")


if __name__ == "__main__":
    main()
