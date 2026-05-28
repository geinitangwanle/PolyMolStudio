#!/usr/bin/env python
import json
import sys
from pathlib import Path
import torch
import pandas as pd
import matplotlib.pyplot as plt
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
from transformers import AutoModel

# 确保可以导入仓库内模块
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.predictor.GeoGATModel import GeoGATModel
from utils.GraphDataset import GraphDataset
from models.generator.tokenizer import PolyBertTokenizer

ckpt_path = Path("checkpoints/best_rmse_35.089K_ep032.pt")  # 替换你的 best ckpt
manifest = Path("data/graphs_tg2/manifest.csv")
root_dir = Path("data/graphs_tg2")
batch_size = 64
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1) 读 checkpoint 配置并构建模型/Tokenizer
ckpt = torch.load(ckpt_path, map_location="cpu")
cfg = ckpt["config"]
tokenizer = PolyBertTokenizer(cfg.get("polybert_name", "kuelumbus/polyBERT")) if cfg.get("use_polybert") else None
model = GeoGATModel(use_polybert=cfg.get("use_polybert", False),
                    polybert=AutoModel.from_pretrained(cfg["polybert_name"]) if cfg.get("use_polybert") else None,
                    freeze_polybert=cfg.get("freeze_polybert", True),
                    seq_max_length=cfg.get("seq_max_length", 256),
                    cross_attn_heads=cfg.get("cross_attn_heads", 4),
                    cross_attn_dim=cfg.get("cross_attn_dim"),
                    **{k: cfg[k] for k in cfg if k in {"layers_in_conv","channels","num_edge_types","num_convs","num_fc_layers","geom_K","geom_rmax","concat_original_edge"}})
model.load_state_dict(ckpt["model_state"])
model.to(device).eval()

y_mean = torch.tensor(ckpt["y_mean"], device=device)
y_std = torch.tensor(ckpt["y_std"], device=device)

# 2) 构建 Dataset/Loader（与训练相同的标准化/psmiles列）
# 使用与训练一致的 stratified split
df = pd.read_csv(manifest)
test_split = 0.1
val_split = 0.1
seed = 42
try:
    bins = pd.qcut(df["label"], q=10, duplicates="drop")
    stratify_labels = bins.astype(str)
except Exception:
    stratify_labels = None

train_val_df, test_df = train_test_split(
    df, test_size=test_split, random_state=seed, stratify=stratify_labels
)
if stratify_labels is not None:
    train_val_bins = pd.qcut(train_val_df["label"], q=10, duplicates="drop")
    stratify_trainval = train_val_bins.astype(str)
else:
    stratify_trainval = None
train_df, val_df = train_test_split(
    train_val_df,
    test_size=val_split / (1 - test_split),
    random_state=seed,
    stratify=stratify_trainval if stratify_trainval is not None else None,
)

dataset = GraphDataset(test_df, root=root_dir, standardize_y=True,
                       tokenizer=tokenizer, psmiles_col="psmiles", seq_max_length=cfg.get("seq_max_length", 256))
loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

# 3) 推理
y_true, y_pred = [], []
with torch.no_grad():
    for batch in loader:
        batch = batch.to(device)
        pred_norm = model(batch).view(-1)
        pred = pred_norm * y_std + y_mean
        y = batch.y.view(-1) * y_std + y_mean  # 反标准化
        y_true.append(y.cpu())
        y_pred.append(pred.cpu())
y_true = torch.cat(y_true).numpy()
y_pred = torch.cat(y_pred).numpy()

# 4) 计算 R2/MAE/RMSE & 绘图
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
r2 = float(r2_score(y_true, y_pred))
mae = float(mean_absolute_error(y_true, y_pred))
rmse = float(mean_squared_error(y_true, y_pred, squared=False))
print(json.dumps({"R2": r2, "MAE": mae, "RMSE": rmse}, indent=2))

plt.figure(figsize=(6,6))
plt.scatter(y_true, y_pred, s=20, alpha=0.6, color="#1f77b8")
plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', label="ideal")
plt.xlabel("True Tg (K)")
plt.ylabel("Pred Tg (K)")
plt.title(f"Cross-Attn Tg Predictor\nR2={r2:.3f}, MAE={mae:.2f}K, RMSE={rmse:.2f}K")
plt.legend()
plt.tight_layout()
plt.savefig("demo/at2_r2_scatter.png", dpi=300)
plt.close()
