"""
Quick latent space visualizer for model_v4 base checkpoints.

It loads a ConditionalVAESmiles checkpoint, encodes a CSV of pSMILES+Tg,
reduces the latent vectors to 2D (PCA by default, UMAP/TSNE if available),
and saves scatter plots colored by Tg.
"""

import argparse
import math
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import AutoModel

try:
    import umap.umap_ as umap

    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False

try:
    from sklearn.manifold import TSNE

    HAS_TSNE = True
except ImportError:
    HAS_TSNE = False

from sklearn.decomposition import PCA

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV = REPO_ROOT / "data" / "raw" / "PSMILES_Tg_only.csv"
DEFAULT_CKPT = REPO_ROOT / "checkpoints" / "pretrain_modelv4.pt"
DEFAULT_POLYBERT = REPO_ROOT / "polybert"
DEFAULT_OUT = REPO_ROOT / "demo" / "analysis_outputs" / "v4_latent.png"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.generator.dataset_tg import TgStats, make_loader_with_tg  # noqa: E402
from models.generator.modelv4 import ConditionalVAESmiles  # noqa: E402
from models.generator.modelv4_medium import ConditionalVAESmiles as ConditionalVAESmilesMedium  # noqa: E402
from models.generator.modelv4_premium import ConditionalVAESmiles as ConditionalVAESmilesPremium  # noqa: E402
from models.generator.tokenizer import PolyBertTokenizer  # noqa: E402
from models.generator.train import set_seed  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="可视化 model_v4 latent space（支持多参数对比）")
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_CKPT, help="模型 checkpoint 路径")
    p.add_argument("--polybert-dir", type=Path, default=DEFAULT_POLYBERT, help="polyBERT 权重/分词器目录或 HF 名称")
    p.add_argument("--csv", type=Path, default=DEFAULT_CSV, help="包含 pSMILES 与 Tg 的 CSV")
    p.add_argument("--col-smiles", type=str, default="PSMILES", help="SMILES 列名")
    p.add_argument("--col-tg", type=str, default="Tg", help="Tg 列名")
    p.add_argument("--max-len", type=int, default=256, help="token 最大长度")
    p.add_argument("--batch-size", type=int, default=64, help="编码批大小")
    p.add_argument("--num-workers", type=int, default=0, help="dataloader workers 数量")
    p.add_argument("--max-samples", type=int, default=4000, help="最多可视化多少条（减少大数据集内存/时间）")
    p.add_argument("--method", choices=["pca", "umap", "tsne"], default="pca", help="降维方法")
    p.add_argument("--seed", type=int, default=42, help="随机种子")
    p.add_argument("--model-size", choices=["base", "medium", "premium"], default=None, help="强制模型体量，不填则从 ckpt 读取")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT, help="输出图片路径")
    p.add_argument("--keep-norm", action="store_true", help="使用标准化 Tg 着色（默认反标准化）")
    p.add_argument(
        "--latent-kind",
        choices=["base", "cond", "concat"],
        default="base",
        help="可视化哪种潜变量：base=mu, cond=条件潜变量, concat=拼接潜变量",
    )
    p.add_argument(
        "--run",
        action="append",
        default=[],
        help="多参数对比项，格式: label::/path/to/ckpt.pt 或 label::/path/to/ckpt.pt::model_size",
    )
    p.add_argument("--single-output", action="store_true", help="多 run 模式下仍保存单图（每个 run 一张）")
    return p.parse_args()


def prepare_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _pick_model_cls(size: Optional[str]):
    if size == "medium":
        return ConditionalVAESmilesMedium
    if size == "premium":
        return ConditionalVAESmilesPremium
    return ConditionalVAESmiles


def load_model(args, device, checkpoint: Path, model_size: Optional[str]) -> Tuple[ConditionalVAESmiles, PolyBertTokenizer, Optional[TgStats]]:
    set_seed(args.seed)
    try:
        ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(checkpoint, map_location=device)

    tokenizer_name = ckpt.get("tokenizer_name", str(args.polybert_dir))
    tokenizer = PolyBertTokenizer(tokenizer_name)
    polybert = AutoModel.from_pretrained(tokenizer_name).to(device)

    size = model_size or args.model_size or ckpt.get("model_size") or "base"
    ModelCls = _pick_model_cls(size)

    if "model_kwargs" in ckpt:
        model_kwargs = ckpt["model_kwargs"]
        model_kwargs.update(
            {
                "vocab_size": tokenizer.vocab_size,
                "polybert": polybert,
                "use_polybert": True,
                "pad_id": tokenizer.pad_id,
                "bos_id": tokenizer.bos_id,
                "eos_id": tokenizer.eos_id,
            }
        )
        model = ModelCls(**model_kwargs).to(device)
    else:
        model = ModelCls(
            vocab_size=tokenizer.vocab_size,
            emb_dim=256,
            encoder_hid_dim=polybert.config.hidden_size,
            decoder_hid_dim=512,
            z_dim=128,
            cond_dim=1,
            cond_latent_dim=32,
            pad_id=tokenizer.pad_id,
            bos_id=tokenizer.bos_id,
            eos_id=tokenizer.eos_id,
            drop=0.1,
            use_polybert=True,
            polybert=polybert,
            freeze_polybert=True,
            polybert_pooling="cls",
            use_tg_regression=False,
        ).to(device)

    model.load_state_dict(ckpt["model"])
    model.eval()

    tg_stats = TgStats(**ckpt["tg_stats"]) if "tg_stats" in ckpt else None
    return model, tokenizer, tg_stats


@torch.no_grad()
def collect_latents(model, loader, device, latent_kind: str):
    zs, tg_vals = [], []
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attn = batch["attention_mask"].to(device)
        tg = batch["tg"].to(device)

        mu, _ = model.encode(input_ids, attention_mask=attn)
        tg_in = tg.view(-1, 1)
        cond_latent = model.cond_encoder(tg_in)

        if latent_kind == "cond":
            z_now = cond_latent
        elif latent_kind == "concat":
            z_now = torch.cat([mu, cond_latent], dim=-1)
        else:
            z_now = mu

        zs.append(z_now.cpu())
        tg_vals.append(tg.cpu())

    z_base = torch.cat(zs, dim=0)
    tg = torch.cat(tg_vals, dim=0)
    return z_base, tg


def subsample(z, tg, max_samples: int, seed: int):
    if max_samples is None or z.size(0) <= max_samples:
        return z, tg
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(z.size(0), generator=g)[:max_samples]
    return z[idx], tg[idx]


def reduce_dim(z, method: str, seed: int):
    if method == "umap":
        if not HAS_UMAP:
            print("UMAP 未安装，退回 PCA")
        else:
            reducer = umap.UMAP(random_state=seed)
            return reducer.fit_transform(z)
        method = "pca"

    if method == "tsne":
        if not HAS_TSNE:
            print("TSNE 未安装，退回 PCA")
        else:
            return TSNE(n_components=2, random_state=seed, init="pca", learning_rate="auto").fit_transform(z)
        method = "pca"

    return PCA(n_components=2, random_state=seed).fit_transform(z)


def plot_embeddings(z2d, tg, out_path: Path, method: str, tg_label: str, latent_kind: str):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 4))
    scatter = ax.scatter(z2d[:, 0], z2d[:, 1], c=tg, cmap="viridis", s=8, alpha=0.8)
    ax.set_title(f"{latent_kind} latent ({method})")
    ax.set_xlabel("dim 1")
    ax.set_ylabel("dim 2")
    cbar = fig.colorbar(scatter, ax=ax, orientation="vertical", fraction=0.046, pad=0.04)
    cbar.set_label(tg_label)
    fig.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"[+] 图已保存到 {out_path}")


def plot_compare_embeddings(run_results, out_path: Path, method: str, tg_label: str, latent_kind: str):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = len(run_results)
    ncols = min(3, n)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5.2 * ncols, 4.2 * nrows), squeeze=False)
    cmap = "viridis"
    vmin = min(float(r["tg"].min()) for r in run_results)
    vmax = max(float(r["tg"].max()) for r in run_results)

    last_scatter = None
    for i, run in enumerate(run_results):
        r, c = divmod(i, ncols)
        ax = axes[r][c]
        last_scatter = ax.scatter(
            run["z2d"][:, 0],
            run["z2d"][:, 1],
            c=run["tg"],
            cmap=cmap,
            s=8,
            alpha=0.8,
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_title(run["label"])
        ax.set_xlabel("dim 1")
        ax.set_ylabel("dim 2")

    for j in range(n, nrows * ncols):
        r, c = divmod(j, ncols)
        axes[r][c].axis("off")

    if last_scatter is not None:
        cbar = fig.colorbar(last_scatter, ax=axes.ravel().tolist(), orientation="vertical", fraction=0.025, pad=0.02)
        cbar.set_label(tg_label)

    fig.suptitle(f"Latent Space Comparison ({latent_kind}, {method})", y=0.995)
    fig.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"[+] 对比图已保存到 {out_path}")


def parse_run_specs(run_specs: List[str], default_checkpoint: Path) -> List[Tuple[str, Path, Optional[str]]]:
    if not run_specs:
        return [("default", default_checkpoint, None)]

    runs: List[Tuple[str, Path, Optional[str]]] = []
    for spec in run_specs:
        parts = spec.split("::")
        if len(parts) not in (2, 3):
            raise ValueError(f"run 配置格式错误: {spec}，应为 label::checkpoint 或 label::checkpoint::model_size")
        label, ckpt = parts[0].strip(), parts[1].strip()
        size = parts[2].strip() if len(parts) == 3 else None
        if size is not None and size not in {"base", "medium", "premium"}:
            raise ValueError(f"run 的 model_size 必须是 base/medium/premium，收到: {size}")
        runs.append((label or Path(ckpt).stem, Path(ckpt), size))
    return runs


def main():
    args = parse_args()
    device = prepare_device()
    print(f"[info] device: {device}")
    runs = parse_run_specs(args.run, args.checkpoint)
    run_results = []

    for idx, (label, ckpt_path, run_model_size) in enumerate(runs):
        print(f"[info] run[{idx}] label={label}, ckpt={ckpt_path}, model_size={run_model_size or args.model_size or 'auto'}")
        model, tokenizer, tg_stats_ckpt = load_model(args, device, ckpt_path, run_model_size)
        loader, tg_stats_data = make_loader_with_tg(
            args.csv,
            tokenizer,
            col_smiles=args.col_smiles,
            col_tg=args.col_tg,
            max_len=args.max_len,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            tg_stats=tg_stats_ckpt,
        )
        tg_stats = tg_stats_ckpt or tg_stats_data

        z_latent, tg = collect_latents(model, loader, device, args.latent_kind)
        z_latent, tg = subsample(z_latent, tg, args.max_samples, args.seed)
        tg_np = tg.numpy()

        tg_to_plot = tg_np if args.keep_norm or tg_stats is None else tg_stats.denormalize(tg_np)
        tg_label = "Tg (normalized)" if args.keep_norm or tg_stats is None else "Tg (K)"

        run_results.append({"label": label, "z": z_latent.numpy(), "tg": tg_to_plot})

    if len(run_results) == 1:
        z2d = reduce_dim(run_results[0]["z"], args.method, args.seed)
        plot_embeddings(z2d, run_results[0]["tg"], args.out, args.method, tg_label, args.latent_kind)
        return

    all_z = [r["z"] for r in run_results]
    all_z_cat = all_z[0] if len(all_z) == 1 else np.concatenate(all_z, axis=0)
    all_z2d = reduce_dim(all_z_cat, args.method, args.seed)
    start = 0
    for run in run_results:
        n_i = run["z"].shape[0]
        run["z2d"] = all_z2d[start : start + n_i]
        start += n_i

    plot_compare_embeddings(run_results, args.out, args.method, tg_label, args.latent_kind)

    if args.single_output:
        for run in run_results:
            single_out = args.out.with_name(f"{args.out.stem}_{run['label']}{args.out.suffix}")
            plot_embeddings(run["z2d"], run["tg"], single_out, args.method, tg_label, args.latent_kind)


if __name__ == "__main__":
    main()
