"""
Generate polymers with the VAE generator and score them with the Tg predictor.
Pipeline:
  1) Load generator checkpoint (v4 uncond) and sample pSMILES.
  2) Convert sampled pSMILES to graphs.
  3) Load Tg GNN checkpoint and predict Tg for each sample.
  4) Save samples and scored results.
"""

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Optional, Tuple

import pandas as pd
import torch
from torch_geometric.loader import DataLoader


REPO_ROOT = Path(__file__).resolve().parent.parent
for p in (REPO_ROOT, REPO_ROOT / "utils"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from models.generator.tokenizer import PolyBertTokenizer  # noqa: E402
from models.predictor.GeoGATModel import GeoGATModel  # noqa: E402
from predict.predict import build_model_from_ckpt, predict_loader  # noqa: E402
from sample.sample_v4_uncond import load_model as load_gen_model  # noqa: E402
from sample.sample_v4_uncond import sample_smiles  # noqa: E402
from utils.PSMILES_to_graph import convert_csv_to_graphs  # noqa: E402
from predict.predict import set_seed  # noqa: E402


def load_predictor(ckpt_path: Path, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device)
    model, tokenizer, cfg = build_model_from_ckpt(ckpt, device)
    state = ckpt.get("model_state") or ckpt.get("model") or ckpt
    model.load_state_dict(state)
    model.to(device).eval()
    y_mean = torch.tensor(ckpt["y_mean"], device=device) if "y_mean" in ckpt else None
    y_std = torch.tensor(ckpt["y_std"], device=device) if "y_std" in ckpt else None
    return model, tokenizer, cfg, y_mean, y_std


def run(args):
    # 生成模型可用 CUDA/MPS，预测模型默认走 CPU（torch_sparse 对 MPS 支持有限）
    gen_device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    if args.predict_device == "auto":
        pred_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        pred_device = torch.device(args.predict_device)

    set_seed(args.seed)

    # === Step 1: load generator and sample ===
    gen_args = SimpleNamespace(
        checkpoint=args.gen_checkpoint,
        polybert_dir=args.polybert_dir,
        model_size=args.model_size,
        max_len=args.max_len,
        seed=args.seed,
    )
    gen_model, gen_tokenizer = load_gen_model(gen_args, gen_device)
    samples = sample_smiles(
        model=gen_model,
        tokenizer=gen_tokenizer,
        device=gen_device,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        max_len=args.max_len,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
    )

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    samples_csv = output_dir / args.samples_file
    pd.DataFrame({"PSMILES": samples}).to_csv(samples_csv, index=False)

    # === Step 2: convert to graphs ===
    graphs_dir = output_dir / "graphs"
    graphs_dir.mkdir(parents=True, exist_ok=True)
    _, manifest_df = convert_csv_to_graphs(
        csv_path=samples_csv,
        label_col=None,
        PSMILES_col="PSMILES",
        save_dir=graphs_dir,
    )

    # === Step 3: load predictor and score ===
    predictor, pred_tokenizer, pred_cfg, y_mean, y_std = load_predictor(args.predictor_ckpt, pred_device)
    # Build dataset on the fly using same GraphDataset logic as predict.py
    from utils.GraphDataset import GraphDataset  # localized import to avoid circular deps

    dataset = GraphDataset(
        manifest_df,
        root=graphs_dir,
        standardize_y=False,
        tokenizer=pred_tokenizer if pred_cfg.get("use_polybert", False) else None,
        psmiles_col="psmiles",
        seq_max_length=pred_cfg.get("seq_max_length", 256),
    )
    loader = DataLoader(dataset, batch_size=args.predict_batch_size, num_workers=args.num_workers, shuffle=False)

    preds = predict_loader(predictor, loader, pred_device, y_mean, y_std)
    preds_df = pd.DataFrame(preds)
    if "mol_id" not in preds_df:
        preds_df["mol_id"] = range(len(preds_df))
    samples_df = pd.DataFrame({"mol_id": range(len(samples)), "PSMILES": samples})
    merged = samples_df.merge(preds_df, on="mol_id", how="left")
    scored_csv = output_dir / args.scored_file
    merged.to_csv(scored_csv, index=False)

    print(f"Saved samples to {samples_csv}")
    print(f"Saved scored results to {scored_csv}")


def build_parser():
    p = argparse.ArgumentParser(description="Generate pSMILES with generator and score Tg with predictor.")
    p.add_argument("--gen-checkpoint", type=Path, default=REPO_ROOT / "checkpoints/pretrain_modelv4.pt")
    p.add_argument("--polybert-dir", type=Path, default=REPO_ROOT / "polybert")
    p.add_argument("--model-size", type=str, default="base", choices=["base", "medium", "premium"])
    p.add_argument("--num-samples", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--max-len", type=int, default=256)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-k", type=int, default=None)
    p.add_argument("--top-p", type=float, default=None)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--predictor-ckpt", type=Path, required=True, help="Path to Tg predictor checkpoint (.pt).")
    p.add_argument("--predict-batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--predict-device", type=str, default="cpu", choices=["auto", "cpu", "cuda"], help="Device for Tg predictor.")

    p.add_argument("--output-dir", type=Path, default=REPO_ROOT / "design_outputs")
    p.add_argument("--samples-file", type=str, default="generated_samples.csv")
    p.add_argument("--scored-file", type=str, default="generated_scored.csv")
    return p


if __name__ == "__main__":
    parser = build_parser()
    run(parser.parse_args())
