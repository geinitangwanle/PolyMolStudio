#!/usr/bin/env python3
"""DPO training script for ConditionalVAESmiles (modelv4).

核心循环：sample → reward → logp(agent/prior) → DPO loss → backward。
默认 reward 为 1，占位使用；请在 compute_reward 中集成真实奖励（Tg 预测器、合法性、新颖度等）。
"""

import argparse
import random
from pathlib import Path
import sys

import numpy as np
import torch
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.generator.modelv4_premium import ConditionalVAESmiles
from models.generator.tokenizer import PolyBertTokenizer
from models.generator.dataset_tg import make_loader_with_tg
import models.generator.rl_losses as rl_losses
from train.RL.reward_utils import RewardConfig, compute_reward
from train.RL.tg_predictor import TgGraphPredictor


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_model(args) -> ConditionalVAESmiles:
    model = ConditionalVAESmiles(
        vocab_size=args.vocab_size,
        emb_dim=args.emb_dim,
        encoder_hid_dim=args.encoder_hid_dim,
        decoder_hid_dim=args.decoder_hid_dim,
        z_dim=args.z_dim,
        cond_dim=1,
        cond_latent_dim=args.cond_latent_dim,
        pad_id=args.pad_id,
        bos_id=args.bos_id,
        eos_id=args.eos_id,
        drop=args.dropout,
        use_polybert=args.use_polybert,
        polybert_name=args.polybert_name,
        freeze_polybert=args.freeze_polybert,
        polybert_pooling=args.polybert_pooling,
        max_len=args.max_len,
        num_decoder_layers=args.num_decoder_layers,
        decoder_nhead=args.decoder_nhead,
        decoder_ff_mult=args.decoder_ff_mult,
        use_tg_regression=args.use_tg_regression,
        tg_hidden_dim=args.tg_hidden_dim,
    )
    if args.ckpt:
        state = torch.load(args.ckpt, map_location="cpu")
        state_dict = None
        if isinstance(state, dict):
            state_dict = state.get("model_state") or state.get("model") or state
        else:
            state_dict = state
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing or unexpected:
            print(f"[warn] ckpt loaded with missing keys: {missing} | unexpected: {unexpected}")
    return model


def compute_reward(tokens: torch.Tensor, conditions: torch.Tensor) -> torch.Tensor:
    return torch.ones(tokens.size(0), device=tokens.device)


def parse_args():
    p = argparse.ArgumentParser(description="DPO training for modelv4")
    # data
    p.add_argument("--data", type=str, required=True, help="CSV with PSMILES+Tg columns")
    p.add_argument("--col_smiles", type=str, default="PSMILES")
    p.add_argument("--col_tg", type=str, default="Tg")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=4)
    # model/tokenizer
    p.add_argument("--polybert_name", type=str, default="kuelumbus/polyBERT")
    p.add_argument("--use_polybert", action="store_true", default=True)
    p.add_argument("--freeze_polybert", action="store_true", default=False)
    p.add_argument("--polybert_pooling", type=str, default="cls", choices=["cls", "mean"])
    p.add_argument("--emb_dim", type=int, default=256)
    p.add_argument("--encoder_hid_dim", type=int, default=512)
    p.add_argument("--decoder_hid_dim", type=int, default=None)
    p.add_argument("--z_dim", type=int, default=128)
    p.add_argument("--cond_latent_dim", type=int, default=32)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--max_len", type=int, default=256)
    p.add_argument("--num_decoder_layers", type=int, default=4)
    p.add_argument("--decoder_nhead", type=int, default=8)
    p.add_argument("--decoder_ff_mult", type=int, default=4)
    p.add_argument("--use_tg_regression", action="store_true", default=False)
    p.add_argument("--tg_hidden_dim", type=int, default=128)
    p.add_argument("--ckpt", type=str, default=None, help="initialize agent/prior from checkpoint")
    # training
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--clip", type=float, default=1.0)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top_k", type=int, default=None)
    p.add_argument("--top_p", type=float, default=None)
    p.add_argument("--sigma", type=float, default=1.0, help="DPO sigma")
    # reward weights
    p.add_argument("--w_tg", type=float, default=1.0)
    p.add_argument("--w_valid", type=float, default=0.5)
    p.add_argument("--w_sa", type=float, default=0.25)
    p.add_argument("--w_novelty", type=float, default=0.25)
    p.add_argument("--sa_floor", type=float, default=0.0)
    p.add_argument("--sa_ceil", type=float, default=10.0)
    p.add_argument("--tg_ckpt", type=str, default=None, help="GeoGAT checkpoint for Tg prediction reward")
    p.add_argument("--tg_match_target", action="store_true", help="Use -|pred-target| as Tg reward term")
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save_dir", type=str, default="checkpoints_rl")
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    tokenizer = PolyBertTokenizer(args.polybert_name)
    args.vocab_size = tokenizer.vocab_size
    args.pad_id = tokenizer.pad_id
    args.bos_id = tokenizer.bos_id or tokenizer.pad_id
    args.eos_id = tokenizer.eos_id or tokenizer.pad_id

    loader, _ = make_loader_with_tg(
        args.data,
        tokenizer,
        col_smiles=args.col_smiles,
        col_tg=args.col_tg,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_len=args.max_len,
        shuffle=True,
    )

    prior = load_model(args).to(device)
    for p in prior.parameters():
        p.requires_grad = False
    prior.eval()

    agent = load_model(args).to(device)
    agent.train()

    optimizer = torch.optim.AdamW([p for p in agent.parameters() if p.requires_grad], lr=args.lr)

    reward_cfg = RewardConfig(
        w_tg=args.w_tg,
        w_valid=args.w_valid,
        w_sa=args.w_sa,
        w_novelty=args.w_novelty,
        sa_floor=args.sa_floor,
        sa_ceil=args.sa_ceil,
    )

    tg_predictor = None
    if args.tg_ckpt:
        tg_predictor = TgGraphPredictor(args.tg_ckpt, device=device)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        pbar = tqdm(loader, desc=f"Epoch {epoch:03d}")
        for batch in pbar:
            input_ids = batch["input_ids"].to(device)
            conditions = batch["tg"].unsqueeze(1).to(device)

            tokens, _, _, _ = agent.sample_with_logprob(
                num_samples=input_ids.size(0),
                conditions=conditions,
                max_len=args.max_len,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
            )

            reward = compute_reward(
                tokens,
                conditions,
                tokenizer,
                reward_cfg,
                tg_predict_fn=tg_predictor if tg_predictor is not None else None,
                match_target=args.tg_match_target,
            )
            logp_agent, _ = agent.log_prob(tokens, conditions)
            with torch.no_grad():
                logp_prior, _ = prior.log_prob(tokens, conditions)

            loss = rl_losses.dpo_loss(logp_agent, logp_prior, reward, sigma=args.sigma, reduction="mean")

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.parameters(), args.clip)
            optimizer.step()

            pbar.set_postfix({"loss": float(loss.item())})

        ckpt_path = save_dir / f"dpo_epoch{epoch:03d}.pt"
        torch.save({"model_state": agent.state_dict(), "config": vars(args)}, ckpt_path)
        print(f"Saved: {ckpt_path}")


if __name__ == "__main__":
    main()
