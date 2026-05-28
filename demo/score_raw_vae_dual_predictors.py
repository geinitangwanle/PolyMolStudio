#!/usr/bin/env python3
"""Chunked graph conversion + scoring with two Tg predictors.

This is intended for a large raw/oversampled VAE library. For each CSV chunk it:
  1) optionally checks RDKit validity and metadata filters,
  2) converts pSMILES to graph .npz files,
  3) scores the same graphs with two predictor checkpoints,
  4) writes a full scored CSV and threshold-filtered candidate CSVs.

The default threshold criterion is conservative: both predictors must be above
the threshold. The output also includes mean prediction and absolute delta.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def is_valid_psmiles(value: Any) -> bool:
    if not isinstance(value, str) or not value.strip():
        return False
    try:
        from rdkit import Chem
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "RDKit is required for --rdkit-validity-check. Use py38forGNN or pass "
            "--trust-validity-column to rely on the CSV is_valid flag."
        ) from exc
    try:
        return Chem.MolFromSmiles(value, sanitize=True) is not None
    except Exception:
        return False


def detect_column(columns: list[str], candidates: list[str], required: bool = True) -> str | None:
    lowered = {c.strip().lower(): c for c in columns}
    for candidate in candidates:
        hit = lowered.get(candidate.strip().lower())
        if hit is not None:
            return hit
    if required:
        raise ValueError(f"Could not find any of these columns: {candidates}")
    return None


def prefilter(df: pd.DataFrame, args: argparse.Namespace, smiles_col: str) -> pd.DataFrame:
    out = df
    if args.valid_only and "is_valid" in out.columns:
        out = out[out["is_valid"].astype(str).str.lower().eq("true")]
    if args.rdkit_validity_check:
        out = out[out[smiles_col].map(is_valid_psmiles)]
    if args.unique_only and "is_unique" in out.columns:
        out = out[out["is_unique"].astype(str).str.lower().eq("true")]
    if args.novel_only and "in_training_set" in out.columns:
        out = out[~out["in_training_set"].astype(str).str.lower().eq("true")]
    return out


def load_predictor(ckpt_path: Path, device: Any):
    import torch

    from predict.predict import build_model_from_ckpt

    checkpoint = torch.load(str(ckpt_path), map_location="cpu")
    model, tokenizer, cfg = build_model_from_ckpt(checkpoint, device)
    state_dict = checkpoint.get("model_state") or checkpoint.get("state_dict") or checkpoint
    model.load_state_dict(state_dict, strict=False)
    model.to(device).eval()
    y_mean = checkpoint.get("y_mean")
    y_std = checkpoint.get("y_std")
    y_mean_t = torch.tensor(y_mean, dtype=torch.float32, device=device) if y_mean is not None else None
    y_std_t = torch.tensor(y_std, dtype=torch.float32, device=device) if y_std is not None else None
    return model, tokenizer, cfg, y_mean_t, y_std_t


def predict_manifest(
    model: Any,
    tokenizer: Any,
    cfg: dict[str, Any],
    y_mean: Any,
    y_std: Any,
    manifest_df: pd.DataFrame,
    graph_dir: Path,
    args: argparse.Namespace,
    device: Any,
) -> pd.DataFrame:
    from torch_geometric.loader import DataLoader

    from predict.predict import predict_loader
    from utils.GraphDataset import GraphDataset

    dataset = GraphDataset(
        manifest_df,
        root=graph_dir,
        standardize_y=False,
        tokenizer=tokenizer if cfg.get("use_polybert", False) else None,
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
    return pd.DataFrame(predict_loader(model, loader, device, y_mean, y_std))


def threshold_mask(scored: pd.DataFrame, threshold: float, criterion: str) -> pd.Series:
    if criterion == "both":
        return (scored["pred_model1"] >= threshold) & (scored["pred_model2"] >= threshold)
    if criterion == "mean":
        return scored["pred_mean"] >= threshold
    if criterion == "either":
        return (scored["pred_model1"] >= threshold) | (scored["pred_model2"] >= threshold)
    raise ValueError(f"Unknown threshold criterion: {criterion}")


def append_csv(path: Path, df: pd.DataFrame, wrote_header: bool) -> bool:
    if df.empty:
        return wrote_header
    df.to_csv(path, index=False, mode="a" if wrote_header else "w", header=not wrote_header)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Score an oversampled VAE library with two Tg predictors.")
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--smiles-col", default=None)
    parser.add_argument("--ckpt-model1", type=Path, required=True)
    parser.add_argument("--ckpt-model2", type=Path, required=True)
    parser.add_argument("--model1-name", default="model1")
    parser.add_argument("--model2-name", default="model2")
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "analysis_outputs/raw_vae_100k_dual_scored")
    parser.add_argument("--thresholds", type=float, nargs="+", default=[500.0, 600.0, 700.0])
    parser.add_argument("--threshold-criterion", choices=["both", "mean", "either"], default="both")
    parser.add_argument("--max-delta-k", type=float, default=None, help="Optional extra Delta triage for threshold files.")
    parser.add_argument("--chunk-size", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--tmp-dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit-rows", type=int, default=None)
    parser.add_argument("--valid-only", action="store_true", default=True)
    parser.add_argument("--all-validity-states", dest="valid_only", action="store_false")
    parser.add_argument("--rdkit-validity-check", action="store_true", default=True)
    parser.add_argument("--trust-validity-column", dest="rdkit_validity_check", action="store_false")
    parser.add_argument("--unique-only", action="store_true", default=True)
    parser.add_argument("--all-duplicate-states", dest="unique_only", action="store_false")
    parser.add_argument("--novel-only", action="store_true", default=True)
    parser.add_argument("--include-training-set", dest="novel_only", action="store_false")
    args = parser.parse_args()

    import torch

    from predict.predict import set_seed
    from utils.PSMILES_to_graph import convert_csv_to_graphs

    set_seed(args.seed)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.tmp_dir is not None:
        args.tmp_dir.mkdir(parents=True, exist_ok=True)

    columns = pd.read_csv(args.input_csv, nrows=0).columns.tolist()
    smiles_col = args.smiles_col or detect_column(columns, ["smiles", "PSMILES"])

    model1 = load_predictor(args.ckpt_model1, device)
    model2 = load_predictor(args.ckpt_model2, device)

    scored_path = args.out_dir / "dual_predictor_scored.csv"
    threshold_paths = {t: args.out_dir / f"monomers_Tg_ge_{t:g}K_{args.threshold_criterion}.csv" for t in args.thresholds}
    wrote_scored = False
    wrote_threshold = {t: False for t in args.thresholds}
    selected_counts = {t: 0 for t in args.thresholds}
    total_rows = 0
    kept_rows = 0
    scored_rows = 0

    for chunk_idx, chunk in enumerate(pd.read_csv(args.input_csv, chunksize=args.chunk_size), start=1):
        total_rows += len(chunk)
        chunk = prefilter(chunk, args, smiles_col)
        if args.limit_rows is not None:
            remaining = args.limit_rows - kept_rows
            if remaining <= 0:
                break
            chunk = chunk.head(remaining)
        kept_rows += len(chunk)
        if chunk.empty:
            continue

        with tempfile.TemporaryDirectory(dir=args.tmp_dir) as tmp:
            tmp_dir = Path(tmp)
            chunk_csv = tmp_dir / "chunk.csv"
            graph_dir = tmp_dir / "graphs"
            chunk = chunk.reset_index(drop=False).rename(columns={"index": "source_row"})
            chunk.to_csv(chunk_csv, index=False)

            _, manifest_df = convert_csv_to_graphs(
                csv_path=chunk_csv,
                label_col=None,
                PSMILES_col=smiles_col,
                save_dir=graph_dir,
            )
            if manifest_df.empty:
                continue

            pred1 = predict_manifest(*model1, manifest_df, graph_dir, args, device)
            pred2 = predict_manifest(*model2, manifest_df, graph_dir, args, device)
            pred1_by_id = pred1.set_index("mol_id")["pred"]
            pred2_by_id = pred2.set_index("mol_id")["pred"]
            scored = chunk.copy()
            scored["pred_model1"] = scored.index.map(pred1_by_id)
            scored["pred_model2"] = scored.index.map(pred2_by_id)
            scored = scored.dropna(subset=["pred_model1", "pred_model2"])
            scored["pred_mean"] = scored[["pred_model1", "pred_model2"]].mean(axis=1)
            scored["pred_min"] = scored[["pred_model1", "pred_model2"]].min(axis=1)
            scored["pred_max"] = scored[["pred_model1", "pred_model2"]].max(axis=1)
            scored["pred_delta_abs"] = (scored["pred_model1"] - scored["pred_model2"]).abs()
            scored["model1_name"] = args.model1_name
            scored["model2_name"] = args.model2_name
            scored_rows += len(scored)

            wrote_scored = append_csv(scored_path, scored, wrote_scored)
            for threshold in args.thresholds:
                mask = threshold_mask(scored, threshold, args.threshold_criterion)
                if args.max_delta_k is not None:
                    mask &= scored["pred_delta_abs"] <= args.max_delta_k
                selected = scored.loc[mask].sort_values(["pred_mean", "pred_min"], ascending=False)
                selected_counts[threshold] += len(selected)
                wrote_threshold[threshold] = append_csv(threshold_paths[threshold], selected, wrote_threshold[threshold])

        print(
            f"[chunk {chunk_idx}] input={total_rows:,} kept={kept_rows:,} scored={scored_rows:,} "
            + ", ".join(f">={t:g}K:{selected_counts[t]:,}" for t in args.thresholds),
            flush=True,
        )
        if args.limit_rows is not None and kept_rows >= args.limit_rows:
            break

    summary = {
        "input_csv": str(args.input_csv),
        "scored_csv": str(scored_path),
        "model1_name": args.model1_name,
        "model1_ckpt": str(args.ckpt_model1),
        "model2_name": args.model2_name,
        "model2_ckpt": str(args.ckpt_model2),
        "threshold_criterion": args.threshold_criterion,
        "max_delta_k": args.max_delta_k,
        "n_input_rows_seen": total_rows,
        "n_rows_after_filters": kept_rows,
        "n_scored": scored_rows,
        "thresholds": [
            {
                "threshold_k": t,
                "n_selected": selected_counts[t],
                "fraction_of_scored": selected_counts[t] / scored_rows if scored_rows else None,
                "output_csv": str(threshold_paths[t]),
            }
            for t in args.thresholds
        ],
    }
    (args.out_dir / "dual_predictor_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    pd.DataFrame(summary["thresholds"]).to_csv(args.out_dir / "dual_predictor_threshold_summary.csv", index=False)
    print(f"Saved scored CSV to {scored_path}")
    print(f"Saved summary to {args.out_dir / 'dual_predictor_summary.json'}")


if __name__ == "__main__":
    main()
