#!/usr/bin/env python3
"""Score/filter a raw VAE-generated polymer-monomer library by predicted Tg.

The input library can either already contain a prediction column, or it can be
scored on the fly in chunks with a GeoGAT predictor checkpoint. Chunked scoring
keeps the 2M-row oversampling library manageable: each chunk is converted to
graphs, predicted, filtered, and then its temporary graph files are removed.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def detect_column(columns: list[str], candidates: list[str], required: bool = True) -> str | None:
    lowered = {c.strip().lower(): c for c in columns}
    for candidate in candidates:
        hit = lowered.get(candidate.strip().lower())
        if hit is not None:
            return hit
    if required:
        raise ValueError(f"Could not find any of these columns: {candidates}")
    return None


def is_valid_psmiles(value: Any) -> bool:
    if not isinstance(value, str) or not value.strip():
        return False
    try:
        from rdkit import Chem
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "RDKit is required for --rdkit-validity-check. Use the project environment "
            "that contains RDKit, or pass --trust-validity-column to rely on the CSV flag."
        ) from exc
    try:
        mol = Chem.MolFromSmiles(value, sanitize=True)
    except Exception:
        return False
    return mol is not None


def apply_pre_filters(df: pd.DataFrame, args: argparse.Namespace, smiles_col: str) -> pd.DataFrame:
    filtered = df
    if args.valid_only and "is_valid" in filtered.columns:
        filtered = filtered[filtered["is_valid"].astype(str).str.lower().eq("true")]
    if args.rdkit_validity_check:
        filtered = filtered[filtered[smiles_col].map(is_valid_psmiles)]
    if args.unique_only and "is_unique" in filtered.columns:
        filtered = filtered[filtered["is_unique"].astype(str).str.lower().eq("true")]
    if args.novel_only and "in_training_set" in filtered.columns:
        filtered = filtered[~filtered["in_training_set"].astype(str).str.lower().eq("true")]
    return filtered


def load_predictor(ckpt_path: Path, device: torch.device) -> tuple[torch.nn.Module, Any, dict[str, Any], Any, Any]:
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


def write_threshold_outputs(
    df: pd.DataFrame,
    pred_col: str,
    thresholds: list[float],
    out_dir: Path,
    mode: str,
    header_state: dict[float, bool],
) -> dict[float, int]:
    counts = {}
    scores = pd.to_numeric(df[pred_col], errors="coerce")
    for threshold in thresholds:
        selected = df.loc[scores >= threshold].copy()
        counts[threshold] = int(len(selected))
        if selected.empty:
            continue
        selected = selected.sort_values(pred_col, ascending=False)
        out_path = out_dir / f"monomers_Tg_ge_{threshold:g}K.csv"
        selected.to_csv(
            out_path,
            index=False,
            mode=mode if not header_state[threshold] else "a",
            header=not header_state[threshold],
        )
        header_state[threshold] = True
    return counts


def filter_existing_predictions(args: argparse.Namespace, pred_col: str, smiles_col: str) -> None:
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.input_csv)
    before = len(df)
    df = apply_pre_filters(df, args, smiles_col)
    if args.limit_rows is not None:
        df = df.head(args.limit_rows)
    header_state = {t: False for t in args.thresholds}
    counts = write_threshold_outputs(df, pred_col, args.thresholds, out_dir, "w", header_state)
    write_summary(out_dir, args, before, len(df), len(df[pred_col].dropna()), counts)


def score_and_filter_chunks(args: argparse.Namespace, smiles_col: str) -> None:
    import torch
    from torch_geometric.loader import DataLoader

    from predict.predict import predict_loader, set_seed
    from utils.GraphDataset import GraphDataset
    from utils.PSMILES_to_graph import convert_csv_to_graphs

    if args.ckpt_path is None:
        raise ValueError("Input has no prediction column. Provide --ckpt-path to score the raw library.")

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    model, tokenizer, cfg, y_mean, y_std = load_predictor(args.ckpt_path, device)

    header_state = {t: False for t in args.thresholds}
    total_rows = 0
    kept_rows = 0
    scored_rows = 0
    counts = {t: 0 for t in args.thresholds}

    reader = pd.read_csv(args.input_csv, chunksize=args.chunk_size)
    for chunk_idx, chunk in enumerate(reader, start=1):
        if args.max_chunks is not None and chunk_idx > args.max_chunks:
            break
        total_rows += len(chunk)
        chunk = apply_pre_filters(chunk, args, smiles_col)
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
            pred_rows = pd.DataFrame(predict_loader(model, loader, device, y_mean, y_std))
            if pred_rows.empty:
                continue
            scored = chunk.merge(pred_rows[["mol_id", "pred"]], left_index=True, right_on="mol_id", how="inner")
            scored_rows += len(scored)
            chunk_counts = write_threshold_outputs(
                scored.drop(columns=["mol_id"]),
                "pred",
                args.thresholds,
                out_dir,
                "a",
                header_state,
            )
            for threshold, count in chunk_counts.items():
                counts[threshold] += count

        print(
            f"[chunk {chunk_idx}] input={total_rows:,} kept={kept_rows:,} "
            f"scored={scored_rows:,} hits="
            + ", ".join(f">={t:g}K:{counts[t]:,}" for t in args.thresholds),
            flush=True,
        )
        if args.limit_rows is not None and kept_rows >= args.limit_rows:
            break

    write_summary(out_dir, args, total_rows, kept_rows, scored_rows, counts)


def write_filtered_subset(args: argparse.Namespace, smiles_col: str) -> None:
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    subset_path = args.subset_out_csv or out_dir / "valid_unique_novel_subset.csv"
    wrote_header = False
    total_rows = 0
    kept_rows = 0

    for chunk_idx, chunk in enumerate(pd.read_csv(args.input_csv, chunksize=args.chunk_size), start=1):
        if args.max_chunks is not None and chunk_idx > args.max_chunks:
            break
        total_rows += len(chunk)
        chunk = apply_pre_filters(chunk, args, smiles_col)
        if args.limit_rows is not None:
            remaining = args.limit_rows - kept_rows
            if remaining <= 0:
                break
            chunk = chunk.head(remaining)
        if chunk.empty:
            continue
        chunk.to_csv(subset_path, index=False, mode="a" if wrote_header else "w", header=not wrote_header)
        wrote_header = True
        kept_rows += len(chunk)
        print(f"[chunk {chunk_idx}] input={total_rows:,} subset={kept_rows:,}", flush=True)
        if args.limit_rows is not None and kept_rows >= args.limit_rows:
            break

    summary = pd.DataFrame(
        [
            {
                "input_csv": str(args.input_csv),
                "subset_csv": str(subset_path),
                "n_input_rows_seen": total_rows,
                "n_subset_rows": kept_rows,
                "limit_rows": args.limit_rows,
                "rdkit_validity_check": args.rdkit_validity_check,
                "valid_only": args.valid_only,
                "unique_only": args.unique_only,
                "novel_only": args.novel_only,
            }
        ]
    )
    summary.to_csv(out_dir / "subset_summary.csv", index=False)
    print(f"Saved subset to {subset_path}")
    print(f"Saved summary to {out_dir / 'subset_summary.csv'}")


def write_summary(
    out_dir: Path,
    args: argparse.Namespace,
    total_rows: int,
    filtered_rows: int,
    scored_rows: int,
    counts: dict[float, int],
) -> None:
    summary = pd.DataFrame(
        [
            {
                "threshold_k": threshold,
                "n_selected": counts.get(threshold, 0),
                "fraction_of_scored": counts.get(threshold, 0) / scored_rows if scored_rows else np.nan,
                "output_csv": str(out_dir / f"monomers_Tg_ge_{threshold:g}K.csv"),
            }
            for threshold in args.thresholds
        ]
    )
    summary.insert(0, "input_csv", str(args.input_csv))
    summary["n_input_rows_seen"] = total_rows
    summary["n_rows_after_valid_unique_novel_filters"] = filtered_rows
    summary["n_scored"] = scored_rows
    summary["limit_rows"] = args.limit_rows
    summary["rdkit_validity_check"] = args.rdkit_validity_check
    summary.to_csv(out_dir / "threshold_filter_summary.csv", index=False)
    print(f"Saved summary to {out_dir / 'threshold_filter_summary.csv'}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Filter raw VAE-generated monomer candidates at Tg thresholds such as 500/600/700 K."
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=REPO_ROOT / "data/raw/Raw VAE-Generated Polymer-MonomerCandidateLibrary.csv",
    )
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "analysis_outputs/raw_vae_tg_thresholds")
    parser.add_argument("--smiles-col", default=None)
    parser.add_argument("--pred-col", default=None)
    parser.add_argument("--thresholds", type=float, nargs="+", default=[500.0, 600.0, 700.0])
    parser.add_argument("--valid-only", action="store_true", default=True)
    parser.add_argument("--all-validity-states", dest="valid_only", action="store_false")
    parser.add_argument("--rdkit-validity-check", action="store_true", default=True)
    parser.add_argument("--trust-validity-column", dest="rdkit_validity_check", action="store_false")
    parser.add_argument("--unique-only", action="store_true", default=True)
    parser.add_argument("--all-duplicate-states", dest="unique_only", action="store_false")
    parser.add_argument("--novel-only", action="store_true", default=True)
    parser.add_argument("--include-training-set", dest="novel_only", action="store_false")
    parser.add_argument(
        "--limit-rows",
        type=int,
        default=None,
        help="Only score/filter the first N rows after validity, uniqueness, and novelty filters.",
    )

    parser.add_argument("--ckpt-path", type=Path, default=None, help="Predictor checkpoint for scoring if no pred column exists.")
    parser.add_argument("--chunk-size", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--tmp-dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-chunks", type=int, default=None, help="Debug option for a small dry run.")
    parser.add_argument(
        "--subset-only",
        action="store_true",
        help="Only write the prefiltered subset; do not score or threshold by Tg.",
    )
    parser.add_argument("--subset-out-csv", type=Path, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.tmp_dir is not None:
        args.tmp_dir.mkdir(parents=True, exist_ok=True)

    columns = pd.read_csv(args.input_csv, nrows=0).columns.tolist()
    smiles_col = args.smiles_col or detect_column(columns, ["smiles", "PSMILES"])
    pred_col = args.pred_col or detect_column(columns, ["pred", "prediction", "pred_tg", "Tg_pred"], required=False)

    if args.subset_only:
        write_filtered_subset(args, smiles_col)
    elif pred_col is not None:
        filter_existing_predictions(args, pred_col, smiles_col)
    else:
        score_and_filter_chunks(args, smiles_col)

    if args.tmp_dir is not None and args.tmp_dir.exists():
        for path in args.tmp_dir.iterdir():
            if path.is_dir() and path.name.startswith("tmp"):
                shutil.rmtree(path, ignore_errors=True)


if __name__ == "__main__":
    main()
