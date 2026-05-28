#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run train/train_predictor.py over multiple data split seeds and summarize metrics.

Example:
  python scripts/repeated_split_train_predictor.py \
    --split_seeds 1,2,3,4,5,6,7,8,9,10 \
    --model_seed_mode same_as_split \
    --run_prefix crossattn_repeated \
    --summary_dir analysis_outputs/repeated_split_crossattn \
    -- \
    --data_path data/processed/graphs_tg/manifest.csv \
    --root_dir data/processed/graphs_tg \
    --batch_size 32 --epochs 50 --lr 1e-3 \
    --use_polybert --polybert-dir ./polybert \
    --polybert_lr 1e-5 --seq_max_length 256 \
    --cross_attn_heads 4 --cross_attn_dim 64 \
    --device auto
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parent.parent
TRAIN_SCRIPT = REPO_ROOT / "train" / "train_predictor.py"
TEST_RE = re.compile(
    r"\[TEST(?:-BEST ep(?P<epoch>\d+))?\]\s+MAE\(K\)\s+"
    r"(?P<mae>[-+]?\d*\.?\d+)\s+\|\s+RMSE\(K\)\s+"
    r"(?P<rmse>[-+]?\d*\.?\d+)"
)


def parse_seed_list(text: str) -> list[int]:
    seeds: list[int] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_s, end_s = part.split("-", 1)
            start, end = int(start_s), int(end_s)
            step = 1 if end >= start else -1
            seeds.extend(range(start, end + step, step))
        else:
            seeds.append(int(part))
    if not seeds:
        raise argparse.ArgumentTypeError("At least one split seed is required.")
    return seeds


def has_arg(extra_args: Iterable[str], name: str) -> bool:
    prefix = name + "="
    return any(arg == name or arg.startswith(prefix) for arg in extra_args)


def append_default_arg(extra_args: list[str], name: str, value: str | Path) -> None:
    if not has_arg(extra_args, name):
        extra_args.extend([name, str(value)])


def get_arg_value(extra_args: list[str], name: str) -> str | None:
    prefix = name + "="
    for idx, arg in enumerate(extra_args):
        if arg.startswith(prefix):
            return arg[len(prefix):]
        if arg == name and idx + 1 < len(extra_args):
            return extra_args[idx + 1]
    return None


def model_seed_for_split(split_seed: int, args: argparse.Namespace) -> int:
    if args.model_seed_mode == "same_as_split":
        return split_seed
    if args.model_seed_mode == "fixed":
        return args.model_seed
    raise ValueError(f"Unknown model_seed_mode: {args.model_seed_mode}")


def latest_log(log_dir: Path, run_name: str, split_seed: int, model_seed: int) -> Path | None:
    pattern = f"{run_name}_train_*_split{split_seed}_model{model_seed}.log"
    matches = sorted(log_dir.glob(pattern), key=lambda p: p.stat().st_mtime)
    return matches[-1] if matches else None


def parse_test_metrics(log_path: Path) -> dict[str, float | int | str]:
    text = log_path.read_text(encoding="utf-8", errors="replace")
    matches = list(TEST_RE.finditer(text))
    if not matches:
        raise RuntimeError(f"No final test metric line found in {log_path}")
    match = matches[-1]
    epoch_s = match.group("epoch")
    return {
        "mae_k": float(match.group("mae")),
        "rmse_k": float(match.group("rmse")),
        "best_epoch": int(epoch_s) if epoch_s is not None else "",
    }


def sample_std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return math.sqrt(sum((x - mean) ** 2 for x in values) / (len(values) - 1))


def t_critical_975(n: int) -> float:
    # Two-sided 95% CI critical values for df=n-1. Falls back to normal.
    table = {
        1: 12.706,
        2: 4.303,
        3: 3.182,
        4: 2.776,
        5: 2.571,
        6: 2.447,
        7: 2.365,
        8: 2.306,
        9: 2.262,
        10: 2.228,
        11: 2.201,
        12: 2.179,
        13: 2.160,
        14: 2.145,
        15: 2.131,
        16: 2.120,
        17: 2.110,
        18: 2.101,
        19: 2.093,
        20: 2.086,
        21: 2.080,
        22: 2.074,
        23: 2.069,
        24: 2.064,
        25: 2.060,
        26: 2.056,
        27: 2.052,
        28: 2.048,
        29: 2.045,
        30: 2.042,
    }
    df = max(n - 1, 1)
    return table.get(df, 1.96)


def summarize_metric(values: list[float]) -> dict[str, float | int]:
    n = len(values)
    mean = sum(values) / n
    std = sample_std(values)
    sem = std / math.sqrt(n) if n else float("nan")
    ci95 = t_critical_975(n) * sem if n > 1 else 0.0
    return {
        "n": n,
        "mean": mean,
        "std": std,
        "sem": sem,
        "ci95_half_width": ci95,
        "ci95_low": mean - ci95,
        "ci95_high": mean + ci95,
    }


def write_rows_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "run_name",
        "split_seed",
        "model_seed",
        "mae_k",
        "rmse_k",
        "best_epoch",
        "log_path",
        "checkpoint_dir",
        "status",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Repeated train/val/test splits for train_predictor.py with metric summaries."
    )
    parser.add_argument(
        "--split_seeds",
        type=parse_seed_list,
        default=parse_seed_list("1,2,3,4,5,6,7,8,9,10"),
        help="Comma-separated seeds and/or ranges, e.g. '1,2,3' or '1-10'.",
    )
    parser.add_argument(
        "--model_seed_mode",
        choices=["same_as_split", "fixed"],
        default="same_as_split",
        help="Use the same seed for model initialization as split_seed, or a fixed model seed.",
    )
    parser.add_argument("--model_seed", type=int, default=42, help="Used only with --model_seed_mode fixed.")
    parser.add_argument("--run_prefix", default="repeated_split", help="Prefix for per-run names.")
    parser.add_argument(
        "--summary_dir",
        type=Path,
        default=REPO_ROOT / "analysis_outputs" / "repeated_split_predictor",
        help="Directory for repeated_split_results.csv and repeated_split_summary.json.",
    )
    parser.add_argument(
        "--log_dir",
        type=Path,
        default=None,
        help="Override log directory passed to train_predictor.py. Defaults to summary_dir/logs.",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=Path,
        default=None,
        help="Override checkpoint directory passed to train_predictor.py. Defaults to summary_dir/checkpoints.",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable used to launch train/train_predictor.py.",
    )
    parser.add_argument(
        "--keep_going",
        action="store_true",
        help="Continue remaining split seeds if one run fails.",
    )
    parser.add_argument(
        "train_args",
        nargs=argparse.REMAINDER,
        help="Arguments forwarded to train/train_predictor.py. Put them after '--'.",
    )
    args = parser.parse_args()

    train_args = list(args.train_args)
    if train_args and train_args[0] == "--":
        train_args = train_args[1:]
    if not train_args:
        raise SystemExit("No train_predictor.py arguments were provided. Add them after '--'.")

    summary_dir = args.summary_dir.resolve()
    train_log_dir = get_arg_value(train_args, "--log_dir")
    train_checkpoint_dir = get_arg_value(train_args, "--checkpoint_dir")
    log_dir = Path(args.log_dir or train_log_dir or summary_dir / "logs").resolve()
    checkpoint_dir = Path(args.checkpoint_dir or train_checkpoint_dir or summary_dir / "checkpoints").resolve()
    summary_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    append_default_arg(train_args, "--log_dir", log_dir)
    append_default_arg(train_args, "--checkpoint_dir", checkpoint_dir)

    rows: list[dict[str, object]] = []
    started_at = datetime.now().isoformat(timespec="seconds")

    for split_seed in args.split_seeds:
        model_seed = model_seed_for_split(split_seed, args)
        run_name = f"{args.run_prefix}_split{split_seed}_model{model_seed}"
        cmd = [
            args.python,
            str(TRAIN_SCRIPT),
            *train_args,
            "--split_seed",
            str(split_seed),
            "--model_seed",
            str(model_seed),
            "--run_name",
            run_name,
        ]
        print(f"\n=== Running {run_name} ===", flush=True)
        print(" ".join(cmd), flush=True)

        row: dict[str, object] = {
            "run_name": run_name,
            "split_seed": split_seed,
            "model_seed": model_seed,
            "status": "ok",
        }
        proc = subprocess.run(cmd, cwd=REPO_ROOT)
        if proc.returncode != 0:
            row["status"] = f"failed_returncode_{proc.returncode}"
            rows.append(row)
            write_rows_csv(summary_dir / "repeated_split_results.csv", rows)
            if not args.keep_going:
                raise SystemExit(proc.returncode)
            continue

        log_path = latest_log(log_dir, run_name, split_seed, model_seed)
        if log_path is None:
            row["status"] = "failed_no_log"
            rows.append(row)
            write_rows_csv(summary_dir / "repeated_split_results.csv", rows)
            if not args.keep_going:
                raise SystemExit(f"Could not find log for {run_name} in {log_dir}")
            continue

        try:
            metrics = parse_test_metrics(log_path)
            row.update(metrics)
            row["log_path"] = str(log_path)
            row["checkpoint_dir"] = str(checkpoint_dir / log_path.stem)
        except Exception as exc:  # noqa: BLE001
            row["status"] = f"failed_parse_metrics: {exc}"
            if not args.keep_going:
                rows.append(row)
                write_rows_csv(summary_dir / "repeated_split_results.csv", rows)
                raise

        rows.append(row)
        write_rows_csv(summary_dir / "repeated_split_results.csv", rows)

    ok_rows = [r for r in rows if r.get("status") == "ok"]
    if not ok_rows:
        raise SystemExit("No successful runs to summarize.")

    mae_values = [float(r["mae_k"]) for r in ok_rows]
    rmse_values = [float(r["rmse_k"]) for r in ok_rows]
    summary = {
        "started_at": started_at,
        "finished_at": datetime.now().isoformat(timespec="seconds"),
        "run_prefix": args.run_prefix,
        "split_seeds": args.split_seeds,
        "model_seed_mode": args.model_seed_mode,
        "model_seed": args.model_seed,
        "n_successful": len(ok_rows),
        "n_failed": len(rows) - len(ok_rows),
        "mae_k": summarize_metric(mae_values),
        "rmse_k": summarize_metric(rmse_values),
        "results_csv": str(summary_dir / "repeated_split_results.csv"),
        "log_dir": str(log_dir),
        "checkpoint_dir": str(checkpoint_dir),
    }
    summary_path = summary_dir / "repeated_split_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n=== Summary ===")
    print(f"Successful runs: {summary['n_successful']} / {len(rows)}")
    print(
        "MAE(K): "
        f"{summary['mae_k']['mean']:.3f} ± {summary['mae_k']['ci95_half_width']:.3f} "
        f"(95% CI), std={summary['mae_k']['std']:.3f}"
    )
    print(
        "RMSE(K): "
        f"{summary['rmse_k']['mean']:.3f} ± {summary['rmse_k']['ci95_half_width']:.3f} "
        f"(95% CI), std={summary['rmse_k']['std']:.3f}"
    )
    print(f"Results CSV: {summary_dir / 'repeated_split_results.csv'}")
    print(f"Summary JSON: {summary_path}")


if __name__ == "__main__":
    main()
