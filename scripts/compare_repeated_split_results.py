#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Paired comparison for two repeated_split_results.csv files."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


def read_results(path: Path) -> dict[int, dict[str, float]]:
    rows: dict[int, dict[str, float]] = {}
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("status") != "ok":
                continue
            split_seed = int(row["split_seed"])
            rows[split_seed] = {
                "mae_k": float(row["mae_k"]),
                "rmse_k": float(row["rmse_k"]),
            }
    return rows


def sample_std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return math.sqrt(sum((x - mean) ** 2 for x in values) / (len(values) - 1))


def t_critical_975(n: int) -> float:
    table = {
        1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
        6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
        11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131,
        16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
        21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060,
        26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042,
    }
    return table.get(max(n - 1, 1), 1.96)


def paired_summary(values: list[float]) -> dict[str, float | int]:
    n = len(values)
    mean = sum(values) / n
    std = sample_std(values)
    sem = std / math.sqrt(n) if n else float("nan")
    ci95 = t_critical_975(n) * sem if n > 1 else 0.0
    t_stat = mean / sem if sem > 0 else float("inf")
    return {
        "n": n,
        "mean": mean,
        "std": std,
        "sem": sem,
        "ci95_half_width": ci95,
        "ci95_low": mean - ci95,
        "ci95_high": mean + ci95,
        "paired_t_statistic": t_stat,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare repeated split results by matched split_seed.")
    parser.add_argument("--baseline_csv", type=Path, required=True, help="CSV for the baseline model.")
    parser.add_argument("--candidate_csv", type=Path, required=True, help="CSV for the candidate model.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("analysis_outputs/repeated_split_comparison.json"),
        help="JSON output path.",
    )
    args = parser.parse_args()

    baseline = read_results(args.baseline_csv)
    candidate = read_results(args.candidate_csv)
    split_seeds = sorted(set(baseline) & set(candidate))
    if not split_seeds:
        raise SystemExit("No matched split_seed values found between the two CSV files.")

    # Positive delta means candidate error is lower than baseline error.
    delta_mae = [baseline[s]["mae_k"] - candidate[s]["mae_k"] for s in split_seeds]
    delta_rmse = [baseline[s]["rmse_k"] - candidate[s]["rmse_k"] for s in split_seeds]
    result = {
        "baseline_csv": str(args.baseline_csv),
        "candidate_csv": str(args.candidate_csv),
        "matched_split_seeds": split_seeds,
        "delta_definition": "baseline - candidate; positive means candidate is better",
        "delta_mae_k": paired_summary(delta_mae),
        "delta_rmse_k": paired_summary(delta_rmse),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Matched split seeds: {len(split_seeds)}")
    print(
        "Delta MAE(K): "
        f"{result['delta_mae_k']['mean']:.3f} ± {result['delta_mae_k']['ci95_half_width']:.3f} "
        "(95% CI; positive means candidate better)"
    )
    print(
        "Delta RMSE(K): "
        f"{result['delta_rmse_k']['mean']:.3f} ± {result['delta_rmse_k']['ci95_half_width']:.3f} "
        "(95% CI; positive means candidate better)"
    )
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
