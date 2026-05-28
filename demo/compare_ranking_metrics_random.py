#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare several ranking metrics against exact random top-k selection.

This extends the Delta-ranking analysis to uncertainty metrics such as deep
ensemble standard deviation. The top-k candidates are selected by ascending
metric values, then evaluated by MD relative deviation and successful-candidate
enrichment.
"""

import argparse
import itertools
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


DEFAULT_CSV = Path("demo/analysis_outputs/uncertainty_delta_relationship_new.csv")
DEFAULT_OUT_DIR = Path("demo/analysis_outputs/ranking_metric_vs_random_new_success8")
DEFAULT_K = 7
DEFAULT_SUCCESS_THRESHOLD = 8.0
DEFAULT_METRICS = [
    "Δ /K",
    "graph_model_uncertainty_k",
    "multi_model_uncertainty_k",
]


def find_column(columns: List[str], candidates: List[str], contains: Optional[str] = None) -> str:
    normalized = {col.strip().lower(): col for col in columns}
    for name in candidates:
        key = name.strip().lower()
        if key in normalized:
            return normalized[key]
    if contains is not None:
        token = contains.lower()
        matches = [col for col in columns if token in col.strip().lower()]
        if matches:
            return matches[0]
    raise ValueError(f"Could not find a matching column. Available columns: {columns}")


def safe_name(name: str) -> str:
    return (
        name.strip()
        .replace(" ", "_")
        .replace("/", "")
        .replace("%", "pct")
        .replace("Δ", "delta")
        .replace("__", "_")
        .strip("_")
        .lower()
    )


def random_baseline(values: np.ndarray, successes: np.ndarray, k: int) -> Dict[str, np.ndarray]:
    combos = itertools.combinations(range(len(values)), k)
    means = []
    success_counts = []
    for combo in combos:
        idx = list(combo)
        means.append(values[idx].mean())
        success_counts.append(successes[idx].sum())
    return {
        "means": np.asarray(means, dtype=float),
        "success_counts": np.asarray(success_counts, dtype=float),
    }


def summarize_metric(
    df: pd.DataFrame,
    metric_col: str,
    md_dev_col: str,
    test_id_col: str,
    random_means: np.ndarray,
    random_success_counts: np.ndarray,
    k: int,
    success_threshold: float,
) -> Dict[str, object]:
    ranked = df.sort_values(metric_col, ascending=True).reset_index(drop=True)
    top = ranked.head(k).copy()
    top["successful_candidate"] = top[md_dev_col] <= success_threshold

    mean_md_dev = float(top[md_dev_col].mean())
    success_count = int(top["successful_candidate"].sum())
    success_rate = success_count / k
    random_expected_success_count = float(random_success_counts.mean())
    random_expected_success_rate = random_expected_success_count / k
    enrichment_factor = (
        success_rate / random_expected_success_rate
        if random_expected_success_rate > 0
        else float("nan")
    )

    return {
        "ranking_metric": metric_col,
        "ranking_order": "ascending",
        "top_k_test_ids": ",".join(str(x) for x in top[test_id_col].tolist()),
        "top_k_mean_ranking_metric": float(top[metric_col].mean()),
        "top_k_mean_md_relative_deviation_percent": mean_md_dev,
        "top_k_success_count": success_count,
        "top_k_success_rate": float(success_rate),
        "enrichment_factor": float(enrichment_factor),
        "top_k_mean_md_deviation_percentile_among_random_lower_is_better": float(
            100.0 * np.mean(random_means <= mean_md_dev)
        ),
        "prob_random_mean_md_deviation_lower_or_equal_top_k": float(np.mean(random_means <= mean_md_dev)),
        "prob_random_success_count_greater_or_equal_top_k": float(
            np.mean(random_success_counts >= success_count)
        ),
    }


def analyze(
    csv_path: Path,
    out_dir: Path,
    metrics: List[str],
    k: int,
    success_threshold: float,
) -> Dict[str, object]:
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    md_dev_col = find_column(df.columns.tolist(), ["relative deviation %", "MD deviation %"], contains="deviation")
    test_id_col = find_column(df.columns.tolist(), ["Test id", "id"], contains="id")

    needed_cols = [test_id_col, md_dev_col] + metrics
    missing = [col for col in needed_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Available columns: {df.columns.tolist()}")

    df = df.copy()
    for col in [md_dev_col] + metrics:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=[md_dev_col] + metrics).reset_index(drop=True)

    if k <= 0 or k > len(df):
        raise ValueError(f"k must be between 1 and the number of valid candidates ({len(df)}). Got k={k}.")

    values = df[md_dev_col].to_numpy(dtype=float)
    successes = values <= success_threshold
    baseline = random_baseline(values, successes, k)
    random_means = baseline["means"]
    random_success_counts = baseline["success_counts"]

    common_summary: Dict[str, object] = {
        "source": str(csv_path),
        "n_candidates": int(len(df)),
        "k": int(k),
        "md_relative_deviation_column": md_dev_col,
        "successful_candidate_definition": f"{md_dev_col} <= {success_threshold:g}%",
        "success_threshold_md_relative_deviation_percent": float(success_threshold),
        "n_successful_candidates_overall": int(successes.sum()),
        "overall_success_rate": float(successes.mean()),
        "n_random_combinations": int(random_means.size),
        "random_expected_mean_md_relative_deviation_percent": float(random_means.mean()),
        "random_expected_success_count": float(random_success_counts.mean()),
        "random_expected_success_rate": float(random_success_counts.mean() / k),
    }

    rows = []
    out_dir.mkdir(parents=True, exist_ok=True)
    for metric_col in metrics:
        row = summarize_metric(
            df=df,
            metric_col=metric_col,
            md_dev_col=md_dev_col,
            test_id_col=test_id_col,
            random_means=random_means,
            random_success_counts=random_success_counts,
            k=k,
            success_threshold=success_threshold,
        )
        rows.append(row)

        ranked = df.sort_values(metric_col, ascending=True).head(k).copy()
        ranked["successful_candidate"] = ranked[md_dev_col] <= success_threshold
        ranked.to_csv(out_dir / f"top{k}_{safe_name(metric_col)}.csv", index=False, encoding="utf-8-sig")

    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(out_dir / "ranking_metric_vs_random_summary.csv", index=False)

    summary = {**common_summary, "metric_summary": rows}
    (out_dir / "ranking_metric_vs_random_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    pd.DataFrame({f"random_top{k}_mean_md_relative_deviation_percent": random_means}).to_csv(
        out_dir / f"all_random_combination_means_k{k}.csv",
        index=False,
    )
    pd.DataFrame({f"random_top{k}_success_count_at_{success_threshold:g}pct": random_success_counts}).to_csv(
        out_dir / f"all_random_combination_success_counts_k{k}.csv",
        index=False,
    )

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare ranking metrics against exact random top-k selection.")
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV, help="CSV containing ranking metrics and MD deviation.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR, help="Output directory.")
    parser.add_argument("--k", type=int, default=DEFAULT_K, help="MD budget: number of candidates selected.")
    parser.add_argument(
        "--success-threshold",
        type=float,
        default=DEFAULT_SUCCESS_THRESHOLD,
        help="Successful candidate threshold for MD relative deviation in percent.",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=DEFAULT_METRICS,
        help="Ranking metric columns. Smaller values are selected first.",
    )
    args = parser.parse_args()

    summary = analyze(args.csv, args.out_dir, args.metrics, args.k, args.success_threshold)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
