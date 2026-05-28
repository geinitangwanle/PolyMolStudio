#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare Delta-ranking selection with random selection under a fixed MD budget.

The default analysis answers: if only k candidates can be sent to MD, does
choosing the k lowest-Delta candidates produce lower MD relative deviation than
choosing k candidates uniformly at random?

For small candidate sets the random baseline is exact: all n-choose-k
combinations are enumerated.
"""

import argparse
import itertools
import json
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_CSV = Path("demo/paper_candidate_new.csv")
DEFAULT_OUT_DIR = Path("demo/analysis_outputs/delta_ranking_vs_random_new")
DEFAULT_SUCCESS_THRESHOLD = 8.0


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


def summarize_random_means(random_means: np.ndarray, delta_mean: float) -> Dict[str, float]:
    return {
        "random_expected_mean_md_relative_deviation_percent": float(random_means.mean()),
        "delta_minus_random_expected_percent_point": float(delta_mean - random_means.mean()),
        "random_mean_distribution_std_percent_point": float(random_means.std(ddof=1)),
        "random_mean_distribution_p05": float(np.quantile(random_means, 0.05)),
        "random_mean_distribution_p50": float(np.quantile(random_means, 0.50)),
        "random_mean_distribution_p95": float(np.quantile(random_means, 0.95)),
        "delta_rank_percentile_among_random_means_lower_is_better": float(
            100.0 * np.mean(random_means <= delta_mean)
        ),
        "prob_random_mean_lower_or_equal_delta_rank": float(np.mean(random_means <= delta_mean)),
        "prob_random_mean_higher_or_equal_delta_rank": float(np.mean(random_means >= delta_mean)),
    }


def summarize_success_enrichment(
    random_success_counts: np.ndarray,
    delta_success_count: int,
    k: int,
) -> Dict[str, float]:
    delta_success_rate = delta_success_count / k
    random_expected_success_count = float(random_success_counts.mean())
    random_expected_success_rate = random_expected_success_count / k
    enrichment_factor = (
        delta_success_rate / random_expected_success_rate
        if random_expected_success_rate > 0
        else float("nan")
    )

    return {
        "delta_ranking_success_count": int(delta_success_count),
        "delta_ranking_success_rate": float(delta_success_rate),
        "random_expected_success_count": random_expected_success_count,
        "random_expected_success_rate": float(random_expected_success_rate),
        "enrichment_factor": float(enrichment_factor),
        "random_success_count_distribution_std": float(random_success_counts.std(ddof=1)),
        "random_success_count_distribution_p05": float(np.quantile(random_success_counts, 0.05)),
        "random_success_count_distribution_p50": float(np.quantile(random_success_counts, 0.50)),
        "random_success_count_distribution_p95": float(np.quantile(random_success_counts, 0.95)),
        "prob_random_success_count_greater_or_equal_delta_rank": float(
            np.mean(random_success_counts >= delta_success_count)
        ),
    }


def plot_random_distribution(random_means: np.ndarray, delta_mean: float, out_path: Path) -> None:
    plt.figure(figsize=(6.2, 4.2))
    plt.hist(random_means, bins=36, color="#8aa0b8", edgecolor="white", alpha=0.85)
    plt.axvline(delta_mean, color="#c0392b", linewidth=2.0, label="Delta-ranking top k")
    plt.xlabel("Mean MD relative deviation for selected k candidates / %")
    plt.ylabel("Number of random combinations")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def analyze(csv_path: Path, out_dir: Path, k: int, success_threshold: float) -> Dict[str, object]:
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    delta_col = find_column(df.columns.tolist(), ["Δ /K", "Delta /K", "delta"], contains="δ")
    md_dev_col = find_column(df.columns.tolist(), ["relative deviation %", "MD deviation %"], contains="deviation")

    df = df.copy()
    df[delta_col] = pd.to_numeric(df[delta_col], errors="coerce")
    df[md_dev_col] = pd.to_numeric(df[md_dev_col], errors="coerce")
    df = df.dropna(subset=[delta_col, md_dev_col]).reset_index(drop=True)

    if k <= 0 or k > len(df):
        raise ValueError(f"k must be between 1 and the number of valid candidates ({len(df)}). Got k={k}.")

    ranked = df.sort_values(delta_col, ascending=True).reset_index(drop=True)
    delta_top = ranked.head(k).copy()
    delta_top["successful_candidate"] = delta_top[md_dev_col] <= success_threshold
    delta_mean = float(delta_top[md_dev_col].mean())
    delta_success_count = int(delta_top["successful_candidate"].sum())

    values = df[md_dev_col].to_numpy(dtype=float)
    successes = values <= success_threshold
    random_means = np.fromiter(
        (values[list(combo)].mean() for combo in itertools.combinations(range(len(values)), k)),
        dtype=float,
    )
    random_success_counts = np.fromiter(
        (successes[list(combo)].sum() for combo in itertools.combinations(range(len(values)), k)),
        dtype=float,
    )

    summary: Dict[str, object] = {
        "source": str(csv_path),
        "n_candidates": int(len(df)),
        "k": int(k),
        "columns": {"delta": delta_col, "md_relative_deviation": md_dev_col},
        "successful_candidate_definition": f"{md_dev_col} <= {success_threshold:g}%",
        "success_threshold_md_relative_deviation_percent": float(success_threshold),
        "n_successful_candidates_overall": int(successes.sum()),
        "overall_success_rate": float(successes.mean()),
        "n_random_combinations": int(random_means.size),
        "delta_ranking_mean_md_relative_deviation_percent": delta_mean,
    }
    summary.update(summarize_random_means(random_means, delta_mean))
    summary.update(summarize_success_enrichment(random_success_counts, delta_success_count, k))

    out_dir.mkdir(parents=True, exist_ok=True)
    delta_top.to_csv(out_dir / f"delta_top{k}_candidates.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(
        {f"random_top{k}_mean_md_relative_deviation_percent": random_means}
    ).to_csv(out_dir / f"all_random_combination_means_k{k}.csv", index=False)
    pd.DataFrame(
        {f"random_top{k}_success_count_at_{success_threshold:g}pct": random_success_counts}
    ).to_csv(out_dir / f"all_random_combination_success_counts_k{k}.csv", index=False)
    pd.DataFrame([summary]).to_csv(out_dir / "delta_ranking_vs_random_summary.csv", index=False)
    (out_dir / "delta_ranking_vs_random_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    plot_random_distribution(random_means, delta_mean, out_dir / f"delta_ranking_vs_random_k{k}.png")

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare Delta-ranking top-k selection against random top-k selection.")
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV, help="Candidate CSV with Delta and MD deviation columns.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR, help="Directory for CSV/JSON/PNG outputs.")
    parser.add_argument("--k", type=int, default=7, help="MD budget: number of candidates selected.")
    parser.add_argument(
        "--success-threshold",
        type=float,
        default=DEFAULT_SUCCESS_THRESHOLD,
        help="Successful candidate threshold for MD relative deviation in percent.",
    )
    args = parser.parse_args()

    summary = analyze(args.csv, args.out_dir, args.k, args.success_threshold)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
