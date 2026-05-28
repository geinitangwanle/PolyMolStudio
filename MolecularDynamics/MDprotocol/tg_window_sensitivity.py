#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Window perturbation analysis for Tg uncertainty from MD density-temperature data.

Input:
  TgDensity/Tg_density_*.csv with columns:
    T(K), density(g/cm^3)

Method:
  1) Fit two straight lines to low-T and high-T density data.
     Baseline follows MD_Tg_1.py:
       high-T: first 1/3 of the cooling curve
       low-T:  last 1/3 of the cooling curve
  2) Perturb the equivalent low/high cutoffs by -20, -10, 0, +10, +20 K.
  3) Refit Tg for every valid cutoff pair.
  4) Estimate:
       sigma_bootstrap = std of baseline Tg across the 21 MD repeats
       sigma_window    = std of all window-induced Tg shifts
       sigma_MD        = sqrt(sigma_bootstrap^2 + sigma_window^2)

Outputs are written to TgDensity/window_sensitivity_results/.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path

_MPL_CACHE_DIR = Path(__file__).resolve().parent / ".matplotlib-cache"
_MPL_CACHE_DIR.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CACHE_DIR))
os.environ.setdefault("XDG_CACHE_HOME", str(_MPL_CACHE_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DEFAULT_DELTAS = (-20.0, -10.0, 0.0, 10.0, 20.0)
MIN_POINTS_PER_SEGMENT = 2


@dataclass(frozen=True)
class FitResult:
    run_id: int
    low_cutoff: float
    high_cutoff: float
    n_low: int
    n_high: int
    a_low: float
    b_low: float
    a_high: float
    b_high: float
    tg_k: float


def run_id_from_path(path: Path) -> int:
    match = re.search(r"Tg_density_(\d+)\.csv$", path.name)
    if not match:
        raise ValueError(f"Cannot parse run id from {path.name}")
    return int(match.group(1))


def read_density_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.genfromtxt(path, delimiter=",", names=True)
    if data.size == 0:
        raise ValueError(f"No data found in {path}")

    names = data.dtype.names or ()
    if len(names) < 2:
        raise ValueError(f"Expected at least two columns in {path}")

    t = np.asarray(data[names[0]], dtype=float)
    rho = np.asarray(data[names[1]], dtype=float)
    good = np.isfinite(t) & np.isfinite(rho)
    if good.sum() < 2 * MIN_POINTS_PER_SEGMENT:
        raise ValueError(f"Too few finite points in {path}")

    order = np.argsort(t)
    return t[good][order], rho[good][order]


def line_fit(t: np.ndarray, rho: np.ndarray) -> tuple[float, float]:
    a, b = np.polyfit(t, rho, deg=1)
    return float(a), float(b)


def fit_tg(
    run_id: int,
    t: np.ndarray,
    rho: np.ndarray,
    low_cutoff: float,
    high_cutoff: float,
) -> FitResult | None:
    low_mask = t < low_cutoff
    high_mask = t > high_cutoff
    n_low = int(low_mask.sum())
    n_high = int(high_mask.sum())
    if n_low < MIN_POINTS_PER_SEGMENT or n_high < MIN_POINTS_PER_SEGMENT:
        return None

    a_low, b_low = line_fit(t[low_mask], rho[low_mask])
    a_high, b_high = line_fit(t[high_mask], rho[high_mask])
    denom = a_high - a_low
    if abs(denom) < 1.0e-12:
        return None

    tg = (b_low - b_high) / denom
    if not math.isfinite(tg):
        return None
    if tg < float(np.min(t)) or tg > float(np.max(t)):
        return None

    return FitResult(
        run_id=run_id,
        low_cutoff=low_cutoff,
        high_cutoff=high_cutoff,
        n_low=n_low,
        n_high=n_high,
        a_low=a_low,
        b_low=b_low,
        a_high=a_high,
        b_high=b_high,
        tg_k=float(tg),
    )


def fit_tg_from_masks(
    run_id: int,
    t: np.ndarray,
    rho: np.ndarray,
    low_mask: np.ndarray,
    high_mask: np.ndarray,
    low_cutoff: float,
    high_cutoff: float,
) -> FitResult | None:
    n_low = int(low_mask.sum())
    n_high = int(high_mask.sum())
    if n_low < MIN_POINTS_PER_SEGMENT or n_high < MIN_POINTS_PER_SEGMENT:
        return None

    a_low, b_low = line_fit(t[low_mask], rho[low_mask])
    a_high, b_high = line_fit(t[high_mask], rho[high_mask])
    denom = a_high - a_low
    if abs(denom) < 1.0e-12:
        return None

    tg = (b_low - b_high) / denom
    if not math.isfinite(tg):
        return None
    if tg < float(np.min(t)) or tg > float(np.max(t)):
        return None

    return FitResult(
        run_id=run_id,
        low_cutoff=low_cutoff,
        high_cutoff=high_cutoff,
        n_low=n_low,
        n_high=n_high,
        a_low=a_low,
        b_low=b_low,
        a_high=a_high,
        b_high=b_high,
        tg_k=float(tg),
    )


def baseline_fit_from_thirds(
    run_id: int, t: np.ndarray, rho: np.ndarray
) -> FitResult | None:
    n = len(t)
    high_count = n // 3
    low_start = 2 * n // 3
    if high_count < MIN_POINTS_PER_SEGMENT or n - low_start < MIN_POINTS_PER_SEGMENT:
        return None

    low_mask = np.zeros(n, dtype=bool)
    high_mask = np.zeros(n, dtype=bool)
    low_mask[: n - low_start] = True
    high_mask[n - high_count :] = True

    # Equivalent strict cutoffs, used as the center for perturbation windows.
    low_cutoff = (t[n - low_start - 1] + t[n - low_start]) / 2.0
    high_cutoff = (t[n - high_count - 1] + t[n - high_count]) / 2.0

    return fit_tg_from_masks(
        run_id,
        t,
        rho,
        low_mask,
        high_mask,
        float(low_cutoff),
        float(high_cutoff),
    )


def std(values: list[float] | np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    if values.size < 2:
        return float("nan")
    return float(np.std(values, ddof=1))


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_window_distribution(tg_shifts: list[float], out_path: Path) -> None:
    plt.figure(figsize=(6.2, 4.3), dpi=180)
    plt.hist(tg_shifts, bins=24, color="#4C78A8", edgecolor="white")
    plt.axvline(0.0, color="black", linestyle=":", linewidth=1.5)
    plt.xlabel("Window-induced Tg shift / K")
    plt.ylabel("Count")
    plt.title("Fitting-window sensitivity")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_run_summary(summary_rows: list[dict], out_path: Path) -> None:
    rows = sorted(summary_rows, key=lambda x: x["run_id"])
    x = [row["run_id"] for row in rows]
    y = [row["baseline_tg_k"] for row in rows]
    yerr = [row["sigma_window_run_k"] for row in rows]

    plt.figure(figsize=(7.2, 4.4), dpi=180)
    plt.errorbar(
        x,
        y,
        yerr=yerr,
        fmt="o",
        color="#222222",
        ecolor="#E45756",
        elinewidth=1.2,
        capsize=3,
    )
    plt.xlabel("MD repeat")
    plt.ylabel("Tg / K")
    plt.title("Baseline Tg with per-run window sensitivity")
    plt.xticks(x)
    plt.grid(alpha=0.25, linestyle=":")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run fitting-window perturbation analysis for MD Tg data."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "TgDensity",
        help="Directory containing Tg_density_*.csv files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Default: INPUT_DIR/window_sensitivity_results",
    )
    parser.add_argument(
        "--deltas",
        type=float,
        nargs="+",
        default=list(DEFAULT_DELTAS),
        help="Boundary perturbations around the 1/3 baseline cutoffs in K, e.g. -20 -10 0 10 20.",
    )
    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir or input_dir / "window_sensitivity_results"
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_paths = sorted(input_dir.glob("Tg_density_*.csv"), key=run_id_from_path)
    if not csv_paths:
        raise FileNotFoundError(f"No Tg_density_*.csv files found in {input_dir}")

    all_results: list[FitResult] = []
    baseline_by_run: dict[int, FitResult] = {}
    invalid_windows: list[dict] = []

    for csv_path in csv_paths:
        run_id = run_id_from_path(csv_path)
        t, rho = read_density_csv(csv_path)

        baseline = baseline_fit_from_thirds(run_id, t, rho)
        if baseline is None:
            raise RuntimeError(f"Baseline fit failed for {csv_path}")
        baseline_by_run[run_id] = baseline

        low_cutoffs = [baseline.low_cutoff + delta for delta in args.deltas]
        high_cutoffs = [baseline.high_cutoff + delta for delta in args.deltas]
        for low_cutoff in low_cutoffs:
            for high_cutoff in high_cutoffs:
                result = fit_tg(run_id, t, rho, low_cutoff, high_cutoff)
                if result is None:
                    invalid_windows.append(
                        {
                            "run_id": run_id,
                            "low_cutoff_k": low_cutoff,
                            "high_cutoff_k": high_cutoff,
                        }
                    )
                    continue
                all_results.append(result)

    result_rows = [
        {
            "run_id": r.run_id,
            "low_cutoff_k": f"{r.low_cutoff:.1f}",
            "high_cutoff_k": f"{r.high_cutoff:.1f}",
            "n_low": r.n_low,
            "n_high": r.n_high,
            "a_low": f"{r.a_low:.12g}",
            "b_low": f"{r.b_low:.12g}",
            "a_high": f"{r.a_high:.12g}",
            "b_high": f"{r.b_high:.12g}",
            "tg_k": f"{r.tg_k:.6f}",
        }
        for r in all_results
    ]
    write_csv(
        output_dir / "window_sensitivity_results.csv",
        result_rows,
        [
            "run_id",
            "low_cutoff_k",
            "high_cutoff_k",
            "n_low",
            "n_high",
            "a_low",
            "b_low",
            "a_high",
            "b_high",
            "tg_k",
        ],
    )

    summary_rows: list[dict] = []
    tg_shifts: list[float] = []
    for run_id in sorted(baseline_by_run):
        baseline_tg = baseline_by_run[run_id].tg_k
        run_tgs = [r.tg_k for r in all_results if r.run_id == run_id]
        run_shifts = [tg - baseline_tg for tg in run_tgs]
        tg_shifts.extend(run_shifts)
        summary_rows.append(
            {
                "run_id": run_id,
                "baseline_tg_k": baseline_tg,
                "window_mean_tg_k": float(np.mean(run_tgs)),
                "sigma_window_run_k": std(run_shifts),
                "window_min_tg_k": float(np.min(run_tgs)),
                "window_max_tg_k": float(np.max(run_tgs)),
                "valid_windows": len(run_tgs),
            }
        )

    write_csv(
        output_dir / "per_run_window_summary.csv",
        [
            {
                "run_id": row["run_id"],
                "baseline_tg_k": f"{row['baseline_tg_k']:.6f}",
                "window_mean_tg_k": f"{row['window_mean_tg_k']:.6f}",
                "sigma_window_run_k": f"{row['sigma_window_run_k']:.6f}",
                "window_min_tg_k": f"{row['window_min_tg_k']:.6f}",
                "window_max_tg_k": f"{row['window_max_tg_k']:.6f}",
                "valid_windows": row["valid_windows"],
            }
            for row in summary_rows
        ],
        [
            "run_id",
            "baseline_tg_k",
            "window_mean_tg_k",
            "sigma_window_run_k",
            "window_min_tg_k",
            "window_max_tg_k",
            "valid_windows",
        ],
    )

    baseline_tgs = [baseline_by_run[run_id].tg_k for run_id in sorted(baseline_by_run)]
    sigma_bootstrap = std(baseline_tgs)
    sigma_window = std(tg_shifts)
    sigma_md = float(math.sqrt(sigma_bootstrap**2 + sigma_window**2))

    overall = {
        "input_dir": str(input_dir),
        "n_md_repeats": len(baseline_tgs),
        "baseline_low_window": "last 1/3 of cooling curve",
        "baseline_high_window": "first 1/3 of cooling curve",
        "perturbation_deltas_k": list(args.deltas),
        "mean_baseline_tg_k": float(np.mean(baseline_tgs)),
        "sigma_bootstrap_k": sigma_bootstrap,
        "sigma_window_k": sigma_window,
        "sigma_md_k": sigma_md,
        "min_baseline_tg_k": float(np.min(baseline_tgs)),
        "max_baseline_tg_k": float(np.max(baseline_tgs)),
        "n_valid_window_fits": len(all_results),
        "n_invalid_window_fits": len(invalid_windows),
        "invalid_windows": invalid_windows,
    }
    with (output_dir / "overall_uncertainty_summary.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(overall, f, indent=2)

    plot_window_distribution(
        tg_shifts, output_dir / "window_tg_shift_distribution.png"
    )
    plot_run_summary(summary_rows, output_dir / "baseline_tg_by_run.png")

    print("Window sensitivity analysis complete.")
    print(f"  MD repeats: {len(baseline_tgs)}")
    print(f"  Mean baseline Tg: {overall['mean_baseline_tg_k']:.2f} K")
    print(f"  sigma_bootstrap: {sigma_bootstrap:.2f} K")
    print(f"  sigma_window: {sigma_window:.2f} K")
    print(f"  sigma_MD: {sigma_md:.2f} K")
    print(f"  Results: {output_dir}")


if __name__ == "__main__":
    main()
