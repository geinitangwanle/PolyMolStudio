#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Single-form nonlinear Tg fits for MD density-temperature data.

This script is intended as a window-free alternative to the two-line Tg fit.
It fits the full density-temperature curve with Tg as an explicit parameter
and reports the standard error from the nonlinear least-squares covariance.

Input:
  TgDensity/Tg_density_*.csv with columns:
    T(K), density(g/cm^3)

Models:
  hyperbola:
    smooth bilinear model whose low- and high-temperature asymptotes meet at Tg
  sigmoid:
    smooth logistic blend between low- and high-temperature asymptotes

Outputs:
  nonlinear_fit_results/per_run_nonlinear_fit.csv
  nonlinear_fit_results/combined_mean_nonlinear_fit.csv
  nonlinear_fit_results/overall_nonlinear_summary.json
  nonlinear_fit_results/combined_mean_<model>_fit.png
  nonlinear_fit_results/nonlinear_tg_by_run.png
  nonlinear_fit_results/run_<id>_<model>_fit.png
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
from scipy.optimize import curve_fit


MIN_POINTS = 8
MODEL_NAMES = ("hyperbola", "sigmoid")


@dataclass(frozen=True)
class NonlinearFit:
    run_id: int
    model: str
    n_points: int
    rho_tg: float
    slope_low: float
    slope_high: float
    tg_k: float
    width_k: float
    tg_se_k: float
    width_se_k: float
    rmse: float
    r2: float
    aic: float
    converged: bool
    message: str


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
    if good.sum() < MIN_POINTS:
        raise ValueError(f"Too few finite points in {path}; need at least {MIN_POINTS}")

    order = np.argsort(t[good])
    return t[good][order], rho[good][order]


def hyperbola_model(
    t: np.ndarray,
    rho_tg: float,
    slope_low: float,
    slope_high: float,
    tg_k: float,
    width_k: float,
) -> np.ndarray:
    x = t - tg_k
    smooth_x = np.sqrt(x * x + width_k * width_k)
    return rho_tg + slope_low * x + 0.5 * (slope_high - slope_low) * (x + smooth_x)


def sigmoid_model(
    t: np.ndarray,
    rho_tg: float,
    slope_low: float,
    slope_high: float,
    tg_k: float,
    width_k: float,
) -> np.ndarray:
    x = t - tg_k
    z = np.clip(x / width_k, -60.0, 60.0)
    s = 1.0 / (1.0 + np.exp(-z))
    low_line = rho_tg + slope_low * x
    high_line = rho_tg + slope_high * x
    return (1.0 - s) * low_line + s * high_line


def model_fn(name: str):
    if name == "hyperbola":
        return hyperbola_model
    if name == "sigmoid":
        return sigmoid_model
    raise ValueError(f"Unknown model: {name}")


def initial_guess(t: np.ndarray, rho: np.ndarray) -> list[float]:
    t_min = float(np.min(t))
    t_max = float(np.max(t))
    t_range = t_max - t_min
    n = len(t)
    edge = max(3, n // 4)

    slope_low, intercept_low = np.polyfit(t[:edge], rho[:edge], deg=1)
    slope_high, intercept_high = np.polyfit(t[-edge:], rho[-edge:], deg=1)
    denom = slope_high - slope_low
    if abs(denom) > 1.0e-12:
        tg0 = float((intercept_low - intercept_high) / denom)
    else:
        tg0 = float(np.median(t))
    tg0 = float(np.clip(tg0, t_min + 0.05 * t_range, t_max - 0.05 * t_range))
    rho_tg0 = float(np.interp(tg0, t, rho))
    width0 = max(10.0, 0.08 * t_range)
    return [rho_tg0, float(slope_low), float(slope_high), tg0, width0]


def fit_one(
    run_id: int,
    t: np.ndarray,
    rho: np.ndarray,
    model: str,
    max_width_k: float | None = 120.0,
) -> NonlinearFit:
    t_min = float(np.min(t))
    t_max = float(np.max(t))
    t_range = t_max - t_min
    rho_min = float(np.min(rho))
    rho_max = float(np.max(rho))
    rho_pad = max(0.05, 0.5 * (rho_max - rho_min))
    slope_bound = max(0.02, 10.0 * abs((rho[-1] - rho[0]) / t_range))

    p0 = initial_guess(t, rho)
    max_width = t_range if max_width_k is None else min(float(max_width_k), t_range)
    if max_width <= 1.0:
        raise ValueError("--max-width-k must be > 1 K when provided")
    p0[4] = min(p0[4], 0.5 * max_width)
    lower = [rho_min - rho_pad, -slope_bound, -slope_bound, t_min, 1.0]
    upper = [rho_max + rho_pad, slope_bound, slope_bound, t_max, max_width]

    try:
        params, cov = curve_fit(
            model_fn(model),
            t,
            rho,
            p0=p0,
            bounds=(lower, upper),
            maxfev=50000,
        )
        fitted = model_fn(model)(t, *params)
        residuals = rho - fitted
        sse = float(np.sum(residuals * residuals))
        rmse = float(np.sqrt(sse / len(t)))
        centered = rho - float(np.mean(rho))
        sst = float(np.sum(centered * centered))
        r2 = float(1.0 - sse / sst) if sst > 0 else float("nan")
        k = len(params)
        aic = float(len(t) * math.log(max(sse / len(t), 1.0e-300)) + 2 * k)

        if cov is None or not np.all(np.isfinite(cov)):
            se = np.full(k, np.nan)
        else:
            se = np.sqrt(np.maximum(np.diag(cov), 0.0))

        return NonlinearFit(
            run_id=run_id,
            model=model,
            n_points=len(t),
            rho_tg=float(params[0]),
            slope_low=float(params[1]),
            slope_high=float(params[2]),
            tg_k=float(params[3]),
            width_k=float(params[4]),
            tg_se_k=float(se[3]),
            width_se_k=float(se[4]),
            rmse=rmse,
            r2=r2,
            aic=aic,
            converged=True,
            message="ok",
        )
    except Exception as exc:
        return NonlinearFit(
            run_id=run_id,
            model=model,
            n_points=len(t),
            rho_tg=float("nan"),
            slope_low=float("nan"),
            slope_high=float("nan"),
            tg_k=float("nan"),
            width_k=float("nan"),
            tg_se_k=float("nan"),
            width_se_k=float("nan"),
            rmse=float("nan"),
            r2=float("nan"),
            aic=float("nan"),
            converged=False,
            message=str(exc),
        )


def std(values: list[float] | np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size < 2:
        return float("nan")
    return float(np.std(values, ddof=1))


def sem(values: list[float] | np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size < 2:
        return float("nan")
    return float(np.std(values, ddof=1) / np.sqrt(values.size))


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_fit(
    t: np.ndarray,
    rho: np.ndarray,
    fit: NonlinearFit,
    out_path: Path,
) -> None:
    t_grid = np.linspace(float(np.min(t)), float(np.max(t)), 400)
    y_grid = model_fn(fit.model)(
        t_grid,
        fit.rho_tg,
        fit.slope_low,
        fit.slope_high,
        fit.tg_k,
        fit.width_k,
    )

    plt.figure(figsize=(6.2, 4.5), dpi=150)
    plt.scatter(t, rho, color="black", s=32, label="MD data", zorder=3)
    plt.plot(t_grid, y_grid, color="#1f77b4", linewidth=2.0, label=f"{fit.model} fit")
    plt.axvline(fit.tg_k, color="#d62728", linestyle=":", linewidth=2.0)
    label = f"Tg = {fit.tg_k:.1f}"
    if math.isfinite(fit.tg_se_k):
        label += f" ± {fit.tg_se_k:.1f} K"
    plt.text(
        fit.tg_k + 3.0,
        float(np.interp(fit.tg_k, t_grid, y_grid)),
        label,
        color="#d62728",
        fontsize=10,
    )
    plt.xlabel("Temperature / K", fontsize=12)
    plt.ylabel("Density / g cm$^{-3}$", fontsize=12)
    plt.title(f"Run {fit.run_id}: {fit.model} Tg fit", fontsize=13)
    plt.xlim(float(np.min(t)) - 10.0, float(np.max(t)) + 10.0)
    plt.ylim(float(np.min(rho)) - 0.02, float(np.max(rho)) + 0.02)
    plt.legend(frameon=True, fontsize=9)
    plt.grid(alpha=0.3, linestyle=":")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_tg_by_run(fits: list[NonlinearFit], out_path: Path) -> None:
    plt.figure(figsize=(7.0, 4.5), dpi=150)
    for model, color, marker in (
        ("hyperbola", "#1f77b4", "o"),
        ("sigmoid", "#ff7f0e", "s"),
    ):
        model_fits = [f for f in fits if f.model == model and f.converged]
        x = [f.run_id for f in model_fits]
        y = [f.tg_k for f in model_fits]
        yerr = [f.tg_se_k if math.isfinite(f.tg_se_k) else 0.0 for f in model_fits]
        plt.errorbar(
            x,
            y,
            yerr=yerr,
            fmt=marker,
            color=color,
            capsize=3,
            linestyle="none",
            label=model,
        )

    plt.xlabel("MD repeat")
    plt.ylabel("Tg / K")
    plt.title("Nonlinear Tg fits by run")
    plt.grid(alpha=0.3, linestyle=":")
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def combined_temperature_mean(
    curves: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    by_temp: dict[float, list[float]] = {}
    for t, rho in curves:
        for ti, rhoi in zip(t, rho):
            by_temp.setdefault(float(ti), []).append(float(rhoi))

    temps = np.asarray(sorted(by_temp), dtype=float)
    means = np.asarray([np.mean(by_temp[temp]) for temp in temps], dtype=float)
    ses = np.asarray(
        [
            np.std(by_temp[temp], ddof=1) / np.sqrt(len(by_temp[temp]))
            if len(by_temp[temp]) > 1
            else np.nan
            for temp in temps
        ],
        dtype=float,
    )
    return temps, means, ses


def as_row(fit: NonlinearFit) -> dict[str, str | int | bool]:
    return {
        "run_id": fit.run_id,
        "model": fit.model,
        "n_points": fit.n_points,
        "converged": fit.converged,
        "Tg_K": f"{fit.tg_k:.6f}",
        "Tg_SE_K": f"{fit.tg_se_k:.6f}",
        "rho_Tg_g_cm3": f"{fit.rho_tg:.9f}",
        "slope_low_g_cm3_K": f"{fit.slope_low:.12f}",
        "slope_high_g_cm3_K": f"{fit.slope_high:.12f}",
        "width_K": f"{fit.width_k:.6f}",
        "width_SE_K": f"{fit.width_se_k:.6f}",
        "rmse_g_cm3": f"{fit.rmse:.9f}",
        "r2": f"{fit.r2:.9f}",
        "aic": f"{fit.aic:.6f}",
        "message": fit.message,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fit Tg using full-curve hyperbola and/or sigmoid nonlinear models."
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
        help="Output directory. Default: INPUT_DIR/nonlinear_fit_results",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(MODEL_NAMES),
        choices=MODEL_NAMES,
        help="Nonlinear model(s) to fit.",
    )
    parser.add_argument(
        "--plot-all",
        action="store_true",
        help="Write one fit plot per run and model.",
    )
    parser.add_argument(
        "--max-width-k",
        type=float,
        default=120.0,
        help=(
            "Upper bound for the smooth transition width. Use a pre-declared "
            "physical value such as 100-120 K; set to 0 for no upper bound."
        ),
    )
    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir or input_dir / "nonlinear_fit_results"
    output_dir.mkdir(parents=True, exist_ok=True)
    max_width_k = None if args.max_width_k == 0 else args.max_width_k

    csv_paths = sorted(input_dir.glob("Tg_density_*.csv"), key=run_id_from_path)
    if not csv_paths:
        raise FileNotFoundError(f"No Tg_density_*.csv files found in {input_dir}")

    fits: list[NonlinearFit] = []
    curves: list[tuple[np.ndarray, np.ndarray]] = []
    for csv_path in csv_paths:
        run_id = run_id_from_path(csv_path)
        t, rho = read_density_csv(csv_path)
        curves.append((t, rho))
        for model in args.models:
            fit = fit_one(run_id, t, rho, model, max_width_k=max_width_k)
            fits.append(fit)
            if args.plot_all and fit.converged:
                plot_fit(t, rho, fit, output_dir / f"run_{run_id:02d}_{model}_fit.png")

    fieldnames = [
        "run_id",
        "model",
        "n_points",
        "converged",
        "Tg_K",
        "Tg_SE_K",
        "rho_Tg_g_cm3",
        "slope_low_g_cm3_K",
        "slope_high_g_cm3_K",
        "width_K",
        "width_SE_K",
        "rmse_g_cm3",
        "r2",
        "aic",
        "message",
    ]
    write_csv(
        output_dir / "per_run_nonlinear_fit.csv",
        [as_row(fit) for fit in fits],
        fieldnames,
    )

    combined_t, combined_rho, combined_rho_se = combined_temperature_mean(curves)
    combined_fits = [
        fit_one(0, combined_t, combined_rho, model, max_width_k=max_width_k)
        for model in args.models
    ]
    write_csv(
        output_dir / "combined_mean_nonlinear_fit.csv",
        [as_row(fit) for fit in combined_fits],
        fieldnames,
    )
    for fit in combined_fits:
        if fit.converged:
            plot_fit(
                combined_t,
                combined_rho,
                fit,
                output_dir / f"combined_mean_{fit.model}_fit.png",
            )

    combined_density_rows = [
        {
            "T_K": f"{t:.6f}",
            "mean_density_g_cm3": f"{rho:.9f}",
            "se_density_g_cm3": f"{rho_se:.9f}",
        }
        for t, rho, rho_se in zip(combined_t, combined_rho, combined_rho_se)
    ]
    write_csv(
        output_dir / "combined_mean_density.csv",
        combined_density_rows,
        ["T_K", "mean_density_g_cm3", "se_density_g_cm3"],
    )

    summary: dict[str, dict[str, float | int | str]] = {}
    for model in args.models:
        model_fits = [f for f in fits if f.model == model and f.converged]
        combined_fit = next((f for f in combined_fits if f.model == model), None)
        tgs = [f.tg_k for f in model_fits]
        tg_ses = [f.tg_se_k for f in model_fits if math.isfinite(f.tg_se_k)]
        summary[model] = {
            "n_total": len([f for f in fits if f.model == model]),
            "n_converged": len(model_fits),
            "combined_mean_fit_Tg_K": combined_fit.tg_k if combined_fit else float("nan"),
            "combined_mean_fit_Tg_SE_K": combined_fit.tg_se_k
            if combined_fit
            else float("nan"),
            "combined_mean_fit_rmse_g_cm3": combined_fit.rmse if combined_fit else float("nan"),
            "combined_mean_fit_r2": combined_fit.r2 if combined_fit else float("nan"),
            "mean_Tg_K": float(np.mean(tgs)) if tgs else float("nan"),
            "sd_Tg_across_runs_K": std(tgs),
            "se_mean_Tg_across_runs_K": sem(tgs),
            "mean_curve_fit_Tg_SE_K": float(np.mean(tg_ses)) if tg_ses else float("nan"),
            "mean_rmse_g_cm3": float(np.mean([f.rmse for f in model_fits]))
            if model_fits
            else float("nan"),
            "mean_r2": float(np.mean([f.r2 for f in model_fits])) if model_fits else float("nan"),
            "mean_aic": float(np.mean([f.aic for f in model_fits])) if model_fits else float("nan"),
        }

    with (output_dir / "overall_nonlinear_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    plot_tg_by_run(fits, output_dir / "nonlinear_tg_by_run.png")

    print(f"Wrote nonlinear Tg fit outputs to {output_dir}")
    for model, stats in summary.items():
        print(
            f"  {model}: combined Tg = {stats['combined_mean_fit_Tg_K']:.2f} ± "
            f"{stats['combined_mean_fit_Tg_SE_K']:.2f} K; "
            f"repeat mean = {stats['mean_Tg_K']:.2f} ± "
            f"{stats['se_mean_Tg_across_runs_K']:.2f} K "
            f"(mean ± SE across runs; n={stats['n_converged']})"
        )


if __name__ == "__main__":
    main()
