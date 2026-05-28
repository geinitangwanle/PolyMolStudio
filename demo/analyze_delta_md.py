#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyze the relationship between inter-branch prediction discrepancy (Delta)
and MD relative deviation for paper candidates.

Default behavior splits candidates into low/mid/high Delta groups by tertiles,
which gives balanced groups for small candidate tables.

Examples:
  python demo/analyze_delta_md.py

  python demo/analyze_delta_md.py \
    --csv demo/paper_candidate.csv \
    --out-dir demo/analysis_outputs/delta_md

  python demo/analyze_delta_md.py --thresholds 5 10
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from scipy.stats import kruskal, pearsonr, spearmanr
except Exception:  # pragma: no cover - scipy is optional for basic summaries
    kruskal = None
    pearsonr = None
    spearmanr = None


DEFAULT_CSV = Path("demo/paper_candidate.csv")
DEFAULT_OUT_DIR = Path("demo/analysis_outputs/delta_md")


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


def assign_delta_group(delta: pd.Series, thresholds: Optional[Tuple[float, float]]) -> Tuple[pd.Series, Dict[str, float]]:
    if thresholds is None:
        low_cut = float(delta.quantile(1 / 3))
        high_cut = float(delta.quantile(2 / 3))
        method = "tertile"
    else:
        low_cut, high_cut = thresholds
        if low_cut >= high_cut:
            raise ValueError("--thresholds must be two increasing numbers, e.g. --thresholds 5 10")
        method = "fixed"

    labels = np.where(delta <= low_cut, "low_delta", np.where(delta <= high_cut, "mid_delta", "high_delta"))
    metadata = {"method": method, "low_cut_K": low_cut, "high_cut_K": high_cut}
    return pd.Series(labels, index=delta.index, name="delta_group"), metadata


def numeric_summary(series: pd.Series) -> Dict[str, float]:
    return {
        "n": int(series.size),
        "mean": float(series.mean()),
        "std": float(series.std(ddof=1)) if series.size > 1 else 0.0,
        "median": float(series.median()),
        "min": float(series.min()),
        "max": float(series.max()),
    }


def correlation(x: pd.Series, y: pd.Series) -> Dict[str, Optional[float]]:
    out: Dict[str, Optional[float]] = {
        "pearson_r": None,
        "pearson_p": None,
        "spearman_rho": None,
        "spearman_p": None,
    }
    if pearsonr is None or spearmanr is None:
        return out
    pearson = pearsonr(x, y)
    spearman = spearmanr(x, y)
    out.update(
        {
            "pearson_r": float(pearson.statistic),
            "pearson_p": float(pearson.pvalue),
            "spearman_rho": float(spearman.statistic),
            "spearman_p": float(spearman.pvalue),
        }
    )
    return out


def linear_fit(x: pd.Series, y: pd.Series) -> Dict[str, Optional[float]]:
    if x.size < 2 or float(x.std(ddof=0)) == 0.0:
        return {"slope_k_per_k": None, "intercept_k": None}
    slope, intercept = np.polyfit(x.to_numpy(dtype=float), y.to_numpy(dtype=float), deg=1)
    return {"slope_k_per_k": float(slope), "intercept_k": float(intercept)}


def residual_vs_md_stats(df: pd.DataFrame, md_tg_col: str, residual_col: str) -> Dict[str, object]:
    return {
        "n": int(len(df)),
        "residual_k": numeric_summary(df[residual_col]),
        "correlation_residual_vs_md_tg": correlation(df[md_tg_col], df[residual_col]),
        "linear_fit_residual_vs_md_tg": linear_fit(df[md_tg_col], df[residual_col]),
    }


def group_test(df: pd.DataFrame, value_col: str) -> Dict[str, Optional[float]]:
    if kruskal is None:
        return {"kruskal_H": None, "kruskal_p": None}
    groups = [group[value_col].to_numpy() for _, group in df.groupby("delta_group", sort=False)]
    if len(groups) < 2:
        return {"kruskal_H": None, "kruskal_p": None}
    stat = kruskal(*groups)
    return {"kruskal_H": float(stat.statistic), "kruskal_p": float(stat.pvalue)}


def plot_delta_scatter(df: pd.DataFrame, delta_col: str, md_dev_col: str, out_path: Path) -> None:
    colors = {"low_delta": "#2a9d8f", "mid_delta": "#e9c46a", "high_delta": "#d95f02"}
    plt.figure(figsize=(6, 4.5))
    for group, sub in df.groupby("delta_group", sort=False):
        plt.scatter(sub[delta_col], sub[md_dev_col], s=56, label=group.replace("_", " "), color=colors[group], alpha=0.85)
    plt.xlabel("Delta between fused and graph predictions / K")
    plt.ylabel("MD relative deviation / %")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_group_box(df: pd.DataFrame, md_dev_col: str, out_path: Path) -> None:
    order = ["low_delta", "mid_delta", "high_delta"]
    data = [df.loc[df["delta_group"] == group, md_dev_col].to_numpy() for group in order]
    plt.figure(figsize=(6, 4.5))
    box = plt.boxplot(data, labels=["Low Delta", "Mid Delta", "High Delta"], patch_artist=True, showmeans=True)
    for patch, color in zip(box["boxes"], ["#2a9d8f", "#e9c46a", "#d95f02"]):
        patch.set_facecolor(color)
        patch.set_alpha(0.45)
    plt.ylabel("MD relative deviation / %")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_residual_vs_md_tg(
    df: pd.DataFrame,
    md_tg_col: str,
    model_tg_col: str,
    residual_col: str,
    uncertainty_col: Optional[str],
    fit: Dict[str, Optional[float]],
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(6, 4.8))
    if uncertainty_col is not None:
        scatter = ax.scatter(
            df[md_tg_col],
            df[residual_col],
            c=df[uncertainty_col],
            s=68,
            cmap="viridis",
            alpha=0.88,
            edgecolor="white",
            linewidth=0.5,
        )
        cbar = fig.colorbar(scatter, ax=ax)
        cbar.set_label(uncertainty_col)
    else:
        ax.scatter(
            df[md_tg_col],
            df[residual_col],
            s=68,
            color="#4c78a8",
            alpha=0.88,
            edgecolor="white",
            linewidth=0.5,
        )

    ax.axhline(0, color="#555555", linestyle="--", linewidth=1.1)

    ax.set_xlabel("MD Tg / K")
    ax.set_ylabel(f"Residual ({model_tg_col} - MD Tg) / K")
    ax.set_title("Residual vs MD Tg")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def analyze(
    csv_path: Path,
    out_dir: Path,
    thresholds: Optional[Tuple[float, float]],
    exclude_test_ids: Optional[List[int]] = None,
    model_tg_col: Optional[str] = None,
    uncertainty_col: Optional[str] = None,
) -> Dict[str, object]:
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    delta_col = find_column(df.columns.tolist(), ["Δ /K", "Delta /K", "delta"], contains="δ")
    md_dev_col = find_column(df.columns.tolist(), ["relative deviation %", "MD deviation %"], contains="deviation")
    md_tg_col = find_column(df.columns.tolist(), ["MD Tg /K", "MD Tg / K", "md_tg_from_curve_k"], contains="md tg")
    if model_tg_col is None:
        model_tg_col = find_column(
            df.columns.tolist(),
            [
                "Multimodal Prediction  Tg /K",
                "Multimodal Prediction Tg /K",
                "multi_mean",
                "pred_ensemble_mean",
                "Unimodal Prediction Tg /K",
                "graph_mean",
            ],
            contains="prediction",
        )
    elif model_tg_col not in df.columns:
        raise ValueError(f"--model-tg-col '{model_tg_col}' was not found. Available columns: {df.columns.tolist()}")

    if uncertainty_col is None:
        uncertainty_candidates = [
            "md_window_uncertainty_k",
            "MD fitting uncertainty /K",
            "MD fitting uncertainty K",
            "multi_model_uncertainty_k",
            "graph_model_uncertainty_k",
            "pred_ensemble_std",
        ]
        try:
            uncertainty_col = find_column(df.columns.tolist(), uncertainty_candidates, contains="uncertainty")
        except ValueError:
            uncertainty_col = None
    elif uncertainty_col not in df.columns:
        raise ValueError(f"--uncertainty-col '{uncertainty_col}' was not found. Available columns: {df.columns.tolist()}")

    test_id_col = find_column(df.columns.tolist(), ["Test id", "id"], contains="id")

    df = df.copy()
    excluded_rows = []
    if exclude_test_ids:
        excluded_mask = df[test_id_col].isin(exclude_test_ids)
        excluded_rows = df.loc[excluded_mask].to_dict(orient="records")
        df = df.loc[~excluded_mask].copy()

    df[delta_col] = pd.to_numeric(df[delta_col], errors="coerce")
    df[md_dev_col] = pd.to_numeric(df[md_dev_col], errors="coerce")
    df[md_tg_col] = pd.to_numeric(df[md_tg_col], errors="coerce")
    df[model_tg_col] = pd.to_numeric(df[model_tg_col], errors="coerce")
    if uncertainty_col is not None:
        df[uncertainty_col] = pd.to_numeric(df[uncertainty_col], errors="coerce")
    df = df.dropna(subset=[delta_col, md_dev_col, md_tg_col, model_tg_col])
    if uncertainty_col is not None:
        df = df.dropna(subset=[uncertainty_col])

    residual_col = "residual_model_minus_md_k"
    df[residual_col] = df[model_tg_col] - df[md_tg_col]

    df["delta_group"], group_meta = assign_delta_group(df[delta_col], thresholds)
    group_order = ["low_delta", "mid_delta", "high_delta"]
    df["delta_group"] = pd.Categorical(df["delta_group"], categories=group_order, ordered=True)
    df = df.sort_values(["delta_group", delta_col]).reset_index(drop=True)

    group_summary = {}
    for group_name, group in df.groupby("delta_group", observed=True):
        group_summary[str(group_name)] = {
            "delta_K": numeric_summary(group[delta_col]),
            "md_relative_deviation_percent": numeric_summary(group[md_dev_col]),
        }

    residual_fit = linear_fit(df[md_tg_col], df[residual_col])
    outlier_idx = df[residual_col].abs().idxmax()
    outlier_row = df.loc[outlier_idx]
    robust_df = df.drop(index=outlier_idx).copy()
    residual_summary = {
        "md_tg_column": md_tg_col,
        "model_tg_column": model_tg_col,
        "uncertainty_column": uncertainty_col,
        "residual_column": residual_col,
        "full_data": residual_vs_md_stats(df, md_tg_col, residual_col),
        "robustness_check_excluding_largest_abs_residual": {
            "excluded_row": {
                key: (
                    float(value)
                    if isinstance(value, (np.floating, float))
                    else int(value)
                    if isinstance(value, (np.integer, int))
                    else str(value)
                )
                for key, value in outlier_row.to_dict().items()
            },
            **residual_vs_md_stats(robust_df, md_tg_col, residual_col),
        },
    }

    results: Dict[str, object] = {
        "source": str(csv_path),
        "n": int(len(df)),
        "columns": {
            "delta": delta_col,
            "md_relative_deviation": md_dev_col,
            "md_tg": md_tg_col,
            "model_tg": model_tg_col,
            "uncertainty": uncertainty_col,
            "residual": residual_col,
        },
        "excluded_test_ids": exclude_test_ids or [],
        "excluded_rows": excluded_rows,
        "grouping": group_meta,
        "overall_correlation": correlation(df[delta_col], df[md_dev_col]),
        "between_group_test_for_md_deviation": group_test(df, md_dev_col),
        "residual_vs_md_tg": residual_summary,
        "group_summary": group_summary,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "delta_md_grouped_candidates.csv", index=False, encoding="utf-8-sig")
    (out_dir / "delta_md_summary.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    plot_delta_scatter(df, delta_col, md_dev_col, out_dir / "delta_vs_md_deviation.png")
    plot_group_box(df, md_dev_col, out_dir / "md_deviation_by_delta_group.png")
    plot_residual_vs_md_tg(
        df,
        md_tg_col,
        model_tg_col,
        residual_col,
        uncertainty_col,
        residual_fit,
        out_dir / "residual_vs_md_tg.png",
    )

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze Delta groups against MD relative deviation.")
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV, help="Candidate CSV with Delta and MD deviation columns.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR, help="Directory for CSV/JSON/PNG outputs.")
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs=2,
        metavar=("LOW_K", "HIGH_K"),
        default=None,
        help="Optional fixed Delta thresholds in K. Default uses tertiles.",
    )
    parser.add_argument(
        "--exclude-test-id",
        type=int,
        nargs="*",
        default=None,
        help="Optional Test id values to exclude before grouping and correlation analysis.",
    )
    parser.add_argument(
        "--model-tg-col",
        default=None,
        help="Optional model Tg column for residual = Tg_model - Tg_MD. Default auto-detects multimodal prediction.",
    )
    parser.add_argument(
        "--uncertainty-col",
        default=None,
        help="Optional uncertainty column for residual plot colors. Default auto-detects MD/window or ensemble uncertainty.",
    )
    args = parser.parse_args()

    thresholds = tuple(args.thresholds) if args.thresholds is not None else None
    results = analyze(
        args.csv,
        args.out_dir,
        thresholds,
        exclude_test_ids=args.exclude_test_id,
        model_tg_col=args.model_tg_col,
        uncertainty_col=args.uncertainty_col,
    )
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
