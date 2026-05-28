#!/usr/bin/env python3
"""Lightweight oversampling/scoring/Delta-triage analysis for reviewer response.

The script compares an unfiltered VAE sampling pool with the final triaged
candidates. It is intentionally CSV-based so it can be run on the existing
design outputs without retraining any model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


def find_col(columns: Iterable[str], candidates: list[str], required: bool = True) -> str | None:
    normalized = {c.strip().lower(): c for c in columns}
    for name in candidates:
        hit = normalized.get(name.strip().lower())
        if hit is not None:
            return hit
    if required:
        raise ValueError(f"Could not find any of these columns: {candidates}")
    return None


def numeric_series(df: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_numeric(df[col], errors="coerce")


def summarize_scores(values: pd.Series, thresholds: list[float]) -> dict:
    values = pd.to_numeric(values, errors="coerce").dropna()
    out = {
        "n_scored": int(values.size),
        "mean_k": float(values.mean()) if values.size else None,
        "median_k": float(values.median()) if values.size else None,
        "p90_k": float(values.quantile(0.90)) if values.size else None,
        "p95_k": float(values.quantile(0.95)) if values.size else None,
        "p99_k": float(values.quantile(0.99)) if values.size else None,
        "max_k": float(values.max()) if values.size else None,
    }
    for t in thresholds:
        count = int((values >= t).sum())
        out[f"n_ge_{t:g}k"] = count
        out[f"frac_ge_{t:g}k"] = float(count / values.size) if values.size else None
    return out


def enrichment(pool_frac: float | None, candidate_frac: float | None) -> float | None:
    if pool_frac is None or candidate_frac is None or pool_frac <= 0:
        return None
    return float(candidate_frac / pool_frac)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Quantify high-Tg enrichment from VAE oversampling plus predictor/Delta triage."
    )
    parser.add_argument("--pool-csv", type=Path, required=True, help="Unfiltered generated/scored pool CSV.")
    parser.add_argument("--candidate-csv", type=Path, required=True, help="Final candidate/triage CSV.")
    parser.add_argument("--out-dir", type=Path, default=Path("analysis_outputs/oversampling_triage"))
    parser.add_argument("--pool-score-col", default=None, help="Predicted Tg column in pool CSV.")
    parser.add_argument("--candidate-score-col", default=None, help="Main predicted Tg column in candidate CSV.")
    parser.add_argument("--candidate-score-col-2", default=None, help="Optional second predictor Tg column.")
    parser.add_argument("--delta-col", default=None, help="Optional Delta/disagreement column in candidate CSV.")
    parser.add_argument("--smiles-col", default=None, help="SMILES column shared by pool/candidates.")
    parser.add_argument("--thresholds", type=float, nargs="+", default=[450.0, 475.0, 500.0])
    parser.add_argument("--delta-thresholds", type=float, nargs="*", default=[5.0, 10.0, 20.0])
    args = parser.parse_args()

    pool = pd.read_csv(args.pool_csv)
    cand = pd.read_csv(args.candidate_csv)

    pool_score_col = args.pool_score_col or find_col(pool.columns, ["pred", "prediction", "pred_tg", "Tg_pred"])
    cand_score_col = args.candidate_score_col or find_col(
        cand.columns,
        ["Multimodal Prediction  Tg /K", "GNN Prediction Tg /K", "pred", "prediction", "pred_tg"],
    )
    cand_score_col_2 = args.candidate_score_col_2 or find_col(
        cand.columns,
        ["Unimodal Prediction Tg /K", "graph_mean", "pred_2"],
        required=False,
    )
    delta_col = args.delta_col or find_col(
        cand.columns,
        ["Delta /K", "Δ /K", "delta_k", "branch_delta_k", "abs_graph_multi_ensemble_delta_k"],
        required=False,
    )
    smiles_col_pool = args.smiles_col or find_col(pool.columns, ["smiles", "PSMILES"], required=False)
    smiles_col_cand = args.smiles_col or find_col(cand.columns, ["smiles", "PSMILES"], required=False)

    pool_scores = numeric_series(pool, pool_score_col)
    cand_scores = numeric_series(cand, cand_score_col)

    rows = []
    for label, df, scores in [
        ("unfiltered_vae_pool", pool, pool_scores),
        ("triaged_candidates", cand, cand_scores),
    ]:
        row = {"set": label, "n_rows": int(len(df))}
        row.update(summarize_scores(scores, args.thresholds))
        rows.append(row)

    summary = pd.DataFrame(rows)
    enrich_rows = []
    for t in args.thresholds:
        pool_frac = float(summary.loc[summary["set"] == "unfiltered_vae_pool", f"frac_ge_{t:g}k"].iloc[0])
        cand_frac = float(summary.loc[summary["set"] == "triaged_candidates", f"frac_ge_{t:g}k"].iloc[0])
        enrich_rows.append(
            {
                "threshold_k": t,
                "pool_fraction": pool_frac,
                "candidate_fraction": cand_frac,
                "fold_enrichment": enrichment(pool_frac, cand_frac),
                "expected_pool_draws_per_hit": float(1.0 / pool_frac) if pool_frac > 0 else None,
            }
        )
    enrichment_df = pd.DataFrame(enrich_rows)

    triage = {}
    if delta_col is not None:
        delta = numeric_series(cand, delta_col).dropna()
        triage["delta_col"] = delta_col
        triage["median_delta_k"] = float(delta.median()) if len(delta) else None
        triage["p90_delta_k"] = float(delta.quantile(0.90)) if len(delta) else None
        for t in args.delta_thresholds:
            triage[f"frac_delta_le_{t:g}k"] = float((delta <= t).mean()) if len(delta) else None
    if cand_score_col_2 is not None:
        s1 = numeric_series(cand, cand_score_col)
        s2 = numeric_series(cand, cand_score_col_2)
        pair = pd.concat([s1, s2], axis=1).dropna()
        if len(pair):
            diff = (pair.iloc[:, 0] - pair.iloc[:, 1]).abs()
            triage["second_score_col"] = cand_score_col_2
            triage["median_two_predictor_abs_delta_k"] = float(diff.median())
            triage["p90_two_predictor_abs_delta_k"] = float(diff.quantile(0.90))

    ranks_path = None
    if smiles_col_pool and smiles_col_cand:
        pool_rank = pool[[smiles_col_pool, pool_score_col]].copy()
        pool_rank[pool_score_col] = pool_scores
        pool_rank = pool_rank.dropna(subset=[pool_score_col])
        pool_rank["pool_score_percentile"] = pool_rank[pool_score_col].rank(pct=True)
        ranks = cand[[smiles_col_cand, cand_score_col]].merge(
            pool_rank,
            left_on=smiles_col_cand,
            right_on=smiles_col_pool,
            how="left",
            suffixes=("_candidate", "_pool"),
        )
        args.out_dir.mkdir(parents=True, exist_ok=True)
        ranks_path = args.out_dir / "candidate_pool_percentiles.csv"
        ranks.to_csv(ranks_path, index=False)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.out_dir / "pool_vs_triaged_summary.csv"
    enrichment_path = args.out_dir / "high_tg_enrichment.csv"
    json_path = args.out_dir / "oversampling_triage_summary.json"
    summary.to_csv(summary_path, index=False)
    enrichment_df.to_csv(enrichment_path, index=False)
    payload = {
        "pool_csv": str(args.pool_csv),
        "candidate_csv": str(args.candidate_csv),
        "pool_score_col": pool_score_col,
        "candidate_score_col": cand_score_col,
        "n_pool_rows": int(len(pool)),
        "n_candidate_rows": int(len(cand)),
        "summary_csv": str(summary_path),
        "enrichment_csv": str(enrichment_path),
        "candidate_pool_percentiles_csv": str(ranks_path) if ranks_path else None,
        "triage": triage,
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(summary.to_string(index=False))
    print()
    print(enrichment_df.to_string(index=False))
    if triage:
        print()
        print(json.dumps({"triage": triage}, indent=2))
    print(f"\nWrote: {summary_path}, {enrichment_path}, {json_path}")


if __name__ == "__main__":
    main()
