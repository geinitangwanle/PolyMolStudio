#!/usr/bin/env python
"""Canonicalize a CSV column of polymer SMILES with RDKit."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import pandas as pd
from rdkit import Chem, RDLogger


def canonicalize_psmiles(smiles: str, *, bracket_dummy: bool = True) -> Optional[str]:
    """Return RDKit canonical SMILES, preserving polymer dummy atoms as [*]."""
    if pd.isna(smiles):
        return None
    text = str(smiles).strip()
    if not text:
        return None

    mol = Chem.MolFromSmiles(text)
    if mol is None:
        return None

    canonical = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    if bracket_dummy:
        canonical = canonical.replace("*", "[*]")
    return canonical


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Input CSV path.")
    parser.add_argument("--output", type=Path, required=True, help="Output CSV path.")
    parser.add_argument("--col", default="PSMILES", help="Column containing pSMILES.")
    parser.add_argument(
        "--rejects",
        type=Path,
        default=None,
        help="Optional CSV path for rows RDKit could not parse.",
    )
    parser.add_argument(
        "--bare-dummy",
        action="store_true",
        help="Keep RDKit's bare * dummy atom output instead of converting it to [*].",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    RDLogger.DisableLog("rdApp.warning")
    RDLogger.DisableLog("rdApp.error")

    df = pd.read_csv(args.input)
    if args.col not in df.columns:
        raise ValueError(f"Column {args.col!r} not found in {args.input}")

    original_count = len(df)
    canonical = df[args.col].map(
        lambda value: canonicalize_psmiles(value, bracket_dummy=not args.bare_dummy)
    )
    valid_mask = canonical.notna()

    out_df = df.loc[valid_mask].copy()
    out_df[args.col] = canonical.loc[valid_mask].values
    out_df = out_df.drop_duplicates(subset=[args.col], keep="first")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.output, index=False)

    rejects_count = int((~valid_mask).sum())
    duplicate_count = int(valid_mask.sum() - len(out_df))
    if args.rejects is not None:
        rejects = df.loc[~valid_mask].copy()
        args.rejects.parent.mkdir(parents=True, exist_ok=True)
        rejects.to_csv(args.rejects, index=False)

    print(f"input rows: {original_count}")
    print(f"valid canonical rows: {int(valid_mask.sum())}")
    print(f"dropped invalid rows: {rejects_count}")
    print(f"dropped duplicate canonical pSMILES: {duplicate_count}")
    print(f"output rows: {len(out_df)}")
    print(f"wrote: {args.output}")
    if args.rejects is not None:
        print(f"rejects: {args.rejects}")


if __name__ == "__main__":
    main()
