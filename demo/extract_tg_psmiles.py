#!/usr/bin/env python
"""
Extract Tg-labeled polymer monomers, convert Tg to Kelvin, and standardize SMILES to pSMILES.

The script filters a CSV to rows with a non-null Tg, adds 273.15 to convert from °C to K,
canonicalizes SMILES with Open Babel, and wraps wildcard atoms as [*] to match pSMILES style.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List, Optional

import pandas as pd


# Matches bare "*" atoms that are not already bracketed
STAR_PATTERN = re.compile(r"(?<!\[)\*(?!\])")


def _normalize_placeholders(smiles: str, placeholders: Iterable[str]) -> str:
    """Replace placeholder element symbols (e.g., Ce, Th) with wildcard '*' prior to wrapping."""
    normalized = smiles
    for ph in placeholders:
        pattern_bracketed = re.compile(rf"\[\s*{re.escape(ph)}\s*\]", flags=re.IGNORECASE)
        pattern_bare = re.compile(rf"\b{re.escape(ph)}\b", flags=re.IGNORECASE)
        normalized = pattern_bracketed.sub("*", normalized)
        normalized = pattern_bare.sub("*", normalized)
    return normalized


def _to_psmiles(smiles: str, canonical_smiles: Optional[str], *, use_canonical: bool) -> str:
    """
    Wrap wildcard atoms with brackets to form pSMILES.
    If `use_canonical` is True and a canonicalized string is provided, prefer it;
    otherwise fall back to the original SMILES to preserve order.
    """
    base = canonical_smiles if use_canonical and canonical_smiles else smiles
    return STAR_PATTERN.sub("[*]", base)


def _canonicalize_with_obabel(smiles_list: Iterable[str]) -> List[Optional[str]]:
    """
    Canonicalize SMILES strings via Open Babel. Returns a list aligned to the input order;
    entries that fail to convert are set to None.
    """
    smiles_lines = [s.strip() for s in smiles_list]
    if not smiles_lines:
        return []

    try:
        proc = subprocess.run(
            ["obabel", "-ismi", "-ocan"],
            input="\n".join(smiles_lines) + "\n",
            text=True,
            capture_output=True,
            check=False,
        )
    except FileNotFoundError:
        print("[warn] Open Babel (obabel) not found; skipping canonicalization.", file=sys.stderr)
        return [None] * len(smiles_lines)

    if proc.returncode != 0:
        print(f"[warn] obabel exited with code {proc.returncode}: {proc.stderr.strip()}", file=sys.stderr)
        return [None] * len(smiles_lines)

    # obabel prints canonical SMILES to stdout, summary to stderr
    lines = [ln.strip() for ln in proc.stdout.splitlines() if ln.strip()]
    if len(lines) != len(smiles_lines):
        missing = len(smiles_lines) - len(lines)
        print(
            f"[warn] obabel returned {len(lines)} lines for {len(smiles_lines)} inputs; "
            f"padding {missing} entries without canonicalization.",
            file=sys.stderr,
        )
        lines.extend([None] * missing)

    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract Tg rows and convert SMILES to pSMILES.")
    parser.add_argument("--input", type=Path, default=Path("data/raw/train.csv"), help="Source CSV path.")
    parser.add_argument("--output", type=Path, default=Path("data/raw/train_tg_psmiles.csv"), help="Output CSV path.")
    parser.add_argument("--smiles-col", type=str, default="SMILES", help="Column containing SMILES.")
    parser.add_argument("--tg-col", type=str, default="Tg", help="Column containing Tg.")
    parser.add_argument(
        "--tg-offset",
        type=float,
        default=273.15,
        help="Value added to Tg column (use 0 if Tg already in Kelvin; default 273.15 assumes °C->K).",
    )
    parser.add_argument(
        "--placeholders",
        nargs="*",
        default=[],
        help="Element symbols to treat as connection placeholders and replace with '*' (e.g., Ce Th).",
    )
    parser.add_argument(
        "--canonicalize",
        action="store_true",
        help="Use Open Babel canonical SMILES before wrapping stars. Default: preserve original order.",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    if args.smiles_col not in df.columns or args.tg_col not in df.columns:
        raise ValueError(f"Missing required columns {args.smiles_col!r} or {args.tg_col!r} in {args.input}")

    df = df[df[args.tg_col].notna()].copy()
    df[args.tg_col] = df[args.tg_col].astype(float) + args.tg_offset

    smiles_list_raw = df[args.smiles_col].astype(str).tolist()
    smiles_list = [
        _normalize_placeholders(smi, args.placeholders) if args.placeholders else smi
        for smi in smiles_list_raw
    ]
    canonical = _canonicalize_with_obabel(smiles_list) if args.canonicalize else [None] * len(smiles_list)
    df["PSMILES"] = [_to_psmiles(smi, can, use_canonical=args.canonicalize) for smi, can in zip(smiles_list, canonical)]

    out_cols = [col for col in ("id", "PSMILES", args.tg_col) if col in df.columns]
    output_df = df[out_cols]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(args.output, index=False)
    print(f"Wrote {len(output_df)} rows to {args.output}")


if __name__ == "__main__":
    main()
