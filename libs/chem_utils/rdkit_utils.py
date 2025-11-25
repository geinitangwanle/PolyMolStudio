"""
Minimal RDKit helpers for validity/normalization.
"""

from __future__ import annotations

from typing import Iterable, List, Set

try:
    from rdkit import Chem
except Exception:  # pragma: no cover - allow environments without RDKit
    Chem = None  # type: ignore


def sanitize_smiles(smiles: str) -> str | None:
    if Chem is None or not smiles:
        return None if not smiles else smiles
    try:
        mol = Chem.MolFromSmiles(smiles, sanitize=True)
        if mol is None:
            return None
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return None


def is_valid_smiles(smiles: str) -> bool:
    return sanitize_smiles(smiles) is not None


def canonicalize_smiles(smiles: str) -> str | None:
    return sanitize_smiles(smiles)


def deduplicate_smiles(smiles_iter: Iterable[str]) -> List[str]:
    seen: Set[str] = set()
    results: List[str] = []
    for smi in smiles_iter:
        canon = canonicalize_smiles(smi)
        if canon is None or canon in seen:
            continue
        seen.add(canon)
        results.append(canon)
    return results
