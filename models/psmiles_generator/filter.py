"""
Sampling post-processing helpers.
"""

from typing import Iterable, List, Set

from libs.chem_utils import canonicalize_smiles, is_valid_smiles


def filter_and_dedup(smiles_list: Iterable[str]) -> List[str]:
    seen: Set[str] = set()
    kept: List[str] = []
    for smi in smiles_list:
        if not smi:
            continue
        if not is_valid_smiles(smi):
            continue
        canon = canonicalize_smiles(smi)
        if canon in seen:
            continue
        seen.add(canon)
        kept.append(canon)
    return kept


__all__ = ["filter_and_dedup"]
