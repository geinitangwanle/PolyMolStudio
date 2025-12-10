"""Evaluate SMILES strings with RDKit for 3D buildability, validity, novelty, and uniqueness.

Usage example:
    python rdkit_3d_eval.py --input samples_tg.csv --output samples_tg_checked.csv \\
        --training molecules.csv --smiles-column smiles --training-smiles-column smiles

The script:
    - Parses each SMILES with RDKit to check chemical validity.
    - Attempts to generate a 3D conformer using ETKDG; reports success/failure.
    - Marks molecules as novel if their canonical SMILES are not present in a training set.
    - Marks molecules as unique if they are the first occurrence in the input batch.
Results are appended as new columns and written to the specified output CSV.
"""

import argparse
from typing import Optional, Set, Tuple

import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem


def canonicalize_smiles(smiles: str) -> Optional[str]:
    """Return canonical SMILES or None if parsing fails."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def can_embed_3d(mol: Chem.Mol, max_attempts: int, random_seed: Optional[int]) -> bool:
    """Try to embed a 3D conformer; return True when RDKit succeeds."""
    mol_h = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.maxAttempts = max_attempts
    params.randomSeed = random_seed if random_seed is not None else -1
    params.numThreads = 0  # use all available threads
    try:
        code = AllChem.EmbedMolecule(mol_h, params=params)
    except Exception:
        return False
    if code != 0:
        return False
    # Geometry optimization is optional; failure here still counts as embeddable.
    try:
        AllChem.UFFOptimizeMolecule(mol_h, maxIters=200)
    except Exception:
        pass
    return True


def load_training_set(path: str, smiles_column: str) -> Set[str]:
    """Load canonical SMILES from a training CSV for novelty checking."""
    df = pd.read_csv(path)
    canonical = set()
    for smi in df[smiles_column].dropna().astype(str):
        can = canonicalize_smiles(smi)
        if can:
            canonical.add(can)
    return canonical


def evaluate_smiles(
    df: pd.DataFrame,
    smiles_column: str,
    training_smiles: Optional[Set[str]],
    max_attempts: int,
    random_seed: Optional[int],
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Compute validity, uniqueness, novelty, and 3D embedding success."""
    seen: Set[str] = set()
    validity = []
    uniqueness = []
    novelty = []
    can_build = []

    for smi in df[smiles_column].astype(str):
        mol = Chem.MolFromSmiles(smi)
        valid = mol is not None
        validity.append(valid)

        can_smiles = canonicalize_smiles(smi) if valid else None
        uniq = False
        is_novel = False
        embed_ok = False

        if valid and can_smiles:
            uniq = can_smiles not in seen
            seen.add(can_smiles)
            is_novel = training_smiles is None or can_smiles not in training_smiles
            embed_ok = can_embed_3d(mol, max_attempts=max_attempts, random_seed=random_seed)

        uniqueness.append(uniq)
        novelty.append(is_novel)
        can_build.append(embed_ok)

    return (
        pd.Series(validity, name="is_valid"),
        pd.Series(uniqueness, name="is_unique"),
        pd.Series(novelty, name="is_novel"),
        pd.Series(can_build, name="can_build_3d"),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate SMILES strings with RDKit.")
    parser.add_argument("--input", required=True, help="Input CSV containing SMILES.")
    parser.add_argument("--output", required=True, help="Where to write the annotated CSV.")
    parser.add_argument(
        "--smiles-column",
        default="smiles",
        help="Column name holding SMILES in the input CSV (default: smiles).",
    )
    parser.add_argument(
        "--training",
        help="Optional CSV containing known molecules for novelty checking.",
    )
    parser.add_argument(
        "--training-smiles-column",
        default="smiles",
        help="SMILES column name in the training CSV (default: smiles).",
    )
    parser.add_argument(
        "--max-embed-attempts",
        type=int,
        default=50,
        help="Attempts for RDKit ETKDG embedding (default: 50).",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        help="Random seed for reproducible embeddings (default: None).",
    )

    args = parser.parse_args()

    df = pd.read_csv(args.input)
    training_smiles = None
    if args.training:
        training_smiles = load_training_set(args.training, args.training_smiles_column)

    is_valid, is_unique, is_novel, can_build_3d = evaluate_smiles(
        df,
        smiles_column=args.smiles_column,
        training_smiles=training_smiles,
        max_attempts=args.max_embed_attempts,
        random_seed=args.random_seed,
    )

    df = df.copy()
    df["is_valid"] = is_valid
    df["is_unique"] = is_unique
    df["is_novel"] = is_novel
    df["can_build_3d"] = can_build_3d
    df.to_csv(args.output, index=False)

    print(f"Wrote results with 3D checks to {args.output}")
    print(
        f"Validity: {is_valid.sum()}/{len(is_valid)} | "
        f"Uniqueness: {is_unique.sum()} | "
        f"Novelty: {is_novel.sum()} | "
        f"3D buildable: {can_build_3d.sum()}"
    )


if __name__ == "__main__":
    main()
