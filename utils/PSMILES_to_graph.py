from __future__ import annotations

from pathlib import Path
from typing import List, Tuple, Dict, Any
import math
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from tqdm.auto import tqdm
from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem import AllChem

def _optimize_geometry(mol: Chem.Mol, maxIters: int = 200) -> tuple[bool, str]:
    """
    尝试优化几何：先 MMFF，再 UFF；都不行就返回 False。
    返回 (是否优化成功, 使用的方法名: 'MMFF'/'UFF'/'none')
    """
    try:
        # MMFF 优先
        if AllChem.MMFFHasAllMoleculeParams(mol):
            ok = AllChem.MMFFOptimizeMolecule(mol, maxIters=maxIters)
            # RDKit 返回 0 表示收敛；非 0 也通常可接受，这里统一认为“尝试过就算成功”
            return True, "MMFF"

        # 其次 UFF（先判断是否有参数）
        if AllChem.UFFHasAllMoleculeParams(mol):
            ok = AllChem.UFFOptimizeMolecule(mol, maxIters=maxIters)
            return True, "UFF"

        # 两种力场都没有参数，跳过优化
        return False, "none"

    except Exception:
        # 任一优化报错，保留 ETKDG 坐标使用
        return False, "none"


def _resolved_positions(
    mol_raw: Chem.Mol,
    conf: Chem.Conformer,
    star_to_carbon: Dict[int, int],
) -> List[Tuple[float, float, float]]:
    """Return 3D coordinates aligned with mol_raw atoms; stars reuse the replacement carbon."""
    coords: List[Tuple[float, float, float]] = []
    for atom in mol_raw.GetAtoms():
        idx = atom.GetIdx()
        mapped_idx = star_to_carbon.get(idx, idx)
        pos = conf.GetAtomPosition(mapped_idx)
        coords.append((float(pos.x), float(pos.y), float(pos.z)))
    return coords


def graph_from_psmiles(psmiles: str) -> Dict[str, pd.DataFrame]:
    """
    Convert a PSMILES string (with star attachment points) into graph-friendly features.

    Returns
    -------
    dict with
        • node_feats: per-atom properties + 3D coordinates
        • edge_index: bidirectional adjacency (COO)
        • edge_attr: bond descriptors (type, conjugation, ring flag, length)
        • coords: raw coordinates aligned with the original atoms
    """
    mol_raw = Chem.MolFromSmiles(psmiles, sanitize=False)
    if mol_raw is None:
        raise ValueError(f"Could not parse PSMILES: {psmiles!r}")
    mol_raw.UpdatePropertyCache(strict=False)

    rw = Chem.RWMol(mol_raw)
    star_to_carbon: Dict[int, int] = {}

    star_indices = sorted(
        (atom.GetIdx() for atom in rw.GetAtoms() if atom.GetAtomicNum() == 0),
        reverse=True,
    )
    for star_idx in star_indices:
        star_atom = rw.GetAtomWithIdx(star_idx)
        neighbors = star_atom.GetNeighbors()
        if len(neighbors) != 1:
            raise ValueError(f"Star atom {star_idx} has {len(neighbors)} neighbors; expected exactly one.")

        neighbor_idx = neighbors[0].GetIdx()
        bond = rw.GetBondBetweenAtoms(star_idx, neighbor_idx)

        new_c_idx = rw.AddAtom(Chem.Atom("C"))
        rw.AddBond(new_c_idx, neighbor_idx, bond.GetBondType())

        star_to_carbon[star_idx] = new_c_idx
        rw.RemoveAtom(star_idx)

    mol_geom = rw.GetMol()
    Chem.SanitizeMol(mol_geom)
    mol_geom = Chem.AddHs(mol_geom)

    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    embed_status = AllChem.EmbedMolecule(mol_geom, params)
    if embed_status != 0:
        params.useRandomCoords = True
        for _ in range(4):
            embed_status = AllChem.EmbedMolecule(mol_geom, params)
            if embed_status == 0:
                break
    if embed_status != 0:
        raise RuntimeError("Failed to embed 3D coordinates for the molecule.")

    _ = _optimize_geometry(mol_geom, maxIters=200)

    conf = mol_geom.GetConformer()
    coords = _resolved_positions(mol_raw, conf, star_to_carbon)

    node_feats = []
    for atom, (x, y, z) in zip(mol_raw.GetAtoms(), coords):
        node_feats.append([
            float(atom.GetAtomicNum()),
            float(atom.GetTotalDegree()),
            float(atom.GetFormalCharge()),
            float(int(atom.GetIsAromatic())),
            x,
            y,
            z,
        ])

    edge_index = [[], []]
    edge_attr = []
    for bond in mol_raw.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        pos_i = coords[i]
        pos_j = coords[j]
        length = math.dist(pos_i, pos_j)

        feat = [
            float(bond.GetBondTypeAsDouble()),
            float(int(bond.GetIsConjugated())),
            float(int(bond.IsInRing())),
            length,
        ]
        edge_index[0].extend([i, j])
        edge_index[1].extend([j, i])
        edge_attr.extend([feat, feat])

    return {
        "node_feats": pd.DataFrame(node_feats, columns=["atomic_num", "degree", "formal_charge", "is_aromatic", "x", "y", "z"]),
        "edge_index": pd.DataFrame(edge_index, index=["row", "col"]),
        "edge_attr": pd.DataFrame(edge_attr, columns=["bond_type", "is_conjugated", "is_in_ring", "bond_length"])
    }
def convert_csv_to_graphs(
    csv_path: str | Path,
    label_col: str | None,
    PSMILES_col: str,
    save_dir: str | Path,
    skip_failed: bool = True,
) -> Tuple[List[Dict[str, Any]], pd.DataFrame]:
    """
    读取 CSV 中的 PSMILES，批量生成图数据，并附带全局标签（若提供）。

    Parameters
    ----------
    csv_path : str or Path
        CSV 文件路径。
    label_col : str or None
        CSV 中的标签列名；若为 None，则不读取/保存标签。
    PSMILES_col : str
        CSV 中的 PSMILES 列名。
    save_dir : str or Path or None
        若提供，将把每个图保存为 `.npz` 文件，便于后续快速加载。
    skip_failed : bool, default True
        如果单条分子 3D 嵌入失败，是否跳过继续后续样本；否则直接抛出错误。

    Returns
    -------
    graphs : list of dict
        每项是 graph_from_psmiles 的返回结果，若提供标签则额外包含键 `label` 与 `mol_id`。
    manifest : pandas.DataFrame
        索引信息，含 `mol_id`、节点/边数量以及（如保存）对应文件路径；若提供标签则包含 `label` 列。
    """
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)

    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

    graphs: List[Dict[str, Any]] = []
    manifest_rows: List[Dict[str, Any]] = []
    failed: List[Tuple[int, str]] = []

    iterator = tqdm(
        df.itertuples(index=True),
        total=len(df),
        desc="Converting PSMILES to graphs",
    )

    for idx, row in enumerate(iterator):
        # 获取 PSMILES
        psmiles = getattr(row, PSMILES_col, None)
        if not isinstance(psmiles, str) or not psmiles.strip():
            continue

        # 尝试获取标签（如果用户提供了列名并且该行包含该属性）
        label_value = None
        if label_col is not None and hasattr(row, label_col):
            raw_label = getattr(row, label_col)
            # 尝试将标签转为 float（如果为缺失会抛出或转为 nan）
            try:
                label_value = float(raw_label)
            except Exception:
                label_value = None

        try:
            graph = graph_from_psmiles(psmiles)
        except Exception as e:
            if skip_failed:
                failed.append((row.Index, str(e)))
                continue
            raise
        mol_id = row.Index

        graph["mol_id"] = mol_id
        if label_value is not None:
            graph["label"] = label_value
        graphs.append(graph)

        record: Dict[str, Any] = {
            "mol_id": mol_id,
            "num_nodes": len(graph["node_feats"]),
            "num_edges": len(graph["edge_attr"]) // 2,
            "csv_row": row.Index,
        }
        if label_value is not None:
            record["label"] = label_value

        if save_dir is not None:
            batch_idx = idx // 1000
            batch_dir = save_dir / f"batch_{batch_idx}"
            batch_dir.mkdir(parents=True, exist_ok=True)
            file_path = batch_dir / f"{mol_id}.npz"

            save_dict = {
                "node_feats": graph["node_feats"].to_numpy(dtype=np.float32),
                "edge_index": graph["edge_index"].to_numpy(dtype=np.int64),
                "edge_attr": graph["edge_attr"].to_numpy(dtype=np.float32),
            }
            if label_value is not None:
                save_dict["label"] = np.array([label_value], dtype=np.float32)

            np.savez_compressed(file_path, **save_dict)
            record["file_path"] = str(file_path)

        manifest_rows.append(record)

    manifest = pd.DataFrame(manifest_rows)
    pd.DataFrame.to_csv(manifest, save_dir / "manifest.csv" , index=False)
    if failed:
        tqdm.write(f"Skipped {len(failed)} molecules due to 3D embed failures.")
        tqdm.write("Example failure: " + str(failed[0]))
    return graphs, manifest

