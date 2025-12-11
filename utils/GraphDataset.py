import os
from pathlib import Path
from typing import Any, Optional, Sequence, Union
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data

def load_graph_npz(
    file_path: Union[str, Path],
    *,
    separate_pos: bool = True,
    # 默认你的 node_feats 列为 [atomic_num, degree, formal_charge, is_aromatic, x, y, z]
    # 如果你的特征列不同，可手动指定：
    feature_cols: Optional[Sequence[int]] = (0, 1, 2, 3),
    coord_cols: Optional[Sequence[int]] = (4, 5, 6),
    dtype: torch.dtype = torch.float32,
) -> Data:
    """
    将保存的 .npz 图数据恢复为 PyG 的 Data 对象。

    期望 .npz 内包含键：
      - node_feats: (N, F) float32
      - edge_index: (2, E) int64
      - edge_attr: (E, A) float32
    可选键：
      - label: (1,) float32
    """
    file_path = Path(file_path)
    with np.load(file_path, allow_pickle=False) as npz:
        node_feats = npz["node_feats"]  # (N, F)
        edge_index = npz["edge_index"]  # (2, E)
        edge_attr = npz["edge_attr"]    # (E, A)
        label = npz["label"] if "label" in npz.files else None

    # 转 tensor
    edge_index_t = torch.as_tensor(edge_index, dtype=torch.long)
    edge_attr_t  = torch.as_tensor(edge_attr,  dtype=dtype)

    if separate_pos and coord_cols is not None:
        pos_t = torch.as_tensor(node_feats[:, coord_cols], dtype=dtype)  # (N, 3)
        if feature_cols is not None and len(feature_cols) > 0:
            x_t = torch.as_tensor(node_feats[:, feature_cols], dtype=dtype)  # (N, Ffeat)
        else:
            x_t = None
    else:
        pos_t = None
        x_t = torch.as_tensor(node_feats, dtype=dtype)

    y_t = None
    if label is not None:
        y_t = torch.as_tensor(label, dtype=dtype).view(-1)

    data = Data(
        x=x_t,
        edge_index=edge_index_t,
        edge_attr=edge_attr_t,
        y=y_t,
        pos=pos_t,
    )
    return data

class GraphDataset(Dataset):
    """
    基于 manifest 的 .npz 图数据集。

    manifest 至少包含列：
      - file_path: 指向 .npz 文件的绝对/相对路径
      - label    : (可选) 若 .npz 已含 label，这里可用于快速筛选/统计
    """
    def __init__(
        self,
        manifest: Union[pd.DataFrame, str, Path],
        root: Optional[Union[str, Path]] = None,
        *,
        separate_pos: bool = True, # 将空间坐标与节点特征分开，有利于构建几何模型
        feature_cols: Optional[Sequence[int]] = (0, 1, 2, 3), # 保存的特征为[atomic_num, degree, formal_charge, is_aromatic, x, y, z]
        coord_cols: Optional[Sequence[int]] = (4, 5, 6),
        dtype: torch.dtype = torch.float32,
        standardize_y: bool = False,
        tokenizer: Optional[Any] = None, # 可选：传入 PolyBERT tokenizer，用于序列编码
        psmiles_col: str = "psmiles",
        seq_max_length: int = 256,
    ):
        if isinstance(manifest, (str, Path)):
            manifest = pd.read_csv(manifest)
        self.manifest = manifest.reset_index(drop=True).copy()

        # 允许 root 指定公共前缀路径（manifest 里的 file_path 可以是相对路径）
        self.root = Path(root) if root is not None else None

        # 参数存储
        self.separate_pos = separate_pos
        self.feature_cols = feature_cols
        self.coord_cols = coord_cols
        self.dtype = dtype
        self.standardize_y = standardize_y
        self.tokenizer = tokenizer
        self.psmiles_col = psmiles_col
        self.seq_max_length = seq_max_length

        # 预先计算 y 的均值方差（若启用标准化）
        self._y_mean = None
        self._y_std = None
        if self.standardize_y:
            y = None
            # 优先从 manifest 获取（若存在且非空）
            if "label" in self.manifest and self.manifest["label"].notna().any():
                y = self.manifest["label"].astype(float).dropna().to_numpy()
            else:
                # 回退到逐个文件读取（只收集存在 label 的文件）
                ys = []
                for fp in self.manifest["file_path"]:
                    full_path = self._resolve_path(fp)
                    try:
                        with np.load(full_path, allow_pickle=False) as npz:
                            if "label" in npz.files:
                                ys.append(float(npz["label"][0]))
                    except Exception:
                        # 忽略无法读取的文件
                        continue
                if len(ys) > 0:
                    y = np.array(ys, dtype=np.float32)

            if y is None or len(y) == 0:
                raise ValueError("standardize_y=True but no labels found in manifest or npz files.")
            self._y_mean = float(np.mean(y))
            self._y_std = float(np.std(y) + 1e-8)
    
    # 把 manifest 里的 file_path 解析成可读的真实路径
    def _resolve_path(self, fp: str) -> Path:
        p = Path(fp)
        if p.is_absolute():
            return p
        if self.root is not None:
            # 若 manifest 里已经带了 root 前缀，避免重复拼接
            p_str = p.as_posix()
            root_str = self.root.as_posix()
            if p_str.startswith(root_str):
                return p
            return self.root / p
        return p

    def __len__(self) -> int:
        return len(self.manifest)

    # 将数据集中的每个样本转换为 PyG 的 Data 对象
    def __getitem__(self, idx: int) -> Data:
        row = self.manifest.iloc[idx]
        file_path = self._resolve_path(row["file_path"])

        data = load_graph_npz(
            file_path,
            separate_pos=self.separate_pos,
            feature_cols=self.feature_cols,
            coord_cols=self.coord_cols,
            dtype=self.dtype,
        )

        # 可选：标准化标签（训练常用）
        if self.standardize_y and data.y is not None:
            data.y = (data.y - self._y_mean) / self._y_std

        # 可选：编码 pSMILES 序列（用于 cross-attention）
        if self.tokenizer is not None:
            if self.psmiles_col not in row.index:
                raise KeyError(f"pSMILES column '{self.psmiles_col}' not found in manifest; cannot tokenize.")
            psmiles = row[self.psmiles_col]
            if not isinstance(psmiles, str):
                raise ValueError(f"Invalid pSMILES at index {idx}: {psmiles!r}")
            base_tokenizer = getattr(self.tokenizer, "tokenizer", self.tokenizer)
            tokens = base_tokenizer(
                psmiles,
                padding="max_length",
                truncation=True,
                max_length=self.seq_max_length,
                return_tensors="pt",
            )
            data.seq_input_ids = tokens["input_ids"].squeeze(0)
            data.seq_attention_mask = tokens["attention_mask"].squeeze(0)

        # 附加元信息（可选）
        data.mol_id = row.get("mol_id", idx)
        data.file_path = str(file_path)
        return data

    @property
    def y_mean(self) -> Optional[float]:
        return self._y_mean

    @property
    def y_std(self) -> Optional[float]:
        return self._y_std
