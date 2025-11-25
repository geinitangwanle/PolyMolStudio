"""
Convenience helpers for building PyG DataLoaders using the polyGeoGAT dataset.
"""

from typing import Tuple
from torch_geometric.loader import DataLoader

from .dataset import GraphDataset


def make_dataloader(
    manifest,
    root_dir,
    *,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = True,
):
    dataset = GraphDataset(
        manifest=manifest,
        root=root_dir,
        separate_pos=True,
        feature_cols=(0, 1, 2, 3),
        coord_cols=(4, 5, 6),
        standardize_y=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return loader, dataset


__all__ = ["make_dataloader"]
