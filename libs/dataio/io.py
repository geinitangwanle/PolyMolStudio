"""
Basic data IO helpers.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple
import pandas as pd
from sklearn.model_selection import train_test_split


def load_csv(path: Path):
    return pd.read_csv(path)


def save_csv(df, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def train_val_test_split(df, *, train_frac: float = 0.8, val_frac: float = 0.1, seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df, temp_df = train_test_split(df, train_size=train_frac, random_state=seed, shuffle=True)
    val_ratio = val_frac / (1 - train_frac)
    val_df, test_df = train_test_split(temp_df, test_size=(1 - val_ratio), random_state=seed, shuffle=True)
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)
