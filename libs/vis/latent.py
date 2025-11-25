"""
Latent space visualization helpers.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA


def plot_latent_pca(latents: np.ndarray, *, save_path: Optional[Path] = None, title: str = "Latent PCA"):
    latents = np.asarray(latents)
    pca = PCA(n_components=2)
    proj = pca.fit_transform(latents)
    plt.figure(figsize=(6, 5))
    plt.scatter(proj[:, 0], proj[:, 1], s=6, alpha=0.6)
    plt.title(title)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=200)
    return proj
