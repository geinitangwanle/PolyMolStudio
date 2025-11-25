"""
Programmatic inference helper for notebooks/interactive use.
"""

from pathlib import Path
from typing import Iterable, Optional
import pandas as pd
import subprocess
import sys
import tempfile


def predict_from_csv(
    ckpt_path: Path,
    csv_path: Path,
    *,
    psmiles_col: str = "PSMILES",
    save_dir: Path | str = "pred_graphs",
    out_csv: Optional[Path] = None,
    device: str = "auto",
) -> pd.DataFrame:
    """
    Run polyGeoGAT.predict on a CSV and return the predictions DataFrame.
    """
    if out_csv is None:
        tmp = tempfile.NamedTemporaryFile(prefix="gnn_preds_", suffix=".csv", delete=False)
        out_csv = Path(tmp.name)
        tmp.close()
    args = [
        sys.executable,
        "-m",
        "models.polyGeoGAT.predict",
        "--ckpt_path",
        str(ckpt_path),
        "--csv_path",
        str(csv_path),
        "--psmiles_col",
        psmiles_col,
        "--save_dir",
        str(save_dir),
        "--out_csv",
        str(out_csv),
        "--device",
        device,
    ]
    subprocess.run(args, check=True)
    return pd.read_csv(out_csv)


__all__ = ["predict_from_csv"]
