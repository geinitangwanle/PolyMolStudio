# PolyGeoGAT
Geometry‑aware Graph Attention for Polymer Tg Prediction

**Overview**
- Convert PSMILES to graphs with 3D coordinates, then learn Tg via a geometry‑enhanced GNN combining GatedGCN and GAT.

**Highlights**
- SGeometry features: RDKit 3D coords, distance RBF, and angle triplets concatenated with original bond features.
- Attention + gating: GAT with edge features plus GatedGraphConv backbone.
- Clean pipeline: CSV → graphs files → PyG dataset; best‑val checkpoint and final test using the best model.

**Structure**
- `data/PSMILES_to_graph.py` – PSMILES → graph NPZ and `manifest.csv`.
- `data/GraphDataset.py` – NPZ graphs → PyG `Data` loader.
- `model/GeoGATModel.py` – geometry‑aware GAT model.
- `train.py` – train/val/test split, logging, checkpoints.
- `predict.py` – build graphs from unlabeled CSV and write predictions.

**Usage**
- Data conversion (from CSV with PSMILES)
  ```bash
  python data.py \
    --csv_path datasets/Train.csv \
    --label_col label \
    --PSMILES_col PSMILES \
    --save_dir graph
  ```

- Train
  ```bash
  python train.py \
    --data_path ./graph/manifest.csv \
    --root_dir . \
    --epochs 50 \
    --batch_size 32 \
    --device auto \
    --log_dir ./logs \
    --checkpoint_dir ./checkpoints
  ```

- Predict (CSV without labels)
  ```bash
  python predict.py \
    --ckpt_path checkpoints/train_*/best_rmse_*.pt \
    --csv_path datasets/Testdataset.csv \
    --psmiles_col PSMILES \
    --save_dir pred_graphs \
    --out_csv preds.csv \
    --device auto
  ```

Notes
- Training logs: `logs/train_*.log` (includes final test from the best‑val checkpoint).
- Checkpoints: `checkpoints/train_*/best_rmse_*.pt` (with normalization stats for denormed predictions).

Authors
- Tianyu Huang, Wenzhu Bi

License
- MIT — see `LICENSE`
