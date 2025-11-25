# PolyMolStudio
Unified home for two polymer ML toolkits:
- `models/polyGeoGAT`: geometry‑aware GAT for Tg prediction (graph‑based property predictor).
- `models/PolySmith`: PSMILES generators (v1–v4) with Tg conditioning and syntax masking.

## Repo layout
- `models/polyGeoGAT/` – graph construction, GeoGAT model, training & prediction CLIs.
- `models/PolySmith/` – VAE/Transformer generators, versioned scripts, unified CLI dispatcher.
- `requirements.txt` – combined dependency entry point (`pip install -r requirements.txt`).

## Quickstart
- Install deps (inside your venv):
  ```bash
  pip install -r requirements.txt
  ```

- PolyGeoGAT (from repo root):
  ```bash
  python -m models.polyGeoGAT.build_graphs --csv_path models/polyGeoGAT/datasets/Train.csv --label_col label --PSMILES_col PSMILES --save_dir models/polyGeoGAT/graph
  python -m models.polyGeoGAT.train --data_path models/polyGeoGAT/graph/manifest.csv --root_dir models/polyGeoGAT --log_dir models/polyGeoGAT/logs --checkpoint_dir models/polyGeoGAT/checkpoints
  python -m models.polyGeoGAT.predict --ckpt_path models/polyGeoGAT/checkpoints/train_*/best_rmse_*.pt --csv_path models/polyGeoGAT/datasets/Testdataset.csv --psmiles_col PSMILES --out_csv models/polyGeoGAT/preds.csv
  ```

- PolySmith generators (from repo root):
  ```bash
  python -m models.PolySmith.unified_cli train --version v4 --mode pretrain -- --csv models/PolySmith/data/PI1M_v2_psmiles.csv
  python -m models.PolySmith.unified_cli train --version v4 --mode finetune -- --csv models/PolySmith/data/PSMILES_Tg_only.csv
  python -m models.PolySmith.unified_cli sample --version v4 --mode tg -- --checkpoint models/PolySmith/checkpoints/finetune_tg_modelv4.pt
  ```

More details live in each submodule README (`models/polyGeoGAT/readme.md`, `models/PolySmith/readme.md`).

## Monorepo layout (modular skeleton)
- `models/gnn_predictor/` and `models/psmiles_generator/` wrap the existing models for reuse.
- `libs/` common utilities (chem utils, tokenizers, metrics, vis, dataio).
- `configs/` YAML configs (`gnn/`, `psmiles/`, `common/`, `pipeline.yaml`).
- `scripts/` entrypoints: `train_gnn.py`, `eval_gnn.py`, `train_psmiles.py`, `sample_psmiles.py`, `pipeline_generate_and_predict.py`.
- `data/` (raw/processed, keep only small samples), `outputs/` (gnn/psmiles/pipeline), `tests/`, `docs/`.

Sample commands for the new wrappers:
```bash
python scripts/train_gnn.py --config configs/gnn/train.yaml
python scripts/eval_gnn.py --config configs/gnn/eval.yaml
python scripts/train_psmiles.py --config configs/psmiles/train.yaml
python scripts/sample_psmiles.py --config configs/psmiles/sample.yaml
python scripts/pipeline_generate_and_predict.py --config configs/pipeline.yaml
```
