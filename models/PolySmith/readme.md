# PolySmith
Transformer/VAE PSMILES generators with Tg‑aware fine‑tuning (v1–v4), now hosted inside the PolyMolStudio monorepo.

**Features**
- HuggingFace polyBERT tokenizer + conditional VAE decoders (v1/v2) and Transformer decoders (v3/v4).
- Optional Tg conditioning (FiLM modulation + syntax mask for validity).
- Unified CLI dispatcher for training and sampling across versions.

**Layout**
- `src/` – core models (v1–v4), datasets, tokenizer, syntax mask utilities.
- `scripts/train` & `scripts/sample` – versioned entrypoints.
- `unified_cli.py` – single CLI wrapper for train/sample.
- `checkpoints/` & `sample_output/` – example artifacts.

**Quickstart (run from repo root)**
- Train v4 pretrain:
  ```bash
  python -m models.PolySmith.unified_cli train --version v4 --mode pretrain -- --csv models/PolySmith/data/PI1M_v2_psmiles.csv
  ```
- Fine‑tune v4 on Tg:
  ```bash
  python -m models.PolySmith.unified_cli train --version v4 --mode finetune -- --csv models/PolySmith/data/PSMILES_Tg_only.csv
  ```
- Sample (Tg‑conditioned v4):
  ```bash
  python -m models.PolySmith.unified_cli sample --version v4 --mode tg -- --checkpoint models/PolySmith/checkpoints/finetune_tg_modelv4.pt
  ```

Extra args go after `--` and are forwarded to the underlying versioned scripts.
