# PolyMolStudio

## Overview 概览
- Generative models (VAE v1–v4) for polymer pSMILES, with pretrain + Tg finetune workflows.
- Tg predictor (GeoGAT) for property scoring and design.
- One-click design pipeline: generate → graph conversion → Tg prediction → export.

## Environment 环境
- Python 3.8+ (tested with `py38` conda env).
- Key deps: PyTorch, torch_geometric/torch_sparse stack, transformers, pandas, rdkit, tqdm.
- Place `polybert` weights under `./polybert` (or use a HF model name).

## Training 训练
- 预训练 v4: `python utils/unified_cli.py train --version v4 --mode pretrain -- --csv data/raw/PI1M_v2_psmiles.csv`
- 微调 v4 Tg: `python utils/unified_cli.py train --version v4 --mode finetune -- --csv data/raw/PSMILES_Tg_only.csv --pretrained checkpoints/pretrain_modelv4.pt`
- 训练 v3 Tg: `python utils/unified_cli.py train --version v3 -- --csv data/raw/PSMILES_Tg_only.csv`
- 训练 v2 / v1: `python utils/unified_cli.py train --version v2` / `python utils/unified_cli.py train --version v1`
- 训练 Tg 预测器: `python train/train_predictor.py --data_path <manifest.csv> --root_dir <npz_root>`

## Sampling 采样
- v4 无条件: `python utils/unified_cli.py sample --version v4 --mode uncond -- --checkpoint checkpoints/pretrain_modelv4.pt`
- v4 Tg 条件: `python utils/unified_cli.py sample --version v4 --mode tg -- --checkpoint checkpoints/finetune_tg_modelv4.pt --target-tg 350 450`
- v3 Tg: `python utils/unified_cli.py sample --version v3 -- --checkpoint checkpoints/modelv3_tg.pt --target-tg 350 450`
- v1/v2: `python utils/unified_cli.py sample --version v2 -- --checkpoint checkpoints/modelv2_best.pt`
- 预测器推理: `python predict/predict.py --ckpt_path <checkpoint.pt> --csv_path <input.csv> --psmiles_col <col> --save_dir pred_graphs`

## Design pipeline 设计流水线
- 生成并评分（v4 生成器 + Tg 预测器）:
```
python design/generate_and_score.py \
  --gen-checkpoint checkpoints/pretrain_modelv4.pt \
  --polybert-dir ./polybert \
  --model-size base \
  --num-samples 20 \
  --batch-size 20 \
  --max-len 256 \
  --temperature 0.7 \
  --top-k 50 \
  --top-p 30 \
  --seed 42 \
  --predictor-ckpt checkpoints/predictor.pt \
  --predict-batch-size 32 \
  --num-workers 0 \
  --predict-device cpu \
  --output-dir design_outputs \
  --samples-file samples_pretrain_base.csv \
  --scored-file scored_pretrain_base.csv
```
- 输出：`samples-file` 为生成的 pSMILES；`scored-file` 为合并 Tg 预测的结果。

## Data 数据
- 示例 CSV 在 `data/raw/`；按需调整 `--csv` 路径。
- 采样时若需新奇度指标，设置 `--data-csv <train_csv>`；留空则跳过。

## Tips 小贴士
- torch_sparse 在 MPS 上有限，预测器默认可用 `--predict-device cpu`；如有 CUDA 可设为 `auto/cuda`。
- 检查 `checkpoints/` 路径是否存在，确保 polyBERT 路径正确。
- 支持 PolyBERT 本地目录或 HF 名称。
