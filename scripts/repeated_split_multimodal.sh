#!/bin/bash

#SBATCH --job-name=repeat_multi
#SBATCH --partition=A800
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a800:1
#SBATCH --output=log/%j_repeat_multi.out
#SBATCH --error=log/%j_repeat_multi.err

source /share/home/u23514/apps/miniconda3/etc/profile.d/conda.sh
conda activate py38forGNN

# Repeated stratified random splits for CrossAttention-PolyGeoGAT.
# Edit data_path/root_dir if your current graph manifest is elsewhere.
python ./scripts/repeated_split_train_predictor.py \
  --split_seeds 1-10 \
  --model_seed_mode same_as_split \
  --run_prefix crossattn_repeated \
  --summary_dir analysis_outputs/repeated_split_crossattn \
  --keep_going \
  -- \
  --data_path data/processed/graphs_tg/manifest.csv \
  --root_dir data/processed/graphs_tg \
  --batch_size 32 \
  --epochs 50 \
  --lr 1e-3 \
  --weight_decay 1e-4 \
  --device auto \
  --use_polybert \
  --polybert-dir ./polybert \
  --polybert_lr 1e-5 \
  --seq_max_length 256 \
  --cross_attn_heads 4 \
  --cross_attn_dim 64
