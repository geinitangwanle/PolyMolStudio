#!/bin/bash

#SBATCH --job-name=repeat_graph
#SBATCH --partition=A800
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a800:1
#SBATCH --output=log/%j_repeat_graph.out
#SBATCH --error=log/%j_repeat_graph.err

source /share/home/u23514/apps/miniconda3/etc/profile.d/conda.sh
conda activate py38forGNN

# Repeated stratified random splits for graph-only PolyGeoGAT baseline.
# Use the same split seeds as repeated_split_multimodal.sh for paired comparison.
python ./scripts/repeated_split_train_predictor.py \
  --split_seeds 1-10 \
  --model_seed_mode same_as_split \
  --run_prefix graph_repeated \
  --summary_dir analysis_outputs/repeated_split_graph \
  --keep_going \
  -- \
  --data_path data/processed/graphs_tg/manifest.csv \
  --root_dir data/processed/graphs_tg \
  --batch_size 32 \
  --epochs 50 \
  --lr 1e-3 \
  --weight_decay 1e-4 \
  --device auto
