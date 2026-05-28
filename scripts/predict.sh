#!/bin/bash

#SBATCH --job-name=predict # 作业名

#SBATCH --partition=A800 # A800 队列

#SBATCH -N 1

#SBATCH --ntasks-per-node=1

#SBATCH --cpus-per-task=4 # 1:4 的 GPU:CPU 配比

#SBATCH --gres=gpu:a800:1 # 1 块 GP

                

#SBATCH --output=log/%j.out

#SBATCH --error=log/%j.err


# 加载conda环境变量
source /share/home/u23514/apps/miniconda3/etc/profile.d/conda.sh

# 加载你的虚拟环境
conda activate py38forGNN

# 执行代码
srun python ./predict/predict.py \
  --ckpt_path checkpoints/best_rmse_35.089K_ep032.pt \
  --csv_path data/raw/MD.csv \
  --psmiles_col PSMILES \
  --batch_size 32 \
  --save_dir pred_graphs \
  --out_csv data/raw/MD_pred.csv


