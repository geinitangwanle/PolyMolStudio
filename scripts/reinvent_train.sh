#!/bin/bash

#SBATCH --job-name=reinvent_train # 作业名

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
srun python ./train/RL/reinvent_train.py \
  --data data/raw/molecules.csv \
  --polybert_name ./polybert \
  --epochs 5 --batch_size 32 --lr 1e-4 \
  --temperature 0.7 --top_p 0.9 \
  --ckpt checkpoints/pretrain_modelv4.pt \
  --tg_ckpt checkpoints/best_rmse_35.089K_ep032.pt \
  --w_tg 1.0 --w_valid 0.5 --w_sa 0.25 --w_novelty 0.25

