#!/bin/bash

#SBATCH --job-name=pretrain_v4 # 作业名

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
srun python ./utils/unified_cli.py train --version v4 --mode pretrain -- --csv data/raw/PI1M_v2_psmiles.csv \
    --col PSMILES \
    --epochs 10 \
    --batch-size 256 \
    --max-len 256 \
    --lr 3e-4 \
    --polybert-lr 1e-5 \
    --polybert-train-last-n 2 \
    --dropout 0.1 \
    --output checkpoints/pretrain_modelv4.pt \
    --polybert-dir ./polybert \
    --num-workers 8 \
    --model-size premium
