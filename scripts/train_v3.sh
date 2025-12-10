#!/bin/bash

#SBATCH --job-name=train_v3 # 作业名

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
srun python ./utils/unified_cli.py train --version v3 -- --csv data/PSMILES_Tg_only.csv \
    --col-smiles PSMILES \
    --col-tg Tg \
    --polybert-dir ./polybert \
    --epochs 20 \
    --batch-size 128 \
    --max-len 256 \
    --lr 3e-4 \
    --polybert-lr 1e-5 \
    --polybert-train-last-n 0 \
    --weight-decay 0.01 \
    --kl-warmup 10 \
    --lambda-tg 0.1 \
    --seed 42 \
    --output checkpoints/modelv3_tg.pt \
    --num-workers 4 \
    --emb-dim 256 \
    --decoder-hid-dim 512 \
    --z-dim 128 \
    --cond-latent-dim 32 \
    --tg-hidden-dim 128 \
    --dropout 0.1 \
    --train-frac 0.8 \
    --val-frac 0.1
