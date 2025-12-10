#!/bin/bash

#SBATCH --job-name=finetune_v4 # 作业名

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
srun python ./utils/unified_cli.py train --version v4 --mode finetune -- --csv data/raw/PSMILES_Tg_only.csv \
    --col-smiles PSMILES \
    --col-tg Tg \
    --epochs 20 \
    --batch-size 128 \
    --max-len 256 \
    --lr 1.5e-4 \
    --polybert-lr 5e-6 \
    --polybert-train-last-n 2 \
    --dropout 0.1 \
    --lambda-tg 0.5 \
    --pretrained checkpoints/pretrain_modelv4.pt \
    --polybert-dir ./polybert \
    --output checkpoints/finetune_tg_modelv4.pt \
    --num-workers 8 \
    --model-size base
