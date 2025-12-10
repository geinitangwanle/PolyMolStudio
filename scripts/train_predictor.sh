#!/bin/bash

#SBATCH --job-name=train_predictor # 作业名

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
srun python train/train_predictor.py --data_path <manifest.csv> \
    --root_dir <npz根目录> \
    --batch_size 32 \
    --test_split 0.1 \
    --val_split 0.1 \
    --seed 42 \
    --epochs 50 \
    --lr 1e-3 \
    --weight_decay 1e-4 \
    --optimizer adamw \
    --loss mse \
    --scheduler cosine \
    --clip_grad_norm 5.0 \
    --device auto \
    --log_dir ./logs \
    --checkpoint_dir ./checkpoints
