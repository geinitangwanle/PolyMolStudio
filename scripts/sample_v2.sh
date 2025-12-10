#!/bin/bash

#SBATCH --job-name=sample_v2 # 作业名

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
srun python ./utils/unified_cli.py sample --version v2 -- --checkpoint checkpoints/modelv2_best.pt \
    --polybert-dir ./polybert \
    --model-version v2 \
    --data-csv data/PSMILES_Tg_only.csv \
    --data-col PSMILES \
    --num-samples 512 \
    --max-len 256 \
    --seed 42 \
    --output-dir outputs \
    --samples-file sampled_smiles.csv \
    --metrics-file sample_metrics.json
