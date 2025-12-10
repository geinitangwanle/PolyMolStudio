#!/bin/bash

#SBATCH --job-name=sample_v4_pretrain # 作业名

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
srun python ./utils/unified_cli.py sample --version v4 --mode uncond -- --checkpoint checkpoints/pretrain_modelv4.pt \
    --polybert-dir ./polybert \
    --model-size base \
    --data-csv data/raw/PI1M_v2_psmiles.csv \
    --data-col PSMILES \
    --num-samples 20 \
    --batch-size 20 \
    --max-len 256 \
    --temperature 0.7 \
    --top-k 50 \
    --top-p 30 \
    --seed 42 \
    --output-dir outputs_pretrain \
    --samples-file samples_pretrain_base.csv \
    --metrics-file metrics_pretrain_base.json
