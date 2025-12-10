#!/bin/bash

#SBATCH --job-name=sample_v4_finetune # 作业名

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
srun python ./utils/unified_cli.py sample --version v4 --mode tg -- --checkpoint checkpoints/finetune_tg_modelv4.pt \
    --polybert-dir ./polybert \
    --data-csv data/raw/PSMILES_Tg_only.csv \
    --data-col PSMILES \
    --col-tg Tg \
    --model-size base \
    --target-tg 350 450 \
    --num-per-target 50 \
    --batch-size 50 \
    --max-len 256 \
    --temperature 0.7 \
    --top-k 50 \
    --top-p 30 \
    --seed 42 \
    --output-dir outputs_masked \
    --samples-file sampled_smiles_masked.csv \
    --metrics-file sample_metrics_masked.json
