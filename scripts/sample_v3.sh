#!/bin/bash

#SBATCH --job-name=sample_v3 # 作业名

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
srun python ./utils/unified_cli.py sample --version v3 -- --checkpoint checkpoints/modelv3_tg.pt \
    --polybert-dir ./polybert \
    --target-tg 350 450 \
    --num-per-target 128 \
    --max-len 256 \
    --temperature 1.0 \
    --top-k "" \
    --top-p "" \
    --output-dir outputs_tg \
    --samples-file samples_tg.csv \
    --metrics-file metrics_tg.json
