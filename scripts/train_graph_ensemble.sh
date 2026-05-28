#!/bin/bash

#SBATCH --job-name=graph_ens # 作业名

#SBATCH --partition=A800 # A800 队列

#SBATCH -N 1

#SBATCH --ntasks-per-node=1

#SBATCH --cpus-per-task=4 # 1:4 的 GPU:CPU 配比

#SBATCH --gres=gpu:a800:1 # 1 块 GPU

#SBATCH --output=log/%j_graph_ens.out

#SBATCH --error=log/%j_graph_ens.err


# 加载conda环境变量
source /share/home/u23514/apps/miniconda3/etc/profile.d/conda.sh

# 加载你的虚拟环境
conda activate py38forGNN

# 必须使用新版 train_predictor.py：固定数据划分，只改变模型随机种子。
python ./train/train_predictor.py --help | grep -q -- "--split_seed" || {
  echo "ERROR: train_predictor.py does not support --split_seed/--model_seed."
  echo "Please sync the updated train/train_predictor.py before running ensemble training."
  exit 1
}

# 固定 split_seed，改变 model_seed 训练 ensemble
for MODEL_SEED in 1 2 3 4 5
do
  echo "Training graph-only ensemble member with MODEL_SEED=${MODEL_SEED}"

  srun python ./train/train_predictor.py \
    --data_path data/graphs_tg2/manifest.csv \
    --root_dir data/graphs_tg2 \
    --batch_size 32 \
    --epochs 50 \
    --lr 1e-3 \
    --weight_decay 1e-4 \
    --split_seed 42 \
    --model_seed ${MODEL_SEED} \
    --run_name graph_seed${MODEL_SEED} \
    --checkpoint_dir checkpoints/ensemble_graph \
    --log_dir logs/ensemble_graph \
    --device auto
done
