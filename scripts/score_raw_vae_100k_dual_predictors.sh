#!/bin/bash

#SBATCH --job-name=vae100k_dual_score
#SBATCH --partition=A800
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a800:1
#SBATCH --output=log/%j_vae100k_dual_score.out
#SBATCH --error=log/%j_vae100k_dual_score.err

set -euo pipefail

source /share/home/u23514/apps/miniconda3/etc/profile.d/conda.sh
conda activate py38forGNN

mkdir -p log

# Override these when submitting if needed, e.g.
#   sbatch --export=ALL,MODEL1_CKPT=ensemble_checkpoints/ensemble_multi/seed1.pt,MODEL2_CKPT=ensemble_checkpoints/ensemble_graph/seed1.pt scripts/score_raw_vae_100k_dual_predictors.sh
INPUT_CSV=${INPUT_CSV:-analysis_outputs/raw_vae_100k/valid_unique_novel_100k.csv}
MODEL1_CKPT=${MODEL1_CKPT:-checkpoints/predictor.pt}
MODEL2_CKPT=${MODEL2_CKPT:-checkpoints/predictor2.pt}
MODEL1_NAME=${MODEL1_NAME:-multimodal}
MODEL2_NAME=${MODEL2_NAME:-graph}
OUT_DIR=${OUT_DIR:-analysis_outputs/raw_vae_100k_dual_scored}
TMP_DIR=${TMP_DIR:-analysis_outputs/raw_vae_100k_dual_scored/tmp}
CHUNK_SIZE=${CHUNK_SIZE:-5000}
BATCH_SIZE=${BATCH_SIZE:-128}
THRESHOLD_CRITERION=${THRESHOLD_CRITERION:-both}
# Optional: set MAX_DELTA_K=20 to require |model1-model2| <= 20 K in threshold files.
MAX_DELTA_ARG=()
if [[ -n "${MAX_DELTA_K:-}" ]]; then
  MAX_DELTA_ARG=(--max-delta-k "${MAX_DELTA_K}")
fi

mkdir -p "${OUT_DIR}" "${TMP_DIR}"

srun python ./scripts/score_raw_vae_dual_predictors.py \
  --input-csv "${INPUT_CSV}" \
  --smiles-col smiles \
  --ckpt-model1 "${MODEL1_CKPT}" \
  --ckpt-model2 "${MODEL2_CKPT}" \
  --model1-name "${MODEL1_NAME}" \
  --model2-name "${MODEL2_NAME}" \
  --out-dir "${OUT_DIR}" \
  --tmp-dir "${TMP_DIR}" \
  --thresholds 500 600 700 \
  --threshold-criterion "${THRESHOLD_CRITERION}" \
  "${MAX_DELTA_ARG[@]}" \
  --chunk-size "${CHUNK_SIZE}" \
  --batch-size "${BATCH_SIZE}" \
  --num-workers 0 \
  --device auto \
  --trust-validity-column
