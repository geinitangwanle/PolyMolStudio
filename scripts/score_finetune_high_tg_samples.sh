#!/bin/bash

#SBATCH --job-name=score_highTg_gen
#SBATCH --partition=A800
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --gres=gpu:a800:1
#SBATCH --output=log/%j.out
#SBATCH --error=log/%j.err

set -euo pipefail

mkdir -p log

CONDA_SH=${CONDA_SH:-/share/home/u23514/apps/miniconda3/etc/profile.d/conda.sh}
CONDA_ENV=${CONDA_ENV:-py38forGNN}

INPUT_DIR=${INPUT_DIR:-demo/finetune_high_Tg}
OUT_DIR=${OUT_DIR:-demo/finetune_high_Tg/scored}
PSMILES_COL=${PSMILES_COL:-smiles}
BATCH_SIZE=${BATCH_SIZE:-64}
NUM_WORKERS=${NUM_WORKERS:-7}

# Default: use the two main trained predictors as a small ensemble.
# You can override with glob patterns, for example:
#   CKPT_PATHS="ensemble_checkpoints/ensemble_graph/seed*.pt ensemble_checkpoints/ensemble_multi/seed*.pt"
CKPT_PATHS=${CKPT_PATHS:-"checkpoints/graph-predictor.pt checkpoints/multi-predictor.pt"}

source "$CONDA_SH"
conda activate "$CONDA_ENV"

mkdir -p "$OUT_DIR"

for split in uncond cond; do
    input_csv="$INPUT_DIR/${split}_samples.csv"
    split_out="$OUT_DIR/${split}"

    if [[ ! -f "$input_csv" ]]; then
        echo "Missing input CSV: $input_csv" >&2
        exit 1
    fi

    echo "=== Scoring ${input_csv} ==="
    srun python ./predict/predict_ensemble.py \
        --ckpt_paths $CKPT_PATHS \
        --csv_path "$input_csv" \
        --psmiles_col "$PSMILES_COL" \
        --save_dir "$split_out/graphs" \
        --batch_size "$BATCH_SIZE" \
        --num_workers "$NUM_WORKERS" \
        --device auto \
        --out_csv "$OUT_DIR/${split}_samples_scored.csv" \
        --summary_json "$OUT_DIR/${split}_score_summary.json"
done

echo "Done."
echo "Scored outputs:"
echo "  $OUT_DIR/uncond_samples_scored.csv"
echo "  $OUT_DIR/cond_samples_scored.csv"
