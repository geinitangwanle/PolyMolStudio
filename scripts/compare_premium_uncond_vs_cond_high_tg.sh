#!/bin/bash

#SBATCH --job-name=v4_prem_cmp
#SBATCH --partition=A800
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --gres=gpu:a800:1
#SBATCH --output=log/%j.out
#SBATCH --error=log/%j.err

set -euo pipefail

mkdir -p log checkpoints outputs_compare/high_tg_premium

# Compare two high-Tg adaptation strategies from the same premium pretrain:
#   1) unconditional continued pretraining on demo/high_Tg.csv
#   2) Tg-conditional finetuning on demo/high_Tg.csv
#
# Override examples:
#   sbatch --export=ALL,FORCE_PRETRAIN=0,COND_EPOCHS=30 scripts/compare_premium_uncond_vs_cond_high_tg.sh

CONDA_SH=${CONDA_SH:-/share/home/u23514/apps/miniconda3/etc/profile.d/conda.sh}
CONDA_ENV=${CONDA_ENV:-py38forGNN}

PRETRAIN_CSV=${PRETRAIN_CSV:-data/raw/PI1M_v2_psmiles.csv}
ADAPT_CSV=${ADAPT_CSV:-demo/high_Tg.csv}
SMILES_COL=${SMILES_COL:-PSMILES}
TG_COL=${TG_COL:-Tg}
POLYBERT_DIR=${POLYBERT_DIR:-./polybert}

PRETRAIN_CKPT=${PRETRAIN_CKPT:-checkpoints/pretrain_modelv4_premium_PI1M.pt}
UNCOND_CKPT=${UNCOND_CKPT:-checkpoints/pretrain_modelv4_premium_highTg_uncond.pt}
COND_CKPT=${COND_CKPT:-checkpoints/finetune_tg_modelv4_premium_highTg_cond.pt}

FORCE_PRETRAIN=${FORCE_PRETRAIN:-0}
PRETRAIN_EPOCHS=${PRETRAIN_EPOCHS:-10}
UNCOND_EPOCHS=${UNCOND_EPOCHS:-20}
COND_EPOCHS=${COND_EPOCHS:-20}
PRETRAIN_BATCH=${PRETRAIN_BATCH:-128}
ADAPT_BATCH=${ADAPT_BATCH:-64}
MAX_LEN=${MAX_LEN:-256}
NUM_WORKERS=${NUM_WORKERS:-7}

NUM_UNCOND_SAMPLES=${NUM_UNCOND_SAMPLES:-1000}
TARGET_TG=${TARGET_TG:-700 750}
NUM_PER_TARGET=${NUM_PER_TARGET:-500}
SAMPLE_BATCH=${SAMPLE_BATCH:-100}
SAMPLE_TEMP=${SAMPLE_TEMP:-0.8}
TOP_K=${TOP_K:-50}
TOP_P=${TOP_P:-0.95}
SEED=${SEED:-42}

OUT_DIR=${OUT_DIR:-outputs_compare/high_tg_premium}
UNCOND_SAMPLES=${UNCOND_SAMPLES:-uncond_samples.csv}
UNCOND_METRICS=${UNCOND_METRICS:-uncond_metrics.json}
COND_SAMPLES=${COND_SAMPLES:-cond_samples.csv}
COND_METRICS=${COND_METRICS:-cond_metrics.json}
SUMMARY_CSV=${SUMMARY_CSV:-comparison_summary.csv}

source "$CONDA_SH"
conda activate "$CONDA_ENV"

if [[ "$FORCE_PRETRAIN" == "1" || ! -f "$PRETRAIN_CKPT" ]]; then
    echo "=== Step 1/5: Premium pretraining on ${PRETRAIN_CSV} ==="
    srun python ./utils/unified_cli.py train --version v4 --mode pretrain -- \
        --csv "$PRETRAIN_CSV" \
        --col "$SMILES_COL" \
        --epochs "$PRETRAIN_EPOCHS" \
        --batch-size "$PRETRAIN_BATCH" \
        --max-len "$MAX_LEN" \
        --lr 2e-4 \
        --polybert-lr 5e-6 \
        --polybert-train-last-n 1 \
        --dropout 0.1 \
        --output "$PRETRAIN_CKPT" \
        --polybert-dir "$POLYBERT_DIR" \
        --num-workers "$NUM_WORKERS" \
        --model-size premium
else
    echo "=== Step 1/5: Reusing existing premium pretrain checkpoint ${PRETRAIN_CKPT} ==="
fi

echo "=== Step 2/5: Unconditional high-Tg continued pretraining ==="
srun python ./utils/unified_cli.py train --version v4 --mode pretrain -- \
    --csv "$ADAPT_CSV" \
    --col "$SMILES_COL" \
    --epochs "$UNCOND_EPOCHS" \
    --batch-size "$ADAPT_BATCH" \
    --max-len "$MAX_LEN" \
    --lr 5e-5 \
    --polybert-lr 0 \
    --polybert-train-last-n 0 \
    --dropout 0.1 \
    --pretrained "$PRETRAIN_CKPT" \
    --output "$UNCOND_CKPT" \
    --polybert-dir "$POLYBERT_DIR" \
    --num-workers "$NUM_WORKERS" \
    --model-size premium

echo "=== Step 3/5: Tg-conditional high-Tg finetuning ==="
srun python ./utils/unified_cli.py train --version v4 --mode finetune -- \
    --csv "$ADAPT_CSV" \
    --col-smiles "$SMILES_COL" \
    --col-tg "$TG_COL" \
    --epochs "$COND_EPOCHS" \
    --batch-size "$ADAPT_BATCH" \
    --max-len "$MAX_LEN" \
    --lr 5e-5 \
    --polybert-lr 0 \
    --polybert-train-last-n 0 \
    --dropout 0.1 \
    --lambda-tg 0.5 \
    --pretrained "$PRETRAIN_CKPT" \
    --polybert-dir "$POLYBERT_DIR" \
    --output "$COND_CKPT" \
    --num-workers "$NUM_WORKERS" \
    --model-size premium

echo "=== Step 4/5: Sampling both adapted generators ==="
srun python ./utils/unified_cli.py sample --version v4 --mode uncond -- \
    --checkpoint "$UNCOND_CKPT" \
    --polybert-dir "$POLYBERT_DIR" \
    --model-size premium \
    --data-csv "$ADAPT_CSV" \
    --data-col "$SMILES_COL" \
    --num-samples "$NUM_UNCOND_SAMPLES" \
    --batch-size "$SAMPLE_BATCH" \
    --max-len "$MAX_LEN" \
    --temperature "$SAMPLE_TEMP" \
    --top-k "$TOP_K" \
    --top-p "$TOP_P" \
    --seed "$SEED" \
    --output-dir "$OUT_DIR" \
    --samples-file "$UNCOND_SAMPLES" \
    --metrics-file "$UNCOND_METRICS"

srun python ./utils/unified_cli.py sample --version v4 --mode tg -- \
    --checkpoint "$COND_CKPT" \
    --polybert-dir "$POLYBERT_DIR" \
    --data-csv "$ADAPT_CSV" \
    --data-col "$SMILES_COL" \
    --col-tg "$TG_COL" \
    --model-size premium \
    --target-tg $TARGET_TG \
    --num-per-target "$NUM_PER_TARGET" \
    --batch-size "$SAMPLE_BATCH" \
    --max-len "$MAX_LEN" \
    --temperature "$SAMPLE_TEMP" \
    --top-k "$TOP_K" \
    --top-p "$TOP_P" \
    --seed "$SEED" \
    --output-dir "$OUT_DIR" \
    --samples-file "$COND_SAMPLES" \
    --metrics-file "$COND_METRICS"

echo "=== Step 5/5: Summarizing comparison ==="
srun python scripts/summarize_generator_comparison.py \
    --uncond-metrics "$OUT_DIR/$UNCOND_METRICS" \
    --cond-metrics "$OUT_DIR/$COND_METRICS" \
    --uncond-samples "$OUT_DIR/$UNCOND_SAMPLES" \
    --cond-samples "$OUT_DIR/$COND_SAMPLES" \
    --output "$OUT_DIR/$SUMMARY_CSV"

echo "Done."
echo "Unconditional checkpoint: ${UNCOND_CKPT}"
echo "Conditional checkpoint: ${COND_CKPT}"
echo "Comparison summary: ${OUT_DIR}/${SUMMARY_CSV}"
