#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# Override any path via environment variable, e.g.:
#   TRAIN_FILE=/path/to/train.jsonl MODEL_PATH=/path/to/model ./run_finetune_continue.sh
# TRAIN_FILE="${TRAIN_FILE:-data/train_with_voxflame_converted.jsonl}"
TRAIN_FILE="${TRAIN_FILE:-/cmy/cmy/Qwen3-ASR/data/finetune_64758dee.jsonl}"
EVAL_FILE="${EVAL_FILE:-finetuning/new_data_finetune/val.jsonl}"
MODEL_PATH="${MODEL_PATH:-/cmy/cmy/Qwen3-ASR/qwen3-asr-finetuning-out-20260509/checkpoint-10432}"
OUTPUT_DIR="${OUTPUT_DIR:-./qwen3-asr-finetuning-out-$(date +%Y%m%d)}"

# NOTE: --resume 1 continues from MODEL_PATH checkpoint.
#       --epochs 2 means train to epoch 2.0 (checkpoint-2200 is at epoch ~0.84, so ~1.16 more epochs).
python finetuning/qwen3_asr_sft.py \
  --model_path "${MODEL_PATH}" \
  --train_file "${TRAIN_FILE}" \
  --eval_file "${EVAL_FILE}" \
  --output_dir "${OUTPUT_DIR}" \
  --resume 1 \
  --epochs 20 \
  --batch_size 8 \
  --grad_acc 4 \
  --lr 1e-5 \
  --save_steps 200 \
  --max_audio_seconds 60 \
  --num_workers 0 \
  --persistent_workers 0
