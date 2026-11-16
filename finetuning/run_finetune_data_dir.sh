#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

MODEL_PATH="${MODEL_PATH:-./Qwen3-ASR-1.7B}"
TRAIN_FILE="${TRAIN_FILE:-/cmy/cmy/Qwen3-ASR/data/combined_filtered_with_0614.jsonl}"
EVAL_FILE="${EVAL_FILE:-./data/val.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-./qwen3-asr-finetuning-all-out-$(date +%Y%m%d)}"
CONDA_ENV="${CONDA_ENV:-qwen3-asr}"

python finetuning/qwen3_asr_sft.py \
  --model_path "${MODEL_PATH}" \
  --train_file "${TRAIN_FILE}" \
  --eval_file "${EVAL_FILE}" \
  --output_dir "${OUTPUT_DIR}" \
  --batch_size 8 \
  --grad_acc 8 \
  --lr 1e-5 \
  --epochs 3 \
  --save_steps 200 \
  --num_workers 8 \
  --pin_memory 1 \
  --persistent_workers 1 \
  --gradient_checkpointing 1 \
  --max_audio_seconds 30
