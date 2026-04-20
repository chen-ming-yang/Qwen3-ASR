#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

MODEL_PATH="${MODEL_PATH:-./Qwen3-ASR-1.7B}"
DATA_DIR="${DATA_DIR:-./data}"
OUTPUT_DIR="${OUTPUT_DIR:-./qwen3-asr-finetuning-out}"

python finetuning/qwen3_asr_sft.py \
  --model_path "${MODEL_PATH}" \
  --data_dir "${DATA_DIR}" \
  --output_dir "${OUTPUT_DIR}" \
  --batch_size 32 \
  --grad_acc 16 \
  --lr 2e-5 \
  --epochs 2 \
  --save_steps 200 \
  --num_workers 0 \
  --pin_memory 0 \
  --persistent_workers 0 \
  --gradient_checkpointing 1 \
  --max_audio_seconds 60