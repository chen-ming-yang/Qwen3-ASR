#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

MODEL_PATH="${MODEL_PATH:-/cmy/cmy/Qwen3-ASR/qwen3-asr-finetuning-out-20260509/checkpoint-10432}"
TRAIN_FILE="${TRAIN_FILE:-/cmy/cmy/SemScore/output/train_aug_finetune_gt0_55.jsonl}"
EVAL_FILE="${EVAL_FILE:-./data/val.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-./qwen3-asr-lora-out-$(date +%Y%m%d)}"
CONDA_ENV="${CONDA_ENV:-qwen3-asr}"

python finetuning/qwen3_asr_sft_lora.py \
  --model_path "${MODEL_PATH}" \
  --train_file "${TRAIN_FILE}" \
  --eval_file "${EVAL_FILE}" \
  --output_dir "${OUTPUT_DIR}" \
  --batch_size 16 \
  --grad_acc 8 \
  --lr 5e-4 \
  --epochs 2 \
  --save_steps 200 \
  --num_workers 4 \
  --pin_memory 1 \
  --persistent_workers 1 \
  --gradient_checkpointing 1 \
  --max_audio_seconds 30 \
  --lora 1 \
  --lora_r 16 \
  --lora_alpha 32 \
  --lora_dropout 0.05 \
  --lora_target_modules q_proj,k_proj,v_proj,o_proj
