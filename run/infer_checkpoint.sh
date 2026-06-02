#!/usr/bin/env bash
# Usage: bash /path/to/infer_checkpoint.sh [options]
# Run inference on a finetuned checkpoint with sensible defaults.
# Can be called from any directory.

set -euo pipefail

# ---- Resolve script location so it works from anywhere ----
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# ---- Defaults (override via env vars) ----
MODEL="${MODEL:-/cmy/cmy/Qwen3-ASR/qwen3-asr-finetuning-all-out-20260524/checkpoint-3948}"
INPUT_JSONL="${INPUT_JSONL:-/cmy/cmy/Qwen3-ASR/data/test.jsonl}"
# INPUT_JSONL="${INPUT_JSONL:-data/train_augmented_clean_filtered.jsonl}"
# RESULT_FILE="${RESULT_FILE:-${REPO_ROOT}/outputs/infer_$(date +%Y%m%d).jsonl}"
RESULT_FILE="${RESULT_FILE:-${REPO_ROOT}/outputs/infer_$(date +%Y%m%d)_train_finetune_again.jsonl}"
DEVICE="${DEVICE:-cuda:0}"
DTYPE="${DTYPE:-bfloat16}"
BATCH_SIZE="${BATCH_SIZE:-32}"
LANGUAGE="${LANGUAGE:-Chinese}"
CONTEXT="${CONTEXT:-}"

# ---- Parse named args (override env vars) ----
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)        MODEL="$2";        shift 2 ;;
    --input)        INPUT_JSONL="$2";  shift 2 ;;
    --output)       RESULT_FILE="$2";  shift 2 ;;
    --device)       DEVICE="$2";       shift 2 ;;
    --dtype)        DTYPE="$2";        shift 2 ;;
    --batch_size)   BATCH_SIZE="$2";   shift 2 ;;
    --language)     LANGUAGE="$2";     shift 2 ;;
    --context)      CONTEXT="$2";      shift 2 ;;
    -h|--help)
      echo "Usage: $0 [options]"
      echo "  --model PATH      Finetuned checkpoint dir  (env: MODEL)"
      echo "  --input FILE      Input JSONL path          (env: INPUT_JSONL)"
      echo "  --output FILE     Result JSONL path         (env: RESULT_FILE)"
      echo "  --device STR      CUDA device               (env: DEVICE, default: cuda:0)"
      echo "  --dtype STR       Model dtype               (env: DTYPE, default: bfloat16)"
      echo "  --batch_size N    Batch size                (env: BATCH_SIZE, default: 32)"
      echo "  --language STR    Language hint             (env: LANGUAGE, default: Chinese)"
      echo "  --context STR     System prompt context     (env: CONTEXT)"
      echo ""
      echo "All options also settable via environment variables."
      echo "Defaults assume checkpoint: qwen3-asr-finetuning-out-YYYYMMDD/checkpoint-latest"
      exit 0 ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

echo "=== Qwen3-ASR Inference ==="
echo "  Model:       ${MODEL}"
echo "  Input:       ${INPUT_JSONL}"
echo "  Output:      ${RESULT_FILE}"
echo "  Device:      ${DEVICE}"
echo "  dtype:       ${DTYPE}"
echo "  batch_size:  ${BATCH_SIZE}"
echo "  language:    ${LANGUAGE}"
echo "  context:     ${CONTEXT:-"(none)"}"

# Create output directory
mkdir -p "$(dirname "${RESULT_FILE}")"

exec python "${SCRIPT_DIR}/infer.py" \
    --input_jsonl "${INPUT_JSONL}" \
    --model "${MODEL}" \
    --device "${DEVICE}" \
    --dtype "${DTYPE}" \
    --batch_size "${BATCH_SIZE}" \
    --language "${LANGUAGE}" \
    ${CONTEXT:+--context "${CONTEXT}"} \
    --result_file "${RESULT_FILE}"
    --skip_cal_wer