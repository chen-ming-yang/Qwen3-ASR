#!/bin/bash
# Run the full ASR evaluation pipeline.
# Usage: bash eval/run_all.sh [--max_samples N]
#
# Steps:
#   1. Latency benchmark (RTF, E2E latency P50/P90/P99)
#   2. Resource profiling (peak memory, GPU VRAM)
#   3. Quantization evaluation (BF16 vs INT8 vs INT4)
#   4. Decision report generation

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# Activate the correct conda environment
source /cmy/cmy/miniconda3/bin/activate qwen3-asr

MAX_SAMPLES=${1:-200}
MODEL="${MODEL:-qwen3-asr-finetuning-out-bak/checkpoint-326}"
DEVICE="${DEVICE:-cuda:0}"

echo "============================================"
echo "  Qwen3-ASR Full Evaluation Pipeline"
echo "============================================"
echo "Model: $MODEL"
echo "Device: $DEVICE"
echo "Max samples: $MAX_SAMPLES"
echo ""

# Step 1: Latency benchmark
echo "[1/4] Running latency benchmark..."
python eval/benchmark_latency.py \
    --model "$MODEL" \
    --max_samples "$MAX_SAMPLES" \
    --device "$DEVICE" \
    --output eval/latency_report.json

echo ""

# Step 2: Resource profiling
echo "[2/4] Running resource profiling..."
python eval/profile_resources.py \
    --model "$MODEL" \
    --max_samples 50 \
    --device "$DEVICE" \
    --output eval/resource_report.json

echo ""

# Step 3: Quantization evaluation
echo "[3/4] Running quantization evaluation..."
python eval/quantization_eval.py \
    --model "$MODEL" \
    --max_samples 100 \
    --device "$DEVICE" \
    --output eval/quantization_report.json

echo ""

# Step 4: Generate decision report
echo "[4/4] Generating decision report..."
python eval/generate_decision_report.py \
    --latency_report eval/latency_report.json \
    --resource_report eval/resource_report.json \
    --quant_report eval/quantization_report.json \
    --output eval/decision_report.txt \
    --output_json eval/decision_report.json

echo ""
echo "============================================"
echo "  Evaluation complete!"
echo "  Reports in: eval/"
echo "    - latency_report.json"
echo "    - resource_report.json"
echo "    - quantization_report.json"
echo "    - decision_report.txt"
echo "    - decision_report.json"
echo "============================================"
