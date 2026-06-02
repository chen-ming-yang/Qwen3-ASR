SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
JSONL_FILE="${1:-/cmy/cmy/Qwen3-ASR/outputs/infer_test_finetune_502_400.jsonl}"
OUT_DIR="${2:-wer_inputs_len2_502_400}"

mkdir -p "$OUT_DIR"
python "$SCRIPT_DIR/jsonl_to_compute_wer.py" "$JSONL_FILE" \
  --out-dir "$OUT_DIR" \
  --ref-out ref.txt \
  --hyp-out hyp.txt \
  --min-ref-chars 2 \
  --strip-hyp-punctuation \
  --run-compute-wer \
  --compute-wer-bin /cmy/cmy/miniconda3/envs/qwen3-asr/bin/compute-wer
# Store a copy of the input JSONL alongside WER results
cp "$JSONL_FILE" "$OUT_DIR/"
