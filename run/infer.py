"""
Qwen3-ASR inference on a directory of audio files with CER evaluation.

Expected directory structure:
    data_dir/
        utt0001.wav
        utt0001.txt      # reference transcript (plain text, no language prefix)
        utt0002.wav
        utt0002.txt
        ...

Each .wav file must have a matching .txt file with the same stem.

Usage:
    python run/infer.py --data_dir ./test_data
    python run/infer.py --data_dir ./test_data --model Qwen/Qwen3-ASR-0.6B
    python run/infer.py --data_dir ./test_data --model ./qwen3-asr-finetuning-out/checkpoint-200
"""

import argparse
import glob
import os

import torch
from qwen_asr import Qwen3ASRModel


def compute_cer(ref: str, hyp: str) -> tuple:
    """Compute Character Error Rate using edit distance. Returns (errors, ref_len)."""
    r = list(ref.replace(" ", ""))
    h = list(hyp.replace(" ", ""))
    n = len(r)
    m = len(h)
    if n == 0:
        return m, 0
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        d[i][0] = i
    for j in range(m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if r[i - 1] == h[j - 1] else 1
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + cost)
    return d[n][m], n


def main():
    parser = argparse.ArgumentParser(description="Qwen3-ASR inference + CER evaluation")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Directory containing .wav and .txt file pairs")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-ASR-1.7B",
                        help="Model name or local checkpoint path")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Max inference batch size")
    args = parser.parse_args()

    # Collect wav/txt pairs
    wav_files = sorted(glob.glob(os.path.join(args.data_dir, "*.wav")))
    if not wav_files:
        print(f"No .wav files found in {args.data_dir}")
        return

    pairs = []
    for wav_path in wav_files:
        txt_path = os.path.splitext(wav_path)[0] + ".txt"
        if not os.path.exists(txt_path):
            print(f"[skip] no matching .txt for {os.path.basename(wav_path)}")
            continue
        with open(txt_path, "r", encoding="utf-8") as f:
            ref_text = f.read().strip()
        pairs.append((wav_path, ref_text))

    if not pairs:
        print("No valid wav/txt pairs found.")
        return

    print(f"Found {len(pairs)} audio/text pairs in {args.data_dir}")

    # Load model
    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    model = Qwen3ASRModel.from_pretrained(
        args.model,
        dtype=dtype_map[args.dtype],
        device_map=args.device,
        max_inference_batch_size=args.batch_size,
    )

    # Inference
    audio_paths = [p[0] for p in pairs]
    ref_texts = [p[1] for p in pairs]

    results = model.transcribe(
        audio=audio_paths,
        language="Chinese",
    )

    # Compute CER
    total_errors = 0
    total_ref_len = 0
    for i, (r, ref) in enumerate(zip(results, ref_texts)):
        hyp = r.text
        errors, ref_len = compute_cer(ref, hyp)
        total_errors += errors
        total_ref_len += ref_len
        cer_i = errors / ref_len * 100 if ref_len > 0 else 0.0
        name = os.path.basename(audio_paths[i])
        print(f"  {name}  CER={cer_i:.2f}%  ref={ref}  hyp={hyp}")

    overall_cer = total_errors / total_ref_len * 100 if total_ref_len > 0 else 0.0
    print(f"\n===== Results =====")
    print(f"Total utterances: {len(pairs)}")
    print(f"Overall CER:      {overall_cer:.2f}%")
    print(f"Total errors:     {total_errors}")
    print(f"Total ref chars:  {total_ref_len}")


if __name__ == "__main__":
    main()
