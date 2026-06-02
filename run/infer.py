"""
Qwen3-ASR inference with CER evaluation.

Supports two input layouts:
1) Flat pairs under one directory:
    data_dir/
        utt0001.wav
        utt0001.txt

2) Audio/Text dataset layout:
    dataset_dir/
        Audio/<shard>/<utt_id>.wav
        Text/<shard>_label.txt   # each line: <utt_id> <transcript>

Usage:
    python run/infer.py --data_dir ./test_data
    python run/infer.py --dataset_dir /cmy/after_catting/10h
"""

import argparse
import glob
import json
import os
import re
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple

import numpy as np
import soundfile as sf

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

import torch
from qwen_asr import Qwen3ASRModel


def load_audio(path: str) -> Tuple[np.ndarray, int]:
    """Load a wav file and return (waveform, sample_rate) tuple."""
    wav, sr = sf.read(path, dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    return wav, sr


def preload_audios(
    paths: List[str], num_workers: int = 8
) -> List[Tuple[np.ndarray, int]]:
    """Load audio files in parallel using a thread pool."""
    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        return list(pool.map(load_audio, paths))


PUNCTUATION = set('\u3002\uff0c\u3001\uff1b\uff1a\uff1f\uff01\u201c\u201d\u2018\u2019\u3010\u3011\u300a\u300b\uff08\uff09(){}[],.;:?!\'"\n\r\t\u00b7\u2026\u2014\u2500')


def strip_punctuation(text: str) -> str:
    """Remove common Chinese and English punctuation from text."""
    return "".join(ch for ch in text if ch not in PUNCTUATION)


def collect_pairs_from_flat_dir(data_dir: str) -> List[Tuple[str, str, str]]:
    wav_files = sorted(glob.glob(os.path.join(data_dir, "*.wav")))
    if not wav_files:
        return []

    pairs: List[Tuple[str, str, str]] = []
    for wav_path in wav_files:
        txt_path = os.path.splitext(wav_path)[0] + ".txt"
        if not os.path.exists(txt_path):
            print(f"[skip] no matching .txt for {os.path.basename(wav_path)}")
            continue
        with open(txt_path, "r", encoding="utf-8") as f:
            ref_text = f.read().strip()
        utt_id = os.path.splitext(os.path.basename(wav_path))[0]
        pairs.append((wav_path, ref_text, utt_id))
    return pairs


def collect_pairs_from_audio_text_dir(
    dataset_dir: str,
    audio_subdir: str,
    text_subdir: str,
    strict_audio: bool,
) -> List[Tuple[str, str, str]]:
    audio_root = os.path.join(dataset_dir, audio_subdir)
    text_root = os.path.join(dataset_dir, text_subdir)

    if not os.path.isdir(audio_root):
        raise ValueError(f"Audio directory not found: {audio_root}")
    if not os.path.isdir(text_root):
        raise ValueError(f"Text directory not found: {text_root}")

    txt_files = sorted(
        [
            os.path.join(text_root, fn)
            for fn in os.listdir(text_root)
            if fn.lower().endswith(".txt") and os.path.isfile(os.path.join(text_root, fn))
        ]
    )
    if not txt_files:
        return []

    pairs: List[Tuple[str, str, str]] = []
    missing_audio = 0
    malformed_lines = 0

    for txt_file in txt_files:
        base_name = os.path.basename(txt_file)
        shard_match = re.match(r"^(\d+)", base_name)
        if shard_match is None:
            print(f"[skip] cannot infer shard from file name: {base_name}")
            continue
        shard = shard_match.group(1)

        with open(txt_file, "r", encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line:
                    continue
                parts = line.split(maxsplit=1)
                if len(parts) < 2:
                    malformed_lines += 1
                    continue

                utt_id, ref_text = parts[0], parts[1]
                wav_path = os.path.join(audio_root, shard, f"{utt_id}.wav")

                if not os.path.exists(wav_path):
                    missing_audio += 1
                    if strict_audio:
                        raise ValueError(f"Missing audio for utt_id={utt_id}: {wav_path}")
                    continue

                pairs.append((wav_path, ref_text, utt_id))

    print(
        "[dataset] loaded from Audio/Text: "
        f"pairs={len(pairs)}, missing_audio={missing_audio}, malformed_lines={malformed_lines}"
    )
    return pairs


def collect_pairs_from_jsonl(input_jsonl: str) -> List[Tuple[str, str, str]]:
    if not os.path.isfile(input_jsonl):
        raise ValueError(f"JSONL file not found: {input_jsonl}")

    pairs: List[Tuple[str, str, str]] = []
    malformed_lines = 0
    missing_audio = 0

    with open(input_jsonl, "r", encoding="utf-8") as f:
        for line_num, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue

            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_num} of {input_jsonl}: {exc}") from exc

            audio_path = (
                item.get("source")
                or item.get("audio")
                or item.get("audio_path")
                or item.get("wav")
                or item.get("path")
            )
            ref_text = (
                item.get("target")
                or item.get("text")
                or item.get("transcript")
                or item.get("label")
            )

            if not audio_path or ref_text is None:
                malformed_lines += 1
                continue

            if not os.path.exists(audio_path):
                missing_audio += 1
                continue

            utt_id = (
                item.get("key")
                or item.get("utt_id")
                or item.get("id")
                or os.path.splitext(os.path.basename(audio_path))[0]
            )
            pairs.append((str(audio_path), str(ref_text), str(utt_id)))

    print(
        "[dataset] loaded from jsonl: "
        f"pairs={len(pairs)}, missing_audio={missing_audio}, malformed_lines={malformed_lines}"
    )
    return pairs


def main():
    parser = argparse.ArgumentParser(description="Qwen3-ASR inference + CER evaluation")
    parser.add_argument("--data_dir", type=str, default="",
                        help="Directory containing flat .wav/.txt file pairs")
    parser.add_argument("--dataset_dir", type=str, default="",
                        help="Dataset root containing Audio/ and Text/ subdirectories")
    parser.add_argument("--input_jsonl", type=str, default="",
                        help="JSONL file containing source/audio path and target text")
    parser.add_argument("--audio_subdir", type=str, default="Audio")
    parser.add_argument("--text_subdir", type=str, default="Text")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-ASR-1.7B",
                        help="Model name or local checkpoint path")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--language", type=str, default="Chinese",
                        help="Force output language. Set empty string to auto-detect")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Max inference batch size")
    parser.add_argument("--max_samples", type=int, default=0,
                        help="Use first N samples only (0 means all)")
    parser.add_argument("--strict_audio", type=int, default=0,
                        help="When dataset_dir mode is used, fail if any transcript has no audio")
    parser.add_argument("--result_file", type=str, default="",
                        help="Optional output jsonl path for per-utterance results")
    parser.add_argument("--backend", type=str, default="transformers",
                        choices=["transformers", "vllm"],
                        help="Backend to use for inference (transformers or vllm)")
    parser.add_argument("--num_workers", type=int, default=8,
                        help="Threads for parallel audio loading")
    parser.add_argument("--compile", action="store_true",
                        help="Use torch.compile() on the model (first batch is slower)")
    parser.add_argument("--flash_attn", action="store_true",
                        help="Use Flash Attention 2 (requires flash-attn package)")
    parser.add_argument("--sort_by_length", action="store_true", default=True,
                        help="Sort audio by length to minimize padding waste (default: True)")
    parser.add_argument("--context", type=str, default="",
                        help="Context/prompt string passed as system message to the model")
    parser.add_argument("--strip_punc", action="store_true", default=True,
                        help="Strip punctuation from hyp before CER computation (default: True)")
    parser.add_argument("--skip_cal_wer", action="store_true",
                        help="Skip the external cal_wer.sh WER computation step")
    args = parser.parse_args()

    input_modes = [bool(args.data_dir), bool(args.dataset_dir), bool(args.input_jsonl)]
    if sum(input_modes) != 1:
        raise ValueError("Use exactly one of --data_dir, --dataset_dir, or --input_jsonl")

    # Collect wav/txt pairs
    if args.input_jsonl:
        pairs = collect_pairs_from_jsonl(args.input_jsonl)
    elif args.dataset_dir:
        pairs = collect_pairs_from_audio_text_dir(
            dataset_dir=args.dataset_dir,
            audio_subdir=args.audio_subdir,
            text_subdir=args.text_subdir,
            strict_audio=(args.strict_audio == 1),
        )
    else:
        pairs = collect_pairs_from_flat_dir(args.data_dir)

    if not pairs:
        print("No valid wav/txt pairs found.")
        return

    if args.max_samples > 0:
        pairs = pairs[:args.max_samples]

    source_desc = args.input_jsonl or args.dataset_dir or args.data_dir
    print(f"Found {len(pairs)} audio/text pairs in {source_desc}")

    # ── Pre-load audio in parallel (biggest I/O speedup) ─────────────
    audio_paths = [p[0] for p in pairs]
    ref_texts = [p[1] for p in pairs]
    utt_ids = [p[2] for p in pairs]

    t0 = time.perf_counter()
    print(f"Pre-loading {len(audio_paths)} audio files ({args.num_workers} threads)...")
    audio_data = preload_audios(audio_paths, num_workers=args.num_workers)
    print(f"Audio loaded in {time.perf_counter() - t0:.1f}s")

    # ── Sort by audio length for efficient padding ───────────────────
    if args.sort_by_length:
        order = sorted(range(len(audio_data)), key=lambda i: len(audio_data[i][0]))
        audio_data = [audio_data[i] for i in order]
        ref_texts = [ref_texts[i] for i in order]
        utt_ids = [utt_ids[i] for i in order]
        audio_paths = [audio_paths[i] for i in order]

    # ── Load model ───────────────────────────────────────────────────
    torch.backends.cudnn.benchmark = True

    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    extra_kwargs = {}
    if args.flash_attn:
        extra_kwargs["attn_implementation"] = "flash_attention_2"

    if args.backend == "transformers":
        model = Qwen3ASRModel.from_pretrained(
            args.model,
            dtype=dtype_map[args.dtype],
            device_map=args.device,
            max_inference_batch_size=args.batch_size,
            **extra_kwargs,
        )
        if args.compile:
            print("Compiling model with torch.compile()...")
            model.model = torch.compile(model.model, mode="reduce-overhead")
    elif args.backend == "vllm":
        model = Qwen3ASRModel.LLM(
            model=args.model,
            max_inference_batch_size=args.batch_size,
        )
    else:
        raise ValueError(f"Unsupported backend: {args.backend}")

    # ── Inference (pass pre-loaded numpy arrays, not file paths) ─────
    batch_size = args.batch_size if args.batch_size > 0 else 32
    results = [None] * len(audio_data)

    t0 = time.perf_counter()
    with torch.inference_mode():
        iterator = range(0, len(audio_data), batch_size)
        if tqdm is not None:
            print(f"Running inference on {len(audio_data)} samples...")
            iterator = tqdm(iterator, desc="Infer", ncols=80)
        else:
            print("[warn] tqdm not installed, progress bar disabled.")

        for i in iterator:
            batch_audio = audio_data[i : i + batch_size]
            batch_results = model.transcribe(
                audio=batch_audio,
                context=args.context,
                language=(args.language if args.language else None),
            )
            for j, r in enumerate(batch_results):
                results[i + j] = r

    print(f"Inference done in {time.perf_counter() - t0:.1f}s")

    # Compute CER
    total_errors = 0
    total_ref_len = 0
    per_utt = []
    cer_iter = zip(results, ref_texts)
    if tqdm is not None:
        cer_iter = tqdm(cer_iter, total=len(results), desc="CER", ncols=80)
    for i, (r, ref) in enumerate(cer_iter):
        hyp = r.text
        name = os.path.basename(audio_paths[i])
        print(f"  {name}  ref={ref}  hyp={hyp}")
        per_utt.append(
            {
                "utt_id": utt_ids[i],
                "audio": audio_paths[i],
                "ref": ref,
                "hyp": hyp,
            }
        )

    if args.result_file:
        out_dir = os.path.dirname(args.result_file)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(args.result_file, "w", encoding="utf-8") as f:
            for item in per_utt:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"Saved per-utterance results to: {args.result_file}")

        # Run cal_wer.sh to compute WER using external tool
        if not args.skip_cal_wer:
            cal_wer_script = "/cmy/cmy/Qwen3-ASR/cal_wer/cal_wer.sh"
            if os.path.exists(cal_wer_script):
                print(f"\nRunning WER computation via {cal_wer_script}...")
                result_basename = os.path.splitext(os.path.basename(args.result_file))[0]
                wer_out_dir = f"wer_inputs_{result_basename}"

                try:
                    subprocess.run(
                        ["bash", cal_wer_script, args.result_file, wer_out_dir],
                        check=True,
                        cwd="/cmy/cmy/Qwen3-ASR/cal_wer"
                    )
                    print(f"WER computation completed. Results in cal_wer/{wer_out_dir}/")
                except subprocess.CalledProcessError as e:
                    print(f"Warning: WER computation failed with exit code {e.returncode}")
            else:
                print(f"Warning: {cal_wer_script} not found, skipping WER computation")
    else:
        print("\nNote: --result_file not specified, skipping WER computation")


if __name__ == "__main__":
    main()
