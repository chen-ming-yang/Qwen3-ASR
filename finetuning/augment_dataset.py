#!/usr/bin/env python3
"""
Pre-generate an augmented ASR dataset using sox tempo/speed perturbation.

Reads a source JSONL, for each entry generates N augmented copies via sox,
saves the augmented audio to disk, and writes a new combined JSONL.

Usage example:
  python finetuning/augment_dataset.py \
      --input_jsonl data/train.jsonl \
      --output_dir data/augmented \
      --output_jsonl data/train_augmented.jsonl \
      --tempo_range 0.15 \
      --speed_range 0.15 \
      --num_copies 3 \
      --num_workers 8
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from functools import partial
from multiprocessing import Pool, cpu_count
from typing import Any, Dict, List, Optional, Tuple

import librosa
import numpy as np
import scipy.io.wavfile as wavfile


def augment_audio_sox(audio_array: np.ndarray, sr: int,
                       tempo_factor: float = 1.0,
                       speed_factor: float = 1.0) -> np.ndarray:
    """Apply tempo/speed perturbation via sox subprocess."""
    if abs(tempo_factor - 1.0) < 1e-6 and abs(speed_factor - 1.0) < 1e-6:
        return audio_array
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f_in:
        in_path = f_in.name
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f_out:
        out_path = f_out.name
    try:
        wavfile.write(in_path, sr, audio_array)
        effects = []
        if abs(tempo_factor - 1.0) > 1e-6:
            effects.extend(["tempo", f"{tempo_factor:.6f}"])
        if abs(speed_factor - 1.0) > 1e-6:
            effects.extend(["speed", f"{speed_factor:.6f}"])
        subprocess.run(["sox", in_path, out_path] + effects,
                       capture_output=True, timeout=30, check=True)
        augmented, _ = librosa.load(out_path, sr=sr, mono=True)
        return augmented
    except Exception as exc:
        print(f"  [warn] sox failed (tempo={tempo_factor:.3f}, speed={speed_factor:.3f}): {exc}")
        return audio_array
    finally:
        for p in (in_path, out_path):
            try:
                os.remove(p)
            except OSError:
                pass


def parse_args():
    p = argparse.ArgumentParser("Augment ASR dataset with sox")
    p.add_argument("--input_jsonl", required=True,
                   help="Source JSONL with 'audio' and 'text' fields")
    p.add_argument("--output_dir", required=True,
                   help="Directory to store augmented audio files")
    p.add_argument("--output_jsonl", required=True,
                   help="Output JSONL (originals + augmented copies)")
    p.add_argument("--sr", type=int, default=16000, help="Sample rate")
    p.add_argument("--tempo_range", type=float, default=0.1,
                   help="Tempo range (0 disables)")
    p.add_argument("--speed_range", type=float, default=0.0,
                   help="Speed range (0 disables)")
    p.add_argument("--num_copies", type=int, default=3,
                   help="Number of augmented copies per original")
    p.add_argument("--keep_originals", type=int, default=1,
                   help="Include original rows in the output JSONL")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_workers", type=int, default=0,
                   help="Number of parallel workers (0 = use all CPUs)")
    return p.parse_args()


def _process_one_source(args: Tuple[int, Dict[str, Any], str, int, float, float, int, str, int]) -> List[Dict[str, Any]]:
    """Worker function for multiprocessing: process one source row.

    Returns list of new rows (one per augmented copy).
    """
    idx, row, audio_keys_str, sr, tempo_range, speed_range, num_copies, out_dir, seed = args
    audio_keys = audio_keys_str.split(",")
    rng = np.random.RandomState(seed + idx * 7 + 13)

    # Find audio path
    audio_path = None
    ak_found = None
    for ak in audio_keys:
        if ak in row and row[ak] and os.path.exists(str(row[ak])):
            audio_path = str(row[ak])
            ak_found = ak
            break
    if audio_path is None:
        print(f"  [skip] row {idx}: no audio file found")
        return []

    # Load original
    audio, _ = librosa.load(audio_path, sr=sr, mono=True)

    local_rows = []
    for copy_idx in range(num_copies):
        tf = 1.0
        if tempo_range > 0:
            tf = 1.0 + rng.uniform(-tempo_range, tempo_range)
        sf = 1.0
        if speed_range > 0:
            sf = 1.0 + rng.uniform(-speed_range, speed_range)

        aug_audio = augment_audio_sox(audio, sr, tf, sf)

        base = os.path.splitext(os.path.basename(audio_path))[0]
        aug_name = f"{base}_c{copy_idx}_t{int(tf*100):03d}_s{int(sf*100):03d}.wav"
        aug_path = os.path.join(out_dir, aug_name)
        wavfile.write(aug_path, sr, (aug_audio * 32767).astype(np.int16))

        new_row = dict(row)
        if ak_found:
            new_row[ak_found] = aug_path
        new_row["audio"] = aug_path
        local_rows.append(new_row)

    return local_rows


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    audio_out_dir = os.path.join(args.output_dir, "audio_aug")
    os.makedirs(audio_out_dir, exist_ok=True)

    # Read source JSONL
    with open(args.input_jsonl, "r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    print(f"[input] {len(rows)} rows from {args.input_jsonl}")

    audio_keys_str = ",".join(["audio", "source", "audio_path", "wav", "path"])
    num_workers = args.num_workers if args.num_workers > 0 else cpu_count()
    print(f"[parallel] using {num_workers} workers")

    # Build task args for each row
    task_args = [
        (i, row, audio_keys_str, args.sr, args.tempo_range, args.speed_range,
         args.num_copies, audio_out_dir, args.seed)
        for i, row in enumerate(rows)
    ]

    out_rows = []
    if args.keep_originals:
        out_rows.extend(rows)

    if num_workers == 1:
        for ta in task_args:
            out_rows.extend(_process_one_source(ta))
    else:
        with Pool(num_workers) as pool:
            for i, result in enumerate(pool.imap_unordered(
                _process_one_source, task_args, chunksize=1
            )):
                out_rows.extend(result)
                if (i + 1) % 50 == 0 or i == 0:
                    print(f"[progress] {i+1}/{len(rows)}")

    # Write output JSONL
    with open(args.output_jsonl, "w", encoding="utf-8") as f:
        for row in out_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    original_count = len(rows) if args.keep_originals else 0
    augmented_count = len(out_rows) - original_count
    print(f"\n[done] Originals: {original_count}")
    print(f"       Augmented copies: {augmented_count}")
    print(f"       Total: {len(out_rows)} rows")
    print(f"       Output JSONL: {args.output_jsonl}")
    print(f"       Audio files: {audio_out_dir}/")


if __name__ == "__main__":
    main()
