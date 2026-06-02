#!/usr/bin/env python3
"""
Profile peak memory and GPU VRAM usage during Qwen3-ASR inference.

Measures:
  - Peak GPU memory allocated / reserved
  - CPU RSS (Resident Set Size) peak
  - Model size on disk
  - Per-inference memory delta
"""

import argparse
import gc
import json
import os
import sys
import time
import tracemalloc
from typing import List, Tuple

import numpy as np
import psutil
import soundfile as sf
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from qwen_asr import Qwen3ASRModel


def load_audio(path: str) -> Tuple[np.ndarray, int]:
    wav, sr = sf.read(path, dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    return wav, sr


def get_model_size_mb(model_path: str) -> float:
    """Calculate total size of model files (safetensors/bin) in MB."""
    total = 0
    if os.path.isdir(model_path):
        for f in os.listdir(model_path):
            if f.endswith((".safetensors", ".bin", ".pt")):
                total += os.path.getsize(os.path.join(model_path, f))
    return total / (1024 * 1024)


def get_gpu_memory_info(device_id: int = 0) -> dict:
    """Get current GPU memory stats from PyTorch."""
    if not torch.cuda.is_available():
        return {}
    return {
        "allocated_mb": torch.cuda.memory_allocated(device_id) / (1024 * 1024),
        "reserved_mb": torch.cuda.memory_reserved(device_id) / (1024 * 1024),
        "max_allocated_mb": torch.cuda.max_memory_allocated(device_id) / (1024 * 1024),
        "max_reserved_mb": torch.cuda.max_memory_reserved(device_id) / (1024 * 1024),
    }


def main():
    parser = argparse.ArgumentParser(description="Profile Qwen3-ASR memory usage")
    parser.add_argument("--model", type=str,
                        default="/cmy/cmy/Qwen3-ASR/qwen3-asr-finetuning-out-bak/checkpoint-326")
    parser.add_argument("--test_file", type=str,
                        default="/cmy/cmy/Qwen3-ASR/data/test.jsonl")
    parser.add_argument("--max_samples", type=int, default=50,
                        help="Samples to profile (0=all)")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--output", type=str, default="eval/resource_report.json")
    args = parser.parse_args()

    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    device_id = int(args.device.split(":")[-1]) if "cuda" in args.device else 0

    process = psutil.Process(os.getpid())

    # Model size on disk
    model_size_mb = get_model_size_mb(args.model)
    print(f"Model size on disk: {model_size_mb:.1f} MB")

    # Baseline memory
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device_id)

    cpu_rss_before = process.memory_info().rss / (1024 * 1024)
    print(f"CPU RSS before model load: {cpu_rss_before:.1f} MB")

    # Load model
    print(f"Loading model from {args.model} ...")
    model = Qwen3ASRModel.from_pretrained(
        args.model,
        dtype=dtype_map[args.dtype],
        device_map=args.device,
    )

    cpu_rss_after_load = process.memory_info().rss / (1024 * 1024)
    gpu_after_load = get_gpu_memory_info(device_id)
    print(f"CPU RSS after model load: {cpu_rss_after_load:.1f} MB "
          f"(delta: +{cpu_rss_after_load - cpu_rss_before:.1f} MB)")
    if gpu_after_load:
        print(f"GPU allocated after load: {gpu_after_load['allocated_mb']:.1f} MB")
        print(f"GPU reserved after load: {gpu_after_load['reserved_mb']:.1f} MB")

    # Load test audio
    pairs = []
    with open(args.test_file, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line.strip())
            audio_path = item.get("source") or item.get("audio")
            if audio_path and os.path.exists(audio_path):
                pairs.append(audio_path)

    if args.max_samples > 0:
        pairs = pairs[:args.max_samples]

    print(f"\nProfiling with {len(pairs)} audio samples...")
    audios = [load_audio(p) for p in pairs]

    # Reset peak stats before inference
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device_id)

    # Run inference and track memory
    cpu_rss_peaks = []
    with torch.inference_mode():
        for i, audio in enumerate(audios):
            model.transcribe(audio=[audio], language="Chinese")
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            cpu_rss_peaks.append(process.memory_info().rss / (1024 * 1024))

    # Final stats
    gpu_after_infer = get_gpu_memory_info(device_id)
    cpu_rss_peak = max(cpu_rss_peaks) if cpu_rss_peaks else cpu_rss_after_load

    report = {
        "model": args.model,
        "device": args.device,
        "dtype": args.dtype,
        "num_samples_profiled": len(audios),
        "model_size_disk_mb": round(model_size_mb, 1),
        "cpu_memory": {
            "rss_before_load_mb": round(cpu_rss_before, 1),
            "rss_after_load_mb": round(cpu_rss_after_load, 1),
            "rss_peak_during_infer_mb": round(cpu_rss_peak, 1),
            "model_load_delta_mb": round(cpu_rss_after_load - cpu_rss_before, 1),
        },
        "gpu_memory": {},
    }

    if gpu_after_infer:
        report["gpu_memory"] = {
            "after_load_allocated_mb": round(gpu_after_load["allocated_mb"], 1),
            "after_load_reserved_mb": round(gpu_after_load["reserved_mb"], 1),
            "peak_allocated_mb": round(gpu_after_infer["max_allocated_mb"], 1),
            "peak_reserved_mb": round(gpu_after_infer["max_reserved_mb"], 1),
            "inference_peak_delta_mb": round(
                gpu_after_infer["max_allocated_mb"] - gpu_after_load["allocated_mb"], 1
            ),
        }

    # Print summary
    print("\n" + "=" * 60)
    print("RESOURCE PROFILING RESULTS")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Model size on disk: {model_size_mb:.1f} MB")
    print(f"\nCPU Memory (RSS):")
    print(f"  Before load: {cpu_rss_before:.1f} MB")
    print(f"  After load:  {cpu_rss_after_load:.1f} MB (+{cpu_rss_after_load - cpu_rss_before:.1f})")
    print(f"  Peak infer:  {cpu_rss_peak:.1f} MB")
    if gpu_after_infer:
        print(f"\nGPU Memory (device {device_id}):")
        print(f"  After load (allocated): {gpu_after_load['allocated_mb']:.1f} MB")
        print(f"  Peak (allocated):       {gpu_after_infer['max_allocated_mb']:.1f} MB")
        print(f"  Peak (reserved):        {gpu_after_infer['max_reserved_mb']:.1f} MB")
        print(f"  Inference delta:        +{gpu_after_infer['max_allocated_mb'] - gpu_after_load['allocated_mb']:.1f} MB")
    print("=" * 60)

    # Save
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\nReport saved to: {args.output}")


if __name__ == "__main__":
    main()
