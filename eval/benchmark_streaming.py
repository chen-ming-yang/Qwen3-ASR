#!/usr/bin/env python3
"""
Benchmark streaming-style latency for Qwen3-ASR.

Since the native streaming API requires vLLM backend, this script simulates
streaming behavior with the transformers backend by:
  1. Splitting audio into chunks (default 2s)
  2. For each chunk, feeding accumulated audio (from start to current chunk end)
     to the model and measuring per-chunk inference time

This measures:
  - First-chunk latency: time to get the first partial result
  - Per-chunk latency: incremental decode time for each subsequent chunk
  - Total streaming latency vs offline latency comparison

This reflects real streaming behavior where the model re-processes all audio
seen so far at each chunk boundary (same as the vLLM streaming implementation).
"""

import argparse
import json
import os
import sys
import time
from typing import List, Tuple

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from qwen_asr import Qwen3ASRModel


SAMPLE_RATE = 16000


def load_audio(path: str) -> Tuple[np.ndarray, int]:
    wav, sr = sf.read(path, dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    return wav, sr


def percentiles(values: List[float]) -> dict:
    if not values:
        return {}
    arr = np.array(values)
    return {
        "min": float(np.min(arr)),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "p99": float(np.percentile(arr, 99)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
    }


def simulate_streaming(model, audio: Tuple[np.ndarray, int],
                       chunk_size_sec: float = 2.0) -> dict:
    """Simulate streaming by feeding progressively longer audio."""
    wav, sr = audio
    chunk_size_samples = int(chunk_size_sec * sr)
    total_samples = len(wav)
    num_chunks = max(1, (total_samples + chunk_size_samples - 1) // chunk_size_samples)

    chunk_latencies = []
    first_chunk_latency = None

    with torch.inference_mode():
        for i in range(num_chunks):
            end_sample = min((i + 1) * chunk_size_samples, total_samples)
            audio_so_far = wav[:end_sample]

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            t0 = time.perf_counter()
            model.transcribe(audio=[(audio_so_far, sr)], language="Chinese")

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            latency = time.perf_counter() - t0
            chunk_latencies.append(latency)

            if i == 0:
                first_chunk_latency = latency

    audio_duration = total_samples / sr
    total_streaming_time = sum(chunk_latencies)

    return {
        "audio_duration_sec": audio_duration,
        "num_chunks": num_chunks,
        "chunk_size_sec": chunk_size_sec,
        "first_chunk_latency_sec": first_chunk_latency,
        "last_chunk_latency_sec": chunk_latencies[-1] if chunk_latencies else 0,
        "avg_chunk_latency_sec": np.mean(chunk_latencies),
        "total_streaming_time_sec": total_streaming_time,
        "streaming_rtf": total_streaming_time / audio_duration if audio_duration > 0 else 0,
        "chunk_latencies": chunk_latencies,
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark streaming latency for Qwen3-ASR")
    parser.add_argument("--model", type=str,
                        default="/cmy/cmy/Qwen3-ASR/qwen3-asr-finetuning-out-bak/checkpoint-326")
    parser.add_argument("--test_file", type=str,
                        default="/cmy/cmy/Qwen3-ASR/data/test.jsonl")
    parser.add_argument("--max_samples", type=int, default=50,
                        help="Max samples to benchmark")
    parser.add_argument("--chunk_size", type=float, default=2.0,
                        help="Chunk size in seconds (default: 2.0)")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--output", type=str, default="eval/streaming_latency_report.json")
    parser.add_argument("--warmup", type=int, default=3)
    args = parser.parse_args()

    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}

    # Load test data
    pairs = []
    with open(args.test_file, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line.strip())
            audio_path = item.get("source") or item.get("audio")
            if audio_path and os.path.exists(audio_path):
                pairs.append(audio_path)

    if args.max_samples > 0:
        pairs = pairs[:args.max_samples]

    print(f"Loaded {len(pairs)} audio paths")

    # Load model
    print(f"Loading model from {args.model} ...")
    model = Qwen3ASRModel.from_pretrained(
        args.model,
        dtype=dtype_map[args.dtype],
        device_map=args.device,
    )
    print("Model loaded.")

    # Pre-load audio
    print("Pre-loading audio...")
    audios = []
    for path in pairs:
        wav, sr = load_audio(path)
        audios.append((wav, sr))

    # Warmup
    print(f"Warmup ({args.warmup} samples)...")
    with torch.inference_mode():
        for i in range(min(args.warmup, len(audios))):
            model.transcribe(audio=[audios[i]], language="Chinese")

    # Also run offline benchmark for comparison
    print(f"\nRunning offline baseline for comparison...")
    offline_latencies = []
    with torch.inference_mode():
        for audio in audios:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            model.transcribe(audio=[audio], language="Chinese")
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            offline_latencies.append(time.perf_counter() - t0)

    # Streaming benchmark
    print(f"\nRunning streaming simulation (chunk_size={args.chunk_size}s)...")
    all_results = []
    first_chunk_latencies = []
    last_chunk_latencies = []
    streaming_rtfs = []

    for i, audio in enumerate(audios):
        result = simulate_streaming(model, audio, chunk_size_sec=args.chunk_size)
        all_results.append(result)
        first_chunk_latencies.append(result["first_chunk_latency_sec"])
        last_chunk_latencies.append(result["last_chunk_latency_sec"])
        streaming_rtfs.append(result["streaming_rtf"])

        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(audios)}] avg first-chunk: "
                  f"{np.mean(first_chunk_latencies):.4f}s")

    # Compute stats
    durations = [len(a[0]) / a[1] for a in audios]

    report = {
        "model": args.model,
        "device": args.device,
        "dtype": args.dtype,
        "chunk_size_sec": args.chunk_size,
        "num_samples": len(audios),
        "total_audio_duration_sec": sum(durations),
        "streaming": {
            "first_chunk_latency": percentiles(first_chunk_latencies),
            "last_chunk_latency": percentiles(last_chunk_latencies),
            "streaming_rtf": percentiles(streaming_rtfs),
        },
        "offline_comparison": {
            "offline_rtf": sum(offline_latencies) / sum(durations),
            "offline_latency": percentiles(offline_latencies),
        },
    }

    # Print summary
    print("\n" + "=" * 65)
    print("STREAMING LATENCY BENCHMARK RESULTS")
    print("=" * 65)
    print(f"Model: {args.model}")
    print(f"Chunk size: {args.chunk_size}s")
    print(f"Samples: {len(audios)}, Total audio: {sum(durations):.1f}s")

    print(f"\n--- 首字延迟 (First-Chunk Latency) ---")
    print(f"  即用户说完第一个 chunk ({args.chunk_size}s) 后，拿到首次结果的时间")
    fc = report["streaming"]["first_chunk_latency"]
    print(f"  P50: {fc['p50']*1000:.1f} ms")
    print(f"  P90: {fc['p90']*1000:.1f} ms")
    print(f"  P99: {fc['p99']*1000:.1f} ms")

    print(f"\n--- 末 chunk 延迟 (Last-Chunk Latency) ---")
    print(f"  即最后一个 chunk 处理完成的时间（含全部累积音频）")
    lc = report["streaming"]["last_chunk_latency"]
    print(f"  P50: {lc['p50']*1000:.1f} ms")
    print(f"  P90: {lc['p90']*1000:.1f} ms")
    print(f"  P99: {lc['p99']*1000:.1f} ms")

    print(f"\n--- 流式 RTF ---")
    sr = report["streaming"]["streaming_rtf"]
    print(f"  P50: {sr['p50']:.4f}")
    print(f"  P90: {sr['p90']:.4f}")
    print(f"  Mean: {sr['mean']:.4f}")
    realtime_ok = sr['p90'] < 1.0
    print(f"  流式实时可行: {'✓' if realtime_ok else '✗'} (P90 RTF {'<' if realtime_ok else '>'} 1.0)")

    print(f"\n--- 对比: 离线 vs 流式 ---")
    off = report["offline_comparison"]
    print(f"  离线 RTF: {off['offline_rtf']:.4f}")
    print(f"  流式 RTF (mean): {sr['mean']:.4f}")
    print(f"  流式开销倍数: {sr['mean'] / off['offline_rtf']:.2f}x")

    print(f"\n--- 用户体感延迟估算 ---")
    user_delay = args.chunk_size + fc['p50']
    print(f"  用户说话 {args.chunk_size}s 后首次出字: "
          f"{args.chunk_size}s (等待chunk) + {fc['p50']*1000:.0f}ms (推理) "
          f"= {user_delay*1000:.0f} ms")
    print("=" * 65)

    # Save
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\nReport saved to: {args.output}")


if __name__ == "__main__":
    main()
