#!/usr/bin/env python3
"""
Quantization evaluation for Qwen3-ASR.

Compares FP16/BF16 baseline vs INT8 (bitsandbytes) vs INT4 (bitsandbytes/GPTQ):
  - CER on test set
  - Inference speed (RTF)
  - Model memory footprint
  - Model size estimate

Requires: bitsandbytes, jiwer
Optional: auto-gptq (for GPTQ INT4)
"""

import argparse
import gc
import json
import os
import sys
import time
import unicodedata
from typing import Dict, List, Tuple

import jiwer
import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_audio(path: str) -> Tuple[np.ndarray, int]:
    wav, sr = sf.read(path, dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    return wav, sr


def remove_punctuation(text: str) -> str:
    return "".join(ch for ch in text if not unicodedata.category(ch).startswith("P"))


def compute_cer(refs: List[str], hyps: List[str]) -> float:
    """Compute character error rate using jiwer."""
    refs_clean = [remove_punctuation(r) for r in refs]
    hyps_clean = [remove_punctuation(h) for h in hyps]
    # Filter out empty refs
    valid = [(r, h) for r, h in zip(refs_clean, hyps_clean) if len(r) > 0]
    if not valid:
        return 0.0
    refs_v, hyps_v = zip(*valid)
    return jiwer.cer(list(refs_v), list(hyps_v))


def load_test_data(test_file: str, max_samples: int) -> List[dict]:
    items = []
    with open(test_file, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line.strip())
            audio_path = item.get("source") or item.get("audio")
            ref = item.get("target") or item.get("text")
            if audio_path and os.path.exists(audio_path) and ref:
                items.append({"audio_path": audio_path, "ref": ref})
    if max_samples > 0:
        items = items[:max_samples]
    return items


def evaluate_model(model, audios, refs, durations) -> dict:
    """Run inference and compute CER + RTF."""
    hyps = []
    latencies = []

    with torch.inference_mode():
        for audio, dur in zip(audios, durations):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            results = model.transcribe(audio=[audio], language="Chinese")
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            latencies.append(time.perf_counter() - t0)
            hyps.append(results[0].text)

    total_audio = sum(durations)
    total_infer = sum(latencies)
    cer = compute_cer(refs, hyps)

    return {
        "cer": round(cer * 100, 2),
        "rtf": round(total_infer / total_audio, 4),
        "total_infer_sec": round(total_infer, 2),
        "total_audio_sec": round(total_audio, 2),
        "avg_latency_sec": round(total_infer / len(audios), 4),
    }


def load_model_baseline(model_path: str, dtype, device: str):
    from qwen_asr import Qwen3ASRModel
    return Qwen3ASRModel.from_pretrained(
        model_path, dtype=dtype, device_map=device
    )


def load_model_int8(model_path: str, device: str):
    """Load model with bitsandbytes INT8 quantization."""
    from transformers import BitsAndBytesConfig
    from qwen_asr import Qwen3ASRModel

    quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    return Qwen3ASRModel.from_pretrained(
        model_path,
        device_map=device,
        quantization_config=quantization_config,
    )


def load_model_int4(model_path: str, device: str):
    """Load model with bitsandbytes INT4 (NF4) quantization."""
    from transformers import BitsAndBytesConfig
    from qwen_asr import Qwen3ASRModel

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    return Qwen3ASRModel.from_pretrained(
        model_path,
        device_map=device,
        quantization_config=quantization_config,
    )


def get_gpu_mem(device_id: int = 0) -> float:
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated(device_id) / (1024 * 1024)
    return 0.0


def main():
    parser = argparse.ArgumentParser(description="Quantization evaluation for Qwen3-ASR")
    parser.add_argument("--model", type=str,
                        default="/cmy/cmy/Qwen3-ASR/qwen3-asr-finetuning-out-bak/checkpoint-326")
    parser.add_argument("--test_file", type=str,
                        default="/cmy/cmy/Qwen3-ASR/data/test.jsonl")
    parser.add_argument("--max_samples", type=int, default=100,
                        help="Max samples for evaluation (0=all)")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--output", type=str, default="eval/quantization_report.json")
    parser.add_argument("--skip_baseline", action="store_true",
                        help="Skip BF16 baseline (use if already measured)")
    parser.add_argument("--skip_int8", action="store_true")
    parser.add_argument("--skip_int4", action="store_true")
    args = parser.parse_args()

    device_id = int(args.device.split(":")[-1]) if "cuda" in args.device else 0

    # Load test data
    items = load_test_data(args.test_file, args.max_samples)
    print(f"Loaded {len(items)} test samples")

    # Pre-load audio
    audios = []
    refs = []
    durations = []
    for item in items:
        wav, sr = load_audio(item["audio_path"])
        audios.append((wav, sr))
        refs.append(item["ref"])
        durations.append(len(wav) / sr)

    results = {}
    configs_to_run = []
    if not args.skip_baseline:
        configs_to_run.append(("bf16_baseline", "bf16"))
    if not args.skip_int8:
        configs_to_run.append(("int8", "int8"))
    if not args.skip_int4:
        configs_to_run.append(("int4_nf4", "int4"))

    for name, quant_type in configs_to_run:
        print(f"\n{'='*60}")
        print(f"Evaluating: {name}")
        print(f"{'='*60}")

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.set_device(device_id)
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device_id)

        try:
            if quant_type == "bf16":
                model = load_model_baseline(args.model, torch.bfloat16, args.device)
            elif quant_type == "int8":
                model = load_model_int8(args.model, args.device)
            elif quant_type == "int4":
                model = load_model_int4(args.model, args.device)
        except Exception as e:
            print(f"  Failed to load {name}: {e}")
            results[name] = {"error": str(e)}
            continue

        gpu_after_load = get_gpu_mem(device_id) if torch.cuda.is_available() else 0

        # Warmup
        with torch.inference_mode():
            model.transcribe(audio=[audios[0]], language="Chinese")

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(device_id)

        # Evaluate
        eval_result = evaluate_model(model, audios, refs, durations)
        peak_gpu = get_gpu_mem(device_id)

        eval_result["gpu_model_mb"] = round(gpu_after_load, 1)
        eval_result["gpu_peak_mb"] = round(peak_gpu, 1)
        results[name] = eval_result

        print(f"  CER: {eval_result['cer']:.2f}%")
        print(f"  RTF: {eval_result['rtf']:.4f}")
        print(f"  GPU model: {gpu_after_load:.1f} MB, peak: {peak_gpu:.1f} MB")

        # Cleanup
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Comparison summary
    report = {
        "model": args.model,
        "num_samples": len(items),
        "total_audio_sec": round(sum(durations), 1),
        "results": results,
    }

    # Print comparison table
    print(f"\n{'='*60}")
    print("QUANTIZATION COMPARISON")
    print(f"{'='*60}")
    print(f"{'Config':<15} {'CER%':<8} {'RTF':<8} {'GPU(MB)':<10} {'Speedup':<8}")
    print("-" * 60)

    baseline_rtf = results.get("bf16_baseline", {}).get("rtf", None)
    baseline_cer = results.get("bf16_baseline", {}).get("cer", None)

    for name, res in results.items():
        if "error" in res:
            print(f"{name:<15} {'ERROR':<8}")
            continue
        speedup = f"{baseline_rtf / res['rtf']:.2f}x" if baseline_rtf else "N/A"
        cer_delta = ""
        if baseline_cer is not None and name != "bf16_baseline":
            delta = res["cer"] - baseline_cer
            cer_delta = f" ({'+' if delta > 0 else ''}{delta:.2f})"
        print(f"{name:<15} {res['cer']:<8.2f}{cer_delta} "
              f"{res['rtf']:<8.4f} {res.get('gpu_peak_mb', 'N/A'):<10} {speedup}")

    print(f"{'='*60}")

    # Save
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\nReport saved to: {args.output}")


if __name__ == "__main__":
    main()
