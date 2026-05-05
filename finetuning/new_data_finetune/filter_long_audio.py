"""
Filter and/or combine multiple JSONL datasets before training.

Each input file is converted to {audio, text} schema, then both filters are
applied independently (OR logic). Results are concatenated into a single output.

Usage:
    # Combine two files and filter
    python filter_long_audio.py \
        --inputs /cmy/cmy/Qwen3-ASR/data/train.jsonl /cmy/cmy/Qwen3-ASR/finetuning/new_data_finetune/voxflame_train.jsonl \
        --output /cmy/cmy/Qwen3-ASR/datacombined_filtered.jsonl \
        --max_duration_sec 60 \
        --max_dur_ratio 4.0

    # Filter a single file
    python filter_long_audio.py \
        --inputs data/train_with_voxflame_converted.jsonl \
        --output data/train_filtered.jsonl \
        --max_duration_sec 60
"""

import argparse
import json
import wave
from pathlib import Path


def get_duration(path: str) -> float:
    try:
        with wave.open(path, "rb") as w:
            return w.getnframes() / w.getframerate()
    except Exception:
        return None


def normalize_to_audio_text(obj: dict) -> dict | None:
    """Convert various JSONL schemas to {audio, text}."""
    audio = obj.get("audio") or obj.get("source") or ""
    text = obj.get("text") or obj.get("target") or ""
    if not audio or not text:
        return None
    return {"audio": audio, "text": text}


def filter_sample(obj: dict, max_dur_sec: float, max_dur_ratio: float) -> tuple[bool, str]:
    """
    Returns (keep: bool, reason: str) for a sample.
    reason is empty if kept.
    """
    audio_path = obj["audio"]
    text = obj["text"].strip()
    char_count = len(text)

    dur = get_duration(audio_path)
    if dur is None:
        return False, "missing_audio"

    if max_dur_sec > 0 and dur > max_dur_sec:
        return False, f"too_long({dur:.0f}s)"

    if max_dur_ratio > 0 and char_count > 0:
        ratio = dur / char_count
        if ratio > max_dur_ratio:
            return False, f"bad_ratio({ratio:.1f}s/char)"

    return True, ""


def main():
    parser = argparse.ArgumentParser(description="Filter and/or combine JSONL datasets")
    parser.add_argument(
        "--inputs", nargs="+", required=True,
        help="One or more input JSONL files (supports mixed schemas)",
    )
    parser.add_argument("--output", required=True, help="Output JSONL file")
    parser.add_argument(
        "--max_duration_sec", type=float, default=60.0,
        help="Skip samples where audio is longer than this (seconds). Default: 60",
    )
    parser.add_argument(
        "--max_dur_ratio", type=float, default=4.0,
        help="Skip samples where duration_sec / text_chars > this ratio. Default: 4.0",
    )
    args = parser.parse_args()

    total = too_long = too_ratio = missing = 0
    kept_lines = []

    for input_path_str in args.inputs:
        input_path = Path(input_path_str).resolve()
        print(f"Processing: {input_path}")

        with open(input_path, encoding="utf-8") as fin:
            for line in fin:
                total += 1
                raw = json.loads(line)

                obj = normalize_to_audio_text(raw)
                if obj is None:
                    missing += 1
                    continue

                keep, reason = filter_sample(obj, args.max_duration_sec, args.max_dur_ratio)
                if not keep:
                    if reason == "missing_audio":
                        missing += 1
                    elif reason.startswith("too_long"):
                        too_long += 1
                    else:
                        too_ratio += 1
                    continue

                kept_lines.append(json.dumps(obj, ensure_ascii=False) + "\n")

    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fout:
        fout.writelines(kept_lines)

    kept = len(kept_lines)
    print(f"\nTotal input:     {total} samples across {len(args.inputs)} file(s)")
    print(f"Filtered out:    {too_long} too-long (> {args.max_duration_sec}s), "
          f"{too_ratio} bad-ratio (> {args.max_dur_ratio}s/char), "
          f"{missing} missing/unparseable")
    print(f"Output:           {kept} samples  ({output_path})")


if __name__ == "__main__":
    main()
