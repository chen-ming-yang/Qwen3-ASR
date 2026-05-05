"""
Convert voxflame-oss-by-account dataset into Qwen3-ASR finetuning JSONL format.

Each output line: {"audio": "/abs/path/to/file.wav", "text": "transcript"}
Compatible with the format expected by finetuning/qwen3_asr_sft.py.

Usage:
    python convert_voxflame.py --src ../voxflame-oss-by-account/oss-by-account \
                                --out voxflame_train.jsonl
"""

import argparse
import json
import os
from pathlib import Path


def find_account_root(manifest_path: str) -> Path | None:
    """Return the account directory (parent of dataset/ or supervised/)."""
    parts = Path(manifest_path).parts
    # manifest lives at: .../{account_id}/dataset/{split}/manifest.jsonl
    try:
        idx = next(i for i, p in enumerate(parts) if p == "dataset" or p == "supervised")
        return Path(*parts[:idx])
    except StopIteration:
        return None


def convert_account(manifest_path: str, min_confidence: float, max_dur_ratio: float) -> list[dict]:
    """Parse one manifest.jsonl and return filtered + reformatted entries."""
    account_root = find_account_root(manifest_path)
    if account_root is None:
        return []

    results = []
    with open(manifest_path, encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)

            eval_signals = obj.get("evaluation", {}).get("clarity_signals", {})

            # Quality filter: alignment must be matched, confidence above threshold
            if eval_signals.get("alignment_status") != "matched":
                continue
            confidence = obj.get("transcript", {}).get("confidence", 0.0)
            if confidence < min_confidence:
                continue

            audio_rel = obj.get("audio", {}).get("path", "")
            if not audio_rel:
                continue

            audio_abs = account_root / audio_rel
            if not audio_abs.exists():
                continue

            text = obj.get("prompt", {}).get("text", "").strip()
            if not text:
                continue

            # Skip entries where audio is far longer than the label warrants
            # (e.g. speaker repeated a 2-char word for 24s, but label only says "老师")
            dur_sec = obj.get("audio", {}).get("duration_ms", 0) / 1000
            if max_dur_ratio > 0 and dur_sec / len(text) > max_dur_ratio:
                continue

            results.append({
                "audio": str(audio_abs.resolve()),
                "text": text,
                "recording_id": obj.get("recording_id", ""),
                "category": obj.get("prompt", {}).get("category", ""),
                "confidence": confidence,
                "clarity_score": eval_signals.get("clarity_score", 0.0),
            })

    return results


def main():
    parser = argparse.ArgumentParser(description="Convert voxflame OSS dataset to Qwen3-ASR JSONL")
    parser.add_argument("--src", default="../voxflame-oss-by-account/oss-by-account",
                        help="Root of oss-by-account (contains account dirs)")
    parser.add_argument("--out", default="voxflame_train.jsonl",
                        help="Output JSONL path")
    parser.add_argument("--min-confidence", type=float, default=0.3,
                        help="Minimum transcript.confidence to include (default 0.3)")
    parser.add_argument("--max-dur-ratio", type=float, default=4.0,
                        help="Max allowed duration_ms / text_len (default 4.0). "
                             "Entries with a higher ratio are skipped.")
    args = parser.parse_args()

    src_root = Path(args.src).resolve()
    manifests = sorted(src_root.glob("**/manifest.jsonl"))

    if not manifests:
        print(f"No manifest.jsonl found under {src_root}")
        return

    all_entries = []
    for mp in manifests:
        account = mp.parts[-4]
        entries = convert_account(str(mp), args.min_confidence, args.max_dur_ratio)
        manifest_count = sum(1 for _ in open(mp))
        print(f"  {account}: {len(entries)} / {manifest_count} passed quality filter")
        all_entries.extend(entries)

    # Deduplicate by recording_id (keep first)
    seen, unique = set(), []
    for e in all_entries:
        if e["recording_id"] not in seen:
            seen.add(e["recording_id"])
            unique.append({"audio": e["audio"], "text": e["text"]})

    out_path = Path(args.out)
    with open(out_path, "w", encoding="utf-8") as f:
        for e in unique:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

    print(f"\nTotal: {len(unique)} samples written to {out_path}")


if __name__ == "__main__":
    main()
