#!/usr/bin/env python3
"""
Merge audio recordings from two source folders for user 64758dee
and generate a finetune JSONL in the format: {"audio": "<abs_path>", "text": "<transcript>"}
"""

import json
import os
from pathlib import Path

# Source roots
VOXFLAME_ROOT = Path("/cmy/cmy/Qwen3-ASR/voxflame-oss-by-account/oss-by-account/2307294809__64758dee")
CUSTOMER_ROOT = Path("/cmy/cmy/Qwen3-ASR/customer_audio/2307294809__64758dee")
OUTPUT_FILE = Path("/cmy/cmy/Qwen3-ASR/data/finetune_64758dee.jsonl")

def load_transcripts(root: Path) -> list[dict]:
    """
    Parse all transcripts.txt files under root.
    Format: recording_id <TAB> relative_wav_path <TAB> text
    The wav files live under root/supervised/... (relative to root).
    """
    records = []
    for txt_file in root.rglob("transcripts.txt"):
        # The wav paths in transcripts.txt are relative to the parent of dataset/
        # e.g. "supervised/mandarin/日常与出行/.../xxx.wav"
        # The actual files sit at root / <relative_path>
        base = root  # wav paths are relative to the account root
        with open(txt_file, encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                if not line:
                    continue
                parts = line.split("\t")
                if len(parts) < 3:
                    continue
                _rec_id, rel_wav, text = parts[0], parts[1], parts[2]
                abs_wav = base / rel_wav
                records.append({"audio": str(abs_wav), "text": text})
    return records


def main():
    seen_audio = set()
    all_records = []

    for root in (VOXFLAME_ROOT, CUSTOMER_ROOT):
        records = load_transcripts(root)
        for rec in records:
            audio_path = rec["audio"]
            if audio_path in seen_audio:
                continue
            if not Path(audio_path).exists():
                print(f"[WARN] missing file: {audio_path}")
                continue
            seen_audio.add(audio_path)
            all_records.append(rec)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for rec in all_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Written {len(all_records)} records to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
