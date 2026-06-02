#!/usr/bin/env python3
"""
Merge all audio training data into a single finetune JSONL.

Sources:
  1. /cmy/cmy/Qwen3-ASR/data/train.jsonl
     Fields: {"key": ..., "source": "<abs_wav_path>", "target": "<text>", ...}

  2. /cmy/cmy/Qwen3-ASR/voxflame-oss-by-account/oss-by-account/
     Each account has transcripts.txt under dataset/<user_id>/transcripts.txt
     Format: <rec_id> TAB <rel_wav_path> TAB <text>
     - Standard accounts: rel_wav_path is relative to the account root
       e.g. "supervised/mandarin/.../xxx.wav"
     - legacy__v_gv7fxwrp: rel_wav_path is relative to the account root
       e.g. "dataset/v_gv7fxwrp/s_xxxx.wav"

  3. /cmy/cmy/Qwen3-ASR/customer_audio/
     Same transcripts.txt format, rel_wav_path relative to account root.

Output: /cmy/cmy/Qwen3-ASR/data/finetune_all.jsonl
Format: {"audio": "<abs_path>", "text": "<transcript>"}
"""

import json
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import soundfile as sf

VOXFLAME_ROOT = Path("/cmy/cmy/Qwen3-ASR/voxflame-oss-by-account/oss-by-account")
CUSTOMER_ROOT = Path("/cmy/cmy/Qwen3-ASR/customer_audio")
TRAIN_JSONL   = Path("/cmy/cmy/Qwen3-ASR/data/train.jsonl")
OUTPUT_FILE   = Path("/cmy/cmy/Qwen3-ASR/data/finetune_all.jsonl")

# Skip these non-account entries in voxflame root
VOXFLAME_SKIP = {"_inventory.json", "_objects.jsonl", "unassigned"}

MAX_DURATION_S = 15.0
NUM_WORKERS    = 16


def get_duration(audio_path: str) -> float:
    """Return audio duration in seconds, or -1 on error."""
    try:
        info = sf.info(audio_path)
        return info.frames / info.samplerate
    except Exception:
        return -1.0


def filter_by_duration(records: list[dict], max_s: float = MAX_DURATION_S) -> list[dict]:
    """Keep only records whose audio is shorter than max_s seconds."""
    paths = [r["audio"] for r in records]
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as pool:
        durations = list(pool.map(get_duration, paths))

    kept, dropped_long, dropped_err = [], 0, 0
    for rec, dur in zip(records, durations):
        if dur < 0:
            dropped_err += 1
        elif dur <= max_s:
            kept.append(rec)
        else:
            dropped_long += 1
    print(f"  duration filter: kept={len(kept)}, dropped_long(>{max_s}s)={dropped_long}, dropped_err={dropped_err}")
    return kept


def load_from_train_jsonl(path: Path) -> list[dict]:
    records = []
    missing = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            audio = item.get("source") or item.get("audio") or item.get("audio_path")
            text  = item.get("target") or item.get("text") or item.get("transcript")
            if not audio or text is None:
                continue
            if not Path(audio).exists():
                missing += 1
                continue
            records.append({"audio": str(audio), "text": str(text)})
    print(f"[train.jsonl] loaded={len(records)}, missing={missing}")
    return records


def load_from_transcripts_txt(txt_file: Path, account_root: Path) -> list[dict]:
    """
    Parse a transcripts.txt and resolve wav paths relative to account_root.
    Returns list of {"audio": abs_path, "text": text}.
    """
    records = []
    missing = 0
    with open(txt_file, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                continue
            _rec_id, rel_wav, text = parts[0], parts[1], parts[2]
            abs_wav = account_root / rel_wav
            if not abs_wav.exists():
                missing += 1
                continue
            records.append({"audio": str(abs_wav), "text": text})
    return records, missing


def load_from_folder(root: Path, skip_names: set = None) -> list[dict]:
    """Walk all account subdirs under root, find transcripts.txt files."""
    all_records = []
    total_missing = 0
    skip_names = skip_names or set()

    for account_dir in sorted(root.iterdir()):
        if not account_dir.is_dir():
            continue
        if account_dir.name in skip_names:
            continue

        txt_files = list(account_dir.rglob("transcripts.txt"))
        if not txt_files:
            print(f"  [warn] no transcripts.txt in {account_dir.name}")
            continue

        account_records = []
        account_missing = 0
        for txt_file in txt_files:
            recs, miss = load_from_transcripts_txt(txt_file, account_dir)
            account_records.extend(recs)
            account_missing += miss

        print(f"  [{account_dir.name}] loaded={len(account_records)}, missing={account_missing}")
        all_records.extend(account_records)
        total_missing += account_missing

    return all_records, total_missing


def main():
    seen = set()
    all_records = []

    def add_records(records: list[dict], source_label: str):
        added = dupes = 0
        for rec in records:
            key = rec["audio"]
            if key in seen:
                dupes += 1
                continue
            seen.add(key)
            all_records.append(rec)
            added += 1
        print(f"  -> added={added}, dupes_skipped={dupes}  [{source_label}]")

    # 1. train.jsonl
    print("\n=== Source 1: train.jsonl ===")
    train_records = load_from_train_jsonl(TRAIN_JSONL)
    add_records(train_records, "train.jsonl")

    # 2. voxflame-oss-by-account
    print("\n=== Source 2: voxflame-oss-by-account ===")
    vox_records, _ = load_from_folder(VOXFLAME_ROOT, skip_names=VOXFLAME_SKIP)
    add_records(vox_records, "voxflame")

    # 3. customer_audio
    print("\n=== Source 3: customer_audio ===")
    cust_records, _ = load_from_folder(CUSTOMER_ROOT)
    add_records(cust_records, "customer_audio")

    # Filter by duration
    print(f"\n=== Filtering: keep audio <= {MAX_DURATION_S}s ===")
    all_records = filter_by_duration(all_records)

    # Write output
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for rec in all_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"\n=== Done: {len(all_records)} total records -> {OUTPUT_FILE} ===")


if __name__ == "__main__":
    main()
