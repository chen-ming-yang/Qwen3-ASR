"""
Find audio samples longer than a given threshold (default 30s) in a JSONL file.
Usage:
    python find_long_audio.py <jsonl_path> [--max_duration 30] [--list_only] [--remove]
"""
import argparse
import json
import subprocess
import sys

try:
    import soundfile as sf
except ImportError:
    sf = None


def get_duration_ffprobe(path: str) -> float:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", path],
        capture_output=True, text=True, timeout=30,
    )
    return float(result.stdout.strip())


def get_duration_soundfile(path: str) -> float:
    info = sf.info(path)
    return info.duration


def main():
    parser = argparse.ArgumentParser(description="Find audio files longer than a threshold")
    parser.add_argument("jsonl_path", help="Path to the JSONL file")
    parser.add_argument("--max_duration", type=float, default=30.0,
                        help="Maximum allowed duration in seconds (default: 30)")
    parser.add_argument("--list_only", action="store_true",
                        help="Only list long audio paths, one per line")
    parser.add_argument("--remove", action="store_true",
                        help="Remove long entries from JSONL and write filtered copy")
    args = parser.parse_args()

    if sf is not None:
        get_duration = get_duration_soundfile
    else:
        get_duration = get_duration_ffprobe

    with open(args.jsonl_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    long_entries = []
    total = len(lines)

    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
        data = json.loads(line)
        audio_path = data.get("source") or data.get("audio_path") or data.get("wav")
        if not audio_path:
            print(f"WARN: line {i+1} has no audio path", file=sys.stderr)
            continue

        try:
            dur = get_duration(audio_path)
        except Exception as e:
            print(f"WARN: {audio_path}: {e}", file=sys.stderr)
            continue

        if dur > args.max_duration:
            long_entries.append((audio_path, dur, data))
            if args.list_only:
                print(audio_path)
            else:
                print(f"[{i+1}/{total}] {dur:.1f}s  {audio_path}")

    if not args.list_only:
        print(f"\n{'='*60}")
        print(f"Total entries: {total}")
        print(f"Entries > {args.max_duration}s: {len(long_entries)}")
        if long_entries:
            total_dur = sum(e[1] for e in long_entries)
            print(f"Total extra time: {total_dur:.0f}s")
            for path, dur, _ in long_entries:
                print(f"  {dur:8.1f}s  {path}")

    if args.remove and long_entries:
        long_keys = set(e[2].get("key", e[0]) for e in long_entries)
        out_path = args.jsonl_path.replace(".jsonl", "_filtered.jsonl")
        kept = 0
        with open(out_path, "w", encoding="utf-8") as fout:
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                key = data.get("key", "") or data.get("source", "")
                if key in long_keys:
                    continue
                fout.write(line + "\n")
                kept += 1
        print(f"\nFiltered JSONL written to: {out_path}")
        print(f"Kept: {kept}, Removed: {len(long_entries)}")


if __name__ == "__main__":
    main()
