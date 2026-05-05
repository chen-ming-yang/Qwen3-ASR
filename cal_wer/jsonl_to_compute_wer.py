#!/usr/bin/env python3
"""Convert Qwen3-ASR JSONL output to compute-wer input files.

Input JSONL line example:
{"utt_id": "xxx", "ref": "...", "hyp": "...", ...}

Output file format for compute-wer:
utt_id text
"""

from __future__ import annotations

import argparse
import json
import subprocess
import unicodedata
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert ASR JSONL to compute-wer ref/hyp text files."
    )
    parser.add_argument("input_jsonl", type=Path, help="Path to JSONL result file")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("."),
        help="Output directory for ref/hyp files (default: current directory)",
    )
    parser.add_argument(
        "--ref-out",
        default="ref_from_jsonl.txt",
        help="Output reference filename under out-dir (default: ref_from_jsonl.txt)",
    )
    parser.add_argument(
        "--hyp-out",
        default="hyp_from_jsonl.txt",
        help="Output hypothesis filename under out-dir (default: hyp_from_jsonl.txt)",
    )
    parser.add_argument(
        "--min-ref-chars",
        type=int,
        default=2,
        help="Only keep samples where ref length >= this value (default: 2)",
    )
    parser.add_argument(
        "--utt-key",
        default="utt_id",
        help="Utterance id field name in JSON (default: utt_id)",
    )
    parser.add_argument(
        "--ref-key",
        default="ref",
        help="Reference text field name in JSON (default: ref)",
    )
    parser.add_argument(
        "--hyp-key",
        default="hyp",
        help="Hypothesis text field name in JSON (default: hyp)",
    )
    parser.add_argument(
        "--strip-hyp-punctuation",
        action="store_true",
        help="Remove all punctuation from hypothesis text before writing output",
    )
    parser.add_argument(
        "--run-compute-wer",
        action="store_true",
        help="Run compute-wer after writing ref/hyp files",
    )
    parser.add_argument(
        "--compute-wer-bin",
        default="compute-wer",
        help="compute-wer executable name/path (default: compute-wer)",
    )
    return parser


def normalize_text(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip()


def remove_punctuation(text: str) -> str:
    # Remove all unicode punctuation characters.
    return "".join(ch for ch in text if not unicodedata.category(ch).startswith("P"))


def main() -> None:
    args = build_parser().parse_args()

    ref_out = args.out_dir / args.ref_out
    hyp_out = args.out_dir / args.hyp_out

    ref_out.parent.mkdir(parents=True, exist_ok=True)
    hyp_out.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    written = 0
    skipped = 0
    filtered = 0

    with args.input_jsonl.open("r", encoding="utf-8") as fin, \
        ref_out.open("w", encoding="utf-8") as fref, \
        hyp_out.open("w", encoding="utf-8") as fhyp:
        for line_no, raw_line in enumerate(fin, start=1):
            line = raw_line.strip()
            if not line:
                continue
            total += 1
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                continue

            if not isinstance(item, dict):
                skipped += 1
                continue

            utt = normalize_text(item.get(args.utt_key))
            ref = normalize_text(item.get(args.ref_key))
            hyp = normalize_text(item.get(args.hyp_key))

            if args.strip_hyp_punctuation:
                hyp = remove_punctuation(hyp)

            if not utt:
                skipped += 1
                continue

            if len(ref) < args.min_ref_chars:
                filtered += 1
                continue

            fref.write(f"{utt} {ref}\n")
            fhyp.write(f"{utt} {hyp}\n")
            written += 1

    print(
        f"Done. total={total}, written={written}, filtered={filtered}, skipped={skipped}"
    )
    print(f"ref file: {ref_out}")
    print(f"hyp file: {hyp_out}")

    if args.run_compute_wer:
        cmd = [args.compute_wer_bin, str(ref_out), str(hyp_out)]
        print(f"Running: {' '.join(cmd)}")
        try:
            subprocess.run(cmd, check=True)
        except FileNotFoundError:
            raise SystemExit(
                "compute-wer command not found. "
                "Use --compute-wer-bin with full path or ensure it is in PATH."
            )
        except subprocess.CalledProcessError as exc:
            raise SystemExit(exc.returncode)


if __name__ == "__main__":
    main()
