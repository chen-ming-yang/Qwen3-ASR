"""
Recalculate CER from an existing jsonl result file with punctuation stripping.

Usage:
    python run/recalc_cer.py --input ./run/infer_10h_prompt.jsonl
"""

import argparse
import json

PUNCTUATION = set("。，、；：？！""''【】《》（）(){}[],.;:?!'\"\n\r\t·…—─")


def strip_punctuation(text: str) -> str:
    return "".join(ch for ch in text if ch not in PUNCTUATION)


def compute_cer(ref: str, hyp: str) -> tuple:
    r = list(ref.replace(" ", ""))
    h = list(hyp.replace(" ", ""))
    n, m = len(r), len(h)
    if n == 0:
        return m, 0
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        d[i][0] = i
    for j in range(m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if r[i - 1] == h[j - 1] else 1
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + cost)
    return d[n][m], n


def main():
    parser = argparse.ArgumentParser(description="Recalculate CER with punctuation stripping")
    parser.add_argument("--input", type=str, required=True, help="Input jsonl file")
    parser.add_argument("--output", type=str, default="", help="Optional output jsonl with recalculated CER")
    args = parser.parse_args()

    items = []
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))

    total_errors_orig = 0
    total_ref_len_orig = 0
    total_errors_new = 0
    total_ref_len_new = 0

    for item in items:
        ref = item["ref"]
        hyp = item["hyp"]

        # Original CER (no stripping)
        e_orig, n_orig = compute_cer(ref, hyp)
        total_errors_orig += e_orig
        total_ref_len_orig += n_orig

        # New CER (with punctuation stripping)
        ref_clean = strip_punctuation(ref)
        hyp_clean = strip_punctuation(hyp)
        e_new, n_new = compute_cer(ref_clean, hyp_clean)
        total_errors_new += e_new
        total_ref_len_new += n_new

        item["cer_orig"] = e_orig / n_orig * 100 if n_orig > 0 else 0.0
        item["cer_new"] = e_new / n_new * 100 if n_new > 0 else 0.0

    cer_orig = total_errors_orig / total_ref_len_orig * 100 if total_ref_len_orig > 0 else 0.0
    cer_new = total_errors_new / total_ref_len_new * 100 if total_ref_len_new > 0 else 0.0

    print(f"Total utterances: {len(items)}")
    print(f"Original CER (with punc):    {cer_orig:.2f}%  (errors={total_errors_orig}, ref_chars={total_ref_len_orig})")
    print(f"New CER (punc stripped):      {cer_new:.2f}%  (errors={total_errors_new}, ref_chars={total_ref_len_new})")
    print(f"CER reduction:               {cer_orig - cer_new:.2f}%")

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            for item in items:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"Saved to: {args.output}")


if __name__ == "__main__":
    main()
