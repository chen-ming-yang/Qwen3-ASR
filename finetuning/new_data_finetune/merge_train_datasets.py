"""
Merge two JSONL datasets into one, normalising all rows to {audio, text} format.

Input formats:
  - train.jsonl:          {key, source, target, speaker}
  - voxflame_train.jsonl: {audio, text}

Output:
  - train_merged.jsonl:   {audio, text}  (unified)
"""
import json
import os

PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TRAIN_PATH = os.path.join(PROJ, "data", "train.jsonl")
VOXFLAME_PATH = os.path.join(PROJ, "finetuning", "new_data_finetune", "voxflame_train.jsonl")
OUTPUT_PATH = os.path.join(PROJ, "data", "train_merged.jsonl")

def main():
    total = 0
    with open(OUTPUT_PATH, "w", encoding="utf-8") as out:
        with open(TRAIN_PATH, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line: continue
                obj = json.loads(line)
                norm = {"audio": obj["source"], "text": obj["target"]}
                out.write(json.dumps(norm, ensure_ascii=False) + "\n")
                total += 1
        print(f"[merge] Normalised + copied {total} lines from {TRAIN_PATH}")
        vc = 0
        with open(VOXFLAME_PATH, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line: continue
                out.write(line + "\n")
                vc += 1
                total += 1
        print(f"[merge] Appended {vc} lines from {VOXFLAME_PATH}")
    print(f"[merge] Total: {total} lines written to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
