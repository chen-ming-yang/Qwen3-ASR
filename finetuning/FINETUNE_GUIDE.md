# Qwen3 ASR Finetuning Guide

## Setup

```bash
pip install -U qwen-asr datasets
pip install -U flash-attn --no-build-isolation   # optional, recommended
```

> If your machine has less than 96GB of RAM and lots of CPU cores:
> ```bash
> MAX_JOBS=4 pip install -U flash-attn --no-build-isolation
> ```

## Prepare Data

Create a JSONL file (`train.jsonl`) with one JSON per line:

```jsonl
{"audio":"/data/wavs/utt0001.wav","text":"language English<asr_text>This is a test sentence."}
{"audio":"/data/wavs/utt0002.wav","text":"language Chinese<asr_text>这是一个测试。"}
{"audio":"/data/wavs/utt0003.wav","text":"language None<asr_text>No language label example."}
```

### Language Prefix

| Scenario | Prefix |
|---|---|
| Have language info (English) | `language English<asr_text>...` |
| Have language info (Chinese) | `language Chinese<asr_text>...` |
| No language info | `language None<asr_text>...` |

> If you set `language None`, the model will not learn language detection from that prefix.

## Single GPU Finetuning

```bash
python finetuning/qwen3_asr_sft.py \
  --model_path Qwen/Qwen3-ASR-1.7B \
  --train_file ./train.jsonl \
  --output_dir ./qwen3-asr-finetuning-out \
  --batch_size 32 \
  --grad_acc 4 \
  --lr 2e-5 \
  --epochs 1 \
  --save_steps 200 \
  --save_total_limit 5
```

## Multi-GPU Finetuning (torchrun)

```bash
export CUDA_VISIBLE_DEVICES=0,1
torchrun --nproc_per_node=2 finetuning/qwen3_asr_sft.py \
  --model_path Qwen/Qwen3-ASR-1.7B \
  --train_file ./train.jsonl \
  --output_dir ./qwen3-asr-finetuning-out \
  --batch_size 32 \
  --grad_acc 4 \
  --lr 2e-5 \
  --epochs 1 \
  --save_steps 200
```

## Resume Training

### Option A: Explicit checkpoint path

```bash
python finetuning/qwen3_asr_sft.py \
  --train_file ./train.jsonl \
  --output_dir ./qwen3-asr-finetuning-out \
  --resume_from ./qwen3-asr-finetuning-out/checkpoint-200
```

### Option B: Auto-resume from latest checkpoint

```bash
python finetuning/qwen3_asr_sft.py \
  --train_file ./train.jsonl \
  --output_dir ./qwen3-asr-finetuning-out \
  --resume 1
```

## All Arguments

| Argument | Default | Description |
|---|---|---|
| `--model_path` | `Qwen/Qwen3-ASR-1.7B` | HuggingFace model ID or local path |
| `--train_file` | `train.jsonl` | Path to training JSONL file |
| `--eval_file` | `""` | Path to evaluation JSONL file (optional) |
| `--output_dir` | `./qwen3-asr-finetuning-out` | Output directory for checkpoints |
| `--sr` | `16000` | Audio sampling rate |
| `--batch_size` | `32` | Per-device training batch size |
| `--grad_acc` | `4` | Gradient accumulation steps |
| `--lr` | `2e-5` | Learning rate |
| `--epochs` | `1` | Number of training epochs |
| `--log_steps` | `10` | Logging interval (steps) |
| `--lr_scheduler_type` | `linear` | Learning rate scheduler type |
| `--warmup_ratio` | `0.02` | Warmup ratio |
| `--num_workers` | `4` | DataLoader workers |
| `--pin_memory` | `1` | Pin memory (0 or 1) |
| `--persistent_workers` | `1` | Persistent workers (0 or 1) |
| `--prefetch_factor` | `2` | Prefetch factor |
| `--save_strategy` | `steps` | Save strategy |
| `--save_steps` | `200` | Save interval (steps) |
| `--save_total_limit` | `5` | Max checkpoints to keep |
| `--resume_from` | `""` | Explicit checkpoint path to resume from |
| `--resume` | `0` | Auto-resume from latest checkpoint (0 or 1) |

## Inference After Finetuning

```python
import torch
from qwen_asr import Qwen3ASRModel

model = Qwen3ASRModel.from_pretrained(
    "qwen3-asr-finetuning-out/checkpoint-200",
    dtype=torch.bfloat16,
    device_map="cuda:0",
)

results = model.transcribe(
    audio="test.wav",
)

print(results[0].language)
print(results[0].text)
```

## One-Click Shell Script

```bash
#!/usr/bin/env bash
set -e

export CUDA_VISIBLE_DEVICES=0,1

MODEL_PATH="Qwen/Qwen3-ASR-1.7B"
TRAIN_FILE="./train.jsonl"
EVAL_FILE="./eval.jsonl"
OUTPUT_DIR="./qwen3-asr-finetuning-out"

torchrun --nproc_per_node=2 finetuning/qwen3_asr_sft.py \
  --model_path ${MODEL_PATH} \
  --train_file ${TRAIN_FILE} \
  --eval_file ${EVAL_FILE} \
  --output_dir ${OUTPUT_DIR} \
  --batch_size 32 \
  --grad_acc 4 \
  --lr 2e-5 \
  --epochs 1 \
  --log_steps 10 \
  --save_strategy steps \
  --save_steps 200 \
  --save_total_limit 5 \
  --num_workers 2 \
  --pin_memory 1 \
  --persistent_workers 1 \
  --prefetch_factor 2
```
