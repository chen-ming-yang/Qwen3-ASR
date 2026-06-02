# Qwen3-ASR Inference Usage

This document explains how to run `run/infer.py`.

## 0) Download model with ModelScope (recommended in Mainland China)

```bash
pip install -U modelscope
modelscope download --model Qwen/Qwen3-ASR-1.7B --local_dir ./Qwen3-ASR-1.7B
```

Then use local path in `--model`, for example: `--model ./Qwen3-ASR-1.7B`.

## 1) JSONL mode

If your dataset looks like [data/test.jsonl](/cmy/cmy/Qwen3-ASR/data/test.jsonl), each line can contain keys such as `source`, `target`, and `key`:

```json
{"key": "S008T118E000N00040", "source": "/path/to/audio.wav", "target": "reference text"}
```

Run:

```bash
cd /cmy/cmy/Qwen3-ASR
python run/infer.py \
  --input_jsonl ./data/test.jsonl \
  --model ./Qwen3-ASR-1.7B \
  --device cuda:0 \
  --dtype bfloat16 \
  --batch_size 32 \
  --language Chinese \
  --result_file ./outputs/infer_test.jsonl

  python run/infer.py \
  --input_jsonl ./data/test.jsonl \
  --model /cmy/cmy/Qwen3-ASR/qwen3-asr-finetuning-out-bak/checkpoint-326 \
  --device cuda:0 \
  --dtype bfloat16 \
  --batch_size 32 \
  --language Chinese \
  --result_file /cmy/cmy/Qwen3-ASR/finetune_outputs/infer_test_finetune_504_326.jsonl

    python run/infer.py \
  --input_jsonl ./data/test.jsonl \
  --model /cmy/cmy/Qwen3-ASR/qwen3-asr-finetuning-out/checkpoint-3600 \
  --device cuda:0 \
  --dtype bfloat16 \
  --batch_size 32 \
  --language Chinese \
  --result_file /cmy/cmy/Qwen3-ASR/finetune_outputs/infer_test_finetune_504.jsonl

      python run/infer.py \
  --input_jsonl ./data/test.jsonl \
  --model /cmy/cmy/Qwen3-ASR/outputs/qwen3-asr-gspo-out/final \
  --device cuda:0 \
  --dtype bfloat16 \
  --batch_size 32 \
  --language Chinese \
  --result_file /cmy/cmy/Qwen3-ASR/finetune_outputs/infer_test_finetune_508.jsonl
```

If you want to run your finetuned checkpoint instead of the base model, replace `--model` with something like:

```bash
--model ./qwen3-asr-finetuning-out/checkpoint-326
```

## 2) Flat wav/txt directory mode

Directory format:

```text
data_dir/
  utt0001.wav
  utt0001.txt
  utt0002.wav
  utt0002.txt
```

Run:

```bash
cd /cmy/cmy/Qwen3-ASR
python run/infer.py \
  --data_dir ./test_data \
  --model ./Qwen3-ASR-1.7B \
  --device cuda:0 \
  --dtype bfloat16 \
  --batch_size 32
```

Run with context prompt:

```bash
python run/infer.py \
  --dataset_dir /cmy/after_catting/10h \
  --audio_subdir Audio \
  --text_subdir Text \
  --model ./Qwen3-ASR-1.7B \
  --device cuda:0 \
  --dtype bfloat16 \
  --batch_size 32 \
  --language Chinese \
  --context "no no hallucinations, you are a dysarthria speech transcript" \
  --result_file ./outputs/infer_10h_prompt.jsonl
```

## 3) Audio/Text dataset mode (your 10h dataset)

Directory format:

```text
/cmy/after_catting/10h/
  Audio/
    01/*.wav
    02/*.wav
    ...
  Text/
    01_label.txt
    02_label.txt
    ...
```

Text file format (each line):

```text
<utt_id> <transcript>
```

`infer.py` maps each line to:

```text
Audio/<shard>/<utt_id>.wav
```

Run on full dataset:

```bash
cd /cmy/cmy/Qwen3-ASR
python run/infer.py \
  --dataset_dir /cmy/after_catting/10h \
  --audio_subdir Audio \
  --text_subdir Text \
  --model ./Qwen3-ASR-1.7B \
  --device cuda:0 \
  --dtype bfloat16 \
  --batch_size 32 \
  --language Chinese \
  --result_file ./outputs/infer_10h.jsonl
```

Quick smoke test:

```bash
python run/infer.py \
  --dataset_dir /cmy/after_catting/10h \
  --model ./Qwen3-ASR-1.7B \
  --max_samples 20 \
  --result_file ./outputs/infer_10h_smoke.jsonl
```

## 4) Important options

- `--context TEXT`: context/prompt string passed as system message to the model (e.g. `--context "no hallucinations, you are a dysarthria speech transcript"`).
- `--language Chinese`: force language. Use empty string for auto-detect.
- `--max_samples N`: run only first N samples (0 means all).
- `--strict_audio 1`: fail if transcript exists but matching wav is missing.
- `--result_file PATH`: save per-utterance output as jsonl.

Each line of `result_file` includes:

```json
{"utt_id":"...","audio":"...","ref":"...","hyp":"...","cer":12.34}
```


