# coding=utf-8
# Copyright 2026 The Alibaba Qwen team.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
import os
import re
import shutil
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import librosa
import torch
from datasets import Dataset, DatasetDict, IterableDataset, IterableDatasetDict, load_dataset
from qwen_asr import Qwen3ASRModel
from transformers import (GenerationConfig, Trainer, TrainerCallback,
                          TrainingArguments)


def patch_outer_forward(model):
    cls = model.__class__
    if getattr(cls, "_forward_patched", False):
        return

    if not hasattr(model, "thinker") or not hasattr(model.thinker, "forward"):
        raise RuntimeError(
            "Cannot patch forward: model has no `.thinker.forward`. "
            "Your qwen3_asr model may be incompatible."
        )

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        input_features=None,
        feature_attention_mask=None,
        labels=None,
        **kwargs,
    ):
        return self.thinker.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            input_features=input_features,
            feature_attention_mask=feature_attention_mask,
            labels=labels,
            **kwargs,
        )

    cls.forward = forward
    cls._forward_patched = True


_CKPT_RE = re.compile(r"^checkpoint-(\d+)$")


def find_latest_checkpoint(output_dir: str) -> Optional[str]:
    if not output_dir or not os.path.isdir(output_dir):
        return None
    best_step = None
    best_path = None
    for name in os.listdir(output_dir):
        m = _CKPT_RE.match(name)
        if not m:
            continue
        step = int(m.group(1))
        path = os.path.join(output_dir, name)
        if os.path.isdir(path) and (best_step is None or step > best_step):
            best_step = step
            best_path = path
    return best_path


def load_audio(path: str, sr: int = 16000):
    wav, _ = librosa.load(path, sr=sr, mono=True)
    return wav


def build_prefix_messages(prompt: str, audio_array):
    return [
        {"role": "system", "content": prompt or ""},
        {"role": "user", "content": [{"type": "audio", "audio": audio_array}]},
    ]


def make_preprocess_fn_prefix_only(processor):
    def _preprocess(ex: Dict[str, Any]) -> Dict[str, Any]:
        prompt = ex.get("prompt", "")
        dummy_audio = None
        prefix_msgs = build_prefix_messages(prompt, dummy_audio)
        prefix_text = processor.apply_chat_template(
            [prefix_msgs], add_generation_prompt=True, tokenize=False
        )[0]
        return {
            "prompt": prompt,
            "audio": ex["audio"],
            "target": ex["text"],
            "prefix_text": prefix_text,
        }

    return _preprocess


def build_dataset_from_audio_text_dir(
    dataset_dir: str,
    audio_subdir: str,
    text_subdir: str,
    default_prompt: str,
    strict_audio: bool,
) -> List[Dict[str, str]]:
    audio_root = os.path.join(dataset_dir, audio_subdir)
    text_root = os.path.join(dataset_dir, text_subdir)

    if not os.path.isdir(audio_root):
        raise ValueError(f"Audio directory not found: {audio_root}")
    if not os.path.isdir(text_root):
        raise ValueError(f"Text directory not found: {text_root}")

    txt_files = sorted(
        [
            os.path.join(text_root, fn)
            for fn in os.listdir(text_root)
            if fn.lower().endswith(".txt") and os.path.isfile(os.path.join(text_root, fn))
        ]
    )
    if not txt_files:
        raise ValueError(f"No .txt files found under: {text_root}")

    records: List[Dict[str, str]] = []
    missing_audio = 0
    malformed_lines = 0

    for txt_file in txt_files:
        # Shard comes from the leading number in filenames like 01_label.txt / 20-label.txt.
        base_name = os.path.basename(txt_file)
        shard_match = re.match(r"^(\d+)", base_name)
        if shard_match is None:
            raise ValueError(
                f"Cannot infer shard from text file name: {base_name}. "
                "Expected a leading numeric prefix like 01_label.txt"
            )
        shard = shard_match.group(1)

        with open(txt_file, "r", encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line:
                    continue
                parts = line.split(maxsplit=1)
                if len(parts) < 2:
                    malformed_lines += 1
                    continue

                utt_id, text = parts[0], parts[1]
                audio_path = os.path.join(audio_root, shard, f"{utt_id}.wav")

                if not os.path.exists(audio_path):
                    missing_audio += 1
                    if strict_audio:
                        raise ValueError(f"Missing audio for utt_id={utt_id}: {audio_path}")
                    continue

                records.append(
                    {
                        "audio": audio_path,
                        "text": text,
                        "prompt": default_prompt,
                    }
                )

    if not records:
        raise ValueError("No valid samples were built from dataset_dir")

    print(
        "[dataset] built from directory: "
        f"samples={len(records)}, missing_audio={missing_audio}, malformed_lines={malformed_lines}"
    )
    return records


def iter_records_from_audio_text_dir(
    dataset_dir: str,
    audio_subdir: str,
    text_subdir: str,
    default_prompt: str,
    strict_audio: bool,
):
    audio_root = os.path.join(dataset_dir, audio_subdir)
    text_root = os.path.join(dataset_dir, text_subdir)

    if not os.path.isdir(audio_root):
        raise ValueError(f"Audio directory not found: {audio_root}")
    if not os.path.isdir(text_root):
        raise ValueError(f"Text directory not found: {text_root}")

    txt_files = sorted(
        [
            os.path.join(text_root, fn)
            for fn in os.listdir(text_root)
            if fn.lower().endswith(".txt") and os.path.isfile(os.path.join(text_root, fn))
        ]
    )
    if not txt_files:
        raise ValueError(f"No .txt files found under: {text_root}")

    for txt_file in txt_files:
        base_name = os.path.basename(txt_file)
        shard_match = re.match(r"^(\d+)", base_name)
        if shard_match is None:
            raise ValueError(
                f"Cannot infer shard from text file name: {base_name}. "
                "Expected a leading numeric prefix like 01_label.txt"
            )
        shard = shard_match.group(1)

        with open(txt_file, "r", encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line:
                    continue
                parts = line.split(maxsplit=1)
                if len(parts) < 2:
                    continue

                utt_id, text = parts[0], parts[1]
                audio_path = os.path.join(audio_root, shard, f"{utt_id}.wav")

                if not os.path.exists(audio_path):
                    if strict_audio:
                        raise ValueError(f"Missing audio for utt_id={utt_id}: {audio_path}")
                    continue

                yield {
                    "audio": audio_path,
                    "text": text,
                    "prompt": default_prompt,
                }


@dataclass
class DataCollatorForQwen3ASRFinetuning:
    processor: Any
    sampling_rate: int = 16000
    max_audio_seconds: float = 0.0

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        audio_paths = [f["audio"] for f in features]
        prefix_texts = [f["prefix_text"] for f in features]
        targets = [f["target"] for f in features]

        eos = self.processor.tokenizer.eos_token or ""
        full_texts = [pfx + tgt + eos for pfx, tgt in zip(prefix_texts, targets)]
        audios = [load_audio(p, sr=self.sampling_rate) for p in audio_paths]
        if self.max_audio_seconds > 0:
            max_samples = int(self.max_audio_seconds * self.sampling_rate)
            if max_samples > 0:
                audios = [a[:max_samples] for a in audios]

        full_inputs = self.processor(
            text=full_texts,
            audio=audios,
            return_tensors="pt",
            padding=True,
            truncation=False,
        )
        prefix_inputs = self.processor(
            text=prefix_texts,
            audio=audios,
            return_tensors="pt",
            padding=True,
            truncation=False,
        )

        prefix_lens = prefix_inputs["attention_mask"].sum(dim=1).tolist()
        labels = full_inputs["input_ids"].clone()
        for i, pl in enumerate(prefix_lens):
            labels[i, :pl] = -100

        pad_id = self.processor.tokenizer.pad_token_id
        if pad_id is not None:
            labels[labels == pad_id] = -100

        full_inputs["labels"] = labels
        return full_inputs


class CastFloatInputsTrainer(Trainer):
    def _prepare_inputs(self, inputs):
        inputs = super()._prepare_inputs(inputs)
        model_dtype = getattr(self.model, "dtype", None)
        if model_dtype is not None:
            for k, v in list(inputs.items()):
                if torch.is_tensor(v) and v.is_floating_point():
                    inputs[k] = v.to(dtype=model_dtype)
        return inputs


def copy_required_hf_files_for_qwen_asr(src_dir: str, dst_dir: str):
    os.makedirs(dst_dir, exist_ok=True)
    required = [
        "config.json",
        "generation_config.json",
        "preprocessor_config.json",
        "processor_config.json",
        "tokenizer_config.json",
        "tokenizer.json",
        "special_tokens_map.json",
        "chat_template.json",
        "merges.txt",
        "vocab.json",
    ]
    for fn in required:
        src = os.path.join(src_dir, fn)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(dst_dir, fn))


class MakeEveryCheckpointInferableCallback(TrainerCallback):
    def __init__(self, base_model_path: str):
        self.base_model_path = base_model_path

    def on_save(self, args: TrainingArguments, state, control, **kwargs):
        if args.process_index != 0:
            return control

        ckpt_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        if not os.path.isdir(ckpt_dir):
            ckpt_dir = kwargs.get("checkpoint", ckpt_dir)

        copy_required_hf_files_for_qwen_asr(self.base_model_path, ckpt_dir)
        return control


def parse_args():
    p = argparse.ArgumentParser("Qwen3-ASR Finetuning")

    # Paths
    p.add_argument("--model_path", type=str, default="Qwen/Qwen3-ASR-1.7B")
    p.add_argument("--train_file", type=str, default="train.jsonl")
    p.add_argument("--eval_file", type=str, default="")
    p.add_argument("--dataset_dir", type=str, default="")
    p.add_argument("--audio_subdir", type=str, default="Audio")
    p.add_argument("--text_subdir", type=str, default="Text")
    p.add_argument("--prompt", type=str, default="")
    p.add_argument("--eval_ratio", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--strict_audio", type=int, default=0)
    p.add_argument("--output_dir", type=str, default="./qwen3-asr-finetuning-out")

    # Audio
    p.add_argument("--sr", type=int, default=16000)

    # Train hyper-params
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--grad_acc", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--epochs", type=float, default=1)
    p.add_argument("--log_steps", type=int, default=10)
    p.add_argument("--lr_scheduler_type", type=str, default="linear")
    p.add_argument("--warmup_ratio", type=float, default=0.02)
    p.add_argument("--max_steps", type=int, default=-1)
    p.add_argument("--gradient_checkpointing", type=int, default=1)
    p.add_argument(
        "--max_audio_seconds",
        type=float,
        default=0.0,
        help="Truncate each training audio to at most this duration (0 disables)",
    )
    p.add_argument(
        "--streaming",
        type=int,
        default=0,
        help="Use streaming/iterable datasets to reduce memory usage",
    )

    # DataLoader
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--pin_memory", type=int, default=1)
    p.add_argument("--persistent_workers", type=int, default=1)
    p.add_argument("--prefetch_factor", type=int, default=2)

    # Save
    p.add_argument("--save_strategy", type=str, default="steps")
    p.add_argument("--save_steps", type=int, default=200)
    p.add_argument("--save_total_limit", type=int, default=5)

    # Resume
    p.add_argument("--resume_from", type=str, default="")
    p.add_argument("--resume", type=int, default=0)

    return p.parse_args()


def main():
    args_cli = parse_args()

    if not args_cli.dataset_dir and not args_cli.train_file:
        raise ValueError("TRAIN_FILE is required (json/jsonl). Needs fields: audio, text, optional prompt")

    if args_cli.eval_ratio < 0 or args_cli.eval_ratio >= 1:
        raise ValueError("--eval_ratio must be in [0, 1)")

    use_streaming = args_cli.streaming == 1
    if use_streaming and args_cli.eval_ratio > 0:
        raise ValueError("--eval_ratio is not supported with --streaming 1; use --eval_file")
    if use_streaming and args_cli.max_steps <= 0:
        raise ValueError("--max_steps must be > 0 when --streaming 1 is enabled")

    use_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8
    asr_wrapper = Qwen3ASRModel.from_pretrained(
        args_cli.model_path,
        dtype=torch.bfloat16 if use_bf16 else torch.float16,
        device_map=None,
    )
    model = asr_wrapper.model
    processor = asr_wrapper.processor

    patch_outer_forward(model)
    model.generation_config = GenerationConfig.from_model_config(model.config)

    use_gradient_checkpointing = args_cli.gradient_checkpointing == 1
    if use_gradient_checkpointing:
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
        if hasattr(model, "config") and hasattr(model.config, "use_cache"):
            model.config.use_cache = False

    if args_cli.dataset_dir:
        if use_streaming:
            train_ds = IterableDataset.from_generator(
                iter_records_from_audio_text_dir,
                gen_kwargs={
                    "dataset_dir": args_cli.dataset_dir,
                    "audio_subdir": args_cli.audio_subdir,
                    "text_subdir": args_cli.text_subdir,
                    "default_prompt": args_cli.prompt,
                    "strict_audio": (args_cli.strict_audio == 1),
                },
            )
            if args_cli.eval_file:
                eval_raw = load_dataset(
                    "json",
                    data_files={"validation": args_cli.eval_file},
                    streaming=True,
                )
                raw_ds = IterableDatasetDict(
                    {"train": train_ds, "validation": eval_raw["validation"]}
                )
            else:
                raw_ds = IterableDatasetDict({"train": train_ds})
        else:
            records = build_dataset_from_audio_text_dir(
                dataset_dir=args_cli.dataset_dir,
                audio_subdir=args_cli.audio_subdir,
                text_subdir=args_cli.text_subdir,
                default_prompt=args_cli.prompt,
                strict_audio=(args_cli.strict_audio == 1),
            )

            train_ds = Dataset.from_list(records)
            if args_cli.eval_file:
                eval_raw = load_dataset("json", data_files={"validation": args_cli.eval_file})
                raw_ds = DatasetDict(
                    {"train": train_ds, "validation": eval_raw["validation"]}
                )
            elif args_cli.eval_ratio > 0:
                split = train_ds.train_test_split(
                    test_size=args_cli.eval_ratio, seed=args_cli.seed
                )
                raw_ds = DatasetDict({"train": split["train"], "validation": split["test"]})
            else:
                raw_ds = DatasetDict({"train": train_ds})
    else:
        raw_ds = load_dataset(
            "json",
            data_files={
                "train": args_cli.train_file,
                **({"validation": args_cli.eval_file} if args_cli.eval_file else {}),
            },
            streaming=use_streaming,
        )

    preprocess_fn = make_preprocess_fn_prefix_only(processor)
    if use_streaming:
        ds = raw_ds.map(preprocess_fn)
    else:
        ds = raw_ds.map(preprocess_fn, num_proc=1)

    keep = {"prompt", "audio", "target", "prefix_text"}
    for split in ds.keys():
        cols = getattr(ds[split], "column_names", None)
        if not cols:
            continue
        drop = [c for c in cols if c not in keep]
        if drop:
            ds[split] = ds[split].remove_columns(drop)

    if args_cli.max_audio_seconds > 0:
        print(f"[train] max_audio_seconds={args_cli.max_audio_seconds} (audio will be truncated)")

    collator = DataCollatorForQwen3ASRFinetuning(
        processor=processor,
        sampling_rate=args_cli.sr,
        max_audio_seconds=args_cli.max_audio_seconds,
    )

    training_args = TrainingArguments(
        output_dir=args_cli.output_dir,
        per_device_train_batch_size=args_cli.batch_size,
        gradient_accumulation_steps=args_cli.grad_acc,
        learning_rate=args_cli.lr,
        num_train_epochs=args_cli.epochs,
        max_steps=args_cli.max_steps,
        logging_steps=args_cli.log_steps,
        lr_scheduler_type=args_cli.lr_scheduler_type,
        warmup_ratio=args_cli.warmup_ratio,
        gradient_checkpointing=use_gradient_checkpointing,
        dataloader_num_workers=args_cli.num_workers,
        dataloader_pin_memory=(args_cli.pin_memory == 1),
        dataloader_persistent_workers=(args_cli.persistent_workers == 1 and args_cli.num_workers > 0),
        dataloader_prefetch_factor=args_cli.prefetch_factor if args_cli.num_workers > 0 else None,
        save_strategy=args_cli.save_strategy,
        save_steps=args_cli.save_steps,
        save_total_limit=args_cli.save_total_limit,
        save_safetensors=True,
        eval_strategy="steps" if ("validation" in ds) else "no",
        eval_steps=args_cli.save_steps,
        do_eval=("validation" in ds),
        bf16=use_bf16,
        fp16=not use_bf16,
        ddp_find_unused_parameters=False,
        remove_unused_columns=False,
        report_to="none",
    )

    trainer = CastFloatInputsTrainer(
        model=model,
        args=training_args,
        train_dataset=ds["train"],
        eval_dataset=ds.get("validation", None),
        data_collator=collator,
        processing_class=processor,
        callbacks=[MakeEveryCheckpointInferableCallback(base_model_path=args_cli.model_path)],
    )

    resume_from = (args_cli.resume_from or "").strip()
    if not resume_from and args_cli.resume == 1:
        resume_from = find_latest_checkpoint(training_args.output_dir) or ""

    if resume_from:
        if trainer.args.process_index == 0:
            print(f"[resume] resume_from_checkpoint = {resume_from}")
        trainer.train(resume_from_checkpoint=resume_from)
    else:
        trainer.train()


if __name__ == "__main__":
    main()
