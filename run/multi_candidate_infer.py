"""
Generate multiple ASR candidates using a fine-tuned Qwen3-ASR checkpoint.
For each audio, runs beam search and outputs all candidates + CER/WER.

Usage:
  python run/multi_candidate_infer.py \
      --model_path ./qwen3-asr-finetuning-out/checkpoint-5240 \
      --input_file data/test.jsonl \
      --output_file outputs/multi_candidate_results.jsonl \
      --num_candidates 4
"""
import argparse
import json
import os
import sys

import librosa
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoConfig, AutoModel, AutoProcessor, GenerationConfig

from qwen_asr import Qwen3ASRModel
from qwen_asr.core.transformers_backend import (
    Qwen3ASRConfig,
    Qwen3ASRForConditionalGeneration,
    Qwen3ASRProcessor,
)
from qwen_asr.inference.utils import SAMPLE_RATE

AutoConfig.register("qwen3_asr", Qwen3ASRConfig)
AutoModel.register(Qwen3ASRConfig, Qwen3ASRForConditionalGeneration)
AutoProcessor.register(Qwen3ASRConfig, Qwen3ASRProcessor)


def load_audio(path, sr=16000):
    wav, _ = librosa.load(path, sr=sr, mono=True)
    return wav


def compute_cer(hyp, ref):
    import jiwer
    return jiwer.cer(ref, hyp)


def compute_wer(hyp, ref):
    import jiwer
    return jiwer.wer(ref, hyp)


def is_zh(text):
    import re
    zh = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaff]")
    if not text:
        return True
    return len(zh.findall(text)) / max(len(text), 1) > 0.5


def build_messages(context, audio_placeholder=""):
    return [
        {"role": "system", "content": context or ""},
        {"role": "user", "content": [{"type": "audio", "audio": audio_placeholder}]},
    ]


def build_text_prompt(processor, context=""):
    msgs = build_messages(context=context)
    return processor.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)


@torch.no_grad()
def generate_candidates(model, processor, audio_array, context="",
                        num_candidates=4, max_new_tokens=256):
    text_prompt = build_text_prompt(processor, context)
    inputs = processor(text=[text_prompt], audio=[audio_array],
                       return_tensors="pt", padding=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items() if hasattr(v, "to")}
    if "input_features" in inputs:
        inputs["input_features"] = inputs["input_features"].to(dtype=model.dtype)
    prompt_len = inputs["input_ids"].shape[1]

    gen = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        num_beams=num_candidates,
        num_return_sequences=num_candidates,
        output_scores=True,
        
        early_stopping=False,
    )

    seqs = gen.sequences[:, prompt_len:]
    scores = gen.scores
    G = seqs.shape[0]
    log_p = torch.zeros(G, device=model.device)
    for g in range(G):
        for t, tok in enumerate(seqs[g]):
            if t < len(scores):
                lp = F.log_softmax(scores[t][g, :], dim=-1)
                log_p[g] += lp[tok]

    texts = processor.batch_decode(seqs, skip_special_tokens=True)
    texts = [t.strip() for t in texts]
    return texts, log_p.cpu().tolist()


def patch_forward(model):
    cls = model.__class__
    if getattr(cls, "_forward_patched", False):
        return
    if not hasattr(model, "thinker") or not hasattr(model.thinker, "forward"):
        raise RuntimeError("No .thinker.forward on model")

    def forward(self, input_ids=None, attention_mask=None,
                input_features=None, feature_attention_mask=None,
                labels=None, **kwargs):
        return self.thinker.forward(
            input_ids=input_ids, attention_mask=attention_mask,
            input_features=input_features,
            feature_attention_mask=feature_attention_mask,
            labels=labels, **kwargs)

    cls.forward = forward
    cls._forward_patched = True


def main():
    parser = argparse.ArgumentParser("Multi-Candidate ASR Inference")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Fine-tuned checkpoint path")
    parser.add_argument("--input_file", type=str, required=True,
                        help="JSONL with {source,target} or {audio,text}")
    parser.add_argument("--output_file", type=str, required=True,
                        help="Output JSONL with all candidates")
    parser.add_argument("--num_candidates", type=int, default=4)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8
    dtype = torch.bfloat16 if use_bf16 else torch.float16

    print(f"[infer] Loading model from {args.model_path}")
    wrapper = Qwen3ASRModel.from_pretrained(args.model_path, dtype=dtype, device_map=None)
    model = wrapper.model
    proc = wrapper.processor
    patch_forward(model)
    model.generation_config = GenerationConfig.from_model_config(model.config)
    model.to(device)
    model.eval()

    samples = []
    with open(args.input_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    print(f"[infer] Loaded {len(samples)} samples from {args.input_file}")

    results = []
    for sample in tqdm(samples, desc="Generating"):
        audio_path = sample.get("source") or sample.get("audio", "")
        ref_text = sample.get("target") or sample.get("text", "")

        if not os.path.exists(audio_path):
            print(f"[infer] WARNING: {audio_path} not found, skipping")
            continue

        audio = load_audio(audio_path, sr=SAMPLE_RATE)
        prompt = sample.get("prompt", "")

        candidates, log_probs = generate_candidates(
            model, proc, audio, prompt,
            num_candidates=args.num_candidates,
            max_new_tokens=args.max_new_tokens,
        )

        best_idx = -1
        best_err = float("inf")
        errs = []
        for i, cand in enumerate(candidates):
            if not cand.strip():
                errs.append(None)
                continue
            if is_zh(ref_text) and is_zh(cand):
                err = compute_cer(cand, ref_text)
            else:
                err = compute_wer(cand, ref_text)
            errs.append(err)
            if err < best_err:
                best_err = err
                best_idx = i

        result = {
            "audio": audio_path,
            "reference": ref_text,
            "candidates": [
                {"text": candidates[i], "log_prob": log_probs[i],
                 "cer" if is_zh(ref_text) else "wer": errs[i]}
                for i in range(len(candidates))
            ],
            "best_candidate_idx": best_idx if best_idx >= 0 else None,
            "best_" + ("cer" if is_zh(ref_text) else "wer"): best_err if best_err != float("inf") else None,
        }
        results.append(result)

    os.makedirs(os.path.dirname(args.output_file) or ".", exist_ok=True)
    with open(args.output_file, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[infer] Done: {len(results)} results written to {args.output_file}")


if __name__ == "__main__":
    main()
