#!/usr/bin/env python3
"""CER regression analysis: baseline (finetune only) vs RL (finetune+GSPO)."""

import json
import sys

with open('/cmy/cmy/Qwen3-ASR/finetune_outputs/infer_test_finetune_baseline.jsonl') as f:
    base = {l['utt_id']: l for l in (json.loads(line) for line in f if line.strip())}

with open('/cmy/cmy/Qwen3-ASR/finetune_outputs/infer_test_finetune_508.jsonl') as f:
    rl = {l['utt_id']: l for l in (json.loads(line) for line in f if line.strip())}

common = [k for k in base if k in rl]
print(f"Common utterances: {len(common)}")

cer_base = sum(base[k].get('cer', 0) for k in common) / len(common)
cer_rl = sum(rl[k].get('cer', 0) for k in common) / len(common)
print(f"\n=== OVERALL CER ===")
print(f"Baseline (finetune):        {cer_base:.4f}")
print(f"RL (finetune + GSPO):       {cer_rl:.4f}")
print(f"Change:                     {(cer_rl - cer_base)*100:+.2f} pp")

worse, better, same = [], [], []
for k in common:
    cb = base[k].get('cer', 0)
    cr = rl[k].get('cer', 0)
    if cr > cb + 1e-6:
        worse.append((cr - cb, cb, cr, base[k]['ref'], base[k]['hyp'], rl[k]['hyp'], k))
    elif cb > cr + 1e-6:
        better.append((cb - cr, cb, cr, base[k]['ref'], base[k]['hyp'], rl[k]['hyp'], k))
    else:
        same.append(k)

print(f"\n=== DISTRIBUTION ===")
print(f"Worsened:  {len(worse)} ({100*len(worse)/len(common):.1f}%)")
print(f"Improved:  {len(better)} ({100*len(better)/len(common):.1f}%)")
print(f"Unchanged: {len(same)} ({100*len(same)/len(common):.1f}%)")

worse.sort(key=lambda x: -x[0])
print(f"\n=== TOP 20 REGRESSIONS ===")
for d, cb, cr, ref, hb, hr, uid in worse[:20]:
    print(f"  +{d:.2f}  b={cb:.2f} rl={cr:.2f}  ref={ref[:30]}  base={hb[:30]}  rl={hr[:30]}")

better.sort(key=lambda x: -x[0])
print(f"\n=== TOP 20 IMPROVEMENTS ===")
for d, cb, cr, ref, hb, hr, uid in better[:20]:
    print(f"  -{d:.2f}  b={cb:.2f} rl={cr:.2f}  ref={ref[:30]}  base={hb[:30]}  rl={hr[:30]}")

empty_base = sum(1 for k in common if not base[k].get('hyp', '').strip())
empty_rl = sum(1 for k in common if not rl[k].get('hyp', '').strip())
print(f"\nEmpty hyp: base={empty_base}  rl={empty_rl}")

hard = sum(1 for k in common if base[k].get('cer', 0) > 0.5)
hw = sum(1 for k in common if base[k].get('cer', 0) > 0.5 and rl[k].get('cer', 0) > base[k].get('cer', 0) + 1e-6)
easy = sum(1 for k in common if base[k].get('cer', 0) < 1e-6)
ew = sum(1 for k in common if base[k].get('cer', 0) < 1e-6 and rl[k].get('cer', 0) > base[k].get('cer', 0) + 1e-6)
print(f"Hard (base CER>0.5): {hard}, worsened: {hw}")
print(f"Perfect (base CER=0): {easy}, worsened: {ew}")

print(f"\n=== PER-SPEAKER ===")
speakers = {}
for k in common:
    spk = k.split('T')[0] if 'T' in k else k[:4]
    s = speakers.setdefault(spk, {'n': 0, 'bc': 0, 'rc': 0})
    s['n'] += 1; s['bc'] += base[k].get('cer', 0); s['rc'] += rl[k].get('cer', 0)
for spk in sorted(speakers):
    s = speakers[spk]
    bc = s['bc']/s['n']; rc = s['rc']/s['n']
    print(f"  {spk:8s} n={s['n']:4d}  base={bc:.4f}  rl={rc:.4f}  {(rc-bc)*100:+.2f}pp")
