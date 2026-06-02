import json, random

best_cers = []
oracle_cer = []
best_log_probs = []
num_cands_set = set()
top1_is_best = 0

with open('/cmy/cmy/Qwen3-ASR/outputs/candidates_results.jsonl') as f:
    lines = f.readlines()

for line in lines:
    d = json.loads(line)
    cers = [c['cer'] for c in d['candidates']]
    lps = [c['log_prob'] for c in d['candidates']]
    num_cands_set.add(len(d['candidates']))
    best_cers.append(d['best_cer'])
    best_log_probs.append(d['candidates'][d['best_candidate_idx']]['log_prob'])
    oracle_cer.append(min(cers))
    if d['best_candidate_idx'] == 0:
        top1_is_best += 1

N = len(best_cers)
sorted_cer = sorted(best_cers)

print(f'Total samples: {N}')
print(f'Candidates per sample: {num_cands_set}')
print()
print(f'Best CER statistics:')
print(f'  Mean:   {sum(best_cers)/N:.4f}')
print(f'  Median: {sorted_cer[N//2]:.4f}')
print(f'  Min:    {min(best_cers):.4f}')
print(f'  Max:    {max(best_cers):.4f}')
print()

exact = sum(1 for c in best_cers if c == 0)
print(f'Exact match (CER=0): {exact}/{N} ({exact/N*100:.1f}%)')

for lo,hi,lbl in [(0,0.05,'0-5%'), (0.05,0.1,'5-10%'), (0.1,0.2,'10-20%'), (0.2,0.3,'20-30%'), (0.3,0.5,'30-50%'), (0.5,1.01,'50-100%')]:
    cnt = sum(1 for c in best_cers if lo <= c < hi)
    print(f'  CER [{lbl}]: {cnt} ({cnt/N*100:.1f}%)')

print()
print(f'Oracle CER (best among all candidates): mean={sum(oracle_cer)/N:.4f}')
print(f'Oracle gap (oracle - best): mean={sum(o-b for o,b in zip(oracle_cer,best_cers))/N:.4f}')
print()

print(f'Top beam (candidate 0) selected as best: {top1_is_best}/{N} ({top1_is_best/N*100:.1f}%)')
print(f'Mean best log_prob: {sum(best_log_probs)/N:.4f}')
print()

# Hardest samples
random.seed(42)
examples = [(i,best_cers[i],oracle_cer[i]) for i in range(N)]
examples.sort(key=lambda x:-x[1])
print('Top 5 hardest samples (highest best CER):')
for idx,cer,o in examples[:5]:
    d = json.loads(lines[idx])
    best = d['candidates'][d['best_candidate_idx']]
    print(f'  #{idx} best_cer={cer:.3f} oracle={o:.3f}')
    print(f'   ref:  {d["reference"][:80]}')
    print(f'    best: {best["text"][:80]}')
    print()

# Best samples
print('5 random good samples (CER <= 0.05):')
good = [(i,best_cers[i]) for i in range(N) if best_cers[i] <= 0.05]
random.shuffle(good)
for idx,cer in good[:5]:
    d = json.loads(lines[idx])
    best = d['candidates'][d['best_candidate_idx']]
    print(f'  #{idx} cer={cer:.3f}')
    print(f'    ref:  {d["reference"][:80]}')
    print(f'    best: {best["text"][:80]}')
    print()
