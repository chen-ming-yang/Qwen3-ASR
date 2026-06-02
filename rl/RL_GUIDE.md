# GSPO: Grouped Supervised Policy Optimization for Qwen3-ASR

## 概述

GSPO（Grouped Supervised Policy Optimization）是 Qwen3-ASR 使用的强化学习微调方法，专门针对 ASR 任务设计。它**不需要价值模型（critic/value model）*)，所有计算都在整句（sequence-level）级别完成，避免训练崩溃和灾难遗忘。

核心思想：对同一条语音，用 beam search 生成多组候选文本，利用识别准确率（CER/WER）作为奖励信号，在组内做相对比较来更新模型。

## 算法流程

### 第 1 步：分组生成（Group Generation）

对一条语�3 `x`，使用束搜索（beam search）生成 `G` 条候选文本：

```
{y1, y2, ..., y_G}  <- beam_search(x; pi_old)
```

- `G = 4`（默认值，可通过 `--group_size` 调整）
- 使用**参考模型 `pi_old`**（当前模型做 forward，不累积梯度）生成
- `output_scores=True` 同时返回每步的 logits，用于计算 `log pi_old(y_i|x)`

### 第 2 步：奖励计算（Reward）

直接使用识别准确率作为奖励：

```
r_i = 1 - CER(y_i, ref)    # 中文
r_i = 1 - WER(y_i, ref)    # 英文
```

- 通过字符分布比例自动判断语言（`_is_mainly_chinese`）
- CER 用 `jiwer.cer`，WER 用 `jiwer.wer`
- 空假设直接得 `r = 0.0`

### 第 3 步：分组相对优势（Group-Relative Advantage）

在一条语音的 `G` 个候选**内部**做标准化，不依赖任何价值模型：

```
A_i = (r_i - mean(r_group)) / (std(r_group) + epsilon)
```

- `epsilon = 1e-8` 防止除零
- **只关心组内相对好坏**：即使全局 CER 偏高，组内最好的候选仍能得到正优势
- 这是 GSPO 稳定性的关键设计

### 第 4 步：序列级重要性权重（Length-Normalized Importance Weight）

```
s_i(theta) = exp((log pi_theta(y_i|x) - log pi_old(y_i|x)) / |y_i|)
```

- `log pi_old`：来自 beam search 时记录的 scores（逐 token log-softmax 后求和）
- `log pi_theta`：当前模型对同一组候选再做一次 forward pass，从 logits 中提取
- **长度归一化**：除以 `|y_i|`，皍保长短句公平更新
- 整句共享一个权重，不逐 token 计算

### 第 5 步：序列级裁剪（Sequence-Level Clipping）

```
L_GSPO = -mean_i[ A_i * clip(s_i, eps_low, eps_high) ]
```

- `eps_low = 3e-4`，`eps_high = 4e-4`
- 整句统一裁剪，不逐字裁剪
- 防止单步更新幅度过大，保证训练稳定性

### 第 6 步：模型更新

```
theta = theta - alpha * grad L_GSPO
```

- 优化器：`AdamW(lr=1e-6)`
- 支持梯度累积（`--gradient_accumulation_steps=4`）
- 梯度裁剪 `max_grad_norm=1.0`
- Cosine LR scheduler

## 完整数据流

```
语音 x
  |
  v
[beam_generate]
  * beam search G=4
  * output_scores=True
  * log pi_old 从 scores 提取
  |
  +-- y1 --+ compute_asr_reward --+ r1
  +-- y2 --+ compute_asr_reward --+ r2
  +-- y3 --+ compute_asr_reward --+ r3
  +-- y4 --+ compute_asr_reward --+ r4
  |
  v
[Group Advantage]
  A_i = (ri - mu) / sigma
  |
  v
[compute_sequence_log_probs]
  pi_theta forward pass
  log pi_theta from logits.gather()
  |
  v
[gspo_loss]
  s_i = exp((pi_theta - pi_old) / |y_i|)
  clip(s_i, 3e-4, 4e-4)
  loss = -mean[A_i * clip(s_i)]
  |
  v
optimizer.step()
```

## 训练数据格式

JSONL 格式，每行一个 JSON 对象（与 SFT 训练兼容）：

```json
{"source": "/path/to/audio.wav", "target": "正确的转写文本"}
```

可选字段：
- `"audio"`: 音频路径（与 `source` 二选一）
- `"text"`: 转写文本（与 `target` 二选一）
- `"prompt"`: 可选的系统提示文本

## 运行命令

```bash
cd Qwen3-ASR
python rl/rl_learning.py \
  --model_path /cmy/cmy/Qwen3-ASR/qwen3-asr-finetuning-out-bak/checkpoint-326 \
  --train_file /cmy/cmy/Qwen3-ASR/data/val.jsonl \
  --output_dir ./outputs/qwen3-asr-gspo-out \
  --batch_size 2 \
  --gradient_accumulation_steps 4 \
  --lr 3e-6 \
  --group_size 4 \
  --eps_low 0.7 \
  --eps_high 1.3 \
  --save_steps 500 \
  --num_epochs 2
```

## 关键超参数

| 参数 | 默认值 | 含义 |
|---|---|---|
| `--group_size` | 4 | 每组候选数 G，也即 beam width |
| `--eps_low` | 3e-4 | 重要性权重裁剪下限 |
| `--eps_high` | 4e-4 | 重要性权重裁剪上限 |
| `--lr` | 1e-6 | 学习率（RL 阶段应远小于 SFT） |
| `--gradient_accumulation_steps` | 4 | 梯度累积步数 |
| `--max_grad_norm` | 1.0 | 梯度裁剪阈值 |
| `--max_new_tokens` | 256 | 每句最多生成 token 数 |

## 与标准 RLHF / PPO 的关键区别

| 维度 | 标准 PPO | GSPO（欬实现） |
|---|---|---|
| 价值模型 | 需要 critic | 不需要�|
| 优势计算 | GAE + value function | 组内相对标准化 |
| 重要性采样 | 逐 token | 整句统一（长度归一化） |
| 裁剪 | PPO clip 逐 token | 序列级 clamp |
| KL 惩罚 | 通常需要 | 由 clip 隐式约束 |
| 奖励模型 | 需要额外训练 | 直接用 CER/WER |

## 为什么 GSPO 适合 ASR

1. **ASR 的奖励天然可计算**：不需要训练 reward model，标注文本本身就能算 CER/WER
2. **组内对比消除噪声**：同一条语音的 beam 结果共享声学条件，相对好坏更可信
3. **序列级更新防止崩溃**：不涉及逐 token 的 credit assignment，训练稳定
4. **无价值模型省显存**：batch 内只需要放模型的一份参数
