# 第 51 节：LoRA 微调让模型适应 Static Summary Memory + Recent Raw

日期：2026-07-03

## 0. 目标

第 50 节直接把原始 Qwen3-0.6B 用在：

```text
[static summary memory] + [recent raw] -> next-token prediction
```

结果显示能加速，但 PPL 明显差于 full raw。原因是原始模型没有训练过这种上下文格式。

本节做一个最小 LoRA adaptation 实验：

```text
训练输入：
  static_hier summary memory + recent raw

训练目标：
  只在 continuation / eval tokens 上计算 causal LM loss

目标：
  让模型学会把 summary memory 当作远程历史上下文使用
```

## 1. 新增脚本

```text
ymluo/projects/learned_hierarchical_summary_memory/src/run_static_summary_lora_adaptation.py
```

服务器输出：

```text
/home/fdong/ymluo/projects/learned_hierarchical_summary_memory/outputs/static_summary_lora_adapt_8k_s120_20260703
```

本地输出：

```text
ymluo/projects/learned_hierarchical_summary_memory/outputs/static_summary_lora_adapt_8k_s120_20260703
```

LoRA adapter：

```text
ymluo/projects/learned_hierarchical_summary_memory/outputs/static_summary_lora_adapt_8k_s120_20260703/adapter
```

adapter size 约 20MB，不是整模型。

## 2. 训练设置

模型：

```text
Qwen3-0.6B
model = /home/fdong/hrj/prove/Qwen3-0.6B
dtype = float16
GPU = RTX 3090
```

数据：

```text
War and Peace
Count of Monte Cristo
```

训练样本格式：

```text
prefix raw tokens = 8192
target continuation tokens = 128
recent raw = 512
block size = 2048
train method = static_hier
```

`static_hier` 格式：

```text
远处 older blocks: summary10
中间 older blocks: summary100
最近的 older block: summary1000
最后拼接 recent raw 512 tokens
```

LoRA：

```text
r = 8
alpha = 16
dropout = 0.05
target modules = q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj
trainable params = 5.05M
trainable fraction = 0.84%
```

训练：

```text
train_steps = 120
wall time = 55s
loss 只算 continuation tokens
prompt / summary tokens 不算 loss
```

评测：

```text
heldout offset = 150000 tokens
eval samples = 4 / text
total eval samples = 8
```

注意：这是小规模 proof-of-concept，不是正式泛化实验。

## 3. 结果

### 3.1 Base Qwen3-0.6B

| 方法 | PPL | input tokens | speedup vs full |
| --- | ---: | ---: | ---: |
| full_raw | 21.9957 | 8320.0 | 1.00x |
| recent_only | 25.1583 | 640.0 | 11.74x |
| static_hier | 23.9290 | 2141.5 | 5.35x |
| static_sum100 | 25.6744 | 1198.6 | 9.74x |
| static_sum1000 | 23.6539 | 5426.2 | 1.88x |

### 3.2 After LoRA adaptation

| 方法 | PPL | input tokens | speedup vs full |
| --- | ---: | ---: | ---: |
| full_raw | 16.4818 | 8320.0 | 1.00x |
| recent_only | 19.8494 | 640.0 | 10.88x |
| static_hier | 17.5598 | 2141.5 | 4.84x |
| static_sum100 | 18.9234 | 1198.6 | 8.88x |
| static_sum1000 | 17.1528 | 5426.2 | 1.70x |

## 4. 关键观察

### 4.1 训练后 static_hier 明显变好

```text
base static_hier:
  PPL = 23.9290

adapted static_hier:
  PPL = 17.5598
```

PPL 大幅下降，说明模型确实学会了使用 `[summary memory] + [recent raw]` 的格式。

### 4.2 static_hier 已经超过未微调 full_raw

```text
base full_raw:
  PPL = 21.9957

adapted static_hier:
  PPL = 17.5598
  speedup = 4.84x
```

这说明“微调适应 summary memory 格式”是有效方向。

### 4.3 和同一个 adapted 模型的 full_raw 比，仍有小差距

```text
adapted full_raw:
  PPL = 16.4818

adapted static_hier:
  PPL = 17.5598
  speedup = 4.84x
```

差距约 1.08 PPL，但换来约 4.84x forward 加速和约 25.7% 输入 token。

### 4.4 static_sum1000 最接近 full_raw，但速度收益较小

```text
adapted static_sum1000:
  PPL = 17.1528
  speedup = 1.70x
```

它保留了更多远程信息，所以 PPL 更接近 full，但输入 token 仍有 65.2% full。

## 5. 结论

可以，微调模型适应 `[summary memory] + [recent raw]` 后，PPL 和速度可以同时变得不错。

当前最重要的结果是：

```text
adapted static_hier:
  PPL = 17.56
  input tokens = 25.7% full
  speedup = 4.84x

adapted full_raw:
  PPL = 16.48
  input tokens = 100% full
  speedup = 1.00x
```

这比第 50 节“不训练直接换上下文格式”的结果强很多。

## 6. 限制

这还是一个小规模 proof：

```text
1. 只训练 120 steps。
2. 只用了两本文学文本。
3. heldout 是同两本书的后续 token 区间，不是跨数据域泛化。
4. summary 仍是 extractive heuristic，不是 learned summarizer。
```

所以它不能直接作为最终论文级结果，但足以说明路线可行。

## 7. 下一步

更正式的实验应该做：

```text
1. 训练更久：1k / 5k / 10k steps。
2. 数据更多：多本文本或 DCLM 子集。
3. summary 换成 trained summarizer 生成，而不是 heuristic extraction。
4. 加 curriculum：
   recent_only -> static_sum1000 -> static_hier
5. 同时评测：
   PPL
   wall time
   LongBench / RULER / QA
   summary route mix
```

如果后续能保持 `PPL gap < 1` 且 `speedup > 4x`，这个方向就很有研究价值。

