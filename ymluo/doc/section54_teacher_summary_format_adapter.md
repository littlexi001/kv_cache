# 第 54 节：teacher summary + teacher-summary-format adapter 实验

日期：2026-07-03

## 0. 目标

第 53 节做的是：

```text
teacher/generative summary 直接替换 learned extractive summary
然后用旧的 learned-extractive LoRA adapter 评估
```

结果是：

```text
reference learned adapter + learned_static_hier:
  PPL = 24.64
  input tokens = 2042
  speedup = 4.56x

reference learned adapter + teacher_static_hier:
  PPL = 25.95
  input tokens = 1247
  speedup = 6.91x
```

teacher summary 更省 token、更快，但 PPL 稍差。一个合理怀疑是：旧 adapter 不是在 teacher summary 格式上训练的，所以不公平。

本节做真正的：

```text
teacher summary + teacher-summary-format adapter
```

也就是用 teacher 生成的 summary memory 作为训练输入，重新训练 LoRA adapter。

## 1. 新增脚本

```text
ymluo/projects/learned_hierarchical_summary_memory/src/run_teacher_summary_lora_adaptation.py
```

这个脚本做三件事：

```text
1. 用 Qwen3-4B-Instruct 为训练/eval block 生成 S10/S100/S1000，并缓存到 jsonl。
2. 构造训练样本：
   [teacher summary memory] + [recent raw] -> continuation
3. 训练 LoRA adapter，并和 base / reference learned adapter 比较。
```

teacher summary cache：

```text
/home/fdong/ymluo/projects/learned_hierarchical_summary_memory/outputs/teacher_summary_lora_4texts_s1000_20260703/teacher_summary_cache.jsonl
```

本次缓存了约 64 个 block summary。后续复跑同一数据和同一 prompt 版本会复用缓存，不需要重新调用 4B teacher。

## 2. 数据和设置

沿用第 53 节 teacher 对照的同一组 4 个文本：

```text
moby_dick
pride_prejudice
origin_species
republic
```

训练/eval 切分：

```text
train:
  每个文本 4 个窗口
  start = 0, 2048, 4096, 6144
  total train examples = 16

eval:
  每个文本 2 个窗口
  start = 40000, 42048
  total eval examples = 8
```

共同设置：

```text
base model = /home/fdong/hrj/prove/Qwen3-0.6B
teacher model = /home/fdong/models/Qwen3-4B-Instruct
prefill_tokens = 8192
target_tokens = 128
block_tokens = 2048
recent_tokens = 512
LoRA r = 8
LoRA alpha = 16
LoRA target modules = q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj
```

teacher 生成格式：

```text
S10: <=10 words keywords
S100: <=100 words dense summary
S1000: <=250 words detailed memory
```

训练输入：

```text
Static memory summaries:
  teacher S10/S100/S1000 routed by static_hier

Recent raw text:
  last 512 raw tokens

Target:
  next 128 tokens
```

loss 只算 target tokens，不算 prompt tokens。

## 3. 对照对象

本节比较三个 phase：

```text
base:
  原始 Qwen3-0.6B，无 LoRA

reference_learned_adapter:
  第 52/53 节最好的 learned-extractive adapter
  不是 teacher summary 格式训练出来的

teacher_summary_adapter:
  本节新训练的 teacher-summary-format adapter
```

评估方法：

```text
full_raw
learned_static_hier
teacher_static_hier
```

## 4. 主要结果

### 4.1 直接替换 teacher summary 的 baseline

这是第 53 节已有结果，也是本节比较基线：

| phase | method | PPL | input tokens | speedup |
| --- | --- | ---: | ---: | ---: |
| reference_learned_adapter | learned_static_hier | 24.64 | 2042 | 4.56x |
| reference_learned_adapter | teacher_static_hier | 25.95 | 1247 | 6.91x |

也就是说，teacher summary 直接替换后：

```text
PPL 比 learned_static_hier 差约 1.31
但 token 从 2042 降到 1247
速度从 4.56x 提到约 6.91x
```

### 4.2 teacher-summary-format adapter: 不同步数

本节尝试了多个训练设置：

| setting | teacher_static_hier PPL | full_raw PPL | 结论 |
| --- | ---: | ---: | --- |
| 5 steps, lr=2e-4 | 26.18 | 23.40 | 略差于直接替换 |
| 10 steps, lr=2e-4 | 25.46 | 22.85 | 当前最好 |
| 20 steps, lr=2e-4 | 25.85 | 23.48 | 开始变差 |
| 30 steps, lr=2e-4 | 28.60 | 26.05 | 明显过拟合 |
| 80 steps, lr=2e-4 | 73.48 | 65.08 | 崩掉 |
| 100 steps, lr=1e-5 | 26.08 | 23.53 | 稳但不如 10 steps |
| 300 steps, lr=5e-5 | 1453.02 | 1149.70 | 崩掉 |
| 1000 steps, lr=2e-4 | 2045.89 | 1304.21 | 崩掉 |

最好的 teacher-summary-format adapter 是：

```text
10 steps, lr=2e-4
```

结果：

| phase | method | PPL | input tokens | token ratio | speedup |
| --- | --- | ---: | ---: | ---: | ---: |
| base | full_raw | 28.32 | 8320 | 100.0% | 1.00x |
| base | teacher_static_hier | 31.02 | 1247 | 15.0% | 8.21x |
| reference_learned_adapter | learned_static_hier | 24.64 | 2042 | 24.5% | 4.56x |
| reference_learned_adapter | teacher_static_hier | 25.95 | 1247 | 15.0% | 6.91x |
| teacher_summary_adapter, 10 steps | teacher_static_hier | 25.46 | 1247 | 15.0% | 6.90x |

## 5. 怎么理解这个结果

### 5.1 teacher adapter 有用，但还没有超过 learned extractive

teacher-summary-format adapter 把 teacher_static_hier 从：

```text
25.95 -> 25.46
```

说明“适配 teacher summary 格式”确实有效。

但它还没有超过：

```text
learned_static_hier = 24.64
```

当前最好点是：

```text
teacher adapter:
  PPL = 25.46
  token ratio = 15.0%
  speedup = 6.90x

learned extractive adapter:
  PPL = 24.64
  token ratio = 24.5%
  speedup = 4.56x
```

所以 teacher adapter 目前是更快、更省 token，但 PPL 仍然略差。

### 5.2 小训练集非常容易过拟合

训练集只有：

```text
4 texts x 4 windows = 16 train examples
```

因此 80/300/1000 steps 都过拟合，甚至把 full_raw PPL 也破坏了：

```text
1000 steps:
  teacher_static_hier PPL = 2045.89
  full_raw PPL = 1304.21
```

这不是 teacher summary 方法本身失败，而是训练 recipe 明显不合理。当前实验只说明：

```text
teacher-summary-format adapter 需要更大的 teacher-summary 训练集和 validation-based early stopping。
```

### 5.3 为什么 10 steps 反而最好

因为它只做了很轻的格式适配：

```text
模型还没有记住 16 个训练窗口；
但已经稍微学会 teacher summary 的输入风格。
```

超过 20-30 steps 后，模型开始快速向这 16 个训练样本过拟合。

## 6. 和 learned extractive 的公平比较

当前 learned extractive adapter 是在更多训练窗口上训练过的，而 teacher adapter 只有 16 个 teacher-summary 训练窗口。

所以这次不是最终公平比较，而是 proof：

```text
teacher summary 可以训练 adapter；
但小数据下必须 early stop；
直接 1k steps 会严重过拟合。
```

更公平的下一步应该是：

```text
1. 为更多自然长文本预生成 teacher summaries。
2. 至少构造数百到数千个 teacher-summary train examples。
3. 加 validation split，每 20-50 steps 评估一次。
4. 保存 validation teacher_static_hier PPL 最好的 checkpoint。
5. 再和 learned-extractive adapter 比：
   - PPL
   - token ratio
   - speed
   - LongBench / RULER
```

## 7. 当前结论

在相同 4 文本数据上：

```text
teacher summary + teacher-summary-format adapter 是有效的，
但当前小数据训练下还没有超过 learned extractive adapter。
```

最好的 teacher adapter：

```text
teacher_static_hier:
  PPL = 25.46
  input tokens = 1247
  token ratio = 15.0%
  speedup = 6.90x
```

最好的 learned extractive adapter：

```text
learned_static_hier:
  PPL = 24.64
  input tokens = 2042
  token ratio = 24.5%
  speedup = 4.56x
```

因此：

```text
如果目标是最低 PPL：
  当前 learned extractive 更好。

如果目标是更高速度和更低 token：
  teacher summary 更有潜力。

如果要让 teacher summary 赢：
  需要更大的 teacher-summary 训练集 + validation-based early stopping。
```

