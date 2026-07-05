# 第 53 节：多数据泛化评估与 teacher/generative summarizer 对照

日期：2026-07-03

## 0. 目标

上一节最好的结果是：

```text
learned summarizer + static_hier + LoRA 1k steps

base full_raw:
  PPL = 26.12
  input tokens = 8320

adapted static_hier:
  PPL = 21.08
  input tokens = 2136, 约 25.7% full_raw
  speedup = 4.41x
```

本节继续验证两个问题：

```text
1. 这个 longbooks 1k adapter 能不能迁移到更多不同文本。
2. 把 summarizer 换成更强的 teacher/generative summarizer 后，和 heuristic / learned extractive 比如何。
```

## 1. 新增脚本

```text
ymluo/projects/learned_hierarchical_summary_memory/src/prepare_public_domain_eval_texts.py
ymluo/projects/learned_hierarchical_summary_memory/src/run_lora_generalization_eval.py
ymluo/projects/learned_hierarchical_summary_memory/src/run_teacher_summary_compare_eval.py
```

其中：

```text
prepare_public_domain_eval_texts.py
  下载并清洗公共领域长文本。

run_lora_generalization_eval.py
  加载已经训练好的 LoRA adapter，在新文本上做 zero-shot PPL/speed 评估。

run_teacher_summary_compare_eval.py
  对比 heuristic_static_hier / learned_static_hier / teacher_static_hier。
```

## 2. 多数据泛化实验

### 2.1 数据

从 Project Gutenberg 下载并清洗 10 个公共领域文本：

| dataset | words | 类型 |
| --- | ---: | --- |
| moby_dick | 212796 | 小说 |
| pride_prejudice | 127359 | 小说 |
| tale_two_cities | 135886 | 小说 |
| sherlock_holmes | 104506 | 短篇侦探小说 |
| dracula | 161321 | 小说 |
| frankenstein | 75042 | 小说 |
| origin_species | 155526 | 科学文本 |
| republic | 216285 | 哲学文本 |
| walden | 115813 | 散文/哲学 |
| time_machine | 32454 | 小说，较短 |

服务器数据目录：

```text
/home/fdong/ymluo/projects/learned_hierarchical_summary_memory/data/public_domain_eval
```

### 2.2 设置

加载上一节的 adapter：

```text
/home/fdong/ymluo/projects/learned_hierarchical_summary_memory/outputs/static_summary_lora_learned_longbooks_s1000_20260703/adapter
```

注意：这次没有在新文本上重新训练 LoRA，只是 zero-shot 评估 adapter 的迁移性。

共同参数：

```text
model = Qwen3-0.6B
summary_backend = learned
prefill_tokens = 8192
eval_tokens = 128
recent_tokens = 512
block_tokens = 2048
samples_per_dataset = 3
eval_start_tokens = 40000
```

`time_machine` 太短，只有 1 个有效 8k 窗口，所以总体样本数是 28，不是 30。

输出目录：

```text
/home/fdong/ymluo/projects/learned_hierarchical_summary_memory/outputs/lora_generalization_public_domain_10texts_20260703
```

本地摘要：

```text
ymluo/projects/learned_hierarchical_summary_memory/outputs/lora_generalization_public_domain_10texts_20260703
```

### 2.3 总体结果

| phase | method | samples | PPL | input tokens | token ratio | speedup vs full |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| base | full_raw | 28 | 29.1872 | 8320.0 | 100.0% | 1.00x |
| base | static_hier | 28 | 30.5898 | 2045.4 | 24.6% | 5.13x |
| base | static_sum100 | 28 | 30.8921 | 1158.8 | 13.9% | 9.28x |
| base | static_sum1000 | 28 | 29.8766 | 5170.5 | 62.1% | 1.90x |
| adapted | full_raw | 28 | 24.8648 | 8320.0 | 100.0% | 1.00x |
| adapted | static_hier | 28 | 25.9926 | 2045.4 | 24.6% | 4.57x |
| adapted | static_sum100 | 28 | 26.7784 | 1158.8 | 13.9% | 7.22x |
| adapted | static_sum1000 | 28 | 25.4190 | 5170.5 | 62.1% | 1.77x |

核心观察：

```text
base full_raw PPL = 29.19
adapted static_hier PPL = 25.99
adapted static_hier input tokens = 24.6% full_raw
adapted static_hier speedup = 4.57x
```

这说明上一节在 War and Peace / Count of Monte Cristo 上训练出来的 adapter，并不是只记住那两本书；它在 10 个新文本上仍然能让 summary memory 格式工作。

### 2.4 分数据集结果

这里看 `adapted static_hier` 和 `base full_raw` 的差值：

| dataset | base full_raw PPL | adapted static_hier PPL | 差值 |
| --- | ---: | ---: | ---: |
| dracula | 24.64 | 22.79 | -1.85 |
| frankenstein | 21.57 | 18.12 | -3.45 |
| moby | 33.11 | 25.34 | -7.77 |
| origin | 18.59 | 15.61 | -2.98 |
| pride | 25.07 | 24.68 | -0.39 |
| republic | 32.53 | 31.70 | -0.83 |
| sherlock | 23.35 | 22.22 | -1.14 |
| tale | 40.87 | 36.99 | -3.88 |
| walden | 47.01 | 38.43 | -8.58 |
| time_machine | 61.85 | 62.80 | +0.95 |

结论：

```text
大部分文本上 adapted static_hier 都优于 base full_raw。
收益最大的是 Moby-Dick、Walden、A Tale of Two Cities。
Pride / Republic 收益较小。
Time Machine 只有 1 个有效窗口，不能作为稳定结论。
```

## 3. Teacher/generative summarizer 对照

### 3.1 方法

teacher 模型：

```text
/home/fdong/models/Qwen3-4B-Instruct
```

对每个 older block，teacher 生成：

```text
S10: <=10 words keywords
S100: <=100 words dense summary
S1000: <=250 words detailed memory
```

然后按同样的 `static_hier` 路由拼接：

```text
远处 block: S10
中间 block: S100
最近 older block: S1000
最后拼 recent raw 512 tokens
```

注意：这次只是“直接替换 summarizer”的 inference-time 对照。LoRA adapter 仍然是上一节在 learned extractive summary 风格上训练的 adapter，并没有重新用 teacher summary 格式训练。

### 3.2 数据和设置

选 4 个不同类型文本：

```text
moby_dick
pride_prejudice
origin_species
republic
```

共同参数：

```text
samples_per_dataset = 2
total samples = 8
prefill_tokens = 8192
eval_tokens = 128
recent_tokens = 512
```

输出目录：

```text
/home/fdong/ymluo/projects/learned_hierarchical_summary_memory/outputs/teacher_summary_compare_4texts_20260703
```

### 3.3 结果

| phase | method | samples | PPL | input tokens | token ratio | speedup vs full |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| base | full_raw | 8 | 28.3155 | 8320.0 | 100.0% | 1.00x |
| base | heuristic_static_hier | 8 | 30.3465 | 2054.2 | 24.7% | 5.39x |
| base | learned_static_hier | 8 | 29.9526 | 2041.9 | 24.5% | 5.19x |
| base | teacher_static_hier | 8 | 31.0190 | 1247.0 | 15.0% | 8.27x |
| adapted | full_raw | 8 | 23.0376 | 8320.0 | 100.0% | 1.00x |
| adapted | heuristic_static_hier | 8 | 24.6466 | 2054.2 | 24.7% | 4.55x |
| adapted | learned_static_hier | 8 | 24.6438 | 2041.9 | 24.5% | 4.56x |
| adapted | teacher_static_hier | 8 | 25.9538 | 1247.0 | 15.0% | 6.91x |

## 4. Teacher summary 结果怎么理解

teacher summary 这次没有赢 PPL：

```text
adapted learned_static_hier:
  PPL = 24.64
  input tokens = 2041.9
  speedup = 4.56x

adapted teacher_static_hier:
  PPL = 25.95
  input tokens = 1247.0
  speedup = 6.91x
```

但它也不是无效：

```text
teacher_static_hier 用更少 token：
  1247 / 8320 = 15.0% full_raw

速度更快：
  6.91x vs adapted full_raw

PPL 仍优于 base full_raw：
  25.95 vs 28.32
```

所以它现在更像是一个更激进的压缩点：牺牲一些 PPL，换更少 token 和更高速度。

这次 teacher 没有超过 learned extractive，主要有三个原因：

```text
1. adapter 是在 learned/extractive summary 格式上训练的，不是 teacher summary 格式。
2. teacher prompt 的 S1000 限制在 <=250 words，实际 token 明显少于 learned_static_hier。
3. Qwen3-4B-Instruct 生成的是自然语言摘要，和 next-token prediction 需要的“保留原文局部统计/措辞线索”不完全一致。
```

换句话说，teacher/generative summarizer 方向仍然值得做，但不能只在 inference 时直接替换。正确的下一步应该是：

```text
teacher/generative summaries -> 重新训练 LoRA adapter -> 再评估 PPL/speed
```

## 5. 本节结论

### 5.1 多数据泛化是正向的

同一个 longbooks 1k adapter 在 10 个新文本上仍然有效：

```text
adapted static_hier:
  PPL = 25.99
  input tokens = 24.6% full_raw
  speedup = 4.57x

base full_raw:
  PPL = 29.19
```

这说明当前 adapter 学到的不只是某两本书的内容，而是一定程度上学会了：

```text
[summary memory] + [recent raw] -> continuation
```

这种输入格式。

### 5.2 teacher/generative summary 直接替换暂时不是最优

teacher summary 对照说明：

```text
learned/extractive static_hier:
  PPL 更好

teacher/generative static_hier:
  token 更少、速度更快
```

因此下一阶段不应该简单宣布 teacher summary 更好，而应该做 teacher-summary-format adapter：

```text
1. 用 Qwen3-4B-Instruct 或更强 teacher 预生成 block summaries。
2. 用这些 teacher summaries 训练 LoRA。
3. 和 learned extractive adapter 在同样数据、同样 token budget 下比较。
4. 增加 validation-based early stopping。
```

如果 teacher-summary-format adapter 能把 PPL 拉回到 learned/extractive 的水平，同时保持 15% token ratio，那会比当前 25% token ratio 的 static_hier 更有价值。

