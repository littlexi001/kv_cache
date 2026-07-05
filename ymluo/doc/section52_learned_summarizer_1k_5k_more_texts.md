# 第 52 节：learned summarizer、更多文本、1k/5k LoRA 训练实验

日期：2026-07-03

## 0. 目标

这一节验证三个问题：

```text
1. 把 heuristic summary 换成 learned summarizer 后，static summary memory 是否还能工作。
2. LoRA 适配从 120 steps 扩到 1k / 5k steps 后，PPL 和速度如何变化。
3. 加入更多文本后，结果是否更稳。
```

结论先写在前面：

```text
最可信的 longbooks 实验里：

base full_raw:
  PPL = 26.12
  input tokens = 8320

1k-step adapted static_hier:
  PPL = 21.08
  input tokens = 2136.2, 约 25.7% full_raw
  speedup = 4.41x vs adapted full_raw

但是：
  5k steps + lr=2e-4 明显过训练，PPL 变差。
  5k steps + lr=5e-5 仍然比 base 好，但不如 1k。
```

所以当前实验支持这个方向，但也说明不能简单地把训练步数加长；需要 validation、学习率 schedule 和更干净的数据。

## 1. 新增和修改的代码

新增或修改：

```text
ymluo/projects/learned_hierarchical_summary_memory/src/run_static_summary_ppl_speed.py
ymluo/projects/learned_hierarchical_summary_memory/src/run_static_summary_lora_adaptation.py
ymluo/projects/learned_hierarchical_summary_memory/src/run_learned_summary_lora_moretexts.sh
ymluo/projects/learned_hierarchical_summary_memory/src/run_learned_summary_lora_longbooks.sh
```

主要新增参数：

```text
--summary_backend heuristic|learned
--learned_summary_train_tokens
--learned_summary_epochs
--learned_summary_hidden_dim
--learned_summary_lr
--learned_summary_max_sentences
--learned_summary_seed
```

如果 `summary_backend=learned`，脚本会先训练一个小的 learned summarizer，再用它生成 `sum10 / sum100 / sum1000`。

后续脚本也会保存 learned summarizer：

```text
learned_summary_scorer.pt
```

里面包括：

```text
state_dict
feature mean/std
metadata
```

## 2. learned summarizer 是什么

这次实现的是一个轻量 learned extractive summarizer：

```text
输入：一个 block 里的候选句子
输出：每个句子的保留分数
模型：MLP sentence scorer
hidden_dim = 32
feature_dim = 13
训练标签：由原 heuristic ranker 产生的 pseudo label
```

它不是 mean pooling，也不是 query-dependent summary。它是 query-independent 的静态知识压缩，符合当前方法：

```text
原文 block -> learned summarizer -> static summary memory
query 前向时只看 summary memory + recent raw
```

但要注意：这还不是最终理想形态。它仍然是 pseudo-label extractive summarizer，不是人工标注或大模型蒸馏出来的 generative summarizer。它的价值是先验证“learned summary memory 这条链路是否可行”。

longbooks 实验里的 summarizer 训练统计：

```text
blocks = 约 118
sentences = 6021
positive_rate = 0.116
train_loss = 1.0573
train_accuracy = 0.597
```

mixed more-texts 实验里的 summarizer 训练统计：

```text
blocks = 50
sentences = 3368
positive_rate = 0.087
train_loss = 1.1063
train_accuracy = 0.562
```

## 3. 实验设置

共同设置：

```text
model = /home/fdong/hrj/prove/Qwen3-0.6B
dtype = float16
GPU = RTX 3090
prefill raw tokens = 8192
target continuation tokens = 128
recent raw tokens = 512
block size = 2048
train method = static_hier
eval methods = full_raw,recent_only,static_hier,static_sum100,static_sum1000
LoRA r = 8
LoRA alpha = 16
LoRA dropout = 0.05
trainable params = 5.05M, 约 0.84%
```

`static_hier` 的拼接方式：

```text
远处 block: sum10
中间 block: sum100
最近的 older block: sum1000
最后拼 recent raw 512 tokens
```

### 3.1 mixed more-texts 数据

数据：

```text
War and Peace
Count of Monte Cristo
topic_texts/science
topic_texts/history
topic_texts/finance
topic_texts/software
topic_texts/literature
topic_texts/mixed_qa
pcic_needle_style_validation
pcic_ruler_style_validation
```

评测：

```text
eval_start_tokens = 12000
eval_samples_per_dataset = 2
total eval samples = 20
```

这个设置覆盖文本更多，但有一个明显问题：topic/PCIC 文本比较模板化，base PPL 已经低到约 1.94，所以它更适合作为“更多文本压力测试”，不适合作为主要 PPL 结论。

### 3.2 longbooks 真实长文本数据

数据：

```text
War and Peace
Count of Monte Cristo
```

评测：

```text
train_span_tokens = 120000
eval_start_tokens = 150000
eval_samples_per_dataset = 8
total eval samples = 16
```

这个设置更干净：train/eval 分离更明显，PPL 也处在正常范围，所以作为主要结论。

## 4. 结果一：mixed more-texts

服务器输出：

```text
/home/fdong/ymluo/projects/learned_hierarchical_summary_memory/outputs/static_summary_lora_learned_moretexts_s1000_20260703
/home/fdong/ymluo/projects/learned_hierarchical_summary_memory/outputs/static_summary_lora_learned_moretexts_s5000_20260703
```

### 4.1 1k steps, lr=2e-4

| phase | method | PPL | input tokens | speedup vs full |
| --- | --- | ---: | ---: | ---: |
| base | full_raw | 1.9437 | 8320.0 | 1.00x |
| base | static_hier | 1.9947 | 2183.9 | 4.88x |
| adapted | full_raw | 2.3147 | 8320.0 | 1.00x |
| adapted | static_hier | 2.3538 | 2183.9 | 4.30x |

### 4.2 5k steps, lr=2e-4

| phase | method | PPL | input tokens | speedup vs full |
| --- | --- | ---: | ---: | ---: |
| base | full_raw | 1.9437 | 8320.0 | 1.00x |
| base | static_hier | 1.9947 | 2183.9 | 4.89x |
| adapted | full_raw | 2.6405 | 8320.0 | 1.00x |
| adapted | static_hier | 2.6748 | 2183.9 | 4.32x |

mixed more-texts 上 LoRA 变差，不应该被解读为方法失败。更合理的解释是：

```text
1. 这个 eval 太容易，base PPL 已经接近 2。
2. topic/PCIC 文本模板化，训练 loss 大量 step 接近 0。
3. 继续训练会把模型推向这些模板分布，导致 heldout PPL 变差。
```

因此 mixed more-texts 的主要价值是暴露数据质量风险，而不是给最终效果背书。

## 5. 结果二：longbooks 真实长文本

服务器输出：

```text
/home/fdong/ymluo/projects/learned_hierarchical_summary_memory/outputs/static_summary_lora_learned_longbooks_s1000_20260703
/home/fdong/ymluo/projects/learned_hierarchical_summary_memory/outputs/static_summary_lora_learned_longbooks_s5000_20260703
/home/fdong/ymluo/projects/learned_hierarchical_summary_memory/outputs/static_summary_lora_learned_longbooks_s5000_lr5e5_20260703
```

### 5.1 1k steps, lr=2e-4

| phase | method | PPL | input tokens | token ratio | speedup vs full |
| --- | --- | ---: | ---: | ---: | ---: |
| base | full_raw | 26.1219 | 8320.0 | 100.0% | 1.00x |
| base | static_hier | 27.9612 | 2136.2 | 25.7% | 5.08x |
| base | static_sum100 | 28.4991 | 1205.1 | 14.5% | 9.16x |
| base | static_sum1000 | 28.1105 | 5474.4 | 65.8% | 1.77x |
| adapted | full_raw | 20.4062 | 8320.0 | 100.0% | 1.00x |
| adapted | static_hier | 21.0762 | 2136.2 | 25.7% | 4.41x |
| adapted | static_sum100 | 21.7994 | 1205.1 | 14.5% | 7.32x |
| adapted | static_sum1000 | 20.8959 | 5474.4 | 65.8% | 1.63x |

最关键的是：

```text
base full_raw PPL = 26.12
adapted static_hier PPL = 21.08
adapted static_hier input tokens = 25.7% full_raw
adapted static_hier speedup = 4.41x vs adapted full_raw
```

这说明 learned summary memory + LoRA adaptation 这条路线是成立的。模型学会了把 `[summary memory] + [recent raw]` 当成长上下文替代品使用。

### 5.2 5k steps, lr=2e-4

| phase | method | PPL | input tokens | speedup vs full |
| --- | --- | ---: | ---: | ---: |
| base | full_raw | 26.1219 | 8320.0 | 1.00x |
| base | static_hier | 27.9612 | 2136.2 | 5.15x |
| adapted | full_raw | 30.0002 | 8320.0 | 1.00x |
| adapted | static_hier | 31.7276 | 2136.2 | 4.46x |

5k steps 用同样的 `lr=2e-4` 会过训练。PPL 不仅没有继续下降，反而比 base 更差。

### 5.3 5k steps, lr=5e-5

| phase | method | PPL | input tokens | speedup vs full |
| --- | --- | ---: | ---: | ---: |
| base | full_raw | 26.1219 | 8320.0 | 1.00x |
| base | static_hier | 27.9612 | 2136.2 | 5.07x |
| adapted | full_raw | 23.1962 | 8320.0 | 1.00x |
| adapted | static_hier | 23.9869 | 2136.2 | 4.45x |
| adapted | static_sum100 | 24.7787 | 1205.1 | 7.68x |
| adapted | static_sum1000 | 23.6922 | 5474.4 | 1.63x |

低学习率 5k 不再崩，但仍然不如 1k：

```text
1k lr=2e-4 adapted static_hier PPL = 21.08
5k lr=5e-5 adapted static_hier PPL = 23.99
```

当前最优不是“训练越久越好”，而是 1k 左右已经接近这个小数据设置下的最佳点。

## 6. 综合结论

### 6.1 learned summarizer 链路成立

learned summarizer 生成的静态 summary memory 可以用于普通文本生成 PPL 任务；再用 LoRA 适配输入格式后，`static_hier` 能在长书评测上同时取得：

```text
更低 PPL：27.96 -> 21.08
更少 token：2136 / 8320 = 25.7%
明显速度收益：4.41x vs adapted full_raw
```

### 6.2 1k 比 5k 更好

这次最好的 checkpoint 是：

```text
longbooks
train_steps = 1000
learning_rate = 2e-4
summary_backend = learned
method = static_hier
```

5k 不是不能做，但需要更正式的训练策略：

```text
validation-based early stopping
learning rate decay
更大的自然文本训练集
混合训练 full_raw / static_sum1000 / static_hier
```

### 6.3 “更多文本”必须注意数据质量

这次 mixed more-texts 说明了一个重要风险：

```text
文本来源更多，不等于评测更可信。
如果里面有模板化数据，PPL 会过低，LoRA 很容易把模型训偏。
```

所以后续更应该加的是自然长文本，例如 PG19、Books3 子集、DCLM 子集、arXiv 长文，而不是大量模板化 needle/PCIC 文本。

## 7. 对这个研究 idea 的判断

现在的证据更支持这个方向：

```text
完整 raw KV cache 不是唯一选择。
把远程上下文压缩成 learned static summary memory，再保留 recent raw，是可以让模型重新适配的。
```

但它还没有到最终论文级别，原因是：

```text
1. learned summarizer 仍是 pseudo-label extractive，不是强 generative summarizer。
2. 数据还太小。
3. 5k 会过拟合，训练 recipe 还没稳定。
4. 当前速度是 prompt token 级别的 wall time，不是 CUDA kernel 级 KV cache 优化。
```

下一步最值得做：

```text
1. 用更强的 teacher 生成 block summaries，训练一个真正的 generative summarizer。
2. 训练集换成更大自然文本集合，保留独立 heldout。
3. LoRA 加 validation early stopping 和 lr schedule。
4. 同时评测 PPL、LongBench/RULER、真实 wall time。
5. 将 summary memory 做成预计算缓存，避免每次 prompt 里重新拼接 summary 文本。
```

