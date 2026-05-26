# Section 12 - Structured Language Synthetic Data Experiment

> 新增日期：2026-05-26  
> 对应项目：`ymluo/projects/qwen3_moe_structured_language`

## 1. 实验背景

之前的 hierarchical synthetic data 有很强的垂直层级结构：

```text
high-level unit -> low-level unit -> raw token
```

在这种数据上，同一个 high-level unit 内部的 token 会反复、整齐地共同出现。因此 attention-cluster 方法很容易学到：

```text
同一个 high-level unit 的 token -> 路由到相同 expert
```

但真实文本没有这么干净。真实文本里，高 attention pair 可能来自多种关系：

- 同一主题；
- 同一实体；
- copy / retrieval；
- 语法依赖；
- function word / 标点 / filler；
- 跨主题 bridge；
- 暂时性的训练噪声。

所以本实验的目标是构造一种更接近真实文本范式的人工数据，用来研究：

```text
high attention pair 是否真的应该被强制路由到同一个 expert？
```

这个项目不是为了追求 synthetic task 上的最高 accuracy，而是为了做一类可控的诊断数据：既保留真实文本中的主题、实体、歧义、噪声、跨主题混合，又能通过 metadata 精确知道每个 token 的 latent label。

## 2. 数据合成方式

本项目新增的数据模式是：

```text
SYNTHETIC_DATA_MODE=structured_language
```

实现位置：

```text
fdong/scripts/utils/data_utils.py
class StructuredLanguageData
```

训练目标仍然是普通 causal LM next-token prediction。数据集先生成 `SEQ_LEN + 1` 个 token，然后构造：

```text
source = token_ids[:-1]
target = token_ids[1:]
```

因此模型看到前面的 token，预测下一个 token。不同之处在于 token 序列不是完全随机，也不是严格层级 block，而是由多个“伪语言模板”拼接而成。

### 2.1 Token 类型

默认小数据配置下，token 被分成几组：

| 类型 | 默认数量 | 含义 |
| --- | ---: | --- |
| topic token | 8 | 主题标记，例如“体育/金融/科技”的抽象版本 |
| private entity token | 8 per topic | 每个 topic 私有的实体 |
| shared entity token | 16 | 多个 topic 都可能使用的歧义实体 |
| verb token | 12 | 动作或关系 token |
| function token | 12 | 类似虚词、连接词、标点 |
| noise token | 32 | filler/noise token |

大数据配置下，这些数量被放大：

| 类型 | 大数据数量 |
| --- | ---: |
| topic token | 32 |
| private entity token | 16 per topic |
| shared entity token | 128 |
| verb token | 32 |
| function token | 32 |
| noise token | 128 |

这些 token 都是整数 id，不经过真实 tokenizer。例如 topic token 可能是 `1..8`，entity token 是后面的若干 id。模型只看到数字序列，但 metadata 会记录每个 token 的 latent role/topic/entity/span/template。

### 2.2 Topic Span

数据不是逐 token 独立采样，而是按 span 生成。每个 span 先采样一个 topic：

```text
topic_id ~ Zipf(alpha=1.1)
```

这意味着 topic 分布不均匀，少数 topic 高频，多数 topic 低频，更接近真实语料中的长尾主题/词频分布。

每个 span 包含若干个模板单元：

```text
STRUCTURED_MIN_SPAN_UNITS=2
STRUCTURED_MAX_SPAN_UNITS=8    # 小数据默认
STRUCTURED_MAX_SPAN_UNITS=10   # 大数据默认
```

一个样本会不断生成 span，直到 token 数量达到 `SEQ_LEN + 1`。

### 2.3 三类模板

每个 span 内部会生成多条短模板。模板有三种：`statement`、`copy`、`bridge`。

#### 2.3.1 Statement Template

statement 是普通陈述结构：

```text
topic function entity [maybe noise] verb object function
```

概念上类似：

```text
[topic=体育] 的 [entity=湖人] [noise] 击败 [object=勇士] 。
```

它提供了最基本的 topic 内相关结构。entity 和 object 通常属于当前 topic，但可能被 shared entity 替换，从而制造歧义。

#### 2.3.2 Copy Template

copy 模板制造延迟复制依赖：

```text
topic entity function/noise ... verb same_entity
```

概念上类似：

```text
[topic=科技] [entity=苹果公司] ... 提到 [entity=苹果公司]
```

最后一个 entity 会复制前面出现过的 entity。这个模板迫使模型利用历史信息，而不只是学习局部 bigram。

#### 2.3.3 Bridge Template

bridge 模板故意把两个 topic 放在一个局部上下文里：

```text
topicA entityA function topicB entityB
```

概念上类似：

```text
[topic=金融] [entity=英伟达股票] 和 [topic=科技] [entity=芯片]
```

这个模板很关键。它打破了“局部共现/高 attention = 同 topic”的简单假设。真实文本里也经常有跨主题引用、比较、转折、实体桥接。

### 2.4 歧义 Entity

数据中有两类 entity：

```text
private entity: 只属于某个 topic
shared entity: 多个 topic 都可能使用
```

采样 entity 时，有一定概率使用 shared entity：

```text
STRUCTURED_AMBIGUITY_RATE=0.35  # 小数据
STRUCTURED_AMBIGUITY_RATE=0.70  # 大数据
```

这模拟真实文本中的多义词/跨领域实体。例如“Apple”可以是水果，也可以是公司。这样 router 不能只根据 token id 判断 topic，必须结合上下文。

### 2.5 Noise Token

模板内部会以一定概率插入 noise token：

```text
STRUCTURED_NOISE_RATE=0.25  # 小数据
STRUCTURED_NOISE_RATE=0.50  # 大数据
```

noise token 的作用是让序列不再像干净 grammar 一样规则。它模拟真实文本中的 filler、标点、连接片段、无明确语义归属 token。attention 可能会看到这些 token，但这些 token 不一定应该参与 topic-level expert 聚类。

### 2.6 Metadata

每个 token 对应 5 个 metadata 字段：

```text
metadata[:, :, 0] = syntax role id
metadata[:, :, 1] = topic id
metadata[:, :, 2] = entity id
metadata[:, :, 3] = span id
metadata[:, :, 4] = template id
```

其中 role id 包括：

| role | 含义 |
| ---: | --- |
| 0 | noise |
| 1 | topic |
| 2 | entity |
| 3 | verb |
| 4 | object |
| 5 | function |
| 6 | copy target |

当前复用旧指标时：

```text
same_higher_by_layer
```

读取的是 `metadata[:, :, 1]`，因此在本项目里表示：

```text
same-topic token pair 被路由到同一个 expert 的比例
```

`higher_mass_by_layer` 也同理，表示 attention 落在历史同 topic token 上的质量。

## 3. 实验设置

本轮实验有四组：

| 组别 | 数据规模 | Attention Cluster | RUN_NAME |
| --- | --- | --- | --- |
| baseline | 小数据 | 关闭 | `structured-baseline-no-attn-cluster` |
| test | 小数据 | 开启 | `structured-language-attn-cluster` |
| big-baseline | 大数据 | 关闭 | `structured-big-baseline` |
| big-test | 大数据 | 开启 | `structured-big-test-attn-cluster` |

小数据默认：

```text
SEQ_LEN=256
TOPIC_COUNT=8
ENTITIES_PER_TOPIC=8
SHARED_ENTITY_COUNT=16
NOISE_RATE=0.25
AMBIGUITY_RATE=0.35
COPY_RATE=0.25
BRIDGE_RATE=0.25
```

大数据默认：

```text
SEQ_LEN=512
TOPIC_COUNT=32
ENTITIES_PER_TOPIC=16
SHARED_ENTITY_COUNT=128
NOISE_RATE=0.50
AMBIGUITY_RATE=0.70
COPY_RATE=0.40
BRIDGE_RATE=0.45
```

Attention-cluster test 组启用：

```text
ATTENTION_CLUSTER_WEIGHT=0.01
```

baseline 组关闭：

```text
ATTENTION_CLUSTER_WEIGHT=0
```

## 4. 实验结果

### 4.1 小数据 Baseline

记录 checkpoint：step 12000。

| metric | value |
| --- | ---: |
| eval_loss | 2.3158 |
| eval_acc | 0.3408 |
| same_higher_same_expert | 0.3776 |
| higher_mass | 0.5348 |
| expert_load | [0.3261, 0.1433, 0.2314, 0.2993] |

Per-layer:

```text
same_higher_by_layer = L0:0.3644 L1:0.3376 L2:0.4308
higher_mass_by_layer = L0:0.4141 L1:0.6197 L2:0.5707
expert_load_by_layer =
  L0:[0.088,0.304,0.507,0.100]
  L1:[0.465,0.121,0.076,0.338]
  L2:[0.425,0.005,0.111,0.460]
```

解释：

- loss/acc 说明小数据任务已经基本进入稳定状态。
- same-topic same-expert 高于随机路由的 0.25，但不强。
- expert load 不均衡，尤其 L2 的 expert 1 几乎不用。
- baseline 中 attention 本身已经较多落在同 topic history 上，但 router 没有强制聚类。

### 4.2 小数据 Test

记录 checkpoint：step 100000。

| metric | value |
| --- | ---: |
| eval_loss | 2.3061 |
| eval_acc | 0.3430 |
| same_higher_same_expert | 0.5247 |
| higher_mass | 0.4990 |
| expert_load | [0.2656, 0.2199, 0.3285, 0.1861] |

Per-layer:

```text
same_higher_by_layer = L0:0.4555 L1:0.7514 L2:0.3671
higher_mass_by_layer = L0:0.4265 L1:0.5595 L2:0.5110
expert_load_by_layer =
  L0:[0.368,0.287,0.346,0.000]
  L1:[0.314,0.000,0.345,0.341]
  L2:[0.116,0.373,0.295,0.217]
```

对比 baseline：

| metric | baseline | test |
| --- | ---: | ---: |
| eval_loss | 2.3158 | 2.3061 |
| eval_acc | 0.3408 | 0.3430 |
| same_higher_same_expert | 0.3776 | 0.5247 |

解释：

- attention-cluster 在小数据上没有伤害 LM loss，反而略微改善。
- same-topic same-expert 明显提高，尤其 L1 达到 0.7514。
- 但 L0/L1 分别出现一个 expert 几乎不用，说明 topic 聚类和 expert balance 之间有张力。
- L2 的 same-topic 聚类反而低于 L1，可能是高层开始服务 next-token prediction 中的 entity/copy/bridge 信息，而不再单纯按 topic 分工。

### 4.3 大数据 Baseline

记录 checkpoint：step 10000。

| metric | value |
| --- | ---: |
| eval_loss | 3.4230 |
| eval_acc | 0.3002 |
| same_higher_same_expert | 0.2610 |
| higher_mass | 0.4618 |
| expert_load | [0.2524, 0.2502, 0.2488, 0.2486] |

Per-layer:

```text
same_higher_by_layer = L0:0.2689 L1:0.2579 L2:0.2561
higher_mass_by_layer = L0:0.4270 L1:0.5265 L2:0.4318
expert_load_by_layer =
  L0:[0.251,0.252,0.251,0.246]
  L1:[0.249,0.250,0.253,0.249]
  L2:[0.257,0.249,0.243,0.251]
```

解释：

- 大数据明显更难，loss 从小数据约 2.31 上升到 3.42。
- same-topic same-expert 接近随机 0.25，说明没有 attention-cluster 时 router 基本没有自发形成 topic-level selectivity。
- expert load 非常均衡，几乎完美 25/25/25/25。
- 这说明在大数据 baseline 上，router 的主要行为是负载均衡，而不是语义/主题分工。

### 4.4 大数据 Test

记录 checkpoint：step 10000。

| metric | value |
| --- | ---: |
| eval_loss | 3.4233 |
| eval_acc | 0.2996 |
| same_higher_same_expert | 0.5475 |
| higher_mass | 0.4518 |
| expert_load | [0.1256, 0.5349, 0.2043, 0.1352] |

Per-layer:

```text
same_higher_by_layer = L0:0.4394 L1:0.4131 L2:0.7900
higher_mass_by_layer = L0:0.3775 L1:0.5243 L2:0.4535
expert_load_by_layer =
  L0:[0.000,0.589,0.102,0.309]
  L1:[0.322,0.515,0.099,0.065]
  L2:[0.055,0.501,0.412,0.032]
```

对比 big-baseline：

| metric | big-baseline | big-test |
| --- | ---: | ---: |
| eval_loss | 3.4230 | 3.4233 |
| eval_acc | 0.3002 | 0.2996 |
| same_higher_same_expert | 0.2610 | 0.5475 |
| higher_mass | 0.4618 | 0.4518 |

解释：

- attention-cluster 在大数据上显著提高 same-topic routing selectivity：0.2610 -> 0.5475。
- 但 LM loss/acc 没有改善，甚至略差。
- expert load 明显塌缩，expert 1 承担超过 53% token，L0 的 expert 0 完全不用。
- 这说明 attention-cluster 确实改变了 router 行为，但这个行为不一定转化为语言建模收益。

## 5. 总体结论

### 5.1 Attention-cluster 能提高 topic-level selectivity

无论小数据还是大数据，开启 attention-cluster 后：

```text
same_higher_same_expert 明显上升
```

尤其大数据：

```text
0.2610 -> 0.5475
```

说明这个正则确实在推动“同 topic token 进同 expert”。

### 5.2 但是 selectivity 不等于更低 LM loss

小数据 test 的 loss 略优于 baseline，但差距很小。大数据 test 的 loss 与 baseline 基本持平，甚至略差：

```text
big-baseline eval_loss = 3.4230
big-test     eval_loss = 3.4233
```

这说明在复杂 synthetic data 上，把 attention 邻居推向同 expert 不一定提升 next-token prediction。

### 5.3 大数据更接近真实数据上的冲突

大数据里有更多：

- topic；
- entity；
- shared entity；
- noise；
- copy；
- bridge；
- 长序列上下文。

因此 high attention pair 和 same-topic pair 的关系不再完全一致。attention-cluster 仍然能制造 routing selectivity，但这种 selectivity 会牺牲一部分 expert balance，并且没有明显改善 loss。

### 5.4 Expert load 是关键副作用

baseline 大数据 expert load 几乎完美均衡：

```text
[0.252, 0.250, 0.249, 0.249]
```

test 大数据则明显偏斜：

```text
[0.126, 0.535, 0.204, 0.135]
```

这可能解释为什么 selectivity 上升但 loss 没改善：router 被正则推向 topic 聚类后，部分 expert 过载，模型容量没有被均匀利用。

## 6. 下一步建议

目前指标只把 `metadata[:, :, 1]` 当作 higher feature，因此主要观察 same-topic selectivity。下一步应该增加更细的诊断指标：

```text
same_role_same_expert
same_entity_same_expert
same_span_same_expert
same_template_same_expert
noise_token_expert_load
bridge_cross_topic_same_expert
copy_source_target_same_expert
```

这些指标可以回答：

- router 到底是在按 topic 分工，还是按 syntax role 分工？
- copy source 和 copy target 是否被路由到同一 expert？
- bridge 模板中的跨 topic token 是否被错误聚到一起？
- noise token 是否污染 expert 分工？
- attention-cluster 的收益/副作用主要发生在哪类 token 上？

此外可以继续做两组 ablation：

1. 降低 attention-cluster 权重：

```text
ATTENTION_CLUSTER_WEIGHT=0.001
```

2. 加入 negative-pair loss，防止不同 topic 过度混合：

```text
ATTENTION_CLUSTER_NEGATIVE_WEIGHT=0.01
```

本轮实验的核心结论是：structured-language synthetic data 比原始 hierarchical data 更能暴露真实数据上的问题。attention-cluster 可以制造 topic-level routing selectivity，但这种 selectivity 在复杂、多关系、带噪声的数据上不自动等价于更好的 LM loss。
