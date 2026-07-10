# 长 needle 年龄实验与 local-rule 实验的合并分析

日期：2026-07-10

## 1. 问题

当前看起来有一个矛盾：

1. 年龄 needle 实验中，`zh prompt + zh filler` 在 64k/128k 明显失败，尤其 128k 三个位置全部失败，evidence attention mass 降到 `1e-4` 量级。
2. `qwen3_local_rule_failure_boundary` 实验中，1k 到 32k 的 candidate accuracy 没有随长度单调下降，结论写成“长度本身不是单调失败边界”。

这个矛盾不是真矛盾。两个实验测到的是不同问题。

## 2. 两个实验的核心差异

| 维度 | 年龄 needle 实验 | local-rule 实验 |
|---|---|---|
| 任务形式 | 自然语言事实检索：小明今年是九岁 | 符号规则链：`VERIFIED RULE ...` |
| 主输出指标 | 自由生成答案是否正确 | candidate logprob 是否把 gold code 排第一 |
| 生成格式污染 | 会直接影响 accuracy | generation accuracy 很低，但主分析不用它 |
| 长度范围 | 8k/16k/32k/64k/128k，256k OOM | 1k/4k/8k/16k/32k |
| 位置延申 | 64k 用 YaRN 2.0，128k 用 YaRN 4.0 | 主要在 32k 内，没有覆盖 64k/128k 失败区 |
| filler 语言 | 中文/英文都测了 | 主要是英文 filler + 英文规则 |
| 干扰结构 | 没有显式 competing fact | 有 distractor、conflict、competitor chains |
| attention 指标 | evidence span 的绝对 mass | gold rules 相对 rule-like spans 的 selectivity |

因此，年龄实验更像在测：

```text
模型能不能在超长上下文中自由生成取回一个自然语言事实。
```

local-rule 实验更像在测：

```text
模型在给定候选答案时，能不能在相似规则/竞争规则中把正确 code 的 logprob 排最高。
```

这两个指标不能直接等同。

## 3. 把中文年龄结果和跨语言结果放在一起看

### 3.1 8k/16k/32k 平均 accuracy

这里把三个 depth 的 accuracy 简单平均，只看主趋势。

| condition | 8k | 16k | 32k |
|---|---:|---:|---:|
| zh prompt + zh filler | 0.93 | 0.73 | 0.60 |
| zh prompt + en filler | 1.00 | 0.80 | 0.80 |
| en prompt + zh filler | 0.93 | 1.00 | 1.00 |
| en prompt + en filler | 1.00 | 1.00 | 0.93 |

这说明一个关键点：

```text
“长度变长必然失败”不是普遍规律。
```

在 8k 到 32k 范围内，中文事实/中文问题 + 中文背景最差；英文事实/英文问题，尤其英文背景，反而很稳。这个结果和 local-rule 的“1k 到 32k 没有单调长度失败”是相容的。

真正明显崩掉的是：

1. 原始中文/中文条件在 64k 前中部、128k 全部位置。
2. 中文事实/中文问题 + 英文背景在 128k 全部位置。
3. 128k 需要 YaRN factor 4.0，这已经不是 native 32k 内部的问题。

### 3.2 64k/128k seed0

| condition | 64k 10% | 64k 50% | 64k 90% | 128k 10% | 128k 50% | 128k 90% |
|---|---:|---:|---:|---:|---:|---:|
| zh prompt + zh filler | 0 | 0 | 1 | 0 | 0 | 0 |
| zh prompt + en filler | 1 | 1 | 1 | 0 | 0 | 0 |
| en prompt + zh filler | 1 | 1 | 1 | 0 | 1 | 0 |
| en prompt + en filler | 1 | 1 | 1 | 1 | 1 | 1 |

这说明 128k 的失败不是单纯由 token 长度决定的。语言组合和文本形态会改变失败边界。

尤其是 `en prompt + en filler` 在 128k seed0 三个 depth 全部正确，说明模型并不是完全没有 128k 检索能力；但 `zh prompt + zh filler` 在 128k 全部 miss，说明中文自然语言事实在当前设置下更容易丢失。

## 4. 为什么 local-rule 结果不推翻年龄 needle 结果

### 4.1 local-rule 只覆盖到 32k

local-rule 的主结果是：

```text
1k, 4k, 8k, 16k, 32k 没有单调下降。
```

而年龄 needle 最强的失败证据在：

```text
64k 和 128k，尤其 128k + YaRN factor 4.0。
```

所以 local-rule 没有测试年龄实验真正崩掉的区间。它最多说明：

```text
在 native 或接近 native 的 32k 内，长度本身不是唯一主因。
```

这和年龄实验中的跨语言 8k-32k 结果一致。

### 4.2 local-rule 主指标是 candidate accuracy，不是自由生成 accuracy

local-rule 文档里已经写了：

```text
generation accuracy = 0.71%
```

它之所以还能分析，是因为主指标是 candidate accuracy：给定一批候选 code，看 gold code 的 NLL 是否最低。

年龄 needle 的 accuracy 是自由生成答案是否正确。自由生成会受到这些因素影响：

1. 模型是否遵守“只根据上文回答”；
2. 是否输出“无法确定/unknown”；
3. 是否先输出错误年龄再补一句正确答案；
4. 中文/英文答案形式的先验概率；
5. 最后 query 对证据 span 的注意力是否足够。

如果把 local-rule 也按自由生成 strict exact match 来看，它其实已经很差。反过来，如果把年龄 needle 改成候选打分，例如：

```text
候选 = 九岁 / 八岁 / 十岁 / 无法确定
```

结果可能会比自由生成 accuracy 稳定。因此两者的 accuracy 不是同一种量。

### 4.3 attention mass 和 selectivity 不是同一个指标

年龄实验的 evidence mass 是：

```text
最后 query 对唯一 evidence span 的绝对 attention mass。
```

local-rule 的 selectivity 是：

```text
gold rules / (gold rules + non-gold rule-like spans)。
```

年龄实验没有显式 rule-like distractors，所以它看的是“证据有没有被关注到”。local-rule 有很多相似规则，所以它看的是“在规则候选集合里是否选对 gold”。一个是绝对检索强度，一个是相对选择性。

因此可以同时成立：

1. 年龄 128k 中文条件下，绝对 evidence mass 几乎归零，模型没有取回事实。
2. local-rule 32k 内，绝对长度不是主因；竞争链和相似干扰才是主因。

### 4.4 英文条件解释了表面冲突

local-rule 基本是英文符号规则实验。和它最接近的年龄实验不是 `zh prompt + zh filler`，而是：

```text
en prompt + en filler
```

这组年龄实验在 8k/16k/32k 非常稳定，128k seed0 也全部正确。这个结果反而支持 local-rule：

```text
英文结构化任务在 Qwen3-0.6B 上比中文自然语言年龄事实更稳。
```

所以矛盾主要来自把 `zh/zh 年龄 needle` 和 `en/en symbolic rule candidate scoring` 直接比较了。

## 5. 更合理的统一解释

当前结果可以统一成下面这个解释：

```text
Qwen3-0.6B 的长上下文失败不是单一的“长度越长越差”。
失败边界由多个因素共同决定：

1. context 是否超过 native 长度并需要 RoPE/YaRN 外推；
2. 证据和问题的语言；
3. 背景文本语言是否会制造同语言干扰；
4. 任务是自由生成还是候选打分；
5. 是否存在相似规则、竞争链或变量绑定压力；
6. evidence span 在最终 query 的 attention 中是否仍有足够质量。
```

对于年龄 needle：

```text
主要失败模式是超长上下文下自然语言事实检索失败，中文条件更明显。
```

对于 local-rule：

```text
主要失败模式是相似规则/竞争链导致的选择与绑定失败，而不是 1k-32k 范围内的长度本身。
```

这两句话并不冲突。

## 6. 当前最应该补的验证

为了把这个问题彻底讲清楚，建议补三组实验。

### A. 年龄 needle 加 candidate scoring

在年龄实验里新增候选答案：

```text
中文候选：九岁 / 八岁 / 十岁 / 无法确定
英文候选：nine years old / eight years old / ten years old / unknown
```

同时保留自由生成 accuracy。这样可以回答：

```text
年龄实验失败到底是检索失败，还是生成格式/答案先验失败？
```

### B. local-rule 跑 64k/128k 的无竞争链版本

固定：

```text
competitor_count = 0
distractor_similarity = low
chain_length = 1
```

只扫：

```text
length = 32k / 64k / 128k
position = 10% / 50% / 90%
```

这可以直接验证：

```text
英文符号规则在 YaRN 64k/128k 是否也会出现年龄中文条件那种 evidence mass 崩塌。
```

### C. local-rule 中文化

把 local-rule 的规则、prompt、filler 改成中文或中英混合，至少做：

```text
zh rule + zh filler
zh rule + en filler
en rule + zh filler
en rule + en filler
```

这可以判断中文条件失败是不是来自：

1. 中文 tokenization；
2. 中文 instruction following；
3. 中文 filler 与中文事实的同语言干扰；
4. Qwen3-0.6B 对英文结构化符号任务更稳。

## 7. 结论

目前不应该说两个实验互相矛盾。更准确的说法是：

```text
local-rule 实验排除了“1k-32k 内 token 数本身是唯一主因”这个简单解释；
年龄 needle 实验证明“在自然语言事实检索，尤其中文条件和 64k/128k RoPE 外推下，证据检索会明显失败”。
```

两个实验合在一起，反而给出更强的结论：

```text
长上下文失败不是一个单变量现象，而是 length × language × task format × distractor structure × decoding/scoring method 的交互。
```

