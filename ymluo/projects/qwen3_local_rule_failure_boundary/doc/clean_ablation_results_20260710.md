# 子问题 3 clean ablation 与子问题 2 初步模型大小对比

日期：2026-07-10

## 1. 目的

之前的 420-case shuffled subset 把长度、位置、干扰数量、相似度、rule gap、chain length、竞争链混在一起，因此 `length` 的 marginal average 不能直接解释成“长度完全无关”。

本轮做了更干净的控制变量实验：

1. clean length：无 distractor、无 competitor，只扫 context length、position、chain length。
2. clean gap：无 distractor、无 competitor，固定 chain=4，只扫相关 rule 之间的 gap。
3. clean interference：固定 length/depth/gap/chain，只扫 distractor count、similarity、competitor count。
4. model size：用 Qwen3-8B 跑 clean length 子集和 8k clean interference 子集，对比 Qwen3-0.6B。

所有实验仍然使用 candidate accuracy 作为主指标：

```text
candidate_correct = 1 当且仅当 gold candidate 的 mean NLL 是所有候选里最低的。
```

generation accuracy 只作为辅助指标，因为 base model 经常不遵守“只输出 code”的格式。

## 2. 输出目录

服务器路径：

```text
/home/fdong/ymluo/projects/qwen3_local_rule_failure_boundary/outputs/clean_length_qwen06_20260710
/home/fdong/ymluo/projects/qwen3_local_rule_failure_boundary/outputs/clean_gap_qwen06_20260710
/home/fdong/ymluo/projects/qwen3_local_rule_failure_boundary/outputs/clean_interference_qwen06_20260710
/home/fdong/ymluo/projects/qwen3_local_rule_failure_boundary/outputs/clean_length_qwen8b_20260710
/home/fdong/ymluo/projects/qwen3_local_rule_failure_boundary/outputs/clean_interference_qwen8b_20260710
```

128k clean-length 额外尝试：

```text
/home/fdong/ymluo/projects/qwen3_local_rule_failure_boundary/outputs/clean_length_qwen06_128k_retry_20260710
/home/fdong/ymluo/projects/qwen3_local_rule_failure_boundary/outputs/clean_length_qwen06_128k_retry6gpu_20260710
```

128k 两次都 OOM，因此不纳入质量分析。

## 3. Clean Length: 0.6B

设置：

```text
lengths = 1k, 4k, 8k, 16k, 32k, 64k
depths = 10%, 50%, 90%
seeds = 0, 1, 2
distractor_count = 0
competitor_count = 0
rule_gap = 0
chain_length = 1, 4
```

| context length | cases | candidate acc | generation acc | mean margin | attention samples | mean selectivity |
|---:|---:|---:|---:|---:|---:|---:|
| 1k | 18 | 1.0000 | 0.0000 | 2.8208 | 18 | 0.9393 |
| 4k | 18 | 1.0000 | 0.0000 | 2.7532 | 18 | 0.9409 |
| 8k | 18 | 1.0000 | 0.0000 | 2.1234 | 18 | 0.9335 |
| 16k | 18 | 0.8889 | 0.0000 | 1.6815 | 18 | 0.8825 |
| 32k | 18 | 0.9444 | 0.0000 | 1.5896 | 0 |  |
| 64k | 18 | 1.0000 | 0.0000 | 1.6366 | 0 |  |

按 chain length 分开看：

| length | chain | cases | candidate acc | mean margin | mean selectivity |
|---:|---:|---:|---:|---:|---:|
| 1k | 1 | 9 | 1.0000 | 2.7748 | 0.9334 |
| 1k | 4 | 9 | 1.0000 | 2.8667 | 0.9452 |
| 4k | 1 | 9 | 1.0000 | 2.6502 | 0.9342 |
| 4k | 4 | 9 | 1.0000 | 2.8562 | 0.9476 |
| 8k | 1 | 9 | 1.0000 | 2.3897 | 0.9242 |
| 8k | 4 | 9 | 1.0000 | 1.8570 | 0.9429 |
| 16k | 1 | 9 | 1.0000 | 2.1750 | 0.8680 |
| 16k | 4 | 9 | 0.7778 | 1.1880 | 0.8969 |
| 32k | 1 | 9 | 1.0000 | 1.5181 |  |
| 32k | 4 | 9 | 0.8889 | 1.6611 |  |
| 64k | 1 | 9 | 1.0000 | 1.5961 |  |
| 64k | 4 | 9 | 1.0000 | 1.6771 |  |

结论：

```text
在无干扰、无竞争链时，0.6B 的 candidate accuracy 到 64k 仍然很高。
这说明简单 local rule 本身不是主要瓶颈，长度本身也没有在 clean 条件下形成单调失败边界。
```

但是 margin 从 1k 的 `2.82` 降到 32k/64k 的约 `1.59-1.64`，说明长上下文下 gold 的概率优势变薄了，只是还没足以造成大规模错误。

## 4. Clean Gap: 0.6B

设置：

```text
lengths = 8k, 32k
depth = 50%
seeds = 0..4
distractor_count = 0
competitor_count = 0
chain_length = 4
rule_gap = 0, 512, 2048, 4096, 8192
```

| length | rule gap | cases | candidate acc | generation acc | mean margin | attention samples | mean selectivity |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 8k | 0 | 5 | 1.0000 | 0.0000 | 2.9166 | 5 | 0.9487 |
| 8k | 512 | 5 | 1.0000 | 0.0000 | 2.7752 | 5 | 0.9533 |
| 8k | 2048 | 5 | 1.0000 | 0.0000 | 2.7667 | 5 | 0.9588 |
| 8k | 4096 | 5 | 1.0000 | 0.0000 | 3.0614 | 5 | 1.0000 |
| 8k | 8192 | 5 | 1.0000 | 0.2000 | 2.8828 | 5 | 1.0000 |
| 32k | 0 | 5 | 1.0000 | 0.0000 | 2.7294 | 0 |  |
| 32k | 512 | 5 | 1.0000 | 0.0000 | 2.7340 | 0 |  |
| 32k | 2048 | 5 | 1.0000 | 0.0000 | 2.6218 | 0 |  |
| 32k | 4096 | 5 | 1.0000 | 0.0000 | 2.8105 | 0 |  |
| 32k | 8192 | 5 | 1.0000 | 0.0000 | 2.7032 | 0 |  |

结论：

```text
在没有干扰和竞争链时，相关 rule 之间的距离从 0 增加到 8192 tokens，没有造成 candidate accuracy 下降。
```

这说明“距离”不是无条件主因。距离更可能在下面这些条件中变重要：

1. 超过 native context 或进入 RoPE/YaRN 外推区；
2. 有相似干扰或竞争证据链；
3. 自由生成而不是 candidate scoring；
4. 自然语言事实检索，而不是格式非常显式的 `VERIFIED RULE`。

## 5. Clean Interference: 0.6B

设置：

```text
lengths = 8k, 32k
depth = 50%
seeds = 0..4
rule_gap = 512
chain_length = 2
distractor_count = 0, 4, 16, 64
distractor_similarity = low, high, conflict
competitor_count = 0, 4
```

### 5.1 竞争链主效应

| length | competitor_count | cases | candidate acc | generation acc | mean margin | attention samples | mean selectivity |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 8k | 0 | 60 | 0.6667 | 0.0167 | 1.3580 | 60 | 0.3518 |
| 8k | 4 | 60 | 0.6167 | 0.0000 | 0.1219 | 60 | 0.0901 |
| 32k | 0 | 60 | 0.7167 | 0.0000 | 1.2873 | 0 |  |
| 32k | 4 | 60 | 0.5167 | 0.0000 | 0.0015 | 0 |  |

竞争链的影响很明显：

```text
8k: 0 competitor 0.6667 -> 4 competitors 0.6167
32k: 0 competitor 0.7167 -> 4 competitors 0.5167
```

更关键的是 margin：

```text
32k: margin 1.2873 -> 0.0015
8k attention selectivity: 0.3518 -> 0.0901
```

这说明竞争链让模型的 gold-rule 选择性大幅下降，正确答案只是在候选中勉强打平甚至被错误链压过。

### 5.2 相似度和竞争链的交互

| length | competitor_count | similarity | cases | candidate acc | mean margin | mean selectivity |
|---:|---:|---|---:|---:|---:|---:|
| 8k | 0 | low | 20 | 1.0000 | 2.8197 | 0.4122 |
| 8k | 0 | high | 20 | 0.5500 | 0.5915 | 0.3113 |
| 8k | 0 | conflict | 20 | 0.4500 | 0.6628 | 0.3317 |
| 8k | 4 | low | 20 | 0.9500 | 0.3354 | 0.1223 |
| 8k | 4 | high | 20 | 0.5000 | 0.0500 | 0.0714 |
| 8k | 4 | conflict | 20 | 0.4000 | -0.0197 | 0.0767 |
| 32k | 0 | low | 20 | 1.0000 | 2.5461 |  |
| 32k | 0 | high | 20 | 0.5500 | 0.6556 |  |
| 32k | 0 | conflict | 20 | 0.6000 | 0.6603 |  |
| 32k | 4 | low | 20 | 0.5000 | 0.0279 |  |
| 32k | 4 | high | 20 | 0.4500 | -0.0327 |  |
| 32k | 4 | conflict | 20 | 0.6000 | 0.0094 |  |

结论：

```text
low-sim 无关干扰基本可控；
high/conflict 干扰会显著降低 accuracy；
competitor chain 会进一步把 margin 压到接近 0 或负数。
```

这比“文本越长越差”更具体：真正危险的是相似、冲突、可竞争绑定的证据，而不是普通 filler token。

### 5.3 干扰数量不是简单单调变量

| length | competitor_count | distractor_count | cases | candidate acc | mean margin | mean selectivity |
|---:|---:|---:|---:|---:|---:|---:|
| 8k | 0 | 0 | 15 | 1.0000 | 2.9688 | 0.9368 |
| 8k | 0 | 4 | 15 | 0.4667 | 0.8165 | 0.2821 |
| 8k | 0 | 16 | 15 | 0.4667 | 0.7099 | 0.1157 |
| 8k | 0 | 64 | 15 | 0.7333 | 0.9369 | 0.0724 |
| 8k | 4 | 0 | 15 | 1.0000 | 0.3486 | 0.1644 |
| 8k | 4 | 4 | 15 | 0.4000 | -0.0224 | 0.0982 |
| 8k | 4 | 16 | 15 | 0.4667 | 0.0116 | 0.0649 |
| 8k | 4 | 64 | 15 | 0.6000 | 0.1498 | 0.0330 |
| 32k | 0 | 0 | 15 | 1.0000 | 2.8014 |  |
| 32k | 0 | 4 | 15 | 0.4667 | 0.7151 |  |
| 32k | 0 | 16 | 15 | 0.7333 | 0.7546 |  |
| 32k | 0 | 64 | 15 | 0.6667 | 0.8782 |  |
| 32k | 4 | 0 | 15 | 0.5333 | 0.0209 |  |
| 32k | 4 | 4 | 15 | 0.5333 | -0.0526 |  |
| 32k | 4 | 16 | 15 | 0.3333 | -0.1185 |  |
| 32k | 4 | 64 | 15 | 0.6667 | 0.1562 |  |

数量本身不是单调解释：

```text
8k, competitor=0: 0 distractor = 1.0000, 4/16 = 0.4667, 64 = 0.7333
32k, competitor=4: 0 = 0.5333, 4 = 0.5333, 16 = 0.3333, 64 = 0.6667
```

所以不能简单写成“干扰越多越差”。更准确是：

```text
只要出现相似/冲突/竞争绑定型干扰，模型就会明显变差；
干扰数量会改变难度，但不是单调主因，干扰结构更关键。
```

## 6. Model Size: Qwen3-0.6B vs Qwen3-8B

### 6.1 Clean length

Qwen3-8B clean length 子集：

```text
lengths = 1k, 8k, 16k
depths = 10%, 50%, 90%
seeds = 0, 1
distractor_count = 0
competitor_count = 0
chain_length = 1, 4
```

| model | length | cases | candidate acc | generation acc | mean margin |
|---|---:|---:|---:|---:|---:|
| Qwen3-8B | 1k | 12 | 1.0000 | 0.3333 | 2.5069 |
| Qwen3-8B | 8k | 12 | 1.0000 | 0.5000 | 2.3553 |
| Qwen3-8B | 16k | 12 | 1.0000 | 0.4167 | 2.3293 |

对比 0.6B clean length：

```text
0.6B 在 1k/8k 也是 1.0000，16k 是 0.8889。
```

因此 clean 条件下，两者差距不大。更大模型不是主要提升“单步/少步局部规则本身”。

### 6.2 Clean interference, fair 8k subset

公平比较使用两边共同的设置：

```text
length = 8k
depth = 50%
seeds = 0, 1, 2
rule_gap = 512
chain_length = 2
distractor_count = 0, 4, 16, 64
distractor_similarity = low, high, conflict
competitor_count = 0, 4
```

总体：

| model | cases | candidate acc | generation acc | mean margin |
|---|---:|---:|---:|---:|
| Qwen3-0.6B | 72 | 0.6528 | 0.0139 | 0.7800 |
| Qwen3-8B | 72 | 0.8750 | 0.0000 | 0.7976 |

按 competitor count：

| model | competitor_count | cases | candidate acc | mean margin |
|---|---:|---:|---:|---:|
| Qwen3-0.6B | 0 | 36 | 0.7222 | 1.4273 |
| Qwen3-0.6B | 4 | 36 | 0.5833 | 0.1326 |
| Qwen3-8B | 0 | 36 | 0.8333 | 1.1511 |
| Qwen3-8B | 4 | 36 | 0.9167 | 0.4441 |

按 competitor 和 similarity：

| model | competitor | similarity | cases | candidate acc | mean margin |
|---|---:|---|---:|---:|---:|
| 0.6B | 0 | low | 12 | 1.0000 | 2.8276 |
| 0.6B | 0 | high | 12 | 0.7500 | 0.7221 |
| 0.6B | 0 | conflict | 12 | 0.4167 | 0.7324 |
| 0.6B | 4 | low | 12 | 0.9167 | 0.3359 |
| 0.6B | 4 | high | 12 | 0.4167 | 0.0462 |
| 0.6B | 4 | conflict | 12 | 0.4167 | 0.0158 |
| 8B | 0 | low | 12 | 1.0000 | 2.1253 |
| 8B | 0 | high | 12 | 0.7500 | 0.6419 |
| 8B | 0 | conflict | 12 | 0.7500 | 0.6860 |
| 8B | 4 | low | 12 | 1.0000 | 0.6463 |
| 8B | 4 | high | 12 | 0.8333 | 0.2912 |
| 8B | 4 | conflict | 12 | 0.9167 | 0.3949 |

结论：

```text
模型大小的主要提升不是 clean local reasoning，而是抗干扰、抗竞争链、候选排序和变量绑定。
```

尤其在 `competitor=4` 且 high/conflict 干扰下：

```text
0.6B: high/conflict = 0.4167 / 0.4167
8B:  high/conflict = 0.8333 / 0.9167
```

这直接支持“更大的模型在长 context 训练/能力中提升的是证据筛选、抑制干扰、绑定正确链，而不只是单步推理”。

严格地说，本轮没有直接操纵 training context length，因为没有同模型大小、同训练数据、不同训练长度的 checkpoint。因此“训练时 context 长度”的因果效应还不能单独下结论。当前能回答的是模型大小带来的能力差异。

## 7. 与中文/英文年龄 needle 实验的关系

年龄 needle 实验显示：

1. 中文事实/中文问题 + 中文背景在 64k/128k 明显失败；
2. 英文事实/英文问题 + 英文背景在 128k seed0 仍可成功；
3. 中文条件的 128k evidence mass 会降到非常低。

local-rule clean ablation 显示：

1. 在英文符号规则、candidate scoring、无干扰条件下，0.6B 到 64k 仍然可以做对；
2. 相关规则之间 gap 到 8192 tokens 也不造成失败；
3. 一旦加入 high/conflict distractor 或 competitor chain，accuracy 和 margin 明显下降。

两者合起来说明：

```text
长上下文失败不是单纯由 token 数决定。
失败更可能来自 length、语言、任务形式、自由生成、attention mass、相似干扰、竞争证据链和变量绑定压力的交互。
```

年龄 needle 的中文 128k 失败更像是自然语言事实检索/attention mass 崩塌；local-rule 的失败更像是相似规则和竞争链造成的筛选/绑定错误。

## 8. 当前能回答的科学问题

科学问题：

```text
如果真实世界中大量推理/决策本身是短程、低阶、局部可组合的，
为什么小模型在长 context 推理中仍然会失败？
```

当前实验支持的回答是：

```text
小模型不是不会做短程局部规则。
在干净短/长 context 中，它可以稳定完成 if A then B 及少步组合。

失败主要来自长 context 中的证据选择问题：
模型需要从大量文本中识别哪些规则是真正相关的，
抑制相似但错误的规则，
把 start code 绑定到正确链，
并在多步组合中维护中间状态。

当存在 high-similarity distractor、conflict rule 或 competitor chain 时，
gold rule 的 attention selectivity 和 candidate margin 会明显下降，
错误候选的概率会接近甚至超过正确候选。
```

因此，“长文本性能差”可以这样写：

```text
长上下文不一定直接削弱模型的局部推理能力；
它增加了检索、筛选、抗干扰和变量绑定的压力。
当文本中出现相似或竞争型干扰时，小模型更容易选错证据，
于是表现为长 context 推理失败。
```

需要避免写成：

```text
文本越长，干扰越多，所以必然越差。
```

因为本轮结果显示：

1. clean length 到 64k 并没有单调失败；
2. clean gap 到 8192 tokens 也没有失败；
3. distractor count 本身不是单调变量；
4. 真正危险的是相似、冲突、竞争、可错误绑定的干扰结构。

## 9. 后续还需要的实验

若要严格回答“训练时 context length 有什么影响”，还需要 matched checkpoints：

```text
same model size
same data/recipe
different training context length
```

当前 Qwen3-0.6B vs Qwen3-8B 只能回答模型大小，不能单独识别训练长度因果。

若要继续加强论文图，建议补：

1. 0.6B/8B 同样设置下的 32k interference 对比；
2. local-rule 的中文版本，验证语言是否改变 failure boundary；
3. 年龄 needle 的 candidate scoring，区分“检索不到”与“自由生成格式错误”。

