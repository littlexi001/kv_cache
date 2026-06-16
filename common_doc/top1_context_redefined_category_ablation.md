# Top1 Context Redefined Category Ablation

本文记录重新定义 `answer / front / end / other` 后的 top1% attention
类别消融结果。

对应实验输出：

```text
ymluo/projects/qwen3_top1_category_ablation/outputs/local_8sample_4k_redefined
```

模型与数据：

```text
model: ymluo/models/Qwen3-0.6B
data: ymluo/projects/qwen3_kcache_l2_neighbor_analysis/data/needle_in_haystack/needle_in_haystack.jsonl
samples: 8
max_context_chars: 4000
top_ratio: 1%
```

## 1. 重新定义四类 token

旧定义的问题是：`end` 是针对每个 answer query row 动态取“当前可见 token
最后 1%”。随着答案 token 逐步生成，question token 会被已经出现的 answer token
挤出最后 1%，于是 `What / best / thing / do / San / Francisco` 等问题提示词会被归到
`other`。这会高估 `other` 的重要性。

新定义改为固定 prompt 区域：

```text
answer = context 中埋进去的标准答案 span
front  = prompt 可见部分前 1%
end    = 最后接上的 Question/Answer prompt 后缀，和 tail 1% 取更大范围
other  = 不属于以上三类的 token
```

因此，最后接上的问题与回答格式，例如：

```text
Question: What is the best thing to do in San Francisco?
Answer:
```

会被归入 `end`，不再混入 `other`。

答案评价只统计标准答案文本 token，不统计 `Question:`、`Answer:`、冒号、问号等
prompt 符号。额外增加 `keyword_recall`，只看答案关键词：

```text
eat / a / sandwich / and / sit / in / dolores / park
```

## 2. Top1% 四类组成

新定义下，top1% token 的类别分布为：

| category | token fraction | attention mass |
| --- | ---: | ---: |
| answer | 7.06% | 1576.99 |
| front | 9.59% | 15224.16 |
| end | 61.04% | 9893.91 |
| other | 22.31% | 1025.34 |

和旧定义相比，`end` 比例明显上升，`other` 比例下降。原因是 question/Answer suffix
现在被正确归入 `end`。

## 3. PPL 与正确率

| mode | mean PPL | token accuracy | keyword recall |
| --- | ---: | ---: | ---: |
| full_attention | 10.09 | 68.06% | 68.75% |
| top1_all | 9.35 | 63.89% | 64.06% |
| answer_only | 932.32 | 22.22% | 26.56% |
| front_only | 9238.44 | 0.00% | 3.13% |
| end_only | 5914323.98 | 0.00% | 18.75% |
| other_only | 5652413.82 | 0.00% | 3.13% |
| drop_answer | 225.26 | 18.06% | 14.06% |
| drop_front | 1110.53 | 38.89% | 37.50% |
| drop_end | 612207.23 | 0.00% | 1.56% |
| drop_other | 135.38 | 56.94% | 62.50% |

## 4. 主要结论

### 4.1 Top1% 仍然接近 full attention

`top1_all` 的 PPL 为 `9.35`，略低于 `full_attention` 的 `10.09`。这继续支持：

```text
oracle attention top1% 已经能恢复答案预测所需的大部分有效信息。
```

### 4.2 旧实验高估了 other 的重要性

旧定义下，`other` 混入了问题词，例如：

```text
What / best / thing / do / San / Francisco
```

因此旧版 `drop_other` 同时删除了真正的 body-other 和 question semantics。

新定义后，`drop_other` 的结果明显改善：

```text
drop_other mean_ppl = 135.38
keyword_recall = 62.50%
```

这说明 `other` 仍有一定作用，但不再是最主要因素。之前 `other` 的重要性主要来自
分类口径不够细，把 question/Answer suffix 错归到了 `other`。

### 4.3 end 是最关键的结构

新定义下的 `end` 包含：

```text
Question suffix + Answer prefix + tail 1%
```

删除它后：

```text
drop_end mean_ppl = 612207.23
token accuracy = 0.00%
keyword_recall = 1.56%
```

这说明 `end` 不只是“最近 token”，而是包含当前任务的 query semantics、回答格式、
以及当前答案生成位置的局部条件。它对答案预测稳定性最关键。

### 4.4 answer 是远程证据，但 answer-only 不够

`drop_answer` 会明显变差：

```text
drop_answer mean_ppl = 225.26
keyword_recall = 14.06%
```

说明埋入 context 的答案证据确实关键。

但 `answer_only` 也很差：

```text
answer_only mean_ppl = 932.32
keyword_recall = 26.56%
```

这说明模型不能只靠远程答案 span 完成预测，还需要 `end` 中的问题语义和生成格式条件。

### 4.5 front 有辅助作用，但不是核心

`drop_front` 后：

```text
drop_front mean_ppl = 1110.53
keyword_recall = 37.50%
```

说明 front token 对概率校准或全局 prompt 状态有辅助作用，但不是恢复答案的核心证据。

## 5. 更新后的解释

当前更合理的功能分工是：

```text
answer = 远程证据
end    = 问题语义 + Answer 格式 + 当前生成位置条件
front  = prompt 起始锚点 / 全局状态辅助
other  = 普通上下文桥接和残余辅助信息
```

因此，`top1%` 的有效性不是因为它只包含 answer，而是因为它同时保留：

```text
远程答案证据 + 末尾 query/format 条件 + 少量全局锚点 + 其他上下文辅助
```

其中最关键的是 `end` 和 `answer` 的配合：

```text
end 决定“现在要回答什么、按什么格式回答”；
answer 决定“远程证据中的正确内容是什么”。
```

## 6. 对后续实验的影响

后续不应再使用过粗的 `other` 定义。建议继续拆分：

```text
end_question
end_answer_prefix
tail_recent_answer_tokens
needle_neighbor
body_other
prompt_instruction
```

尤其需要区分：

```text
Question token 是否重要
Answer prefix 是否重要
context 中真正 body-other 是否重要
needle 周边非答案词是否重要
```

这样才能判断模型到底依赖的是：

```text
query semantics
format control
remote evidence
local generation history
or ordinary context bridge
```

## 7. 需要注意的限制

本实验仍是小规模诊断：

```text
samples = 8
max_context_chars = 4000
model = Qwen3-0.6B
```

因此它适合用于解释机制和修正分类口径，不应直接作为大规模 benchmark 结论。

另外，本实验看的是 final hidden-state / answer PPL，还没有完整覆盖文档中要求的：

```text
QK score -> softmax mass -> weighted V -> attention output direction
```

后续仍需要补：

```text
attention output cosine
relative L2 error
tail10 add-back
V-space direction analysis
```

