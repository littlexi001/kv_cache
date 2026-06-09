# Full Context 不一定最优：实验 1 结果解读

## 1. 实验目的

这个实验想验证一个问题：

> baseline 把全量 context 都放进模型里，是否一定是最优的？

实验结果表明：不一定。

在这个 synthetic QA 设置里，只给正确证据段落的 `gold_only` 明显优于包含所有干扰信息的 full context。也就是说，模型虽然在 full context 下大多数时候还能生成正确答案，但它对正确答案的置信度下降了。

这里最关键的指标是 `mean_loss` 和 `mean_ppl`，而不是单纯看 `contains_answer_rate`。因为 full context 的 `contains_answer_rate` 仍然很高，但 answer NLL 已经明显变差。

## 2. 不同 mode 的含义

| mode | 输入内容 | 它在验证什么 |
| --- | --- | --- |
| `gold_only` | 只给正确证据段落 | 理想最小上下文。模型只看到正确答案相关信息。 |
| `oracle_top_chunk` | 和 `gold_only` 一样，只保留正确证据段落 | 模拟 oracle 压缩：如果完美找到最重要 chunk，效果应该是多少。 |
| `full_gold_begin` | 正确证据放在开头，后面接所有干扰段落 | 验证 full context 中 gold 在开头时效果如何。 |
| `full_gold_middle` | 正确证据放在中间，前后都有干扰段落 | 验证 "lost in the middle" 位置问题。 |
| `full_gold_end` | 干扰段落在前，正确证据放在结尾 | 验证 gold 在结尾时是否更容易被模型利用。 |
| `irrelevant_plus_gold` | 正确证据 + 无关干扰段落 | 验证纯无关噪声是否会伤害模型。 |
| `semantic_plus_gold` | 正确证据 + 语义相关但错误/冲突的干扰段落 | 验证相似但错误的信息是否会伤害模型。 |
| `random_top_chunk` | 随机保留一个 chunk，不保证包含正确证据 | 验证“短上下文本身”是否足够。 |
| `semantic_only_wrong` | 只给语义相关但错误的干扰段落，不给正确证据 | 验证模型看到错误证据时会不会失败。 |

## 3. 实验结果

| mode | sample_count | mean_loss | mean_ppl | mean_delta_loss_vs_gold_only | contains_answer_rate | mean_prompt_token_count |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `gold_only` | 48 | 0.06498 | 1.06714 | 0.00000 | 1.0000 | 81.50 |
| `oracle_top_chunk` | 48 | 0.06498 | 1.06714 | 0.00000 | 1.0000 | 81.50 |
| `semantic_plus_gold` | 48 | 0.12509 | 1.13325 | 0.06011 | 0.9792 | 254.40 |
| `full_gold_end` | 48 | 0.20238 | 1.22432 | 0.13740 | 1.0000 | 425.10 |
| `irrelevant_plus_gold` | 48 | 0.27650 | 1.31850 | 0.21152 | 1.0000 | 248.21 |
| `full_gold_begin` | 48 | 0.33435 | 1.39703 | 0.26937 | 0.9792 | 425.10 |
| `full_gold_middle` | 48 | 0.37171 | 1.45022 | 0.30673 | 0.9792 | 425.10 |
| `semantic_only_wrong` | 48 | 9.30777 | 11023.36268 | 9.24279 | 0.0000 | 80.27 |
| `random_top_chunk` | 48 | 10.31837 | 30283.77688 | 10.25339 | 0.0000 | 64.08 |

## 4. 关键对照

### 4.1 Gold-only vs full context

| mode | mean_loss | 说明 |
| --- | ---: | --- |
| `gold_only` | 0.06498 | 最干净，只有正确证据。 |
| `full_gold_begin` | 0.33435 | 有正确证据，但加入大量干扰后变差。 |
| `full_gold_middle` | 0.37171 | 正确证据在中间，最差。 |
| `full_gold_end` | 0.20238 | 正确证据在结尾，比 begin/middle 好，但仍不如 gold-only。 |

结论：

> full context 虽然包含正确证据，但不如只给正确证据。

这说明 baseline 全量 context 不一定最优。额外 context 会引入噪声和注意力竞争，使模型对正确答案的置信度下降。

### 4.2 位置影响

| mode | gold 位置 | mean_loss |
| --- | --- | ---: |
| `full_gold_begin` | 开头 | 0.33435 |
| `full_gold_middle` | 中间 | 0.37171 |
| `full_gold_end` | 结尾 | 0.20238 |

结论：

> 正确证据在中间时最难被利用，在结尾时最好。

这说明模型存在位置偏置，符合 "Lost in the Middle" 现象。模型并不是均匀地利用长上下文中的所有位置。

### 4.3 噪声类型影响

| mode | 噪声类型 | mean_loss |
| --- | --- | ---: |
| `gold_only` | 无噪声 | 0.06498 |
| `irrelevant_plus_gold` | 无关噪声 | 0.27650 |
| `semantic_plus_gold` | 语义相关冲突噪声 | 0.12509 |

结论：

> 不管是无关噪声还是语义冲突噪声，都会让模型比 `gold_only` 更差。

不过这次实验里 `semantic_plus_gold` 反而比 `irrelevant_plus_gold` 好。一个可能原因是 semantic distractor 写得过于明显，例如包含 `unverified` 或 `should not be used` 这类提示，模型能识别它是假的。因此，后续如果想更真实地测试语义冲突噪声，应增加一种更强的 `implicit_conflict` 设置：不给 warning，直接给出看似可信但错误的答案。

### 4.4 压缩是否真的有效

| mode | 是否包含正确证据 | mean_loss | contains_answer_rate |
| --- | --- | ---: | ---: |
| `oracle_top_chunk` | 是 | 0.06498 | 1.0000 |
| `random_top_chunk` | 不一定，通常没有 | 10.31837 | 0.0000 |
| `semantic_only_wrong` | 没有，只有错误证据 | 9.30777 | 0.0000 |

结论：

> 压缩有效不是因为上下文变短，而是因为保留了正确证据。

`oracle_top_chunk` 和 `gold_only` 完全一致，说明如果 selector 能准确找到关键证据，压缩后的上下文可以达到最优效果。相反，`random_top_chunk` 完全失败，说明随机变短没有意义。

## 5. 总结

这个实验支持三个结论。

第一，`gold_only` 和 `oracle_top_chunk` 最好，说明正确证据本身足够回答问题。

第二，full context 明显变差，说明全量上下文会引入噪声、位置偏置和注意力竞争。baseline 把所有 context 都放进去，并不天然最优。

第三，`random_top_chunk` 和 `semantic_only_wrong` 完全失败，说明 top 1% 有效的关键不是 token 数少，而是选中了真正有用的信息。

因此，这个实验可以作为后续研究的第一块证据：

> 长上下文中的有效信息是稀疏的。全量 context 会带来噪声和干扰，而 oracle 选择出的关键 chunk 可以在更少 token 下达到更好的 answer likelihood。

