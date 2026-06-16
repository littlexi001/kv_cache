# Ming Apple Diary Multi-Evidence Sum Experiment

本文记录 `Ming apple diary` 多证据求和实验。该实验用于测试：当最终答案不能从单个 answer span 直接复制，而必须从多条证据中聚合时，`top1% attention` 以及 `answer / front / end / other` 类别消融的行为。

对应项目目录：

```text
ymluo/projects/qwen3_top1_category_ablation
```

## 1. 数据设计

数据构造脚本：

```text
ymluo/projects/qwen3_top1_category_ablation/src/generate_ming_apple_diary_data.py
```

数据文件：

```text
ymluo/projects/qwen3_top1_category_ablation/data/ming_apple_diary_sum_8_words.jsonl
ymluo/projects/qwen3_top1_category_ablation/data/ming_apple_diary_sum_8_numeric.jsonl
```

每条样本是一段小明日记。日记中有很多普通生活记录，也有若干条 apple evidence，例如：

```text
Ming ate 3 apples after dinner.
```

问题固定为：

```text
How many apples did Ming eat in total?
```

`answer_evidence_texts` 显式标记所有 apple evidence spans。实验脚本已更新为优先使用这些 evidence spans 标记 `answer` 类，而不是只在 context 中查找最终答案字符串。

数字版数据规模：

| item | value |
| --- | ---: |
| samples | 8 |
| context chars | 2628 - 4997 |
| prompt chars | 2809 - 5178 |
| prompt tokens | 667 - 1247 |
| mean prompt tokens | 952 |
| evidence count | 8 - 15 |
| answer token count | 3 |
| answers | 17, 20, 22, 28, 32, 34 |

## 2. 英文数字词版本

英文数字词版本要求输出：

```text
seventeen / twenty / twenty two / ...
```

输出目录：

```text
ymluo/projects/qwen3_top1_category_ablation/outputs/ming_apple_diary_sum_8_top1
```

Qwen3-0.6B 在该版本上表现很差：

| mode | PPL | token acc | recall |
| --- | ---: | ---: | ---: |
| full_attention | 7245.75 | 0.00% | 0.00% |
| top1_all | 9968.31 | 0.00% | 0.00% |
| answer_only | 5430.23 | 12.50% | 18.75% |
| drop_answer | 15697.08 | 0.00% | 0.00% |
| drop_other | 59640.18 | 0.00% | 18.75% |

结论：英文数字词版本把任务变成了“多证据检索 + 多步求和 + 英文数字生成”。对 0.6B 来说过难，`full_attention` 本身也不能正确回答，因此该版本不适合单独分析 pruning 效果。

## 3. 数字版本

数字版本要求输出：

```text
17 / 20 / 22 / ...
```

对应数据：

```text
ymluo/projects/qwen3_top1_category_ablation/data/ming_apple_diary_sum_8_numeric.jsonl
```

### 3.1 Qwen3-0.6B

输出目录：

```text
ymluo/projects/qwen3_top1_category_ablation/outputs/ming_apple_diary_sum_8_numeric_top1
```

| mode | PPL | token acc | exact/recall |
| --- | ---: | ---: | ---: |
| full_attention | 13.07 | 8.33% | 0.00% |
| top1_all | 14.19 | 8.33% | 0.00% |
| answer_only | 19.54 | 25.00% | 0.00% |
| drop_answer | 19.17 | 8.33% | 0.00% |
| drop_front | 207.01 | 12.50% | 0.00% |
| drop_end | 3256.80 | 0.00% | 0.00% |
| drop_other | 18.10 | 37.50% | 12.50% |

0.6B 的数字版 PPL 明显低于英文数字词版，但仍然几乎不能完整求和。

### 3.2 Qwen3-1.7B

模型目录：

```text
ymluo/models/Qwen3-1.7B
```

输出目录：

```text
ymluo/projects/qwen3_top1_category_ablation/outputs/ming_apple_diary_sum_8_numeric_qwen3_1p7b_top1
```

| mode | PPL | token acc | exact/recall |
| --- | ---: | ---: | ---: |
| full_attention | 24.92 | 8.33% | 0.00% |
| top1_all | 23.53 | 16.67% | 0.00% |
| answer_only | 22.00 | 33.33% | 12.50% |
| drop_answer | 53.68 | 8.33% | 0.00% |
| drop_front | 48463.48 | 4.17% | 0.00% |
| drop_end | 10373.47 | 4.17% | 0.00% |
| drop_other | 9.42 | 45.83% | 12.50% |

1.7B 没有稳定解决完整求和。它偶尔在 `answer_only` 或 `drop_other` 下答中一条样本，但 `full_attention` 仍没有 exact match。

常见 greedy 输出类似：

```text
The18
The21
The48
The The6
```

这里的 `The18` 不是一个特殊 token，而是预测 token decode 后直接拼接得到的显示形式，例如：

```text
The + 18
```

### 3.3 Qwen3-4B

模型目录：

```text
ymluo/models/Qwen3-4B
```

输出目录：

```text
ymluo/projects/qwen3_top1_category_ablation/outputs/ming_apple_diary_sum_8_numeric_qwen3_4b_top1
```

| mode | PPL | token acc | exact/recall |
| --- | ---: | ---: | ---: |
| full_attention | 4.23 | 37.50% | 12.50% |
| top1_all | 8.40 | 37.50% | 0.00% |
| answer_only | 5.90 | 50.00% | 25.00% |
| drop_answer | 11.67 | 29.17% | 0.00% |
| drop_front | 6.54 | 25.00% | 0.00% |
| drop_end | 397.84 | 12.50% | 0.00% |
| drop_other | 4.19 | 45.83% | 12.50% |

4B 明显好于 0.6B/1.7B。`full_attention` 完整答对 1/8，`answer_only` 完整答对 2/8。它仍未稳定解决任务，但已经说明模型规模上来后，多证据求和能力开始出现。

## 4. 跨模型对比

| model | mode | PPL | token acc | exact/recall |
| --- | --- | ---: | ---: | ---: |
| Qwen3-0.6B | full_attention | 13.07 | 8.33% | 0.00% |
| Qwen3-1.7B | full_attention | 24.92 | 8.33% | 0.00% |
| Qwen3-4B | full_attention | 4.23 | 37.50% | 12.50% |
| Qwen3-0.6B | top1_all | 14.19 | 8.33% | 0.00% |
| Qwen3-1.7B | top1_all | 23.53 | 16.67% | 0.00% |
| Qwen3-4B | top1_all | 8.40 | 37.50% | 0.00% |
| Qwen3-0.6B | answer_only | 19.54 | 25.00% | 0.00% |
| Qwen3-1.7B | answer_only | 22.00 | 33.33% | 12.50% |
| Qwen3-4B | answer_only | 5.90 | 50.00% | 25.00% |
| Qwen3-0.6B | drop_other | 18.10 | 37.50% | 12.50% |
| Qwen3-1.7B | drop_other | 9.42 | 45.83% | 12.50% |
| Qwen3-4B | drop_other | 4.19 | 45.83% | 12.50% |

## 5. Top1% 类别分布

数字版 top1% token 类别分布：

| model | category | token fraction | attention mass fraction |
| --- | --- | ---: | ---: |
| Qwen3-0.6B | answer | 0.14% | 0.04% |
| Qwen3-0.6B | front | 11.81% | 54.80% |
| Qwen3-0.6B | end | 63.64% | 38.60% |
| Qwen3-0.6B | other | 24.41% | 6.56% |
| Qwen3-1.7B | answer | 0.25% | 0.10% |
| Qwen3-1.7B | front | 12.99% | 52.85% |
| Qwen3-1.7B | end | 59.11% | 38.02% |
| Qwen3-1.7B | other | 27.65% | 9.03% |
| Qwen3-4B | answer | 0.13% | 0.04% |
| Qwen3-4B | front | 13.55% | 53.86% |
| Qwen3-4B | end | 52.42% | 32.18% |
| Qwen3-4B | other | 33.90% | 13.91% |

核心现象：即使在需要多条 apple evidence 的任务里，top1% attention 直接选中 answer evidence 的比例仍然极低。大部分 top1% 注意力集中在 `front` 和 `end`。这说明模型可能没有通过显式逐条 evidence attention 来完成求和，或者当前 top1% 视角没有捕捉到聚合机制。

## 6. 关于 7B/8B

当前本机环境：

```text
RAM: 16GB
GPU: RTX 3050 Laptop, 4GB VRAM
```

Qwen3 官方 dense 档位接近 7B 的是 `Qwen3-8B`。8B fp16 权重本身接近 16GB 级别，运行还需要 activation、KV cache、CPU/GPU offload 额外内存。当前机器直接跑完整 ablation 风险很高，极可能爆内存或非常慢。

因此本轮实际测试了本机仍可运行的 `Qwen3-4B`。4B 已经需要 CPU/GPU offload，完整 8 样本 ablation 约二十多分钟。

## 7. 当前结论

1. 英文数字词版本对 0.6B 过难，不适合作为 pruning 效果主实验。
2. 数字版显著降低生成难度，但 0.6B 和 1.7B 仍不能稳定做多证据求和。
3. 4B 明显改善，说明任务难度确实和模型规模相关。
4. `drop_end` 通常严重破坏结果，说明末尾 question/answer suffix 对输出答案很关键。
5. `drop_other` 在数字版上经常不差，甚至 PPL 很低，说明 answer correctness 不完全由 top1% 中的 `other` 决定。
6. `answer_only` 在 4B 上完整答对 2/8，说明当 top1% 中的 answer evidence 被强行保留时，模型有时可以更直接地利用证据。
7. top1% 直接选中 answer evidence 的比例极低，这提示：多证据聚合可能不适合用“单层单 head top1% 是否命中 evidence”来简单解释。

## 8. 后续建议

如果目标是更纯粹地研究 attention pruning，而不是模型算术能力，可以设计不需要加法的多证据任务，例如：

```text
Which fruit appears in every marked snack entry?
Which label appears most often?
Did Ming eat apples on more than five days?
Output the secret code that appears after every apple entry.
```

这些任务仍然需要多证据，但可以减少多步加法带来的额外能力瓶颈。
