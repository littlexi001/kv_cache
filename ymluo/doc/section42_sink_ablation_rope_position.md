# Section 42: Sink ablation, RoPE position, and KV anchor diagnostics

日期：2026-06-30

## 1. 实验目的

本实验验证 sink 依赖到底来自哪里：

1. 保留 sink 位置，但替换 sink 内容：如果 attention mass 仍高，说明位置/RoPE 成分强。
2. 保留 sink 内容，但把它移到中间或末尾：如果 mass 下降，说明位置成分强。
3. 只保留前 1-2 个 token，去掉后续 sink 格式 token：看 PPL/top2 mass 是否变差。
4. 对 sink KV 做 zero/drop：看 PPL、downstream accuracy、top2 mass。
5. 统计 sink token 的 pre-RoPE/post-RoPE K norm 和 q-k logit。

## 2. 新增代码和输出

新增脚本：

```text
ymluo/projects/qwen3_top2_head_limit3_ppl/src/run_sink_ablation_diagnostics.py
ymluo/projects/qwen3_top2_head_limit3_ppl/scripts/run_sink_ablation_diagnostics_server.sh
```

Attention 诊断输出：

```text
ymluo/projects/qwen3_top2_head_limit3_ppl/outputs/sink_ablation_diagnostics_0630_v1
```

大样本 PPL/accuracy 输出：

```text
ymluo/projects/qwen3_top2_head_limit3_ppl/outputs/sink_ablation_diagnostics_pplacc_0630_v1
```

实验规模：

| run | tasks_per_variant | total condition-task | attention stats | runtime |
|---|---:|---:|---:|---:|
| attention diagnostic | 4 | 128 | yes | 1037.9s |
| PPL/accuracy | 16 | 512 | no | 569.8s |

模型解析：

```text
Qwen3-0.6B
num_attention_heads = 16
head_dim = 64
selected layers = 0,4,8,13,20,27
```

## 3. Conditions

| condition | 含义 |
|---|---|
| baseline | 原始 context |
| replace_sink_content | 保留前 16 个位置，但把前 16 个 token 替换成同一个 `X` token |
| move_sink_middle | 把原前 16 个 token 移到 context 中间 |
| move_sink_end | 把原前 16 个 token 移到 context 末尾 |
| keep_prefix2_text | 只保留前 2 个 token，删除第 3-16 个 sink token |
| zero_sink_kv | query/label 阶段把前 16 个 KV 置零 |
| drop_sink_kv | query/label 阶段 mask 掉前 16 个 KV |
| keep_prefix2_drop_sink_kv | query/label 阶段只 mask 第 3-16 个 KV，保留前 2 个 |

## 4. 大样本 PPL/Accuracy 结果

PPL/accuracy run 使用每个 variant 16 个任务，共 64 个任务，8 个 condition。

| condition | accuracy | query PPL | PPL ratio |
|---|---:|---:|---:|
| baseline | 68.8% | 13.87 | 1.00 |
| replace_sink_content | 39.1% | 13.84 | 1.00 |
| move_sink_middle | 67.2% | 13.03 | 0.94 |
| move_sink_end | 65.6% | 14.73 | 1.06 |
| keep_prefix2_text | 71.9% | 13.73 | 0.99 |
| zero_sink_kv | 32.8% | 2367.04 | 170.62 |
| drop_sink_kv | 39.1% | 2219.56 | 159.99 |
| keep_prefix2_drop_sink_kv | 68.8% | 14.90 | 1.07 |

结论：

1. 文本层面删除第 3-16 个 sink token 几乎不伤 PPL/accuracy：`keep_prefix2_text` 的 PPL ratio 是 `0.99`。
2. KV 层面保留前 2 个、drop 第 3-16 个也基本稳定：PPL ratio `1.07`，accuracy 和 baseline 一样是 `68.8%`。
3. 但完整 zero/drop 前 16 个 sink KV 会让 PPL 爆炸：`170x/160x`。

这说明关键 sink 不是前 16 个格式 token 整体，而是最前 1-2 个全局 anchor KV。

## 5. Attention 诊断结果

Attention diagnostic run 使用每个 variant 4 个任务，共 16 个任务。下面是 overall group attention mass。

| condition | sink_content mass | front_positions mass | evidence_any mass | evidence_label mass |
|---|---:|---:|---:|---:|
| baseline | 45.23% | 45.23% | 2.88% | 0.14% |
| replace_sink_content | 49.54% | 49.54% | 2.91% | 0.12% |
| move_sink_middle | 0.70% | 45.50% | 2.94% | 0.14% |
| move_sink_end | 3.88% | 45.13% | 2.62% | 0.11% |
| keep_prefix2_text | 44.10% | 45.31% | 2.97% | 0.14% |
| zero_sink_kv | 2.48% | 2.48% | 2.32% | 0.14% |
| drop_sink_kv | 0.00% | 0.00% | 2.50% | 0.17% |
| keep_prefix2_drop_sink_kv | 44.61% | 44.61% | 2.93% | 0.14% |

解释：

1. `replace_sink_content`：内容换成 `X` 后，前 16 个位置仍拿到 `49.54%` attention mass。说明 sink mass 不依赖原始 key/value/table 格式内容。
2. `move_sink_middle`：原 sink 内容移到中间后，它自己的 mass 掉到 `0.70%`，但新的前 16 个位置仍是 `45.50%`。
3. `move_sink_end`：原 sink 内容移到末尾后，mass 只有 `3.88%`；新的前 16 个位置仍是 `45.13%`。
4. `keep_prefix2_text`：只保留前 2 个 token，sink_content 仍拿到 `44.10%` mass。
5. `keep_prefix2_drop_sink_kv`：只保留前 2 个 KV，mask 第 3-16 个 KV，sink mass 仍是 `44.61%`。

这组结果强烈支持：

```text
sink 主要是起始位置 anchor，而不是原始 sink 内容。
更精确地说，前 1-2 个位置贡献了绝大多数 sink attention mass。
```

## 6. Pre-RoPE vs Post-RoPE q-k logit

关键对比：

| condition/group | attention mass | post-RoPE mean logit | pre-RoPE mean logit | post K norm | pre K norm |
|---|---:|---:|---:|---:|---:|
| baseline/sink_content | 45.23% | 1.908 | 6.210 | 63.657 | 63.657 |
| move_sink_middle/sink_content | 0.70% | 0.835 | 5.111 | 68.685 | 68.685 |
| move_sink_middle/front_positions | 45.50% | 2.070 | 6.373 | 63.518 | 63.519 |
| move_sink_end/sink_content | 3.88% | 3.049 | 5.223 | 70.158 | 70.155 |
| move_sink_end/front_positions | 45.13% | 2.034 | 6.385 | 63.518 | 63.519 |
| keep_prefix2_text/sink_content | 44.10% | 5.327 | 8.515 | 59.491 | 59.491 |
| zero_sink_kv/sink_content | 2.48% | 0.000 | 5.750 | 0.000 | 63.657 |
| drop_sink_kv/sink_content | 0.00% | 1.279 | 5.737 | 63.657 | 63.657 |

解释：

1. 原 sink 内容被移到中间后，pre-RoPE logit 仍然高：`5.111`，但 post-RoPE logit 降到 `0.835`，attention mass 只有 `0.70%`。
2. 新的前 16 个位置即使内容换了，仍保持高 mass：`45.50%`，post-RoPE logit `2.070`。
3. 这说明 K 向量内容本身确实有一定 pre-RoPE 对齐，但最终能不能成为 sink，强烈依赖 RoPE 后的位置相位。
4. zero/drop 的区别也符合预期：
   - `zero_sink_kv` 把 post K norm 变成 0，post logit 变成 0，但 softmax 仍会给零 logit token 一点 mass；
   - `drop_sink_kv` 直接 mask，所以 mass 为 0。

因此，当前证据支持：

```text
sink = content/format prior + RoPE 起始位置相位 + softmax anchor
```

但其中最决定 attention mass 的是起始位置，尤其是最前 1-2 个 token。

## 7. Evidence 是否因此被替代？

完整 zero/drop sink 后，evidence mass 没有明显恢复：

| condition | evidence_key mass | evidence_label mass | evidence_any mass |
|---|---:|---:|---:|
| baseline | 1.42% | 0.14% | 2.88% |
| zero_sink_kv | 1.06% | 0.14% | 2.32% |
| drop_sink_kv | 1.13% | 0.17% | 2.50% |
| keep_prefix2_drop_sink_kv | 1.45% | 0.14% | 2.93% |

说明：

1. sink 不是简单抢走 evidence attention mass。
2. 完整去掉 sink 后，模型没有自动把 attention 转移到 evidence。
3. PPL 崩坏但 evidence mass 不恢复，说明 sink 更像全局稳定器/normalization anchor，而不是 retrieval evidence 的竞争者。

## 8. 当前结论

最重要的结论：

```text
当前模型极度依赖最前 1-2 个 KV 作为全局 attention anchor。
```

更具体：

1. 保留前 16 个位置但替换内容，sink mass 仍高，支持位置/RoPE 成分强。
2. 保留 sink 内容但移到中间/末尾，sink 内容 mass 大幅下降，支持起始位置是关键。
3. 删除第 3-16 个 sink token 基本不伤 PPL/accuracy，说明“sink 16 token”里真正关键的是前 1-2 个。
4. zero/drop 完整前 16 个 KV 会让 PPL 爆炸，但只 drop 第 3-16 个 KV 不会，进一步确认前 1-2 个 KV 是关键。
5. pre-RoPE logit 显示 sink 内容本身也有对齐，但 move ablation 说明 post-RoPE 位置相位决定它是否真的获得 attention mass。

一句话：

```text
sink 不是答案证据，而是 RoPE/causal attention 下形成的起始位置全局锚点；
它对生成的主要贡献是稳定 attention 分布和提供全局 anchor，而不是提供 retrieval value。
```

## 9. 下一步建议

下一步可以把方法设计从“保护前 16 个 sink”改成更小的 anchor：

```text
always keep first 2 tokens + short recent + evidence/value-aware rescue
```

这样比保护前 16 个 sink 更省预算，同时不破坏 PPL。

