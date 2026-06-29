# PCIC Blockwise Policy Trace Evidence（2026-06-29）

目的：补强 paper 主线中的动态性证据，证明方法不是固定 sparse operator，而是在 block 级别做 online policy selection；同时量化 rescue gate 的必要性和浪费空间。

本分析只读取已有本地 CSV，不跑模型、不访问服务器、不下载数据。

## Trace 总表

| case | blocks | avg ΔPPL | unique combos | switches | extended | initial→final changes | avoidable ext frac | final trace |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Hard-topic b4 top2 | 4 | 0.004371 | 4 | 3 | 1 | 0 | 0.000 | `0,7/2,0,7,12/0,6/0,13` |
| Hard-topic b4 conditional rescue | 4 | -0.049632 | 2 | 2 | 4 | 2 | 0.000 | `0,6/2,0,7,12/0,6/0,6` |
| Hard-topic b8 top2 | 8 | 0.006074 | 5 | 7 | 5 | 3 | 0.000 | `0,7/2,0,7,12/0,6/0,13/2,0,7,12/0,13/7,13/0,13` |
| Hard-topic b8 conditional rescue | 8 | -0.040598 | 4 | 6 | 8 | 5 | 0.000 | `0,6/2,0,7,12/0,6/0,6/7,13/0,6/7,13/2,0` |
| Needle-style top2 | 4 | 0.000118 | 2 | 1 | 4 | 1 | 0.000 | `2,0,7,12/2,0/2,0/2,0` |
| Needle-style conditional rescue | 4 | -0.000166 | 2 | 2 | 4 | 2 | 0.000 | `2,0/2,0/2,0,7,12/2,0` |
| Hard-topic conditional rescue | 4 | -0.049633 | 2 | 2 | 4 | 2 | 0.437 | `0,6/2,0,7,12/0,6/0,6` |
| War easy regime | 2 | -2.135311 | 1 | 0 | 0 | 0 | 0.000 | `0,7/0,7` |
| Monte online selection | 2 | -0.219215 | 2 | 1 | 2 | 1 | 0.400 | `2,7/2,0` |
| Needle-style conditional rescue | 4 | -0.000166 | 2 | 2 | 4 | 2 | 0.462 | `2,0/2,0/2,0,7,12/2,0` |
| RULER-style multi-needle | 3 | 0.000074 | 1 | 0 | 3 | 2 | 0.375 | `2,0/2,0/2,0` |
| RULER-style variable binding | 3 | -0.000564 | 2 | 1 | 3 | 2 | 0.429 | `0,13/2,0/2,0` |
| RULER-style topic switch | 3 | 0.000139 | 2 | 1 | 3 | 1 | 0.669 | `0,13/2,0/2,0` |

## 对创新性的直接支撑

- **动态策略选择存在**：Hard-topic b8 conditional rescue 使用 4 个不同 final combos，发生 6 次 block-to-block switch；RULER-style variable/topic 也从 `0,13` 切到 `2,0`。
- **rescue gate 不是装饰项**：Hard-topic conditional rescue 中 initial→final changes 为 2，说明 early top choice 会被更长 horizon rescue 改写。
- **不同任务选择不同策略**：War 固定为 `0,7`，Monte 为 `2,7/2,0`，RULER multi-needle 固定为 `2,0`，variable/topic 出现 `0,13→2,0`。这比固定 qabs/SparQ-like operator 更像 online policy selection。
- **速度瓶颈有明确方向**：extension waste 表明部分 extension 后 final combo 不变，存在 calibrated skip-gate 空间；但当前主速度路线仍应是 fused/sparse candidate probe。

## 仍然不足

- 这些 trace 主要来自 hard-topic、needle-style、RULER-style synthetic/offline smoke，还不能替代正式 LongBench/RULER。
- 当前 trace 证明“会切换”，但还需要把切换和文本结构、retrieval/variable binding failure 对齐，形成 paper figure。
- corrected gate 成本仍高；创新性主张可以推进，速度主张必须等 fused probe 或真实 kernel 结果。

## 下一步建议

1. 用同一脚本接入正式 benchmark 的 blockwise CSV，生成 standard trace table。
2. 增加每个 block 的文本摘要/任务位置，画 policy trace figure。
3. 对 `initial→final changes` 的 block 做 case study，解释 rescue gate 修复了什么短视错误。

CSV：`docs/pcic_blockwise_policy_trace_2026_06_29.csv`
