# Section 104: Small Block Size 探索计划与当前发现（2026-07-07）

## 背景

上一轮 block-size/topK sweep 说明，block size 会显著影响质量、active token ratio 和速度。256/512 已经明显优于 1024/2048：

- RULER 4k：`block256 top3` 达到 100% score，30.63% token，1.33x e2e。
- RULER 8k：`block256 top3` 达到 100% score，15.27% token，1.70x e2e。
- RULER 16k：`block512 top3` 达到 100% score，10.70% token，1.96x e2e。
- LongBench：overall 最好是 `block512 top4`，但 `block256 top4` 已经超过 full_raw 且 token 更低。

这说明 block size 不应该只是固定超参，而应该进入 RiskKV 的动作空间。

## 已完成的 block-size router 结果

服务器上 `qwen8b_blocksize_router_m3_20260707` 已经完成。它使用 sweep 训练出的 router，在同一批 m3 benchmark 上端到端选择 block size + topK。

| Method | Samples | Score | Token ratio | E2E speed |
|---|---:|---:|---:|---:|
| full_raw | 96 | 0.8116 | 100.00% | 1.000x |
| fixed `recent_plus_span_top3_b0_a0` | 96 | 0.8107 | 18.22% | 1.669x |
| block-size router | 96 | 0.8535 | 11.04% | 1.756x |

这个结果很重要：block-size-aware router 不只是 oracle sweep 上更好，实际 router 也同时提高了 score、降低 token、提升速度。

## 当前判断

新的方法主线可以从 **RiskKV-Floor** 升级成：

> **RiskKV-Block: risk-constrained block-size and budget routing for cache-native long-context inference.**

核心变化：

1. 原来 router 只选 KV budget / topK。
2. 现在 router 同时选择 block size、topK 和 fallback。
3. 小 block 适合 precise evidence retrieval；大 block 适合作为 hard case 或跨边界证据的安全回退。

建议 action space：

- `b32_top{8,16,24,32}`：极细粒度检索，重点看是否能继续压低 token。
- `b64_top{4,8,12,16,24,32}`：可能是 32 和 256 之间的最佳折中。
- `b128_top{2,4,6,8,12,16}`：可能接近 256 的质量，但 token 更低。
- `b256_top{1,2,3,4}`：当前 RULER 默认强基线。
- `b512_top{1,2,3}`：16k hard case 安全回退。
- `b1024_top1`：LongBench 单证据 retrieval 的备选。
- `summary/full fallback`：summary、count、scientific QA 不能强行 sparse span。

## 正在运行的小 block sweep

已启动服务器后台任务：

```bash
scripts/run_qwen8b_smallblock_topk_sweep_m3_20260707.sh
```

输出目录：

```text
outputs/qwen8b_block32_topk_sweep_*_m3_20260707
outputs/qwen8b_block64_topk_sweep_*_m3_20260707
outputs/qwen8b_block128_topk_sweep_*_m3_20260707
```

覆盖：

- block size: 32, 64, 128
- topK: 1, 2, 3, 4, 6, 8, 12, 16, 24, 32
- benchmark: LongBench m3, RULER 4k/8k/16k m3
- recent raw: 512 tokens
- model: Qwen3-8B + LoRA adapter

汇总脚本：

```bash
python scripts/summarize_smallblock_topk_sweep_20260707.py
```

默认会同时读取 32/64/128/256/512/1024/2048，输出：

```text
outputs/smallblock_topk_sweep_m3_summary_20260707.txt
```

## 预期判断规则

如果小 block 成功：

- RULER 8k/16k 应该能把 token ratio 从 15.27% / 10.70% 进一步压到 5-12%。
- 如果 64/128 能保持 100% score 且 token 低于 256，则新版 router 应优先纳入 64/128。
- 如果 32 需要 top32 才稳定，说明 block 太碎，调度成本和检索噪声可能不划算。

如果小 block 失败：

- 32/64 可能因为证据边界被切碎，导致 topK 虽大但语义不完整。
- 这种情况下 128/256 应作为默认，512 作为安全回退。
- 论文里仍可把 block-size sensitivity 写成重要发现：过粗浪费 token，过细破坏 evidence locality，中间粒度由 risk router 自适应选择。

## 下一步

1. 等小 block sweep 完成后运行 `summarize_smallblock_topk_sweep_20260707.py`。
2. 生成新的 oracle label，训练包含 32/64/128 的 block-size-aware router v2。
3. 用 `router_blocksize` 在 m3/m10 上端到端验证。
4. 如果 v2 能稳定超过当前 router 的 `0.8535 score / 11.04% token / 1.756x`，将论文主线升级为 RiskKV-Block。

## Small block sweep 已完成结果

32/64/128 的 m3 sweep 已完成，服务器上没有相关进程，GPU 已空闲。汇总文件：

```text
outputs/smallblock_topk_sweep_m3_summary_20260707.txt
```

本地备份：

```text
ymluo/projects/learned_hierarchical_summary_memory/outputs/remote_runtime_status_20260707/smallblock_topk_sweep_m3_summary_20260707.txt
```

整体推荐点：

| Group | Full score | Best score | Best block/topK | Token | E2E | Min >= full |
|---|---:|---:|---|---:|---:|---|
| LongBench | 0.2465 | 0.3294 | b128 top12 | 15.5% | 1.58x | b128 top6 @ 10.9% |
| RULER 4k | 1.0000 | 1.0000 | b256 top3 | 30.6% | 1.33x | b256 top3 @ 30.6% |
| RULER 8k | 1.0000 | 1.0000 | b256 top3 | 15.3% | 1.70x | b256 top3 @ 15.3% |
| RULER 16k | 0.8750 | 1.0000 | b512 top3 | 10.7% | 1.96x | b64 top4 @ 4.8% |

主要发现：

1. **LongBench 最适合 b128。** `b128 top12` 达到 0.3294，比之前 `b512 top4` 的 0.3262 略高，同时 token 从 19.9% 降到 15.5%。如果只要求超过 full_raw，`b128 top6` 只要 10.9% token。
2. **RULER 4k/8k 的 100% 质量仍然需要 b256。** 32/64/128 更省 token，但整体达不到 full-level 100%。这些可以作为 cost-strict action，不应该作为 quality-strict 默认。
3. **RULER 16k 分成两条线。** 如果要求 100% score，仍然是 `b512 top3`；如果只要求达到 full_raw 0.875，`b64 top4` 只要 4.8% token，并且有 2.18x e2e speed。
4. **b32 太碎。** 在 RULER 4k/8k 上即使 top32 也达不到 100%，LongBench 也没有超过 b128；它更适合作为极低成本候选，不应进入主 router 的高优先动作。

Router v2 推荐 action space：

```text
b64_top4
b64_top8
b64_top16
b128_top4
b128_top6
b128_top8
b128_top12
b256_top2
b256_top3
b256_top4
b512_top2
b512_top3
b1024_top1
summary/full fallback
```

不建议默认纳入 `b32` 和 `b2048`。`b32` 太碎，质量风险高；`b2048` 太粗，token 成本高。它们可以放 appendix 或 exploratory ablation。
