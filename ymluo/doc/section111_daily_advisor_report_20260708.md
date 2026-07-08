# Section 111: 2026-07-08 导师汇报摘要

## 一句话结论

当前最值得继续推进的主线是 **RiskKV-Block: block-size-aware, risk-calibrated KV budget planning**。它不是 oracle，也不是纯 prompt/RAG 方法，而是在 full-context prefill 之后，对 KV pages / blocks 做动态预算选择、RoPE-aware repack 和安全校准。

目前最强的可用证据是：

1. 在 LongBench 和 RULER 4k/8k/16k 上，实际可运行策略可以把 active token/KV 压到约 10%-30%。
2. 在当前 benchmark harness 中，整体端到端速度已经能看到 1.2x-2.0x 级别收益。
3. 在更严格的 KV runtime online 口径下，4k 基本持平，8k 开始加速，16k 可到约 1.67x。
4. 在 attention/KV 子系统 microbenchmark 口径下，长上下文多步 decode 已经可以看到 4x-6x 量级加速。

汇报时需要强调：**质量和 KV 压缩已经比较强，速度收益在长上下文和 fused/page-aware runtime 中更容易兑现；当前 HF/Python 原型会稀释 attention saving。**

## 当前最好可用方法

当前建议汇报的主方法不是 oracle，也不是 free router，而是：

```text
riskkv_block_calibrated / blocksize_calibrated_floor_only
```

核心流程：

1. 将上下文切成 blocks，候选 block size 包括 128 / 256 / 512 等。
2. 对 query 和 blocks 做 evidence scoring，选择 top-k blocks。
3. router / calibrated floor 选择完整 action，而不是只选择 top-k 或只选择 block size。
4. complete-action calibration 给不同任务族和长度一个最小安全动作。
5. 选中的 blocks 被用于后续 generation；在 cache-native runtime 中对应 KV page selection + RoPE-aware repack。

需要特别说明：block size 不是单调安全变量。`b512 top3` 不一定支配 `b256 top3`，`b256 top3` 也不一定支配 `b128 top12`，因为 block 切分会改变 evidence scorer 的排序。因此现在更合理的策略是直接选择完整 action：

```text
a_final = phi(task_family, context_length, risk_features)
```

而不是：

```text
a_final = max_lattice(a_raw, floor)
```

## 当前主要结果

### m20 可运行策略结果

| Setting | Full score | Best practical action | Score | Token ratio | Harness speed |
|---|---:|---|---:|---:|---:|
| LongBench m20 | 0.3596 | b128 top12 | 0.3652 | 15.07% | 1.555x |
| RULER 4k m20 | 1.0000 | b256 top3 | 0.9938 | 30.48% | 1.224x |
| RULER 8k m20 | 1.0000 | b256 top3 | 0.9812 | 15.21% | 1.589x |
| RULER 16k m20 | 0.8688 | b512 top3 | 1.0000 | 10.92% | 1.995x |

四组等权平均：

```text
full_raw score ~= 0.8071
calibrated practical score ~= 0.8351
average token ratio ~= 17.9%
average harness speed ~= 1.58x
```

解释：

1. LongBench 上 score 略高于 full，同时 token ratio 只有 15.07%。
2. RULER 4k / 8k 还有少量 failure，需要 boundary-aware fallback 或更保守 floor。
3. RULER 16k 上质量超过当前 full baseline，token ratio 只有 10.92%，速度接近 2x。

### 质量优先的 refinement 观察

RULER 8k refinement 中，`b512 top3` 可以达到 full-level：

| Setting | Action | Score | Token ratio | Speed |
|---|---|---:|---:|---:|
| RULER 8k partial refinement | b512 top3 | 1.0000 | 22.32% | 1.524x |

RULER 4k 的失败更像 boundary / evidence spillover 问题。failure replay 显示 `b128 top12 b0_a1` 可以修复已知 4k 失败，但这只是小规模 failure replay，还需要完整验证。

## 速度口径说明

今天最容易被问到的问题是：为什么之前 attention 子系统能 3-5x，现在 online speed 只有 0.99x / 1.08x / 1.67x？

答案是：这两个速度不是同一个口径。

### 口径 1: attention/KV subsystem microbenchmark

这个口径只测：

```text
router / scoring / top-k
+ KV gather / compact
+ compact attention
```

它不包含：

```text
full prefill
MLP
lm_head
tokenizer
完整 HF generate
Python decode loop
```

它通常用 64 / 256 / 1024 decode steps，因此一次 page selection / repack 的开销可以被很多 decode token 均摊。

代表性结果：

| Context | Method | Active KV | Speedup |
|---:|---|---:|---:|
| 16k | recent-plus k2 | 2560 | 5.77x |
| 16k | recent-plus k3 | 3584 | 4.50x |
| 20k | recent-plus k2 | 2560 | 6.95x |
| 20k | recent-plus k3 | 3584 | 5.43x |
| 20k | recent-plus k4 | 4608 | 4.47x |

这个口径回答的问题是：

```text
如果 full KV cache 已经存在，并且后续连续 decode 多步，
attention/KV 子系统本身可以加速多少？
```

### 口径 2: runtime online speed

runtime 表里的 online speed 不是单步 decode，也不是纯 attention。它定义为：

```text
full_online =
  full_query_on_cache
+ full_decode

riskkv_online =
  planner
+ gather
+ repack
+ compact_query_on_cache
+ compact_decode
```

其中 decode 是真实任务生成。当前脚本默认：

| Task type | Decode upper bound |
|---|---:|
| QA / RULER / code / classification | 48 tokens |
| summarization | 120 tokens |

因此 online speed 也有一定均摊，但远短于 subsystem benchmark 的 256 / 1024 steps，并且包含更多 HF/Python runtime 成本。

代表性结果：

| Setting | Score | KV ratio | Attention upper bound | Runtime online speed |
|---|---:|---:|---:|---:|
| LongBench m8 conformal planner | 32.50% | 26.24% | 3.81x | 0.988x |
| Mixed13 m2 min-safe | 69.23% | 23.04% | 4.34x | 0.993x |
| RULER 4k floor2 | 100.00% | 26.30% | 3.80x | 0.991x |
| RULER 8k floor2 | 100.00% | 18.25% | 5.48x | 1.075x |
| RULER 16k conformal | 100.00% | 8.44% | 11.85x | 1.669x |

### 为什么理论上界和实测差很多

`Attention upper bound = 1 / KV ratio` 假设所有在线时间都花在 attention 上，并且没有 planner / repack 开销。现实中不是这样。

已有 overhead report：

| Setting | Query saved | Decode saved | Planner | Repack | Net component gain |
|---|---:|---:|---:|---:|---:|
| LongBench m8 | 11.60 ms | 0.96 ms | 22.03 ms | 14.90 ms | -24.37 ms |
| Mixed13 m2 | 8.83 ms | 3.34 ms | 17.58 ms | 14.17 ms | -19.57 ms |
| RULER 4k | 6.22 ms | 2.34 ms | 18.42 ms | 8.63 ms | -18.50 ms |
| RULER 8k | 24.91 ms | 155.82 ms | 19.66 ms | 8.90 ms | +152.17 ms |
| RULER 16k | 305.68 ms | 1425.38 ms | 209.52 ms | 10.52 ms | +1511.02 ms |

结论：

1. 4k / LongBench 中，attention saving 太小，planner + repack 固定开销抵消收益。
2. 8k 开始，decode attention saving 超过 overhead，所以 online speed > 1。
3. 16k 中，attention saving 显著大于 overhead，所以能看到 1.67x online speed。
4. 如果进入 fused/page-aware runtime，并且 generation 更长，subsystem 的 3-5x 更可能转化为端到端收益。

## 和 RAG / prompt rebuild 的边界

主方法应表述为 cache-native KV planning，而不是 RAG：

1. RAG / prompt rebuild 是检索文本块后重新拼 prompt 并重新 prefill。
2. RiskKV-Block 是先 full-context prefill，保留 full context 的 KV cache，然后选择 KV pages / blocks 参与后续 query/decode。
3. RoPE-aware repack 修正 selected KV 在 compact sequence 中的位置编码问题。
4. fallback / safety floor 是为了避免低预算在 multi-evidence 或 boundary case 中丢关键证据。

可以这样讲：

```text
RiskKV-Block does not retrieve external documents or rebuild the prompt.
It performs risk-constrained budget allocation over internal KV pages after full-context prefill.
```

## 长尾词 block matching 的新探索

新 idea 是减少 block scoring 的扫描成本：

1. 每个 block 只保留更稀有的一半 content words。
2. 数字、人名、identifier 等强信号始终保留。
3. 建 inverted index: term -> block ids。
4. query 时只访问命中长尾词的 blocks，而不是扫全体 blocks。

初步 probe 结论：

1. query-time scoring 从毫秒级降到十微秒级量级。
2. index build 在 4k/8k/16k 下大约是 0.003s / 0.0068s / 0.014s 量级。
3. top3 直接替换会有质量风险。
4. top8/top12 或 candidate recall + rerank 更稳。

这条线适合作为系统优化或 appendix，不建议今天汇报成主结果。

## 今天汇报建议

建议按下面顺序汇报：

1. 问题：长上下文 KV cache 很大，直接 full attention 成本高；固定压缩预算不安全。
2. 方法：RiskKV-Block 学习/校准一个 block-size-aware 的安全预算策略，选择完整 action。
3. 结果：实际可用策略在 LongBench/RULER 上保留约 10%-30% token，质量基本接近或超过 full。
4. 速度：attention/KV 子系统在 16k-20k 长上下文可达 4x-6x；当前 HF runtime online 在 8k/16k 兑现到 1.08x/1.67x。
5. 风险：4k/8k 仍有少量 boundary failure，LongBench full baseline 本身较低，真实 fused runtime 还没完全接入当前最优策略。
6. 下一步：补齐 LongBench Table 5 question-aware 对比、训练更稳的 risk-aware router、验证 boundary-aware fallback，并把当前 best policy 接进真正 page-aware KV runtime。

## 下一步优先级

最高优先级：

1. 跑完 LongBench Table 5 question-aware 配置，至少先复现我们方法在同一任务集合上的完整结果。
2. 对当前模型/action space 重新训练 risk-aware router，标签来自 targeted benchmark + combined benchmark。
3. 完整验证 boundary-aware fallback，尤其是 RULER 4k/8k 的最坏 case。
4. 把当前 best RiskKV-Block policy 接入 KV runtime path，重新测 attention/KV subsystem 和 runtime online speed。

论文侧优先级：

1. 明确主 claim：risk-constrained, block-size-aware KV budget planning。
2. 主表同时报告 score / token ratio / online speed。
3. 速度图强调 length scaling: 4k parity, 8k positive, 16k stronger。
4. 和 AdaKV/SnapKV/Pyramid 等 question-aware KV compression 方法做 LongBench 横向比较。
5. 做 multi-model 验证：Qwen3-8B + Llama-3.1-8B-Instruct。

## 目前不能夸大的点

1. 不能说当前 HF/Python 原型已经端到端 3-5x。
2. 不能只拿 RULER synthetic 证明所有长上下文任务都强。
3. 不能把 oracle min-safe / oracle best 当作主方法结果。
4. 不能说 free router 已经安全；m20 中 free router 在 RULER 4k/8k 有明显 failure。
5. 不能说 LongBench 上速度已经明显加速；当前 LongBench runtime online 仍接近持平。

更稳的表述是：

```text
RiskKV-Block already shows strong quality-preserving KV reduction.
The attention/KV subsystem has large long-context speed potential,
and real runtime speed begins to materialize at 8k/16k.
The next engineering step is fused/page-aware runtime integration.
```
