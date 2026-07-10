# Section 109: RiskKV-Block ICLR/ICML 自审与下一步

## 当前最强结果

同一批 Llama-3.1-8B-Instruct / LongBench all-task m20：

```text
method      score     KV ratio   online
full KV     0.372655  100.00%    3.033s
v35         0.349037  35.36%     2.526s
v36         0.367089  51.66%     2.768s
v37         0.370563  56.53%     2.720s
v52 actual  0.373143  58.17%     2.720s
v53 actual  0.375890  62.89%     2.736s
```

v52/v53 已启动 actual full m20，完成后需要确认是否完全复现 stitched。

## 现在最有创新性的版本

建议论文主线从早期的“task policy router”升级为：

```text
RiskKV-Block:
  1. materialized KV page action space
  2. query-aware evidence-flow page scoring
  3. task-family minimum-safe action policy
  4. output-contract verifier
  5. self-grounding verifier
  6. memory-action consistency verifier
  7. cache-native RoPE-aware repacking
```

其中最像顶会创新点的是第 6 点：

```text
memory-action consistency verifier
= 同一个 query 在两个不同 sparse memory action 下生成两个答案；
  如果两个答案不一致，说明当前 memory action 不稳定；
  不看 gold answer，不看 full KV output，只看模型在不同 memory 下的自洽性。
```

这比单纯的 prompt/router/fallback 更有方法味道，因为它提供了一个 label-free counterfactual risk signal。

## 对审稿人的核心故事

建议这样讲：

```text
现有 KV compression 多数问的是 which tokens to keep。
RiskKV-Block 问的是 which memory action is safe。

action 不只是 top-k token：
  - page granularity
  - evidence-flow local support
  - bridge expansion
  - verifier
  - consistency check
  - minimum-safe fallback

因此方法不是 RAG，也不是普通 prompt retrieval：
  - 输入不是外部数据库，而是已 materialized 的 KV cache；
  - 目标不是找文档，而是在 decode 时选择 active KV pages；
  - verifier 不使用检索标签，而是判断当前 KV action 是否安全。
```

## 当前最强证据

1. v35 证明 compact policy 可用：

```text
0.349037 score, 35.36% KV
```

2. v36/v37 证明 minimum-safe action 有效：

```text
v36: 98.5% full score, 51.66% KV
v37: 99.4% full score, 56.53% KV
```

3. v38-v45 负结果证明“简单加预算/窗口”不是答案：

```text
progressive sparse retry 会伤 hotpot/musique；
expanded sparse2048 不能替代 full；
code recent/hybrid 不能超过 v35 compact。
```

4. v46/v47 证明 consistency verifier 是有效新模块：

```text
QA mean:
v35       0.3045
v46       0.3161
v47       0.3333
full QA   0.3466
```

5. v52/v53 stitched 说明新模块能形成更强 Pareto：

```text
v52 actual:   0.373143, 58.17% KV
v53 actual:   0.375890, 62.89% KV
full KV:      0.372655, 100.00% KV
```

## 现在的主要弱点

1. m20 样本太小。
   Reviewer 会质疑 task-family policy 是否对 20 samples/task 过拟合。

2. baseline 横向还不完整。
   需要与 AdaKV / SnapKV / PyramidKV / H2O / StreamingLLM 在同一 LongBench 配置下比较。

3. cache-native 与 prompt-level 结果还没有完全统一。
   当前 LongBench v35-v53 是 harness-level page gather；RULER/attention 子系统有 cache-native/RoPE 证据，但最终论文要把两条线合成一个 deployment story。

4. consistency verifier 有额外 decode 开销。
   需要报告它的 online overhead，并解释为什么在长上下文或多 token decode 下仍然值得。

5. v52/v53 stitched 超过 full KV 需要谨慎。
   这可以作为“compressed context sometimes denoises full context”的观察，但不能过度宣传，必须等 actual 和更大样本确认。

## 短期最高优先级实验

1. 等 v52/v53 actual m20 完成。
   如果复现 stitched，则更新论文主表。

2. 跑更大样本 LongBench：

```text
full KV
v37 high-quality
v52 consistency-quality
v53 consistency-quality-qasper-full
```

建议先 m50，再 m100，最后 full split。

3. 做 action-consistency ablation：

```text
v35 baseline
v35 + output verifier only
v35 + grounding verifier only
v35 + consistency verifier only
v52 full combined
```

4. 做 consistency verifier 的 precision/recall 诊断：

```text
当 consistency disagreement 触发时：
  sparse 是否低于 full？
  fallback 是否修复？
当 consistency 不触发时：
  sparse 是否已经安全？
```

注意这个诊断可以用 full/gold 做离线分析，但部署时不使用。

5. 横向复现 Table 5:

```text
Llama-3.1-8B-Instruct LongBench question-aware
对比 Full Cache / SnapKV / PyramidKV / Ada-SnapKV / Ada-Pyramid
```

## 当前投稿判断

如果只看 v35-v37：

```text
创新性：中等，像 strong engineering + task policy。
性能：不错，但 reviewer 可能说 fallback 太简单。
```

如果 v52/v53 actual 复现，并补上更大样本：

```text
创新性：明显增强，因为有 label-free memory-action consistency verifier。
性能：可能达到 strong reject 边缘到 weak accept 区间，取决于 baseline 和大样本稳定性。
```

真正冲 ICLR/ICML，需要把论文主线压到一句话：

```text
We compress KV cache by routing over complete memory actions, and estimate action risk through counterfactual consistency across sparse memory executions.
```

这句话比“router 选择 block”更有顶会味道。

## 20260709 actual update

v52/v53 actual full m20 已完成，并完全复现 stitched：

```text
v52 actual: 0.373143, 58.17% KV, 2.720s online
v53 actual: 0.375890, 62.89% KV, 2.736s online
full KV:    0.372655, 100.00% KV, 3.033s online
```

因此当前下一步不是继续等 m20，而是扩大到 m50/m100，并补齐横向 baseline。

## 20260709 m50 / overhead update

已启动 m50 稳定性验证，覆盖：

```text
full KV
StreamingLLM sink+recent
v35 compact
v36 balanced
v37 high-quality
v52 consistency-quality
v53 consistency-quality-qasper-full
```

当前还在运行。m50 的目标是验证 v52/v53 超过 full KV 的现象是否能从 m20 扩展到更大样本。

## m50 full validation result

m50 已完成后，m20 的“超过 full KV”现象没有在更大样本上保持：

```text
method        score     KV ratio   online
full KV       0.371970  100.00%    3.283s
StreamingLLM  0.174162    3.31%    2.777s
v35 compact   0.326791   34.25%    2.655s
v36 balanced  0.345119   51.04%    2.652s
v37 quality   0.346527   56.21%    2.720s
v52 consist   0.352895   57.76%    2.735s
v53 +qasper   0.358321   62.47%    2.801s
```

因此论文主 claim 需要从：

```text
compressed KV exceeds full KV
```

改成更稳的：

```text
counterfactual risk control improves the best practical sparse policy
from 93.2% to 96.3% of full-cache score on m50,
while using 62.5% active KV and reducing online time from 3.283s to 2.801s.
```

具体比例：

```text
v37 / full = 93.2%
v52 / full = 94.9%
v53 / full = 96.3%
```

这个结果没有 m20 那么惊艳，但可信度更高：v53 相比 v37 提升 `+0.0118` absolute score，并且仍显著少于 full KV。

对审稿叙事的影响：

```text
1. 不再把 exceed full 作为主卖点，只作为 m20 observation。
2. 主卖点变成 memory-action risk control systematically improves sparse policy quality.
3. m100 仍值得跑，因为 m50 显示 v52/v53 稳定优于 v37。
4. v63/v64 benefit-conformal 是下一步关键：目标是保持 v53 质量，同时降低 verifier check 和 KV/online。
```

同时新增 short-probe consistency 版本：

```text
v55 = v52 + consistency_probe_max_tokens=16
v56 = v53 + consistency_probe_max_tokens=16
```

动机：memory-action consistency verifier 只需要判断两个 sparse action 的短答案是否一致，不一定需要完整 decode。v55/v56 用 16-token probe 降低 verifier 开销。

服务器上已启动 watcher：

```text
scripts/watch_and_launch_v55_v56_probe_20260709.sh
```

它会等待至少两张 GPU 空闲后自动启动 v55/v56 m20。

内部 H2O/SnapKV baseline 尝试在 7.5k context + eager attention 下 OOM，因此当前不把它作为有效 baseline。StreamingLLM 不需要 attention matrix，已作为弱 baseline 跑 m50。

## 20260709 m100 preparation

m100 稳定性脚本已准备并同步到服务器，但暂未启动：

```text
scripts/run_riskkv_v37_v52_v53_m100_20260709.sh
scripts/watch_and_launch_m100_after_m50_20260709.sh
```

建议策略：

```text
1. 等 m50 的 full / v37 / v52 / v53 都完成。
2. 如果 v52/v53 在 m50 仍然稳定优于或接近 full，则启动 m100。
3. m100 跑 full / v37 / v52 / v53 四条即可，v35/v36/StreamingLLM 可以保留 m50 作为辅助证据。
```
