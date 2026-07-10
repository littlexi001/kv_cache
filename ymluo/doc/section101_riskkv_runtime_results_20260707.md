# Section 101: RiskKV 当前 runtime 结果与 ICML 判断（2026-07-07）

## 总体判断

当前 readiness report 给出：**ICML_CANDIDATE**。

这不是说已经稳中，而是说明证据链已经可以作为 ICML 主线候选来写：

- 主方法应写成 **RiskKV input planner**。
- Output verifier 应写成安全闭环、蒸馏标签来源和高风险 fallback，而不是默认 runtime 路径。
- 主要风险仍在 LongBench：m8 扩样后质量超过 full，但 online speed 是 `0.988x`，属于接近持平而不是明显加速。

结果文件位置：

- `outputs/remote_runtime_status_20260707/runtime_scaling_summary_20260707/runtime_scaling_summary.csv`
- `outputs/remote_runtime_status_20260707/runtime_scaling_summary_20260707/icml_tables/main_runtime_table.md`
- `outputs/remote_runtime_status_20260707/runtime_scaling_summary_20260707/icml_figures/speed_scaling.svg`
- `outputs/remote_runtime_status_20260707/runtime_scaling_summary_20260707/icml_figures/accuracy_kv_pareto.svg`
- `outputs/remote_runtime_status_20260707/runtime_scaling_summary_20260707/icml_readiness/icml_readiness_report.md`

## 最强主结果

| Setting | Method | N | Score | KV | Online | E2E |
|---|---|---:|---:|---:|---:|---:|
| LongBench m8 | RiskKV conformal input planner | 40 | 32.50% | 26.24% | 0.988x | 0.992x |
| LongBench m4 | RiskKV conformal input planner | 20 | 25.00% | 26.25% | 0.996x | 0.997x |
| Mixed13 m2 | RiskKV min-safe input planner | 26 | 69.23% | 23.04% | 0.993x | 0.996x |
| Mixed13 m2 | RiskKV conformal input planner | 26 | 65.38% | 15.53% | 0.991x | 0.994x |
| RULER 4k m5 | RiskKV conformal floor2 input planner | 40 | 100.00% | 26.30% | 0.991x | 0.994x |
| RULER 8k m5 | RiskKV conformal floor2 input planner | 40 | 100.00% | 18.25% | 1.075x | 1.035x |
| RULER 16k m3 | RiskKV conformal input planner | 24 | 100.00% | 8.44% | 1.669x | 1.184x |
| RULER 16k m3 | RiskKV bestcal input planner | 24 | 100.00% | 6.41% | 1.659x | 1.177x |

解释：

- LongBench: input-side planner 已经解决 output verifier 的明显亏速问题。m8 上 full 是 `25.00%`，RiskKV conformal 是 `32.50%`，KV 降到 `26.24%`；但 online 是 `0.988x`，所以仍是 borderline。
- Mixed13: min-safe planner 在 m2 上保持 full-level score `69.23%`，KV `23.04%`，online `0.993x`；conformal 更省 KV（`15.53%`），但 score 降到 `65.38%`，因此论文主表里 Mixed13 应优先放 min-safe 或同时报告 safety/efficiency trade-off。
- RULER 4k/8k/16k: input-side planner 证明了真正的 length scaling。4k m5 需要 k2 safety floor 才能从 `95.00%/97.50%` 恢复到 `100.00%`，速度仍接近持平；8k m5 达到 `1.075x`；16k m3 达到 `1.669x`。

## 速度开销解释

最新 overhead report 显示：

| Setting | Query saved | Decode saved | Planner | Repack | Net component gain |
|---|---:|---:|---:|---:|---:|
| LongBench m8 conformal | 11.60 ms | 0.96 ms | 22.03 ms | 14.90 ms | -24.37 ms |
| Mixed13 m2 conformal | 8.83 ms | 3.34 ms | 17.58 ms | 14.17 ms | -19.57 ms |
| RULER 4k m5 conformal floor2 | 6.22 ms | 2.34 ms | 18.42 ms | 8.63 ms | -18.50 ms |
| RULER 8k m5 conformal floor2 | 24.91 ms | 155.82 ms | 19.66 ms | 8.90 ms | 152.17 ms |
| RULER 16k m3 conformal | 305.68 ms | 1425.38 ms | 209.52 ms | 10.52 ms | 1511.02 ms |

这说明当前不是 KV 压缩无效，而是短上下文/短生成时 planner+repack 固定开销超过了 query/decode 节省。长上下文下 query/decode 节省开始主导，所以 8k/16k 能兑现 online speedup。

## 对论文主线的影响

现在可以把论文主 claim 写成：

> RiskKV performs cache-native, risk-constrained KV budget planning after full-context prefill. It preserves full-context behavior while reducing active KV to roughly 15-26% on mixed/LongBench settings, 26% on the conservative RULER 4k floor2 setting, and 8-18% on RULER 8k/16k long-context settings. The input-side planner removes most of the output verifier overhead and yields real online speedups as context length grows.

不要把 claim 写成：

- “所有场景都端到端加速”。
- “LongBench 上显著加速”。
- “模型准确率提升”。

更稳的写法是：

- 长上下文 serving 场景随 context length 增长出现真实 online speedup。
- 4k/mixed 场景主要展示质量保持和 KV 降低，速度接近持平。
- Output verifier 是安全机制，不是默认高效路径。

## 当前风险

1. LongBench m8 online 是 `0.988x`，m4 online 是 `0.996x`，都没有明显超过 full。
2. Mixed13 online 是 `0.993x` 左右，也是接近持平。
3. RULER 仍偏 synthetic，需要把 LongBench/mixed 放在主表里避免被认为只在 needle task 有效。
4. 4k m5 证明 k1 对 multi-evidence case 不安全；必须把 k2 floor 写成 safety mechanism，而不是事后挑结果。

## 下一步

优先级最高：

1. 写方法部分，把 `variable_budget_min_budget=2` 表述为 multi-evidence safety floor / lower-bound budget，而不是结果修补。
2. 对 LongBench 做 planner overhead 消融：planner/repack/query/decode 各自占比，证明为什么短上下文只能持平。
3. 写方法部分：RiskKV input planner + RoPE-aware repack + conformal calibration + optional verifier fallback。
4. 组织主表：LongBench、Mixed13、RULER 8k/m5 或 m3、RULER 16k/m3 或 m2。

如果时间允许：

1. 跑更长上下文，例如 32k，展示 speedup 继续扩大。
2. 跑另一个模型或至少 Qwen3-0.6B/8B 对比，增强泛化。
3. 做风险校准表：selected tau、test failure rate、KV ratio、speed。
