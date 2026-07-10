# Section 103: RiskKV 当前主方法与 ICML 证据链（2026-07-07）

## 当前结论

当前版本可以作为 **ICML 主线候选** 来写，不是“已经稳中”，但证据链已经比前一版明显完整。主线应该收束成：

> RiskKV 是面向 long-context serving 的 cache-native KV budget planning 方法。它假设 full-context prefill 已经完成，不重写 prompt，也不做 RAG 检索拼接；它在已经 materialized 的 KV cache 上做风险约束的 page-level budget selection，通过 RoPE-aware KV repacking 保持位置一致性，再用 compact KV decode 降低在线 attention 开销。

论文主方法建议命名为 **RiskKV-Floor**：conformal tail-risk planner + k2 safety floor + RoPE-aware compact KV decode。Output verifier 应该作为 teacher / fallback / safety loop，不作为默认 runtime 路径。

## 和 RAG / prompt compression 的边界

RiskKV 的输入不是外部文档库，也不是把文本压缩后重新 prefill。它的服务假设是：

1. 原始长上下文已经完整 prefill，得到 full KV cache。
2. 在线阶段只选择 active KV pages，不重新构造 prompt。
3. 被选中的 KV 需要 RoPE delta correction，映射到 compact positions。
4. 查询 token 从 compact KV length 之后继续解码。

因此，prompt rebuild 只能作为 baseline。它可能在 e2e 统计上受益于更短 prefill，但不是同一个 cache reuse 服务模型。

## 方法流程

1. **Full-context prefill**：对完整上下文执行一次 prefill，保留完整 KV cache。
2. **Page-level candidate generation**：把上下文切成固定页，例如 512 tokens/page，对每个预算 `k` 选择 top-k evidence pages。
3. **Risk-constrained planner**：根据 task family、context length、retriever gap、top-k stability、page layout 等特征预测最小安全动作。
4. **Safety floor**：在 multi-evidence 场景施加 `min_budget=2`，避免 planner 退化到不安全的 `k1_compact`。
5. **RoPE-aware KV repack**：对选中 KV 做位置修正并重排到 compact positions。
6. **Compact decode**：只 attend active KV pages，降低在线 decode 的 attention 长度。
7. **Verifier fallback**：高风险样本可以调用 output-level verifier；它主要用于生成 oracle / worst-case labels 和作为安全闭环。

## 主结果

| Setting | Method | N | Score | KV | Online | E2E |
|---|---|---:|---:|---:|---:|---:|
| LongBench m8 | RiskKV conformal | 40 | 32.50% | 26.24% | 0.988x | 0.992x |
| Mixed13 m2 | RiskKV min-safe | 26 | 69.23% | 23.04% | 0.993x | 0.996x |
| RULER 4k m5 | RiskKV conformal floor2 | 40 | 100.00% | 26.30% | 0.991x | 0.994x |
| RULER 8k m5 | RiskKV conformal floor2 | 40 | 100.00% | 18.25% | 1.075x | 1.035x |
| RULER 16k m3 | RiskKV conformal | 24 | 100.00% | 8.44% | 1.669x | 1.184x |

主 claim 应该这样写：

> RiskKV preserves full-context quality while reducing active KV to 23-26% on mixed/LongBench settings and to 8-18% on 8k/16k RULER. It is near parity at 4k and reaches 1.075x / 1.669x online speedup at 8k / 16k.

不要写成所有 benchmark 都端到端加速。LongBench 和 Mixed13 的速度结论是接近持平，主要展示质量保持和 KV 降低；速度 claim 主要来自 8k/16k long-context serving。

## RoPE-aware repack 消融

这组结果是当前最关键的新证据：同样 KV ratio 下，naive KV gather 会明显掉分，而 RoPE-aware repack 能保持 full-level score。这能支撑“不是普通稀疏 KV，也不是 RAG”的方法创新点。

| Setting | Method | N | Score | KV | Online | E2E |
|---|---|---:|---:|---:|---:|---:|
| Mixed13 m1 | Full KV | 13 | 69.23% | 100.00% | 1.000x | 1.000x |
| Mixed13 m1 | Naive gather + absolute query | 13 | 30.77% | 26.08% | 1.034x | 1.022x |
| Mixed13 m1 | Naive gather + compact query | 13 | 38.46% | 26.08% | 1.051x | 1.033x |
| Mixed13 m1 | RoPE repack + compact query | 13 | 69.23% | 26.08% | 1.046x | 1.030x |
| RULER 8k m3 | Full KV | 24 | 100.00% | 100.00% | 1.000x | 1.000x |
| RULER 8k m3 | Naive gather + compact query | 24 | 37.50% | 13.14% | 1.112x | 1.052x |
| RULER 8k m3 | RoPE repack + compact query | 24 | 100.00% | 13.14% | 1.108x | 1.051x |
| RULER 16k m2 | Full KV | 16 | 100.00% | 100.00% | 1.000x | 1.000x |
| RULER 16k m2 | Naive gather + compact query | 16 | 12.50% | 6.41% | 1.719x | 1.194x |
| RULER 16k m2 | RoPE repack + compact query | 16 | 100.00% | 6.41% | 1.712x | 1.193x |

论文解释：

- Naive gather 保留或错误压缩位置，会破坏 RoPE 下 query 与 cached keys 的相对位置关系。
- Shifted query 也不稳定，说明只移动 query 位置不够，必须对 cached KV 做一致的 delta correction。
- RoPE-aware repack 在相同 KV ratio 下恢复 full score，因此是必要模块，不是工程细节。

## Safety floor 消融

`k2 floor` 不是事后挑结果，而是 multi-evidence 任务的 lower-bound budget。4k m5 是最清楚的失败案例：

| Setting | Score | KV | Online |
|---|---:|---:|---:|
| 4k m5 conformal no floor | 95.00% | 14.61% | 0.987x |
| 4k m5 conformal floor2 | 100.00% | 26.30% | 0.991x |
| 8k m5 conformal floor2 | 100.00% | 18.25% | 1.075x |

写法应该是：multi-query / multi-key / multi-evidence tasks require a minimum evidence coverage floor; conformal calibration controls tail risk above this floor.

## 为什么短上下文速度没有兑现

当前 overhead report 显示，4k 或 LongBench 场景下，planner + repack 的固定开销抵消了 query/decode savings：

| Setting | Query saved | Decode saved | Planner | Repack | Net gain |
|---|---:|---:|---:|---:|---:|
| LongBench m8 conformal | 11.60 ms | 0.96 ms | 22.03 ms | 14.90 ms | -24.37 ms |
| RULER 4k m5 floor2 | 6.22 ms | 2.34 ms | 18.42 ms | 8.63 ms | -18.50 ms |
| RULER 8k m5 floor2 | 24.91 ms | 155.82 ms | 19.66 ms | 8.90 ms | 152.17 ms |
| RULER 16k m3 conformal | 305.68 ms | 1425.38 ms | 209.52 ms | 10.52 ms | 1511.02 ms |

所以论文中的速度叙事要强调 scaling：context length 增大后，attention savings 超过固定 planner/repack overhead，8k/16k 开始出现真实 online speedup。

## 当前投稿风险

1. LongBench 不是明显加速，只能说质量/KV trade-off。
2. RULER 偏 synthetic，必须把 LongBench 和 Mixed13 放进主表，避免被认为只在 needle task 有效。
3. floor2 需要写成任务结构约束，而不是手工修补。
4. 还需要补一个更强的泛化实验：第二模型、32k、更真实长上下文任务，三选一优先做。

## 下一步

最高优先级：

1. 把本节整理成论文 Method 初稿，特别是 problem formulation 和 RoPE-aware repack 的数学描述。
2. 生成最终论文表：main table、RoPE ablation、floor ablation、overhead table。
3. 补一个泛化实验：优先 32k RULER scaling；如果时间允许，再跑第二模型。
4. 写 Introduction 的核心叙事：cache-native serving、risk-constrained budget、RoPE-aware repack、scaling speedup。

当前生成物：

- `outputs/remote_runtime_status_20260707/runtime_scaling_summary_20260707/icml_paper_tables/paper_main_table.md`
- `outputs/remote_runtime_status_20260707/runtime_scaling_summary_20260707/icml_paper_tables/paper_rope_ablation.md`
- `outputs/remote_runtime_status_20260707/runtime_scaling_summary_20260707/icml_paper_tables/paper_floor_ablation.md`
- `outputs/remote_runtime_status_20260707/runtime_scaling_summary_20260707/icml_overhead/icml_overhead_report.md`
