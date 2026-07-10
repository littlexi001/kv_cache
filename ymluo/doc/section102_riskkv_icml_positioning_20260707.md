# Section 102: RiskKV ICML 定位、创新性与下一步补强（2026-07-07）

## 当前判断

当前方法已经可以作为 **ICML 主线候选**，但论文主张必须精确：

> RiskKV 是面向 long-context serving 的 cache-native KV budget planner。它不重写 prompt，也不是 RAG；它在 full-context prefill 之后，对 KV cache 做风险约束预算选择、RoPE-aware repack 和 compact decode，并用 conformal calibration / safety floor 控制质量风险。

不要写成：

- 所有 benchmark 都端到端加速。
- LongBench 上显著加速。
- 简单 router 自动学会所有安全预算。

应该写成：

- 短上下文和短生成场景接近持平，主要贡献是质量保持和 active KV 降低。
- 8k/16k 长上下文 serving 场景出现稳定 online speedup。
- k2 safety floor 是 multi-evidence 任务的安全下界，不是事后补丁。

## 最强主方法

论文默认方法建议叫：

**RiskKV-Floor: conformal tail-risk planner with a k2 safety floor**

运行路径：

1. Full-context prefill 一次。
2. Page-level scorer 生成不同 budget 的 KV page candidates。
3. Conformal tail-risk planner 预测最小安全 action。
4. Safety floor 约束：`min_budget=2`，防止 multi-evidence case 退化到不安全的 `k1`。
5. RoPE-aware KV repack 到 compact positions。
6. Compact decode。
7. Output verifier 作为 fallback / distillation teacher，而不是默认 runtime。

这样写的好处：

- 主方法不是 prompt compression。
- 主方法不是单纯 heuristic top-k。
- 主方法不是只会在 synthetic needle 上工作。
- floor2 有失败 case 支撑：4k m5 中无 floor 会掉点，floor2 恢复 full-level。

## 当前主结果

建议主表只放下面这些代表行，避免过多中间版本稀释叙事：

| Setting | Main method | N | Score | KV | Online | 论文解释 |
|---|---|---:|---:|---:|---:|---|
| LongBench m8 | conformal input planner | 40 | 32.50% | 26.24% | 0.988x | 质量和 KV 强，速度接近持平 |
| Mixed13 m2 | min-safe input planner | 26 | 69.23% | 23.04% | 0.993x | 混合任务 match full |
| RULER 4k m5 | conformal floor2 | 40 | 100.00% | 26.30% | 0.991x | safety sanity check |
| RULER 8k m5 | conformal floor2 | 40 | 100.00% | 18.25% | 1.075x | 开始兑现 speedup |
| RULER 16k m3 | conformal input planner | 24 | 100.00% | 8.44% | 1.669x | 长上下文核心结果 |

主 claim 可以写：

> RiskKV preserves full-context quality while reducing active KV to 23-26% on mixed/LongBench settings and to 8-18% on 8k/16k RULER. It is near parity at 4k and reaches 1.075x/1.669x online speedup at 8k/16k.

## 创新点评估

创新性不是靠单个模块，而是靠组合后的问题定义和系统闭环：

1. **Cache-native problem formulation**
   - 不是检索文档重新拼 prompt。
   - 不是 prompt compression。
   - 明确假设 full-context prefill 已经发生，优化后续 online decode 的 active KV。

2. **RoPE-aware KV repacking**
   - 不是简单 gather KV。
   - 需要把选中 KV 映射到 compact positions，并让 query 从 compact length 继续。
   - 这是和普通 KV pruning / retrieval 的关键技术边界。

3. **Risk-constrained variable budget planner**
   - 输出不是固定 top-k，而是最小安全预算。
   - 特征包括 task family、context length、page layout、retriever gap/top-k stability。
   - 标签来自 oracle / worst-case / targeted benchmark。

4. **Conformal calibration + safety floor**
   - Conformal tail-risk 给可解释的风险阈值。
   - k2 floor 处理 multi-evidence lower bound。
   - 4k m5 的 no-floor 掉点和 floor2 恢复，是很好的 ablation。

5. **Output verifier as teacher/fallback**
   - Output verifier 慢，但能提供安全标签和 fallback。
   - Input planner 是部署路径，verifier 是安全闭环。

这个创新组合足够支撑主会投稿，但前提是论文必须把它写成系统性 cache-native serving 方法，而不是“训练了一个 router”。

## 审稿风险与应对

### 风险 1：LongBench online 没有超过 1.0x

应对：

- 主 claim 不说 LongBench 加速。
- LongBench 用来证明非 synthetic 任务上的质量/KV trade-off。
- Overhead table 解释 planner/repack 固定开销抵消短上下文收益。

### 风险 2：RULER 偏 synthetic

应对：

- 主表必须同时放 LongBench m8 和 Mixed13 m2。
- RULER 只承担 length-scaling 论点。
- 写清楚“8k/16k serving”是目标场景。

### 风险 3：floor2 像手工补丁

应对：

- 写成 multi-evidence lower-bound budget。
- 放 no-floor vs floor2 消融：
  - 4k m5 conformal auto: 95.00% / 14.61% KV / 0.987x。
  - 4k m5 conformal floor2: 100.00% / 26.30% KV / 0.991x。
  - 8k m5 conformal floor2: 100.00% / 18.25% KV / 1.075x。
- 解释 k1 对 multi-query/multi-key 证据覆盖不足。

### 风险 4：方法像 RAG

应对：

- Figure 1 画 KV cache flow，不画 document retrieval flow。
- 明确“full prefill already materialized”。
- Prompt rebuild 只作为 baseline，不是同一服务模型。

## 下一步优先级

1. 写完整方法章节：Problem formulation、RoPE-aware repack、Risk planner、Conformal + safety floor、Verifier fallback。
2. 生成论文主表，只保留代表行，不把所有 exploratory rows 放主表。
3. 做 floor ablation 表：no floor / min-safe / conformal floor2。
4. 做 RoPE ablation 表：naive gather / compact query / RoPE-aware repack。
5. 如果还有算力，补一个第二模型或更真实长上下文任务；这会显著提升 ICML 抗审稿风险。

## 当前投稿判断

如果只看性能：

- CCFB：已经比较稳。
- ICML：可以冲，但需要论文叙事非常克制，主打 long-context serving scaling。

如果补齐方法写作和两个关键消融：

- `floor2` safety ablation。
- RoPE-aware repack ablation。

那么这条线具备主会投稿的合理强度。
