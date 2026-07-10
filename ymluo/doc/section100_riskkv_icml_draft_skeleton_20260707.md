# Section 100: RiskKV ICML 草稿骨架（2026-07-07）

## 标题候选

1. **RiskKV: Risk-Constrained KV Cache Budget Planning for Long-Context Language Models**
2. **Conformal RiskKV: Calibrated KV Cache Budgeting with RoPE-Aware Repacking**
3. **Cache-Native Long-Context Compression via Risk-Constrained KV Budget Planning**

当前建议用第 1 个。第 2 个只有在 conformal-auto runtime 结果足够好时再升级。

## 摘要草稿

长上下文语言模型服务中，完整 KV cache 能保持模型质量，但在线解码阶段会带来显著的 attention 开销。现有 prompt compression 或 retrieval 方法通常通过重构文本输入降低长度，但这改变了服务形态，也难以复用已经完成的 full-context prefill。我们提出 RiskKV，一个 cache-native 的 KV cache 预算控制器：它先对完整上下文做 prefill，再基于页面级证据、top-k 稳定性、任务族和上下文长度预测最小安全 KV budget，并通过 RoPE-aware cache repacking 在压缩 KV 上执行 compact decode。为降低质量风险，我们进一步引入输出级 verifier、conformal risk calibration 和 lower-bound safety floor 作为安全闭环。实验表明，RiskKV 能在保持 full-context 输出质量的同时显著降低 active KV；在 Mixed13/LongBench 设置中，active KV 可降至约 15-26% 且在线速度接近 full；在 RULER 8k/16k 长上下文设置中，active KV 可降至约 18%/8%，并带来真实在线解码加速。

当前可用数值：

- LongBench m8 conformal input planner: 32.50% score / 26.24% KV / 0.988x online。
- Mixed13 m2 min-safe input planner: 69.23% score / 23.04% KV / 0.993x online。
- Mixed13 m2 conformal input planner: 65.38% score / 15.53% KV / 0.991x online。
- RULER 4k m5 conformal floor2 input planner: 100.00% score / 26.30% KV / 0.991x online。
- RULER 8k m5 conformal floor2 input planner: 100.00% score / 18.25% KV / 1.075x online。
- RULER 16k m3 conformal input planner: 100.00% score / 8.44% KV / 1.669x online。

## 引言结构

### 第 1 段：问题

长上下文 LLM 的瓶颈不只在 prefill，也在后续 decode 对完整 KV cache 的反复 attention。对于服务场景，full-context prefill 往往已经完成；如果后续每一步都 attend 到完整 KV，成本随上下文长度增长。

### 第 2 段：为什么已有方法不够

Prompt compression / RAG 会重构输入文本，适合从零生成 prompt，但不直接解决已经 prefill 的 KV cache 如何复用和压缩。简单裁剪 KV 又会遇到位置编码不一致和质量风险。

### 第 3 段：核心想法

RiskKV 把问题改写成 risk-constrained KV budget planning：

- 在 KV cache 中选择页面，而不是重写 prompt。
- 用 RoPE-aware repack 保持 compact position。
- 用 input-side planner 预测最小安全预算。
- 用 k2 safety floor 避免 multi-evidence case 退化到不安全的 k1。
- 用 output-level verifier / conformal calibration 处理高风险样本。

### 第 4 段：贡献

建议写 4 点：

1. 提出 cache-native KV budget planning 任务定义，区别于 text compression 和 RAG。
2. 提出 RoPE-aware KV repack，使 compact KV decode 在位置上保持一致。
3. 提出 risk-constrained planner，包括 input-side planner、output-level verifier 和 conformal-auto 校准。
4. 在 Qwen3-8B 上给出 runtime evidence，展示 KV reduction 和长上下文 online speedup。

## 方法章节骨架

### 3.1 Problem formulation

给定上下文 token 序列 `x_1...x_n` 和查询 `q`，full-context prefill 得到 KV cache：

`K,V = f_prefill(x_1...x_n)`

目标是在不重新构造 prompt 的情况下，选择 active KV 子集 `S`，使：

- 质量风险 `Pr[y_S != y_full] <= alpha`
- active KV ratio `|S| / n` 尽可能小
- online decode latency 降低

### 3.2 Page scoring and candidate budgets

把上下文切成固定页面，例如 512 tokens/page。对每个 budget `k`，根据 lexical/evidence scores 选 top-k pages。

候选 action：

- `k1_compact`
- `k2_compact`
- `k3_compact`
- `k4_compact`
- `k6_compact`
- `k8_compact`
- `full`

部署时可以设置 lower-bound budget，例如 `min_budget=2`。这个 floor 对 multi-evidence retrieval 尤其重要：4k m5 中无 floor 的 conformal planner 会因 k1 选择出现掉点，而 floor2 能恢复 full-level score。

### 3.3 RoPE-aware KV repacking

普通 gather 会保留原位置，compact query position 会和 KV position 不匹配。RiskKV 对选中 token 的 KV 做 RoPE delta correction，把它们映射到 compact positions，并让 query 从 compact length 继续。

要强调：

- 不是简单 gather。
- 不是 prompt rebuild。
- 不需要重新 prefill 选中文本。

### 3.4 Input-side risk planner

输入特征：

- context length / query length。
- task family。
- page layout features。
- retriever gap。
- top-k stability。
- candidate KV ratio。

输出：

- 最小安全 action。

训练标签：

- oracle min-safe。
- oracle best。
- worst-case targeted labels。

当前 runtime 待验证：

- best-calibrated tail=0.35。
- min-safe tail=0.35。
- conformal-auto selected tau。

### 3.5 Output-level verifier and fallback

输出级 verifier 解码候选后判断该候选是否安全。它更稳，但会产生多候选 decode 开销。因此最终系统中应作为：

- 训练标签蒸馏来源。
- 高风险 fallback。
- 安全闭环，而不是所有样本默认路径。

### 3.6 Runtime modes

三种模式：

1. full baseline。
2. input-side planner：只 decode 一次，是最终部署路径。
3. output verifier prefix：用于风险较高或分布外场景。

## 实验章节骨架

### 4.1 Setup

模型：

- Qwen3-8B。

Benchmark：

- LongBench: HotpotQA, 2WikiMQA, Musique, Passage Retrieval, Passage Count。
- RULER: niah / vt / cwe / fwe 等。
- Mixed13: 5 LongBench + 8 RULER。

指标：

- Score / exact match。
- Active KV ratio。
- Online speedup。
- E2E speedup。

### 4.2 Main results

主表由 `scripts/make_icml_runtime_tables_20260707.py` 生成；主图由 `scripts/plot_icml_runtime_figures_20260707.py` 生成；投稿判断由 `scripts/make_icml_readiness_report_20260707.py` 生成。

当前主表应优先填：

- LongBench m8 conformal 和 m4 conformal。
- Mixed13 m2 min-safe，并把 conformal 作为更激进 KV trade-off。
- RULER 8k m3 conformal。
- RULER 16k m2 conformal/bestcal；m3 case2 正在补。

### 4.3 Scaling with context length

重点展示：

- 4k 速度收益不明显。
- 8k 开始有真实 online speedup。
- 16k speedup 进一步扩大。

当前已有：

- RULER 4k m5 conformal floor2: 100.00% score / 26.30% KV / 0.991x online。
- RULER 8k m5 conformal floor2: 100.00% score / 18.25% KV / 1.075x online。
- RULER 16k m3 conformal: 100.00% score / 8.44% KV / 1.669x online。

### 4.4 Ablations

必须包括：

- Prompt rebuild vs KV-native。
- RoPE-aware compact vs naive/shifted。
- output verifier vs input-side planner。
- no floor vs floor2 vs conformal-auto。

### 4.5 Risk calibration

如果 conformal-auto 结果可用：

- 报告 selected tau。
- 报告 calibration upper bound。
- 报告 test failure rate。
- 报告 KV/speed tradeoff。

## 结果解释模板

### 如果 input-side planner 成功

可以写：

Input-side RiskKV removes the repeated candidate decoding overhead of output-level verification while preserving the same full-context quality constraint. On LongBench, it improves quality over the full baseline while reducing active KV to about 26%, but its online speed remains close to parity rather than a clear speedup. On RULER 4k/8k/16k, a k2 safety floor preserves full-level accuracy; the method is near parity at 4k and reaches 1.075x/1.669x online speedup at 8k/16k, showing that KV budget planning becomes increasingly beneficial as context length grows.

### 如果 input-side planner 在 LongBench 不够好

改写为：

RiskKV is most effective in long-context serving regimes where attention over the active KV dominates online decoding. LongBench and mixed short-generation settings remain challenging because planner/repack fixed overhead can dominate the savings from KV reduction. The main empirical contribution is therefore cache-native long-context runtime scaling, with output verification providing a conservative safety mechanism.

## 当前写作判断

当前 readiness 已经是 `ICML_CANDIDATE`，可以开始写完整论文初稿，但主张必须收窄为 long-context serving scaling，而不是声称所有 benchmark 都端到端加速。

如果不满足，应转为：

- 强化 RULER/long-context serving 方向。
- 投 CCFB 或 workshop 更稳。
- 继续训练更强的 input-side planner，再冲主会。

## 下一步文件

实验结果回来后运行：

```powershell
powershell -ExecutionPolicy Bypass -File ymluo/projects/learned_hierarchical_summary_memory/scripts/collect_icml_runtime_status_20260707.ps1
python ymluo/projects/learned_hierarchical_summary_memory/scripts/make_icml_runtime_tables_20260707.py --summary_csv ymluo/projects/learned_hierarchical_summary_memory/outputs/remote_runtime_status_20260707/runtime_scaling_summary_20260707/runtime_scaling_summary.csv
python ymluo/projects/learned_hierarchical_summary_memory/scripts/plot_icml_runtime_figures_20260707.py --summary_csv ymluo/projects/learned_hierarchical_summary_memory/outputs/remote_runtime_status_20260707/runtime_scaling_summary_20260707/runtime_scaling_summary.csv
python ymluo/projects/learned_hierarchical_summary_memory/scripts/make_icml_readiness_report_20260707.py --summary_csv ymluo/projects/learned_hierarchical_summary_memory/outputs/remote_runtime_status_20260707/runtime_scaling_summary_20260707/runtime_scaling_summary.csv
```

然后把生成的：

- `main_runtime_table.md`
- `main_runtime_table.tex`
- `best_runtime_rows.md`
- `speed_scaling.svg`
- `accuracy_kv_pareto.svg`
- `icml_readiness_report.md`

填回 section99 和论文草稿。
