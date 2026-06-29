# Horizon-PCIC Paper Blueprint（2026-06-29）

> Corrected 主结果表见：`docs/horizon_pcic_corrected_key_results_2026_06_29.md`。后续 speed/gate claim 以 corrected gate 口径为准。

> Fixed / online / rescue / oracle 主线证据见：`docs/pcic_mainline_fixed_online_rescue_2026_06_29.md`。该表用于支撑“不是固定 combo 小改，而是 online policy selection + rescue gate”的论文叙述。

> Blockwise policy trace 证据见：`docs/pcic_blockwise_policy_trace_2026_06_29.md`。该表用于支撑动态策略切换和 rescue gate 必要性。

> Method spec / 论文算法定义见：`docs/horizon_pcic_method_spec_2026_06_29.md`。该文档给出正式符号、算法伪代码和可审稿 claim 边界。

> Component evidence matrix 见：`docs/pcic_component_evidence_matrix_2026_06_29.md`。该表用于区分已支撑 claim、负面消融和 ICML 级缺口。

> Paper skeleton 见：`docs/horizon_pcic_paper_skeleton_2026_06_29.md`。该文档给出保守 abstract、章节结构、图表计划和 limitations。

## 论文主线

```text
Pairwise-CIC + online blockwise policy selection + horizon-aware top-k cascade rescue gate
```

建议方法名：

```text
Horizon-PCIC: Horizon-aware Pairwise Counterfactual Policy Selection for KV Cache Compression
```

一句话定位：

```text
本文不提出另一个固定 sparse attention / KV eviction 规则；
本文提出一个在线反事实策略选择框架，在每个 long-context block 中动态选择压缩策略，
并用 horizon-aware cascade rescue gate 控制短窗口 calibration、长期 memory prior 和未来 token 风险之间的冲突。
```

## 核心创新点

### 1. 从“压缩规则”转向“策略选择器”

已有方法通常回答：

```text
当前 token 应该保留哪些 KV token/head/layer？
```

Horizon-PCIC 回答：

```text
给定一组候选 KV 压缩 policy，当前 block 应该信任哪一个 policy？
当 local calibration、memory prior 和 horizon probe 冲突时，如何仲裁？
```

这使方法区别于 SparQ/H2O/StreamingLLM/SnapKV/PyramidKV/AdaKV/QUEST 一类固定或半固定压缩启发式。

### 2. Pairwise-CIC

Pairwise-CIC 使用短 calibration window 比较候选 policy 的反事实 loss：

```text
Δ_i,t = loss(policy_i, token_t) - loss(full, token_t)
```

对每个候选 policy 记录：

```text
mean_delta_loss
max_loss_gap
positive_ratio
tail_risk
```

这不是直接预测全局最优 policy，而是提供 block-local counterfactual evidence。

### 3. Counterfactual risk memory

短 calibration window 会漂移，因此引入按 policy 累积的 risk memory：

```text
M_i <- historical counterfactual risk records of policy_i
```

memory prior 提供跨 block 稳定性，但 hard-topic 实验证明 memory anchor 也会过保守。

### 4. Horizon-aware rescue gate

Horizon gate 用 eval-prefix sentinel probe 估计候选 policy 对未来 horizon 的风险：

```text
L_i^h = horizon_loss(policy_i, next h tokens)
gain_i = L_memory^h - L_i^h
```

门控规则：

```text
select horizon best iff
    gain >= tau_gain
    and gain / max(margin, epsilon) >= tau_ratio
otherwise keep memory anchor
```

当前默认实验使用宽松门控：

```text
tau_gain = 0
tau_ratio = 0
```

用于验证 horizon evidence 的上界和稳定性。

### 5. Top-k cascade gate

为了降低 gate 成本，使用两阶段候选评估：

```text
Stage 1:
    all candidates, short horizon s=32

Stage 2:
    if confidence low, extend only top-k candidates plus memory/min-loss anchors to h=64
```

默认配置：

```text
sentinel_tokens = 64
sentinel_cascade_initial_tokens = 32
sentinel_cascade_accept_margin = 0.012
sentinel_cascade_extend_topk = 2
```

这可以写成“反事实候选评估预算分配”，比单纯调 sparse attention 规则更有创新性。

## 算法伪代码

```text
Algorithm: Horizon-PCIC

Input:
    candidate policy set P = {p_1, ..., p_K}
    block sequence B_1, ..., B_T
    calibration length c
    horizon length h
    cascade short horizon s < h

For each block B_t:
    1. Run full baseline on calibration tokens.
    2. For each policy p_i:
           run compressed policy on calibration tokens
           compute counterfactual risk features R_i,t
    3. Select:
           p_min = argmin calibration loss
           p_mem = argmin memory risk score among safe candidates
    4. Build candidate set C_t.
    5. Run short-horizon sentinel for all p_i in C_t.
    6. If short-horizon confidence is high:
           accept best short-horizon policy
       Else:
           extend top-k short-horizon policies + p_mem + p_min to long horizon h
           select horizon best if risk gate accepts, else p_mem
    7. Evaluate remaining block tokens using selected policy.
    8. Update counterfactual risk memory.
```

## 当前最强默认配置

```text
combo_select_policy = risk_memory_horizon_gate
risk_memory_seed_tokens = 64
risk_memory_loss_slack = 0.2
sentinel_tokens = 64
sentinel_cascade_initial_tokens = 32
sentinel_cascade_accept_margin = 0.012
sentinel_cascade_extend_topk = 2
horizon_gate_min_gain = 0.0
horizon_gate_min_ratio = 0.0
```

## 关键实验结果

自动生成表：

```text
docs/horizon_pcic_key_results_2026_06_29.md
docs/horizon_pcic_key_results_2026_06_29.csv
```

主结果：

| run | avg_delta_ppl | serial_total/base | batched_proxy/base | gate_s | combos |
| --- | ---: | ---: | ---: | ---: | --- |
| Hard-topic eval64 top2 | -0.076940 | 4.028 | 1.054 | 25.392 | `0,7;0,6;0,13;2,0,7,12` |
| Hard-topic eval128 top2 | 0.004371 | 2.648 | 1.039 | 27.085 | `0,7;2,0,7,12;0,6;0,13` |
| War top2 | -2.135311 | 2.596 | 1.039 | 6.544 | `0,7;0,7` |
| Monte top2 | -0.219215 | 2.596 | 1.054 | 6.573 | `2,7;2,0` |

Gate 成本下降：

| dataset | raw_s64_gate_s | top2_gate_s | reduction | quality_same |
| --- | ---: | ---: | ---: | --- |
| Hard-topic eval64 | 62.010 | 25.392 | 59.1% | True |
| Hard-topic eval128 | 62.204 | 27.085 | 56.5% | True |
| War | 13.120 | 6.544 | 50.1% | True |
| Monte | 13.098 | 6.573 | 49.8% | True |

## 必须保留的消融

### A. PCIC-CR / conffast_s8

证明旧版 risk memory + confidence gate 能降低 drift：

```text
hard-topic eval64:
    conffast_s8 = 0.003316
    horizon top2 = -0.076940
```

### B. raw s64 horizon oracle

证明 horizon evidence 本身有效：

```text
hard-topic eval64 raw_s64 = -0.076940
hard-topic eval128 raw_s64 = 0.004371
```

### C. s32 failure

证明短 horizon 会误导，cascade 是必要的：

```text
Monte s32 raw = 0.431207
Monte s64/top2 = -0.219215
```

### D. top-k cascade

证明 top2 可以保持 raw_s64 质量并显著降低 gate cost。

### E. batched proxy / functional prototype

证明系统速度有明确工程路径，但不能夸大当前结果。

## 当前不能声称的内容

不能写：

```text
1. batched gate 已经真实 wall-clock 加速。
2. 当前实现已经比 baseline 端到端更快。
3. 方法已经在标准 LongBench/RULER/Needle 上验证充分。
4. 这是一个新的 sparse attention primitive。
```

可以写：

```text
1. serial Horizon-PCIC 显著改善 PPL drift。
2. top-k cascade 保持 raw_s64 质量，并减少约 50%–59% 的串行 gate 成本。
3. batched_proxy 显示，如果 candidate probe 被 batch/fuse，端到端速度可接近 baseline。
4. functional batch-row prototype 已证明语义可行，但还不是最终 speed implementation。
```

## 下一步最高优先级

### 0. 复现入口

新增统一入口：

```text
scripts/run_horizon_pcic_core_suite.sh
```

默认安全模式：

```bash
RUN_EXPERIMENTS=0 RUN_SMOKE=0 HF_HUB_OFFLINE=1 bash scripts/run_horizon_pcic_core_suite.sh
```

默认行为：

```text
1. 检查核心源码、脚本、数据、模型路径；
2. py_compile 核心 Python 文件；
3. 不启动重实验；
4. 不启动 smoke；
5. 只从已有 outputs 重新生成 key result 表格。
```

如需补全缺失实验：

```bash
RUN_EXPERIMENTS=1 RUN_SMOKE=0 HF_HUB_OFFLINE=1 bash scripts/run_horizon_pcic_core_suite.sh
```

如需验证 batched gate smoke：

```bash
RUN_EXPERIMENTS=0 RUN_SMOKE=1 HF_HUB_OFFLINE=1 bash scripts/run_horizon_pcic_core_suite.sh
```

远端 dry-run 已验证：

```text
Horizon-PCIC core files OK
Generated:
  docs/horizon_pcic_key_results_2026_06_29.md
  docs/horizon_pcic_key_results_2026_06_29.csv
```

### 1. 系统实现

```text
实现 vectorized/fused candidate probe：
    - 避免复制完整 KV cache 到 candidate batch；
    - 对候选维度 C 做 fused loss probe；
    - 或至少为 landmark/recent budget 写专门 vectorized branch。
```

目标：

```text
top2 cascade real wall-clock/base <= 1.15
```

### 2. 标准 benchmark

最少补：

```text
LongBench small subset
RULER/Needle small subset
hard-topic stress
War/Monte long text sanity
```

### 3. 多模型验证

至少再补一个模型族：

```text
Llama 或 Mistral 小模型
```

### 4. 论文写作骨架

建议章节：

```text
1. Introduction
2. Related Work
3. Problem: Online Policy Selection for KV Cache Compression
4. Pairwise-CIC
5. Horizon-aware Cascade Rescue Gate
6. Adaptive Candidate Evaluation Budget
7. Experiments
8. Limitations
```

## 当前结论

```text
Horizon-PCIC 已经具备 paper 方法雏形：

创新性：
    从固定压缩规则转向在线反事实策略选择和预算分配。

效果：
    在 hard-topic / War / Monte 上保持或提升 PPL，并显著降低 gate 评估成本。

短板：
    系统速度还需要真正 fused/batched probe；
    标准 benchmark 和多模型验证还不够。
```
## 相关工作与创新性风险补充

- 详细分析见：`docs/horizon_pcic_related_work_novelty_2026_06_29.md`

## 下一组关键证据：fixed / online / oracle

- 实验入口：`scripts/run_pcic_fixed_online_oracle_suite.sh`
- 汇总脚本：`scripts/summarize_pcic_fixed_online_oracle.py`
- 汇总文档：`docs/pcic_fixed_online_oracle_2026_06_29.md`
- 目的：证明 Horizon-PCIC 不是一个 best fixed combo 可以替代的小改，而是能接近 blockwise oracle 的在线策略选择器。
- 服务器推荐命令：`HF_HUB_OFFLINE=1 RUN_EXPERIMENTS=1 bash scripts/run_pcic_fixed_online_oracle_suite.sh`

## Hard-topic eval128 delayed-rescue 补充

- 实验入口：`scripts/run_pcic_hardtopic_eval128_delayed_rescue.sh`
- 汇总文档：`docs/pcic_delayed_rescue_eval128_2026_06_29.md`
- 关键结果：`64→128 top2 + anchor06` 达到 blockwise oracle，平均 ΔPPL 从 `0.004371` 改善到 `-0.049633`。
- 成本结果：相对 `s128 all-candidate`，`method/base` 从 `8.291` 降到 `4.159`，gate 从 `123.331s` 降到 `52.291s`。
- 论文意义：短 horizon PCIC 会错过 delayed-win policy；horizon-anchor rescue gate 是必要组件。
- 当前风险：`anchor06` 仍是手写 anchor，下一步必须自动化为 validation prior / diversity anchor / uncertainty anchor。

## Validation-prior auto-anchor 补充

- Anchor 选择脚本：`scripts/select_pcic_validation_anchor.py`
- 自动 anchor 实验入口：`scripts/run_pcic_hardtopic_eval128_auto_anchor.sh`
- 汇总文档：`docs/pcic_auto_anchor_eval128_2026_06_29.md`
- validation prior：从 Hard-topic eval64 fixed-combo 结果按 `avg_delta_ppl` 排序，自动选出 `0,6`。
- eval128 结果：auto-anchor ΔPPL = `-0.049633`，达到 blockwise oracle。
- 成本结果：auto-anchor `method/base = 4.164`，gate `51.997s`，远低于 s128 all-candidate gate `123.331s`。
- 论文意义：`anchor06` 已从手写常量推进为 validation-prior anchor，主线可表述为 `Pairwise-CIC + online blockwise selection + validation-prior horizon-anchor rescue gate`。

## War / Monte auto-anchor 负面检查

- 实验入口：`scripts/run_pcic_warmonte_auto_anchor_check.sh`
- 汇总文档：`docs/pcic_warmonte_auto_anchor_check_2026_06_29.md`
- War auto-anchor 自动选出 `0,7`，ΔPPL 保持 `-2.135311`，gate 从 `6.544s` 增到 `8.863s`。
- Monte auto-anchor 自动选出 `2,0`，ΔPPL 保持 `-0.219215`，gate 从 `6.573s` 增到 `6.670s`。
- 结论：auto-anchor 不破坏质量，但不应无条件启用；下一版应做 conditional auto-anchor，只在短 horizon 不确定或与 prior 冲突时加入 anchor。

## Conditional auto-anchor 收敛版本

- 实验入口：`scripts/run_pcic_conditional_auto_anchor_suite.sh`
- 汇总文档：`docs/pcic_conditional_auto_anchor_2026_06_29.md`
- 当前阈值：`sentinel_cascade_accept_margin = 0.012`
- Hard-topic eval128：ΔPPL 保持 `-0.049633`，修复原 top2 的 `0.004371` 失败例。
- War：ΔPPL 保持 `-2.135311`，gate 从 unconditional auto-anchor 的 `8.863s` 降到 `6.659s`，接近 top2 baseline `6.544s`。
- Monte：ΔPPL 保持 `-0.219215`，gate `6.641s`，接近 top2 baseline `6.573s`。
- 当前 paper 主线建议固定为：`Pairwise-CIC + online blockwise selection + conditional validation-prior horizon-anchor rescue gate`。

## Conditional margin grid

- 实验入口：`scripts/run_pcic_conditional_auto_anchor_margin_grid.sh`
- 汇总文档：`docs/pcic_conditional_auto_anchor_margin_grid_2026_06_29.md`
- 网格：`0.008 / 0.010 / 0.012 / 0.015`
- 结论：`0.012` 是当前最均衡阈值；`0.008/0.010` 漏掉 Hard-topic delayed-win，`0.015` 在 War 上过度 extension。
- 下一步：把固定 `0.012` 发展成 adaptive margin，或先用 `0.012` 进入更长 blocks / 标准 benchmark smoke。

## Hard-topic b8 更长 block 验证

- 实验入口：`scripts/run_pcic_hardtopic_b8_conditional_auto_anchor.sh`
- 汇总文档：`docs/pcic_hardtopic_b8_condautoanchor_2026_06_29.md`
- b8 top2：平均 ΔPPL = `0.006074`，gate = `49.737s`。
- b8 conditional auto-anchor：平均 ΔPPL = `-0.040598`，gate = `107.643s`。
- 结论：conditional rescue 的质量收益在 8 blocks 上仍存在，不是 b4 偶然；但 gate 成本进一步证明需要 batched/fused probe。
- 下一步优先级：标准 benchmark smoke 与系统侧 probe batching 二选一；如果目标是 paper 创新性，先做 benchmark smoke；如果目标是速度 claim，先做 batching。

## Needle-style benchmark smoke

- 实验入口：`scripts/run_pcic_needle_smoke_condautoanchor.sh`
- 汇总文档：`docs/pcic_needle_smoke_condautoanchor_2026_06_29.md`
- 说明：服务器未发现现成 LongBench / RULER / Needle 数据；本实验为无下载 synthetic needle-style smoke，不等同正式标准 benchmark。
- validation-prior anchor 自动选出 `2,0`。
- needle top2：平均 ΔPPL = `0.000118`，gate = `13.206s`。
- needle conditional auto-anchor：平均 ΔPPL = `-0.000166`，gate = `39.909s`。
- 结论：在 easy retrieval / needle-style 场景，top2 已足够稳定，conditional rescue 质量略好但代价过高；这证明下一版需要 adaptive margin / stronger early-exit。

## Anchor-match early-exit 负面消融

- 实验入口：`scripts/run_pcic_anchor_match_early_exit_suite.sh`
- 汇总文档：`docs/pcic_anchor_match_early_exit_ablation_2026_06_29.md`
- 新参数：`--sentinel_cascade_anchor_accept_on_match`
- 结果：Hard-topic 质量保持但 gate `53.648s -> 62.742s`；Needle extension 减少但 gate `39.909s -> 59.459s`，质量略退。
- 结论：该 heuristic 不进入主方法，只保留负面消融。
- 当前默认：`sentinel_cascade_accept_margin = 0.012`，`sentinel_cascade_anchor_accept_on_match = false`。

## Low-spread early-exit 负面消融

- 实验入口：`scripts/run_pcic_low_spread_early_exit_suite.sh`
- 汇总文档：`docs/pcic_low_spread_early_exit_ablation_2026_06_29.md`
- 新参数：`--sentinel_cascade_accept_low_spread`
- 结果：Needle extension 从 `4` 降到 `0`，但 gate `39.909s -> 61.147s`，质量从 `-0.000166` 轻微退到 `0.000003`。
- 结论：减少 extension 次数没有转化为 wall-clock 收益；该 heuristic 不进入主方法。
- 当前默认：`sentinel_cascade_accept_low_spread = 0.0`。

## 当前速度方向判断

- 主方法仍固定为 `Pairwise-CIC + online blockwise selection + conditional validation-prior horizon-anchor rescue gate`。
- 推荐配置仍为 `sentinel_cascade_accept_margin = 0.012`，`sentinel_cascade_anchor_accept_on_match = false`，`sentinel_cascade_accept_low_spread = 0.0`。
- 两个 early-exit heuristic 负消融说明，当前瓶颈更可能在串行 sentinel probe / Python overhead / 未 fused 的候选前缀计算。
- 下一步若要支撑 speed claim，应优先做 batched/fused sentinel probe；若要支撑 paper quality claim，应优先补 LongBench / RULER 标准 benchmark。

## Conditional auto-anchor batched gate 主线对比

- 实验入口：`scripts/run_pcic_condautoanchor_batched_gate_suite.sh`
- 汇总文档：`docs/pcic_condautoanchor_batched_gate_2026_06_29.md`
- 代码修复：`eval_segment_batched_candidates` 现在支持 extension 阶段传入已经 batched 的 KV cache / logits，避免二次 repeat batch。
- 语义结果：Hard / War / Monte 的 batched combos 与 serial 完全一致。
- 质量结果：Hard `-0.049633 -> -0.049778`，War `-2.135311 -> -2.133787`，Monte `-0.219215 -> -0.251805`。
- 速度结果：当前 eager batch-row 更慢，Hard gate `53.648s -> 79.524s`，War `6.659s -> 8.689s`，Monte `6.641s -> 8.466s`。
- 结论：batch-row 表达和主流程接入已经成立，但不是最终速度实现；paper 只能说它证明 fused candidate probe 的工程路径，而不能说当前 batched gate 已加速。

## 更新后的速度方向判断

- 不再继续做简单 early-exit heuristic。
- 不再把候选简单堆到 batch 维作为 speed claim。
- 下一步速度实现必须减少候选维 KV/cache 复制和 Python/eager mask 开销：`candidate-level fused probe` 或 `tensorized grouped sparse attention`。
- 论文写法应分开：算法质量由 serial/eager Horizon-PCIC 证明；系统潜力由 batch-row semantic prototype + fused-probe design/prototype 证明。

## Batch-row dispatch 优化补充

- 实验入口：`scripts/run_pcic_condautoanchor_batched_gate_optdispatch_suite.sh`
- 汇总文档：`docs/pcic_condautoanchor_batched_gate_optdispatch_2026_06_29.md`
- 代码优化：当一层所有 batch rows 使用同一 budget 时跳过 `index_select` / `index_copy_`，并缓存 batch-row index tensor。
- Hard：batched gate `79.524s -> 61.139s`，仍高于 serial `53.648s`。
- War：batched gate `8.689s -> 7.462s`，仍高于 serial `6.659s`。
- Monte：batched gate `8.466s -> 7.467s`，仍高于 serial `6.641s`。
- 语义结果：optimized batched 的 combos 与 serial 完全一致。
- 结论：dispatch 优化有效，但速度瓶颈已经暴露为候选维 cache 复制和未 fused sparse attention；下一步必须做 fused/tensorized probe。

## Batch-row budget group 分析

- 分析入口：`scripts/analyze_pcic_batch_row_budget_groups.py`
- 汇总文档：`docs/pcic_batch_row_budget_groups_2026_06_29.md`
- 下一步设计：`docs/pcic_mixed_layer_fused_probe_design_2026_06_29.md`
- Hard optdispatch：all-same layer fraction = `0.817`，avg groups/layer = `1.183`。
- War optdispatch：all-same layer fraction = `0.857`，avg groups/layer = `1.143`。
- Monte optdispatch：all-same layer fraction = `0.857`，avg groups/layer = `1.143`。
- 结论：batch-row gate 中绝大多数层已经不需要 row-wise 分派；系统瓶颈集中在少数 mixed-budget layers。
- 下一步 kernel 目标应收窄为 mixed layers 的 fused row-wise full/landmark attention，而不是全模型 attention 重写。

## Mixed dense Stage 1 结果

- 实验入口：`scripts/run_pcic_condautoanchor_batched_gate_mixeddense_suite.sh`
- 汇总文档：`docs/pcic_condautoanchor_batched_gate_mixeddense_2026_06_29.md`
- 开关：`LAYER_BUDGET_MIXED_DENSE=1`
- Hard：optdispatch gate `61.139s`，mixeddense gate `61.700s`，serial `53.648s`。
- War：optdispatch gate `7.462s`，mixeddense gate `7.507s`，serial `6.659s`。
- Monte：optdispatch gate `7.467s`，mixeddense gate `7.217s`，serial `6.641s`。
- 语义结果：combos 与 serial 完全一致。
- 结论：dense mixed path 语义正确，但速度收益不稳定；默认不启用。下一步应直接做 sparse gather / fused candidate probe。

## RULER-style offline smoke

- 实验入口：`scripts/run_pcic_ruler_style_smoke.sh`
- 汇总文档：`docs/pcic_ruler_style_smoke_2026_06_29.md`
- validation prior：`docs/pcic_ruler_style_validation_prior_2026_06_29.md`
- 说明：该实验不下载外部 benchmark，只生成 multi-needle / variable binding / topic switch 三类长上下文文本；它不是正式 RULER / LongBench。
- validation-prior anchor 自动选出 `2,0`。
- Multi-needle：top2 与 conditional 都是 `0.000074`，质量持平但 conditional 更慢。
- Variable binding：top2 `0.017397`，conditional `-0.000564`，conditional 明显修复 PPL drift。
- Topic switch：top2 `0.000302`，conditional `0.000139`，conditional 小幅改善。
- 结论：conditional validation-prior rescue 在多个 synthetic 长上下文模式上没有破坏质量，并在 variable/topic-switch 场景带来改善；但 gate 成本仍是主要问题。

## Cascade extension waste 后验分析

- 分析入口：`scripts/analyze_pcic_extension_waste.py`
- 汇总文档：`docs/pcic_extension_waste_2026_06_29.md`
- skip-gate 设计：`docs/pcic_calibrated_skip_gate_design_2026_06_29.md`
- 说明：该分析只读已有 CSV，不跑模型；它是 skip-gate 的后验上界，不是当前主方法。

| case | extended blocks | no-change ext | avoidable seconds | avoidable fraction |
| --- | ---: | ---: | ---: | ---: |
| hard | 4 | 2 | 15.622 | 0.437 |
| monte | 2 | 1 | 2.211 | 0.400 |
| needle | 4 | 2 | 13.314 | 0.462 |
| ruler multi-needle | 3 | 1 | 4.972 | 0.375 |
| ruler topic-switch | 3 | 2 | 6.620 | 0.669 |
| ruler variable | 3 | 1 | 4.950 | 0.429 |

结论：大量 extension 最终没有改变 combo，说明仍有 `37%–67%` 的 extension 秒数后验可省。下一步速度方法应从 fused kernel 与 learnable/calibrated skip-gate 两条线并行推进。

## Skip-gate rule search

- 搜索入口：`scripts/search_pcic_skip_gate_rules.py`
- 汇总文档：`docs/pcic_skip_gate_rule_search_2026_06_29.md`
- 当前最简洁 zero false-skip 规则：

```text
if initial_selected_combo in validation_prior_anchors
and sentinel_horizon_gain_ratio <= 0:
    skip extension
```

- 后验效果：选择 6 个 block，节省 extension `29.856s`，占总 extension 秒数 `28.5%`，当前样本 false skip 为 0。
- 覆盖场景：Needle / RULER-style easy-regime，不覆盖 Hard-topic delayed-win blocks。
- 结论：这是可接入 runner 的保守规则候选；必须实测在线 ΔPPL / gate_s 后才能进入主方法。

## Skip-gate online 负面消融

- 实验入口：`scripts/run_pcic_skip_gate_anchor_gain_suite.sh`
- 汇总文档：`docs/pcic_skip_gate_anchor_gain_online_2026_06_29.md`
- corrected gate 文档：`docs/pcic_corrected_gate_skipanchor_gain_2026_06_29.md`
- 参数：`--sentinel_cascade_skip_anchor_nonpositive_gain true`
- 口径修正：旧 `gate_s` 漏算 extended cascade 中未扩展候选的 initial probe，因此 extended base 的 gate 被低估；后续 speed claim 应使用 corrected gate。
- Hard：质量不变，corrected gate `89.624s -> 88.033s`，但没有触发 skip。
- Needle：ΔPPL 轻微退化 `+0.000137`，corrected gate `81.705s -> 67.423s`，skipped `3`。
- RULER multi-needle：质量持平，corrected gate `42.868s -> 39.222s`，skipped `1`。
- RULER variable：ΔPPL 轻微退化 `+0.000026`，corrected gate `41.103s -> 35.593s`，skipped `2`。
- RULER topic-switch：ΔPPL 轻微退化 `+0.000163`，corrected gate `39.543s -> 34.918s`，skipped `3`。
- 结论：该规则能省 corrected probe 成本，但质量/选择不够稳，不进入主方法；skip-gate 方向保留，下一版必须做更强校准。
