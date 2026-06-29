# Horizon-PCIC ICML Readiness Checklist（2026-06-29）

> Corrected 主结果表见：`docs/horizon_pcic_corrected_key_results_2026_06_29.md`。后续 speed/gate claim 以 corrected gate 口径为准。

> Fixed / online / rescue / oracle 主线证据见：`docs/pcic_mainline_fixed_online_rescue_2026_06_29.md`。该表用于判断当前创新性是否能站住。

> Blockwise policy trace 证据见：`docs/pcic_blockwise_policy_trace_2026_06_29.md`。该表用于判断 online selection 是否有非平凡动态性。

> Method spec / 论文算法定义见：`docs/horizon_pcic_method_spec_2026_06_29.md`。该文档用于检查方法贡献是否已形成可投稿叙述。

> Component evidence matrix 见：`docs/pcic_component_evidence_matrix_2026_06_29.md`。该表用于判断哪些 claim 可以写，哪些还必须补实验。

> Minimal component ablation suite 见：`docs/pcic_minimal_component_ablation_2026_06_29.md`。该表是后续补 `no-rescue / no-memory / no-pairwise` 的执行入口。

> Minimal component ablation runbook 见：`docs/pcic_minimal_component_ablation_runbook_2026_06_29.md`。该文档给出离线运行、判据和同步边界。

> Paper readiness gate 见：`docs/pcic_paper_readiness_gate_2026_06_29.md`。该表自动汇总当前 pass / missing gate，用于判断是否接近 ICML-ready。

> Paper skeleton 见：`docs/horizon_pcic_paper_skeleton_2026_06_29.md`。该文档把当前证据转成可写论文结构。

## 结论先行

当前状态：

```text
方法主线已经成型，创新点足够继续推进成 paper；
但还没有达到 ICML 完整投稿证据标准。
```

可以作为 paper 主线继续投入：

```text
Pairwise-CIC + online blockwise policy selection + horizon-aware top-k cascade rescue gate
```

暂时不应声称：

```text
1. 端到端速度已经超过 baseline；
2. 已在标准长上下文 benchmark 上充分验证；
3. batched gate 已经真实 wall-clock 加速；
4. 方法已经具备最终 ICML 级实验完整性。
```

## Readiness 总览

| 维度 | 当前状态 | 证据 | 风险 |
| --- | --- | --- | --- |
| 方法创新性 | 较强 | 从固定压缩规则转向在线反事实策略选择与预算分配 | 需要更清楚区分与 AdaKV/QUEST/SparQ 的关系 |
| 算法实现 | 已有可跑原型 | `risk_memory_horizon_gate`、top-k cascade、risk memory | 代码仍是研究原型 |
| 质量结果 | 有积极证据 | hard-topic / War / Monte 上 PPL drift 改善 | benchmark 覆盖不足 |
| gate 成本 | 有明显下降 | top2 cascade 降低 serial gate 约 50%–59% | serial_total 仍慢 |
| 系统速度 | 只有 proxy + functional prototype | `batched_proxy/base≈1.04–1.05`，batch-row smoke 跑通 | 真实 wall-clock 尚未加速 |
| 复现链路 | 初步完整 | `run_horizon_pcic_core_suite.sh` | 还缺 clean README 与环境说明 |
| 投稿成熟度 | 约 55%–65% | 方法、实验、文档都有雏形 | 标准 benchmark / 多模型 / 真速度仍缺 |

## 已满足的关键证据

### 1. 方法不是简单 SparQ 变体

当前方法核心不是每 token query-aware top-k retrieval，而是：

```text
policy-level online selection
counterfactual calibration
risk memory
horizon-aware rescue arbitration
adaptive candidate-evaluation budget allocation
```

这与 SparQ/H2O/SnapKV/QUEST 的主要区别：

```text
SparQ/QUEST:
    如何近似注意力或选择 KV token；

Horizon-PCIC:
    在多个候选 KV compression policies 之间，如何在线选择、验证和仲裁。
```

### 2. s32 failure 支持 cascade 必要性

Monte 上短 horizon 会误导：

```text
Monte s32 raw: 0.431207
Monte s64/top2: -0.219215
```

这证明 cascade 不是工程小技巧，而是有实际失败模式支撑：

```text
short horizon can be myopic
```

### 3. top-k cascade 保持质量并降低 gate 成本

关键表：

| dataset | raw_s64_gate_s | top2_gate_s | reduction | quality_same |
| --- | ---: | ---: | ---: | --- |
| Hard-topic eval64 | 62.010 | 25.392 | 59.1% | True |
| Hard-topic eval128 | 62.204 | 27.085 | 56.5% | True |
| War | 13.120 | 6.544 | 50.1% | True |
| Monte | 13.098 | 6.573 | 49.8% | True |

这支撑论文中的 adaptive compute claim：

```text
all-candidate short horizon + top-k long horizon + memory/min-loss anchors
```

### 4. Batched proxy 显示系统方向有潜力

当前 top2 cascade：

| run | serial_total/base | batched_proxy/base |
| --- | ---: | ---: |
| Hard-topic eval64 top2 | 4.028 | 1.054 |
| Hard-topic eval128 top2 | 2.648 | 1.039 |
| War top2 | 2.596 | 1.039 |
| Monte top2 | 2.596 | 1.054 |

解释：

```text
如果 candidate probe 能 batch/fuse，速度可以接近 baseline；
但这目前是 proxy，不是真实 wall-clock。
```

### 5. Functional batched gate prototype 已跑通

已完成：

```text
batch_maps JSON
batch-row layerbudgetattn
eval_segment_batched_candidates
--sentinel_batched_candidates
```

已验证：

```text
batched candidate smoke:
    serial loss == batched loss
    MAX_DIFF=0
```

War batched gate smoke：

```text
serial_top2:
    avg_delta_ppl = -2.135311
    gate_s = 6.544

batched_top2 grouped:
    avg_delta_ppl = -2.133787
    gate_s = 8.708
```

结论：

```text
语义路径可行；
但当前 grouped prototype 仍慢于 serial，不能声称速度完成。
```

Conditional auto-anchor 主线 batched gate 对比：

```text
hard:
    combos unchanged
    gate_s = 53.648 -> 79.524

war:
    combos unchanged
    gate_s = 6.659 -> 8.689

monte:
    combos unchanged
    gate_s = 6.641 -> 8.466
```

这说明 batch-row budget map 已经可以接入完整主方法并保持选择语义；但当前 eager batch-row path 因 KV/cache batch 复制和 Python/mask 开销仍更慢。ICML 速度 claim 不能停在 batch-row，需要 fused candidate probe 或 tensorized grouped sparse attention。

Dispatch 优化后：

```text
hard:
    gate_s = 79.524 -> 61.139
    serial gate_s = 53.648

war:
    gate_s = 8.689 -> 7.462
    serial gate_s = 6.659

monte:
    gate_s = 8.466 -> 7.467
    serial gate_s = 6.641
```

该结果把系统证据推进了一步：batch-row 的一部分 overhead 可以被工程优化消除，但真实速度瓶颈仍在候选维 cache 复制和 sparse attention 未 fused。

Batch-row budget group 分析：

```text
hard optdispatch:
    all-same layer fraction = 0.817
    avg groups/layer = 1.183

war optdispatch:
    all-same layer fraction = 0.857
    avg groups/layer = 1.143

monte optdispatch:
    all-same layer fraction = 0.857
    avg groups/layer = 1.143
```

含义：大多数层已经可以整批 forward，真正需要系统创新的是少数 mixed-budget layers 的 row-wise full/landmark fused attention。

对应实现设计见：`docs/pcic_mixed_layer_fused_probe_design_2026_06_29.md`

Mixed dense Stage 1 实测：

```text
hard:
    optdispatch gate_s = 61.139
    mixeddense gate_s = 61.700

war:
    optdispatch gate_s = 7.462
    mixeddense gate_s = 7.507

monte:
    optdispatch gate_s = 7.467
    mixeddense gate_s = 7.217
```

该结果说明：dense tensorization 保持语义，但不是稳定 speed path；下一步应做 sparse gather / fused mixed-budget kernel。

Cascade extension waste 后验分析：

```text
hard: avoidable extension fraction = 0.437
monte: avoidable extension fraction = 0.400
needle: avoidable extension fraction = 0.462
ruler multi-needle: avoidable extension fraction = 0.375
ruler topic-switch: avoidable extension fraction = 0.669
ruler variable: avoidable extension fraction = 0.429
```

这不是在线规则，只是上界；但它说明速度路线除了 fused kernel，还应并行设计 calibrated skip-gate。

Zero false-skip rule search：

```text
rule:
    anchor_hit and sentinel_horizon_gain_ratio <= 0

posterior result:
    selected blocks = 6
    saved extension seconds = 29.856
    saved extension fraction = 0.285
    false skip = 0
```

该规则目前只是后验候选；ICML 级证据需要把它接入在线 runner，并在 hard / RULER-style / Needle 上复测质量和 gate。

在线复测结果：

```text
hard:
    skipped = 0
    corrected gate_s change = -1.591
    ΔPPL change = 0

needle:
    skipped = 3
    corrected gate_s change = -14.282
    ΔPPL change = +0.000137

ruler variable:
    skipped = 2
    corrected gate_s change = -5.510
    ΔPPL change = +0.000026
```

结论：修正 gate 口径后，该单规则 skip-gate 确实能省 probe 成本，但质量/选择不够稳，不进入主方法。Skip-gate 仍可作为未来方向，但当前速度证据仍主要依赖 fused/sparse probe。

## 仍缺的 ICML 级证据

### A. 标准 benchmark

必须补：

```text
LongBench small subset
RULER / Needle small subset
至少一个 QA / retrieval / summarization 类型任务
```

当前 War/Monte/hard-topic 只能证明机制，不足以支撑广泛有效性。

已补一个无下载 synthetic RULER-style smoke：

```text
multi-needle:
    top2 ΔPPL = 0.000074
    conditional ΔPPL = 0.000074

variable binding:
    top2 ΔPPL = 0.017397
    conditional ΔPPL = -0.000564

topic switch:
    top2 ΔPPL = 0.000302
    conditional ΔPPL = 0.000139
```

这说明 conditional rescue 在多个 synthetic 长上下文模式上至少不破坏质量，并能修复 variable binding 的 top2 failure。但它仍不是正式 RULER / LongBench，不能替代标准 benchmark。

### B. 多模型

当前主要是：

```text
Qwen3-0.6B
```

建议至少补：

```text
Llama-family small model
或 Mistral-family small model
```

### C. 真实系统速度

当前不能过关：

```text
serial_total/base 仍是 2.6x–4.0x；
grouped batched gate 仍慢于 serial；
conditional auto-anchor batched gate 仍慢于 serial；
dispatch-optimized batched gate 仍慢于 serial；
真实 wall-clock 还没有低于或接近 baseline。
```

必须推进：

```text
mixed-budget-layer fused candidate probe
或专门 sparse-gather row-wise full/landmark branch
以及 calibrated extension skip-gate
```

### D. 更严格消融

需要补齐：

```text
min-loss only
risk-memory only
horizon gate no cascade
s32 only
s64 raw
top1/top2/top3 cascade
without memory anchor
without min-loss anchor
```

### E. 统计稳定性

当前 block 数偏少：

```text
War/Monte: 2 blocks
Hard-topic: 4 blocks
```

需要：

```text
更多 offsets
更多 blocks
mean / worst / std
```

## Paper claim 分级

### 现在可以写

```text
1. We formulate KV cache compression as online blockwise policy selection.
2. We propose Pairwise-CIC for counterfactual policy comparison.
3. We introduce risk memory to stabilize local calibration.
4. We propose horizon-aware top-k cascade rescue gate.
5. In controlled text stress tests, top2 cascade preserves raw s64 quality while reducing serial gate cost by about 50%–59%.
6. A functional batch-row prototype validates the feasibility of batched candidate gates.
```

### 现在不能写

```text
1. Our method is faster than baseline end-to-end.
2. Our batched implementation achieves the proxy speed.
3. Our method outperforms all KV compression baselines on LongBench/RULER.
4. Horizon-PCIC is a new attention kernel.
```

### 目标 claim

最终希望写：

```text
Horizon-PCIC improves quality under aggressive KV compression while keeping gate overhead small through adaptive top-k horizon evaluation, and with fused candidate probing it approaches baseline wall-clock cost.
```

## 下一步执行优先级

### P0：补标准 benchmark 最小闭环

脚本目标：

```text
scripts/run_horizon_pcic_longbench_smoke.sh
```

原则：

```text
不下载大数据；
优先找服务器本地已有 benchmark；
没有则用本地 synthetic QA / needle-like text 先做可控验证。
```

### P1：系统速度

目标：

```text
实现 vectorized landmark/recent grouped branch
```

成功标准：

```text
War top2 batched gate_s <= serial gate_s
```

### P2：消融矩阵自动化

目标：

```text
scripts/summarize_horizon_pcic_ablation_matrix.py
```

输出：

```text
quality table
gate-cost table
claim-safe table
```

### P3：论文初稿

先写：

```text
method + experimental setup + limitations
```

不要先写强 claim 的 abstract。

## 当前总评

```text
Horizon-PCIC 值得继续沿 paper 主线投入。

创新性：
    足够形成投稿方向；
    但需要 benchmark 和相关工作边界加强。

效果：
    controlled experiments 很有希望；
    但还不是完整 benchmark evidence。

速度：
    proxy 有希望；
    真实实现还没达标。

建议：
    继续推进，不要换方向；
    下一阶段重点从“想方法”转到“补 benchmark + fused/batched gate”。
```
