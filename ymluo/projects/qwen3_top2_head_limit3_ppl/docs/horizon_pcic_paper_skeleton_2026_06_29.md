# Horizon-PCIC Paper Skeleton（2026-06-29）

本文档把当前方法和证据整理成论文初稿骨架。原则：只写已有证据能支撑的 claim；速度和正式 benchmark 作为待补证据或 limitation。

## 0. Working Title

```text
Horizon-PCIC: Online Counterfactual Policy Selection for Long-Context KV Cache Compression
```

备选标题：

```text
Horizon-PCIC: Horizon-Aware Pairwise Counterfactual Selection of Sparse Attention Policies
```

不建议标题包含：

```text
qabs / top2 / landmark / head limit / SparQ
```

这些容易让审稿人把方法理解成固定 sparse operator 的小改。

## 1. Conservative Abstract Draft

```text
Efficient long-context inference often relies on hand-designed sparse attention or KV-cache compression rules. 
However, a fixed compression rule can be brittle across non-stationary contexts: the policy that is safe for one block may fail on a later block, while short calibration windows can be myopic.

We propose Horizon-PCIC, an online counterfactual policy-selection framework for KV-cache compression. 
Instead of committing to a single sparse attention operator, Horizon-PCIC compares a set of candidate compression policies using pairwise counterfactual loss, maintains a policy-level risk memory across blocks, and invokes a horizon-aware rescue gate when local calibration and historical priors disagree.

On controlled long-context evaluations, Horizon-PCIC improves over the best fixed policy and repairs short-horizon selection failures; on hard-topic and RULER-style variable-binding settings, conditional horizon rescue closes the gap to a blockwise oracle in key cases. 
Our analysis shows that the method provides a distinct policy-selection layer that can in principle sit above existing sparse attention operators.

Current results also reveal a systems limitation: unfused candidate probing remains more expensive than baseline inference. 
We therefore present Horizon-PCIC as an algorithmic framework for robust online policy selection, and identify fused candidate probing as the next step toward end-to-end speedups.
```

注意：

- 这里没有声称 `faster than baseline`。
- 这里没有声称正式 LongBench/RULER 已完成。
- 这里把贡献放在 `online policy-selection framework`，不是 fixed sparse pattern。

## 2. Introduction Structure

### Paragraph 1: Long-context compression is necessary but brittle

要点：

- KV cache / attention cost grows with context length；
- sparse attention / KV compression 是主流方向；
- 现有方法通常设计固定或半固定 selection rule。

可引用方向：

```text
SparQ / Quest / H2O / SnapKV / PyramidKV / Ada-KV / MInference
```

### Paragraph 2: Fixed policies fail under non-stationarity

要点：

- 不同 block 的信息需求不同；
- short calibration 可能短视；
- best fixed combo 在 Monte 上明显不如 online selection；
- Hard-topic eval128 中 top2 short-horizon selection 与 oracle 有 gap。

证据：

```text
docs/pcic_mainline_fixed_online_rescue_2026_06_29.md
```

### Paragraph 3: Our reframing

核心句：

```text
We reframe KV-cache compression as online counterfactual policy selection under a bounded probe budget.
```

解释：

- candidate policies 是底层 sparse operators；
- PCIC 是上层 selector；
- rescue gate 是防止 short-horizon myopia 的仲裁器。

### Paragraph 4: Contributions

建议写三条：

1. **Pairwise-CIC**：用 pairwise counterfactual loss 比较 candidate compression policies。
2. **Online blockwise selection with risk memory**：把 long-context compression 建模为 non-stationary blockwise decision problem。
3. **Conditional horizon rescue gate**：在 bounded probe budget 下修复 short-horizon failure。

保守补一句：

```text
We also provide corrected gate accounting and identify candidate-probe fusion as the main systems bottleneck.
```

## 3. Method Section

主引用：

```text
docs/horizon_pcic_method_spec_2026_06_29.md
```

建议结构：

### 3.1 Candidate Compression Policies

写法：

- 定义 `P = {p_1, ..., p_K}`；
- 每个 `p_i` 是一种 KV/attention compression policy；
- 本文实验使用 qabs / landmark / full+landmark 等候选；
- 框架不绑定具体候选。

### 3.2 Pairwise Counterfactual Influence Calibration

写法：

```text
d_i,t = L_i(C_t) - L_full(C_t)
A_i,j,t = L_j(C_t) - L_i(C_t)
```

强调：

- 不是 token importance ranking；
- 是 policy-level counterfactual comparison。

### 3.3 Risk Memory

写法：

- 记录每个 policy 的 historical risk；
- 给出 `M_i,t`；
- 解释为什么短窗口需要 memory prior。

注意：

```text
严格 no-memory 消融还没跑完，因此正文先写为 design motivation；
必要性 claim 等消融后再加强。
```

### 3.4 Conditional Horizon Rescue Gate

写法：

- short horizon 先筛；
- low confidence / anchor conflict 时扩展 top-k；
- validation-prior anchor 只作为 rescue candidate，不是固定策略。

默认配置：

```text
sentinel_cascade_accept_margin = 0.012
extend_topk = 2
anchor_accept_on_match = false
low_spread_early_exit = false
skip_anchor_nonpositive_gain = false
```

### 3.5 Algorithm

直接使用：

```text
docs/horizon_pcic_method_spec_2026_06_29.md
Algorithm 1: Horizon-PCIC
```

## 4. Experiments Section

### 4.1 Setup

当前可写：

- model: Qwen3-0.6B；
- datasets/smoke: hard-topic, War, Monte, Needle-style, RULER-style synthetic；
- metrics: ΔPPL, selected method/base, corrected gate_s, combo trace；
- no external downloads for synthetic/offline smoke。

必须标注：

```text
RULER-style smoke is not a substitute for formal RULER/LongBench.
```

### 4.2 Fixed vs Online vs Oracle

主表：

```text
docs/pcic_mainline_fixed_online_rescue_2026_06_29.md
```

核心结果：

- Hard-topic eval128：conditional rescue reaches oracle；
- Monte：online / conditional 明显优于 best fixed；
- War：easy regime，fixed=online=oracle。

论文解释：

```text
The benefit of online selection appears when context is non-stationary; easy regimes may not require it.
```

### 4.3 Blockwise Policy Trace

主表：

```text
docs/pcic_blockwise_policy_trace_2026_06_29.md
```

可做 Figure 1：

```text
x-axis: block id
y-axis: selected policy combo
color: policy family
markers: rescue extension / early accept
```

要展示：

- Hard-topic b8 有多次 switch；
- RULER variable/topic 从 `0,13` 切到 `2,0`；
- War 是固定策略的 easy case。

### 4.4 Component Ablations

当前状态：

```text
docs/pcic_minimal_component_ablation_2026_06_29.md
```

已具备：

- historical fallback 填充 top2/main；
- strict 消融 suite 已准备。

待跑：

```text
memory_only_no_rescue
no_history_memory
no_pairwise_probe
```

论文中最终需要一张表：

| task | main | memory-only | no-memory | no-pairwise | conclusion |
| --- | ---: | ---: | ---: | ---: | --- |

### 4.5 Corrected Cost Accounting

主表：

```text
docs/horizon_pcic_corrected_key_results_2026_06_29.md
```

可写：

- corrected gate 口径更保守；
- conditional rescue 成本比旧 gate 口径高；
- 当前 unfused implementation 不应 claim speedup。

不可写：

```text
our method is faster than baseline
```

可写成 limitation / future system work。

## 5. Related Work Positioning

主引用：

```text
docs/horizon_pcic_related_work_novelty_2026_06_29.md
```

建议分类：

### Sparse attention / query-aware retrieval

- SparQ；
- Quest；
- Loki；
- MInference。

区别：

```text
They propose sparse operators or retrieval mechanisms; Horizon-PCIC selects among candidate policies online.
```

### KV-cache compression

- H2O；
- SnapKV；
- PyramidKV；
- Ada-KV；
- Scissorhands / Keyformer。

区别：

```text
They focus on token/head/cache retention; Horizon-PCIC focuses on policy-level counterfactual arbitration.
```

### Routing / adaptive inference

可以对齐：

- online model/router selection；
- bandit-like policy selection；
- uncertainty-triggered computation。

注意：

不要过度声称理论 bandit regret；当前没有理论证明。

## 6. Limitations

必须坦诚写：

1. **Formal benchmark missing**：当前 RULER-style 是 synthetic smoke，不是正式 RULER/LongBench。
2. **Speed not solved**：corrected gate 后候选 probe 成本仍高。
3. **Ablation not complete**：strict no-memory/no-pairwise/no-rescue 需要补齐。
4. **Single main model**：当前主要是 Qwen3-0.6B。
5. **Candidate set dependence**：selector 的收益依赖候选 policy 集合质量。

## 7. Figure / Table Plan

| id | content | source | status |
| --- | --- | --- | --- |
| Figure 1 | Horizon-PCIC overview | method spec | ready to draw |
| Figure 2 | blockwise policy trace | `pcic_blockwise_policy_trace` | ready, needs visualization |
| Table 1 | fixed vs online vs oracle | `pcic_mainline_fixed_online_rescue` | ready |
| Table 2 | corrected key results | `horizon_pcic_corrected_key_results` | ready |
| Table 3 | component ablation | `pcic_minimal_component_ablation` | missing strict runs |
| Table 4 | related work comparison | `horizon_pcic_related_work_novelty` | ready as text |
| Table 5 | formal benchmark | not available | missing |

## 8. Current Readiness

自动检查：

```text
docs/pcic_paper_readiness_gate_2026_06_29.md
```

当前状态：

```text
pass = 6
missing = 3
```

缺口：

```text
strict component ablation
formal benchmark
real speed / fused probe
```

## 9. Next Writing Step

优先顺序：

1. 跑 P0 strict ablation：

```bash
ONLY_CASES="hard_memoryonly hard_nohistory hard_nopairwise" \
RUN_EXPERIMENTS=1 \
bash scripts/run_pcic_minimal_component_ablation_suite.sh
```

2. 如果 P0 支持主线，写论文 Introduction + Method 初稿。
3. 如果 P0 不支持主线，收缩 claim：

```text
Pairwise-CIC / memory 是辅助稳定组件；
核心贡献保留为 online horizon-aware rescue selection。
```

## 10. Bottom Line

当前主线值得继续：

```text
Pairwise-CIC + online blockwise selection + rescue gate
```

创新性足够形成 paper 方向，但还不是完整 ICML-ready。最关键的下一步不是再想新方法，而是补齐 strict ablation 与 formal benchmark，使贡献从“合理且有证据”变成“审稿人难以归类为工程拼装”。
