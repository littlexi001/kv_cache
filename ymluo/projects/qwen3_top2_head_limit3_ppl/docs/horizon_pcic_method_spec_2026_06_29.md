# Horizon-PCIC Method Spec（2026-06-29）

本文档把当前 paper 主线整理成可以直接写进论文 Method 章节的形式：

```text
Pairwise-CIC + online blockwise policy selection + conditional horizon rescue gate
```

核心定位：

> Horizon-PCIC 不是提出一个新的固定 sparse attention pattern，而是提出一个在线策略选择器：给定多个 KV/attention compression policies，在每个 block 上用反事实 loss、历史 risk memory 和未来 horizon probe 选择当前最可靠的 policy。

## 1. 问题定义

给定长上下文序列，按 block 切分：

```text
X = [B_1, B_2, ..., B_T]
```

给定候选压缩策略集合：

```text
P = {p_1, p_2, ..., p_K}
```

每个 `p_i` 可以是任意训练无关的 KV/attention policy，例如：

- qabs / SparQ-like channel selection；
- landmark / block representative；
- recent / sink / retrieval-style policy；
- full+landmark / mixed layer budget；
- future work 中也可以接 Quest/Loki/MInference-like candidate。

Horizon-PCIC 的目标不是学习一个固定 `p_i`，而是在每个 block `B_t` 在线选择：

```text
a_t = select(P, history, calibration_t, horizon_t)
```

使得压缩后的 loss drift 小，同时把候选 probe 成本控制在可接受范围内。

## 2. Pairwise-CIC：反事实策略校准

对 block `B_t` 的 calibration window `C_t`，先计算 full attention baseline loss：

```text
L_full(C_t)
```

再对每个候选 policy `p_i` 计算压缩 loss：

```text
L_i(C_t) = L(model under policy p_i on C_t)
```

定义 counterfactual drift：

```text
d_i,t = L_i(C_t) - L_full(C_t)
```

Pairwise-CIC 不只看单个 policy 的绝对 drift，还看候选之间的 pairwise advantage：

```text
A_i,j,t = L_j(C_t) - L_i(C_t)
```

若 `A_i,j,t > 0`，说明在当前 calibration window 上 `p_i` 优于 `p_j`。

实际实现中可记录以下风险特征：

```text
mean_delta_loss_i,t
max_loss_gap_i,t
positive_ratio_i,t
tail_risk_i,t
pairwise_margin_i,t
```

这一步的意义：

- 把 KV compression 从 token/head 重要性排序，提升为 policy-level counterfactual comparison；
- 允许不同候选 policy 在不同 block 获胜；
- 为后续 risk memory 和 rescue gate 提供可解释的局部证据。

## 3. Counterfactual Risk Memory

短 calibration window 会有短视误选，因此维护每个 policy 的历史风险：

```text
M_i,t = update(M_i,t-1, risk_features_i,t)
```

一个简单 score 可以写成：

```text
S_i,t = mean_recent(d_i) + lambda_tail * tail_risk_i - lambda_win * win_rate_i
```

得到 memory prior：

```text
p_mem = argmin_i S_i,t
```

同时保留当前 calibration 最优：

```text
p_min = argmin_i L_i(C_t)
```

论文里应强调：

- `p_min` 是短窗口局部最优；
- `p_mem` 是历史稳健 prior；
- 二者冲突时，不能简单相信其中一个，需要 horizon rescue 仲裁。

## 4. Conditional Horizon Rescue Gate

给定未来 sentinel / horizon window `H_t`，用较长 horizon 估计候选 policy 对未来 token 的风险：

```text
L_i(H_t)
```

对 early-selected policy `p_e` 和 rescue candidate `p_r`，定义 horizon gain：

```text
gain(r, e) = L_e(H_t) - L_r(H_t)
```

若 `gain > 0`，说明 `p_r` 在未来 horizon 上比 early choice 更好。

当前主线使用 conditional validation-prior anchor：

```text
p_anchor = best policy on validation prior
```

触发条件：

```text
if short_horizon_margin < tau_margin
or early choice conflicts with validation/risk anchor:
    run rescue probe
else:
    accept early choice
```

当前推荐实验默认：

```text
tau_margin = 0.012
anchor_accept_on_match = false
low_spread_early_exit = false
skip_anchor_nonpositive_gain = false
```

注意：`skip_anchor_nonpositive_gain` 的 corrected gate 结果显示有潜力省 probe，但质量不够稳，因此不进入主方法。

## 5. Top-k Cascade：控制候选评估成本

直接对所有候选跑长 horizon 过贵，因此使用 cascade：

```text
Stage 1:
    evaluate all candidates on short horizon s
    rank candidates by short-horizon risk

Stage 2:
    if confidence is low:
        extend only top-k candidates
        plus p_mem / p_min / validation anchor
    else:
        accept short-horizon winner
```

默认：

```text
short horizon s = 32
long horizon h = 64 or 128
extend_topk = 2
candidate anchors = {p_mem, p_min, p_validation}
```

该设计是论文创新点之一：它不是只提出一个 sparse attention operator，而是提出 bounded-probe-budget 下的 policy arbitration。

## 6. Algorithm

```text
Algorithm 1: Horizon-PCIC

Input:
    block sequence B_1 ... B_T
    candidate policies P = {p_1 ... p_K}
    calibration window length c
    short horizon s
    long horizon h
    cascade top-k k
    risk memory M
    validation-prior anchors A_val

For each block B_t:
    C_t <- calibration prefix of B_t
    H_t^s <- first s sentinel tokens after C_t
    H_t^h <- first h sentinel tokens after C_t

    Run full attention on C_t to get L_full(C_t)

    For each policy p_i in P:
        Run policy p_i on C_t
        Compute d_i,t = L_i(C_t) - L_full(C_t)
        Compute pairwise margins A_i,j,t
        Update local risk features R_i,t

    p_min <- argmin_i L_i(C_t)
    p_mem <- argmin_i memory_score(M_i, R_i,t)

    C_short <- P
    Evaluate all p_i in C_short on H_t^s
    p_early <- best short-horizon policy
    margin <- short-horizon best-vs-runner-up margin

    If margin >= tau_margin and no anchor conflict:
        a_t <- p_early
    Else:
        C_long <- top-k short-horizon policies
                  union {p_min, p_mem}
                  union A_val
        Evaluate policies in C_long on H_t^h
        a_t <- horizon-risk winner under rescue gate

    Evaluate remaining tokens of B_t using selected policy a_t
    Update risk memory M with observed counterfactual risks

Return selected policy trace a_1 ... a_T
```

## 7. 论文贡献写法

建议写成三条贡献：

1. **Pairwise counterfactual policy calibration**：用反事实 loss 比较多个 KV/attention policies，而不是设计单个固定 sparse pattern。
2. **Online blockwise policy selection with risk memory**：把长上下文压缩建模为 non-stationary blockwise decision problem，允许策略随文本段变化。
3. **Conditional horizon rescue gate under bounded probe budget**：用 top-k cascade 在低候选预算下修复 short-horizon myopia。

不建议写成：

- “我们提出 qabs8cand3attn”；
- “我们提出 landmark attention 改进版”；
- “我们已经端到端快于 baseline”；
- “这是一个新的 attention kernel”。

## 8. 当前证据链

已完成的关键证据：

- Fixed / online / rescue / oracle：`docs/pcic_mainline_fixed_online_rescue_2026_06_29.md`
- Blockwise policy trace：`docs/pcic_blockwise_policy_trace_2026_06_29.md`
- Component evidence matrix：`docs/pcic_component_evidence_matrix_2026_06_29.md`
- Minimal component ablation suite：`docs/pcic_minimal_component_ablation_2026_06_29.md`
- Minimal component ablation runbook：`docs/pcic_minimal_component_ablation_runbook_2026_06_29.md`
- Paper readiness gate：`docs/pcic_paper_readiness_gate_2026_06_29.md`
- Paper skeleton：`docs/horizon_pcic_paper_skeleton_2026_06_29.md`
- Corrected key results：`docs/horizon_pcic_corrected_key_results_2026_06_29.md`
- Related work novelty：`docs/horizon_pcic_related_work_novelty_2026_06_29.md`

当前最强 claim：

```text
Horizon-PCIC 将 KV/attention compression 从固定稀疏规则提升为在线反事实策略选择问题；
在 hard-topic、Monte、Needle/RULER-style smoke 上，online selection 和 rescue gate 能修复固定策略或 short-horizon top2 的失败；
但真实端到端速度仍需要 fused/sparse candidate probe 才能成为强系统 claim。
```

## 9. 仍需补的 ICML 级证据

P0：

- 正式 LongBench/RULER 子集上的 `best fixed / online / oracle` 对比；
- blockwise trace figure，与文本结构或任务位置对齐；
- no-rescue / no-memory / no-pairwise 的消融，其中 no-rescue 应使用 `memory_only_no_rescue` 关闭 sentinel/horizon arbitration，no-memory 应使用 `--risk_memory_use_history false` 关闭跨 block 历史，no-pairwise 应使用 `--pairwise_candidate_probe false` 关闭候选间 sentinel/horizon 对比。

P1：

- fused/sparse candidate probe，降低 corrected gate；
- 多模型验证；
- 与 H2O / SnapKV / PyramidKV / QUEST / SparQ-like baseline 的同预算对比。

结论：

> 这条主线可以继续推进为 paper，但当前应把创新性 claim 放在 online counterfactual policy selection，而不是速度超过 baseline。速度是下一阶段必须补齐的系统证据。
