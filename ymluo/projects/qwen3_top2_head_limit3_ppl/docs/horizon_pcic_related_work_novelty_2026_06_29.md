# Horizon-PCIC 与已有方法的相似性和创新性风险分析（2026-06-29）

> Corrected 主结果表见：`docs/horizon_pcic_corrected_key_results_2026_06_29.md`。后续 speed/gate claim 以 corrected gate 口径为准。

> Fixed / online / rescue / oracle 主线证据见：`docs/pcic_mainline_fixed_online_rescue_2026_06_29.md`。相关工作对比时应强调 online policy selection，而不是固定 sparse operator。

> Blockwise policy trace 证据见：`docs/pcic_blockwise_policy_trace_2026_06_29.md`。相关工作对比时应强调跨 block 的动态选择，而不是单个 sparse pattern。

> Method spec / 论文算法定义见：`docs/horizon_pcic_method_spec_2026_06_29.md`。相关工作对比应围绕该算法定义展开。

> Component evidence matrix 见：`docs/pcic_component_evidence_matrix_2026_06_29.md`。相关工作和审稿 rebuttal 应严格区分已验证贡献和未完成速度/消融。

## 结论先行

当前方法 **Horizon-PCIC = Pairwise-CIC + online blockwise policy selection + horizon-aware top-k cascade rescue gate**，不是单纯的 SparQ Attention 复现，也不只是 Landmark / H2O / SnapKV / PyramidKV 的小改。

但如果论文只写成“用若干候选集合做近似注意力，然后用少量 token 选择更好的候选”，创新性会显得不足，容易被审稿人归类为：

1. query-aware sparse attention；
2. KV-cache/token selection；
3. retrieval-head / streaming-head 分流；
4. block-level landmark/routing 的工程组合。

要支撑 ICML 级别投稿，核心 novelty 需要明确上升为：

> **在推理时把 KV/attention 近似看作一个在线策略选择问题：用 Pairwise Counterfactual Influence Calibration 估计候选 attention policy 的未来风险，并用 horizon-aware cascade 在低成本下做可恢复的策略切换。**

也就是说，论文重点不应是“某个固定稀疏模式”，而应是“如何在线选择稀疏注意力策略，并避免短视选择导致的长程 PPL/任务退化”。

## 与已有工作的相似点

### 1. 与 SparQ Attention 的相似点

SparQ Attention 的核心是利用 query 中较大的维度，选择性读取 cached history，从而降低 attention 中的数据搬运成本。我们的 qabs / 候选通道思想与 SparQ 的相似点是：

- 都是 **query-aware sparse attention**；
- 都试图避免全量读取 KV cache；
- 都在 attention 层内部做近似，而不是训练新模型；
- 如果只说“用 query 的少数维度筛 token/head/channel”，会非常像 SparQ。

关键区别应写清楚：

- SparQ 主要解决“如何用 query 近似找到重要 token / 减少带宽”；
- Horizon-PCIC 关注“多个候选稀疏策略之间，哪一个在未来 horizon 上风险最低”；
- 当前方法的 Pairwise-CIC / risk memory / rescue gate 是 **策略选择与校准层**，不是单个 sparse attention operator。

创新风险判断：

- 如果论文主张只是 qabs8cand3attn，本质上很容易被认为接近 SparQ；
- 如果论文主张是 Horizon-PCIC 的在线 counterfactual policy selection，和 SparQ 的差异才足够清楚。

### 2. 与 Landmark Attention 的相似点

Landmark Attention 使用 landmark token 表示 block，并让模型通过 landmark 检索相关 block。我们的 `landmark` 实验也有 block / representative / routing 的味道。

相似点：

- 都有 block-level routing / block-level proxy；
- 都希望避免对完整上下文做全量 attention；
- 都可以解释为先粗筛再精算。

关键区别：

- Landmark Attention 依赖 landmark token，并通常涉及训练或专门适配；
- Horizon-PCIC 不训练模型，不引入新的 landmark token 语义；
- 当前 `landmark` 更像候选策略中的一个 proxy，而不是整个方法的唯一核心。

创新风险判断：

- 如果命名或叙述把 `landmark` 放在中心，会显得很像已有 Landmark Attention；
- 建议把 landmark 降级为 “candidate proposal / cheap probe” 组件，而不是论文标题级贡献。

### 3. 与 H2O / Scissorhands / Keyformer 的相似点

H2O、Scissorhands、Keyformer 都在不同形式上利用 attention 重要性或历史重要性，保留关键 token，丢弃不重要 token。

相似点：

- 都基于“少数 token/head 对输出贡献更大”的稀疏性；
- 都是训练无关或弱训练相关的推理时优化；
- 都存在“过去重要的 token 未来仍可能重要”的假设。

关键区别：

- H2O / Scissorhands / Keyformer 更像 **token retention / eviction policy**；
- Horizon-PCIC 不只是决定 token 留不留，而是在多个 attention policy 之间做在线选择；
- Pairwise-CIC 的重点是比较候选策略导致的 counterfactual loss / PPL 风险，而不是直接用 attention score 当重要性。

创新风险判断：

- 如果只展示“某些 token 更重要，所以保留它们”，创新性不够；
- 需要强调 Pairwise-CIC 是校准候选策略的 **pairwise counterfactual decision rule**。

### 4. 与 SnapKV / PyramidKV / Ada-KV 的相似点

SnapKV 利用 observation window 选择每个 head 的重要 KV 位置；PyramidKV / Ada-KV 则强调层间或 head-wise budget 分配。

相似点：

- 都是 KV cache compression / budget allocation；
- 都可能有 per-head、per-layer、observation-window 或 adaptive-budget 机制；
- 都会用 LongBench / Needle / perplexity 来证明精度保持。

关键区别：

- SnapKV/PyramidKV/Ada-KV 的主问题是“压缩 KV cache 时如何分配预算”；
- Horizon-PCIC 的主问题是“当前 block/horizon 下，哪一种候选 attention policy 的风险最低”；
- top-k cascade rescue gate 是为了控制在线选择代价，而不是单纯改预算。

创新风险判断：

- 与 Ada-KV 的“adaptive budget allocation”表述很接近，论文中应避免只说 adaptive budget；
- 应把贡献写成 “counterfactual policy calibration + horizon rescue”，而不是 “adaptive KV allocation”。

### 5. 与 Quest / Loki / MInference 的相似点

Quest、Loki、MInference 都是 long-context sparse attention 方向的重要相关工作：

- Quest 是 query-aware KV page selection；
- Loki 用 low-rank key space / PCA 近似 attention score；
- MInference 识别 long-context attention 的稀疏 pattern，并用 GPU kernel 加速 prefill。

相似点：

- 都利用 attention 的动态稀疏性；
- 都强调 query/head/block/pattern 的选择；
- 都需要真实系统加速才能形成强论文。

关键区别：

- Quest/Loki/MInference 更偏 sparse index/operator/kernel；
- Horizon-PCIC 更偏在线选择器：给定多个候选 sparse operators，选择当前 horizon 风险最低的一个；
- 这也意味着 Horizon-PCIC 可以作为上层 selector，理论上组合 SparQ/Quest/Loki/MInference 类型候选。

创新风险判断：

- 如果没有真实速度，审稿人会质疑这是“昂贵的元选择器”；
- 必须用 batched/fused candidate probe 或 cached probe 证明选择开销可控。

### 6. 与 DuoAttention / retrieval-head 方法的相似点

DuoAttention 把 attention head 分成 retrieval heads 和 streaming heads，对不同 head 用不同 KV cache。

相似点：

- 都意识到 head 之间功能不同；
- 都可能选择某些 head/层走更完整的 attention，其他 head 走轻量路径；
- 都是 inference-time long-context efficiency。

关键区别：

- DuoAttention 更像 offline/head-type classification；
- Horizon-PCIC 是 online/blockwise/horizon-aware policy selection；
- Pairwise-CIC 可以在不同文本段、不同 horizon 上动态改变策略，而不是固定 head 类型。

创新风险判断：

- 如果最终策略总是固定选 `0,7` 或 `2,0`，动态性证据会变弱；
- 需要展示不同任务、不同 block、不同 horizon 下确实发生非平凡策略切换。

## 当前方法真正有希望成为论文贡献的部分

### 贡献 1：Pairwise Counterfactual Influence Calibration

把候选 attention policy 的选择写成 pairwise counterfactual 比较，而不是直接 ranking token/head：

- 对同一 block 比较候选策略 A/B；
- 用 sentinel/horizon loss 估计 A 相对 B 的未来风险；
- 把选择问题变成可校准的 pairwise decision。

这是最像“方法论贡献”的部分。

### 贡献 2：Risk memory

短 horizon probe 容易误选，risk memory 用过去 block 的风险积累防止反复选择局部最优候选。

这部分如果能证明：

- 去掉 risk memory 会在 hard-topic / Monte 退化；
- 加上 risk memory 可以稳定跨文本；
- risk memory 不显著增加计算；

就能形成清楚的 algorithmic value。

### 贡献 3：Horizon-aware top-k cascade rescue gate

已有实验已经说明 s32 在 Monte 上失败，而 s64/top2 cascade 可以恢复质量。这一点很重要，因为它证明方法不是简单调参：

- 短 horizon probe 有系统性失败模式；
- 需要 horizon-aware rescue；
- top-k cascade 可以只扩展少数候选，降低 gate cost，同时保持质量。

这可以作为论文中的核心 empirical insight。

### 贡献 4：Selector/operator 解耦

Horizon-PCIC 不应该绑定某一个 qabs 候选。更强的论文叙述是：

- 候选 operator 可以是 qabs、landmark、Quest-like page selection、SparQ-like channel selection；
- PCIC 负责在线选择；
- rescue gate 负责把选择成本压低。

这样能避免被说成 “SparQ 的一个 variant”。

## 目前创新性不足的地方

### 1. 真实系统速度还没打穿

当前 top2 cascade 的 gate cost 降低 50%–59%，但 batched grouped prototype 还没有比 serial 更快。论文不能直接声称已经端到端快于 baseline。

必须补：

- batched/fused candidate probe；
- 更合理的 kernel 或至少 torch-level grouped evaluation；
- end-to-end tokens/s、attention latency、prefill/decode 拆分。

### 2. 标准 benchmark 不足

目前 hard-topic / War / Monte 能说明方向，但不够支撑 ICML。

必须补：

- LongBench 或 RULER；
- Needle-in-a-Haystack；
- 至少 2 个模型；
- 至少 2 个长度区间；
- 与 H2O / SnapKV / PyramidKV / Quest / SparQ-like baseline 对比。

### 3. 动态选择证据还不够

如果最终很多数据都选固定 combo，审稿人会问为什么不直接离线选一个固定策略。

必须补：

- blockwise policy trace；
- 不同文本段的策略切换图；
- oracle gap：固定策略 vs online PCIC vs full oracle；
- 证明 online selector 在非平稳文本中有收益。

### 4. 理论表述还需要收敛

目前“CIC / Pairwise-CIC / risk memory / horizon gate”概念较多，论文容易显得拼装。

建议统一成一个主问题：

> Online counterfactual selection of sparse attention policies under bounded probe budget.

然后所有组件围绕这个问题解释。

## 建议论文定位

不建议标题或摘要强调：

- qabs8cand3attn；
- landmark；
- top2 head limit；
- 某个固定 combo；
- “比 SparQ 更快”但没有系统速度证据。

建议强调：

- Horizon-aware sparse attention policy selection；
- Pairwise counterfactual calibration；
- Low-cost cascade rescue for long-horizon robustness；
- Plug-in selector over existing sparse attention candidates。

一个更合适的论文题目方向：

> Horizon-PCIC: Counterfactual Online Policy Selection for Robust Sparse Attention in Long-Context LLM Inference

## Paper 创新性判断

当前状态：

- **作为 workshop / arXiv 方法探索：够。**
- **作为 ICML 正会投稿：方向有潜力，但证据还不够。**
- **如果只讲 qabs / landmark / fixed combo：创新性不足。**
- **如果证明 PCIC 是通用在线策略选择框架，并补齐速度与 benchmark：有机会。**

关键判断标准：

1. 是否能明确区别于 SparQ/Quest/Loki：不是 sparse score approximation，而是 online policy calibration；
2. 是否能明确区别于 SnapKV/PyramidKV/Ada-KV：不是预算分配，而是 counterfactual policy selection；
3. 是否能明确区别于 DuoAttention：不是固定 head 类型，而是 blockwise/horizon-aware 动态策略；
4. 是否能证明 selector 开销低到不会吃掉收益；
5. 是否能在标准 benchmark 上稳定保持质量。

## 后续最重要实验

### 已补：fixed / online / oracle 对比

最新汇总见：`docs/pcic_fixed_online_oracle_2026_06_29.md`

关键结果：

| dataset | best fixed | fixed ΔPPL | online ΔPPL | oracle ΔPPL | online-oracle gap | 结论 |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| Hard-topic eval64 | `0,6` | -0.012719 | -0.076940 | -0.076940 | 0.000000 | online 等于 oracle，强支持动态选择 |
| Hard-topic eval128 | `0,6` | -0.020744 | 0.004371 | -0.049633 | 0.054004 | online 未追上 oracle，是当前主要失败例 |
| War and Peace | `0,7` | -2.135311 | -2.135311 | -2.135311 | 0.000000 | online 等于 best fixed / oracle，但动态性证据弱 |
| Count of Monte Cristo | `2,0` | -0.009005 | -0.219215 | -0.219215 | 0.000000 | online 明显优于 best fixed，并等于 oracle |

这组结果对 paper 主线很关键：

- 正面证据：Hard-topic eval64 和 Monte 都证明 online PCIC 不是固定 combo 的简单替代，能达到 blockwise oracle。
- 中性证据：War 的 best fixed、online、oracle 相同，说明该文本上策略空间太简单，不能单独证明动态选择价值。
- 负面证据：Hard-topic eval128 暴露 horizon gate 仍有短视/误选问题，后续必须针对长 eval window 做 rescue gate 改进。

因此，当前最合理的论文叙述是：

> Pairwise-CIC 提供接近 blockwise oracle 的在线策略选择信号，但长 horizon 下仍需要更强的 rescue / uncertainty handling；这正是方法中 horizon-aware cascade 的必要性来源。

### 已补：Hard-topic eval128 delayed-rescue

最新汇总见：`docs/pcic_delayed_rescue_eval128_2026_06_29.md`

针对 Hard-topic eval128 的 online-oracle gap，新增实验显示：

| run | avg_delta_ppl | method/base | gate_s | combos |
| --- | ---: | ---: | ---: | --- |
| top2 32→64 | 0.004371 | 2.648 | 27.085 | `0,7;2,0,7,12;0,6;0,13` |
| s128 all-candidate | -0.049633 | 8.291 | 123.331 | `0,6;2,0,7,12;0,6;0,6` |
| 64→128 top2 + anchor06 | -0.049633 | 4.159 | 52.291 | `0,6;2,0,7,12;0,6;0,6` |

这说明：

- 原失败例可以通过更长 horizon rescue 修复；
- full-horizon all-candidate gate 太贵；
- top2 + long-horizon anchor 可以达到 oracle，并把 s128 all-candidate 的 gate 成本降低约 `57.6%`；
- 但 `anchor06` 目前仍是手写 anchor，下一步必须自动化为 validation prior / diversity anchor / uncertainty anchor。

### 已补：validation-prior auto-anchor

最新汇总见：`docs/pcic_auto_anchor_eval128_2026_06_29.md`

自动 anchor 流程：

```text
validation fixed-combo results -> rank by avg_delta_ppl -> top-k anchor -> eval horizon rescue
```

在 Hard-topic eval64 validation 上，自动排序选出 `0,6`；在 Hard-topic eval128 上：

| run | avg_delta_ppl | method/base | gate_s | anchors | combos |
| --- | ---: | ---: | ---: | --- | --- |
| top2 32→64 | 0.004371 | 2.648 | 27.085 | `` | `0,7;2,0,7,12;0,6;0,13` |
| manual anchor06 | -0.049633 | 4.158 | 53.710 | `0,6` | `0,6;2,0,7,12;0,6;0,6` |
| auto anchor | -0.049633 | 4.164 | 51.997 | `0,6` | `0,6;2,0,7,12;0,6;0,6` |

这把方法从手工修补推进到可写成论文组件的版本：

> validation-prior horizon-anchor rescue gate

仍需注意：目前 auto-anchor 只在 Hard-topic eval128 上验证，仍需 War/Monte 和更长 blocks 的负面检查。

### 已补：War / Monte auto-anchor 负面检查

最新汇总见：`docs/pcic_warmonte_auto_anchor_check_2026_06_29.md`

结果：

| dataset | baseline ΔPPL | auto-anchor ΔPPL | baseline gate_s | auto-anchor gate_s | auto anchor | 结论 |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| War | -2.135311 | -2.135311 | 6.544 | 8.863 | `0,7` | 质量不变，开销上升 |
| Monte | -0.219215 | -0.219215 | 6.573 | 6.670 | `2,0` | 质量不变，开销基本不变 |

这说明 auto-anchor 不是固定 `0,6`，而是随 validation prior 变化；同时也暴露出新的系统问题：

- 当原 top2 cascade 已经稳定时，强制加入 anchor 会增加 gate 开销；
- 最终方法应采用 conditional auto-anchor，而不是 unconditional anchor。

建议最终算法写成：

```text
Pairwise-CIC online selection
+ cheap top-k cascade
+ conditional validation-prior anchor rescue
```

触发条件可以是：

1. short-horizon best margin 低；
2. early best 与 validation-prior anchor / risk-memory anchor 冲突；
3. delayed-win detector 认为历史长 horizon 排名和当前短 horizon 排名不一致。

### 已补：conditional auto-anchor

最新汇总见：`docs/pcic_conditional_auto_anchor_2026_06_29.md`

使用 `sentinel_cascade_accept_margin = 0.012` 后：

| dataset | top2 ΔPPL | unconditional anchor ΔPPL | conditional anchor ΔPPL | top2 gate_s | unconditional gate_s | conditional gate_s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Hard-topic eval128 | 0.004371 | -0.049633 | -0.049633 | 27.085 | 51.997 | 53.648 |
| War | -2.135311 | -2.135311 | -2.135311 | 6.544 | 8.863 | 6.659 |
| Monte | -0.219215 | -0.219215 | -0.219215 | 6.573 | 6.670 | 6.641 |

这组结果说明：

- Hard-topic 的 delayed-win failure 仍被修复；
- War 的无效 anchor 开销基本被消除；
- Monte 质量不变，开销基本不变；
- 最终方法应写成 `conditional validation-prior horizon-anchor rescue gate`。

当前仍需补强的是阈值选择：`0.012` 是经验阈值，下一步要做小网格或自适应 margin。

### 已补：conditional margin grid

最新汇总见：`docs/pcic_conditional_auto_anchor_margin_grid_2026_06_29.md`

| margin | Hard-topic eval128 ΔPPL | War gate_s | Monte gate_s | 结论 |
| ---: | ---: | ---: | ---: | --- |
| 0.008 | -0.023677 | 6.691 | 6.688 | Hard-topic 仍漏 delayed-win |
| 0.010 | -0.023677 | 6.631 | 6.629 | Hard-topic 仍漏 delayed-win |
| 0.012 | -0.049633 | 6.659 | 6.641 | 当前最均衡 |
| 0.015 | -0.049633 | 7.699 | 6.566 | War 出现过度 extension |

因此当前推荐阈值为 `0.012`。这组网格证明 rescue gate 有可解释的质量/代价边界，不是单点偶然结果。

### 已补：Hard-topic b8 更长 block 验证

最新汇总见：`docs/pcic_hardtopic_b8_condautoanchor_2026_06_29.md`

| run | blocks | avg_delta_ppl | method/base | gate_s | extended | early |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| b4 top2 | 4 | 0.004371 | 2.648 | 27.085 | 1 | 3 |
| b4 conditional auto-anchor | 4 | -0.049633 | 4.144 | 53.648 | 4 | 0 |
| b8 top2 | 8 | 0.006074 | 2.512 | 49.737 | 5 | 3 |
| b8 conditional auto-anchor | 8 | -0.040598 | 4.161 | 107.643 | 8 | 0 |

结论：

- b8 上 top2 仍有多个 short-horizon miss；
- conditional auto-anchor 将 b8 平均 ΔPPL 从 `0.006074` 改善到 `-0.040598`；
- 这说明 rescue gate 的收益不是 4-block 偶然；
- 但 b8 gate 达到 `107.643s`，系统速度仍是主要短板。

当前 paper 风险因此更清晰：方法创新性和质量证据在增强，但真实加速证据仍不足。

### 已补：Needle-style benchmark smoke

最新汇总见：`docs/pcic_needle_smoke_condautoanchor_2026_06_29.md`

服务器本地没有现成 LongBench / RULER / Needle 数据；为避免网络下载，本轮生成了本地 needle-style validation/eval 文本。

| run | avg_delta_ppl | method/base | gate_s | anchors | combos |
| --- | ---: | ---: | ---: | --- | --- |
| needle top2 | 0.000118 | 1.831 | 13.206 | `` | `2,0,7,12/2,0/2,0/2,0` |
| needle conditional auto-anchor | -0.000166 | 3.401 | 39.909 | `2,0` | `2,0/2,0/2,0,7,12/2,0` |

结论：

- top2 在简单 needle-style smoke 上已经很稳；
- conditional auto-anchor 质量略好但代价明显更高；
- rescue gate 应针对 hard / delayed-win regime 自适应触发，而不是在 easy retrieval regime 无条件扩展；
- 这进一步支持 adaptive margin / early-exit 的必要性。

该实验不是正式 LongBench/RULER，只能作为无下载 smoke。正式 paper 仍需要标准 benchmark。

### 已补：anchor-match early-exit 负面消融

最新汇总见：`docs/pcic_anchor_match_early_exit_ablation_2026_06_29.md`

测试 heuristic：

```text
if early_selected_combo == validation_prior_anchor:
    accept early
```

结果：

| run | avg_delta_ppl | method/base | gate_s | extended | early |
| --- | ---: | ---: | ---: | ---: | ---: |
| hard cond | -0.049633 | 4.144 | 53.648 | 4 | 0 |
| hard anchor-match early | -0.049633 | 4.809 | 62.742 | 3 | 1 |
| needle cond | -0.000166 | 3.401 | 39.909 | 4 | 0 |
| needle anchor-match early | -0.000029 | 4.566 | 59.459 | 1 | 3 |

结论：

- anchor-match early-exit 可以减少 extension 次数，但没有降低实际 wall-clock；
- Needle 上质量略退且代价更高；
- 该 heuristic 不应进入 paper 主方法，只保留为负面消融；
- 当前推荐仍是 `sentinel_cascade_accept_margin = 0.012` 且 `sentinel_cascade_anchor_accept_on_match = false`。

### 已补：low-spread early-exit 负面消融

最新汇总见：`docs/pcic_low_spread_early_exit_ablation_2026_06_29.md`

测试 heuristic：

```text
if early best-vs-runner-up loss spread <= 0.001:
    accept early
```

结果：

| run | avg_delta_ppl | method/base | gate_s | extended | early | low_spread_early |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| hard cond | -0.049633 | 4.144 | 53.648 | 4 | 0 | 0 |
| hard low-spread | -0.049633 | 4.161 | 52.526 | 4 | 0 | 0 |
| needle cond | -0.000166 | 3.401 | 39.909 | 4 | 0 | 0 |
| needle low-spread | 0.000003 | 4.692 | 61.147 | 0 | 4 | 4 |

结论：

- low-spread early-exit 可以在 Needle/easy-regime 中把 extension 从 `4` 降到 `0`；
- 但 wall-clock 反而更差，`gate_s` 从 `39.909s` 升到 `61.147s`；
- 质量也从 `-0.000166` 轻微退到 `0.000003`；
- 该 heuristic 同样不应进入主方法，当前推荐保持 `sentinel_cascade_accept_low_spread = 0.0`。

两组 early-exit 负面消融说明，速度问题不是简单减少 extension 次数能解决的；下一步应转向 batch-row/fused sentinel probe，或先补 LongBench / RULER 质量证据。

### 已补：conditional auto-anchor batched gate 主线对比

最新汇总见：`docs/pcic_condautoanchor_batched_gate_2026_06_29.md`

该实验直接测试当前 paper 主线：

```text
Pairwise-CIC
+ online blockwise selection
+ conditional validation-prior horizon-anchor rescue gate
+ --sentinel_batched_candidates true
```

结果：

| dataset | serial ΔPPL | batched ΔPPL | serial gate_s | batched gate_s | same combos |
| --- | ---: | ---: | ---: | ---: | --- |
| hard | -0.049633 | -0.049778 | 53.648 | 79.524 | True |
| war | -2.135311 | -2.133787 | 6.659 | 8.689 | True |
| monte | -0.219215 | -0.251805 | 6.641 | 8.466 | True |

结论：

- batch-row budget map 已经能接入完整 conditional rescue gate 主流程；
- 三个数据集选择的 blockwise combos 与 serial 完全一致；
- PPL 差异很小，说明 selector 语义基本保持；
- 但当前 eager batch-row path 更慢，不能作为 speed claim；
- 下一步速度创新必须是 fused candidate probe / tensorized mask path，而不是单纯把候选堆到 batch 维。

### 已补：batch-row dispatch 优化

最新汇总见：`docs/pcic_condautoanchor_batched_gate_optdispatch_2026_06_29.md`

优化内容：

```text
如果某一层所有 batch rows 使用同一个 layer budget：
    直接走整批 forward
    跳过 index_select / index_copy_

缓存 batch-row index tensor：
    减少每 token / layer 的小张量构造
```

结果：

| dataset | old batched gate_s | optimized batched gate_s | serial gate_s | same combos |
| --- | ---: | ---: | ---: | --- |
| hard | 79.524 | 61.139 | 53.648 | True |
| war | 8.689 | 7.462 | 6.659 | True |
| monte | 8.466 | 7.467 | 6.641 | True |

结论：

- dispatch 优化显著降低了 batch-row overhead；
- 但 optimized batched 仍慢于 serial；
- 说明主要瓶颈已经从“分组调度开销”推进到“候选维 KV/cache 复制 + landmark/recent attention 本身未 fused”；
- 下一步必须做 candidate-level fused probe 或 tensorized grouped sparse attention，才能支撑真实 speed claim。

### 已补：batch-row budget group 瓶颈定位

最新汇总见：`docs/pcic_batch_row_budget_groups_2026_06_29.md`

该分析不跑模型，只解析已有 `batch_maps/*.json`：

| output | maps | row_counts | all-same layer frac | avg groups/layer | mixed layers |
| --- | ---: | --- | ---: | ---: | ---: |
| hard optdispatch | 8 | `2,4,5,8` | 0.817 | 1.183 | 41 |
| monte optdispatch | 4 | `2,3,4` | 0.857 | 1.143 | 16 |
| war optdispatch | 2 | `4` | 0.857 | 1.143 | 8 |

解释：

- 约 `81.7%–85.7%` 的层在 batch-row candidate gate 中已经是 all-same budget；
- 这些层已经可以走整批 forward，dispatch 优化正是针对这部分；
- 真正需要新 kernel 的不是所有层，而是少数 mixed-budget layers；
- 这把系统方向收窄为：对 mixed layers 做 row-wise full/landmark 的 fused/tensorized attention，而不是重写全模型 attention。

### 已补：mixed dense Stage 1 负/弱正结果

最新汇总见：`docs/pcic_condautoanchor_batched_gate_mixeddense_2026_06_29.md`

实现：

```text
LAYER_BUDGET_MIXED_DENSE=1
```

结果：

| dataset | optdispatch gate_s | mixeddense gate_s | serial gate_s | same combos |
| --- | ---: | ---: | ---: | --- |
| hard | 61.139 | 61.700 | 53.648 | True |
| war | 7.462 | 7.507 | 6.659 | True |
| monte | 7.467 | 7.217 | 6.641 | True |

结论：

- Stage 1 dense mixed path 没有改变选择语义；
- 但 Hard/War 更慢，Monte 只小幅变快；
- 说明单纯 dense tensorization 不够，full-score FLOPs 抵消了 dispatch 收益；
- 当前默认不启用该路径，下一步直接做 sparse gather / fused kernel。

### 已补：RULER-style offline smoke

最新汇总见：`docs/pcic_ruler_style_smoke_2026_06_29.md`

该实验不下载外部数据，生成三类 synthetic RULER-style 长上下文文本：

1. multi-needle；
2. variable binding；
3. topic switch。

结果：

| task | top2 ΔPPL | conditional ΔPPL | top2 gate_s | conditional gate_s |
| --- | ---: | ---: | ---: | ---: |
| multi-needle | 0.000074 | 0.000074 | 11.081 | 16.650 |
| variable binding | 0.017397 | -0.000564 | 11.857 | 13.249 |
| topic switch | 0.000302 | 0.000139 | 6.530 | 9.931 |

结论：

- validation-prior anchor 自动选出 `2,0`；
- conditional rescue 在 multi-needle 上质量持平；
- 在 variable binding 上明显修复 top2 的 PPL drift；
- 在 topic switch 上小幅改善；
- 这不是正式 RULER / LongBench，但支持主线不是只在 hard-topic / 小说文本上成立；
- 成本仍然更高，因此系统速度仍需 fused sparse probe。

### 已补：cascade extension waste 后验上界

最新汇总见：`docs/pcic_extension_waste_2026_06_29.md`

该分析只解析已有 CSV，不跑模型；它衡量 extension 后最终 combo 是否仍等于 early initial combo。

| case | extended | no-change ext | avoidable s | avoidable frac |
| --- | ---: | ---: | ---: | ---: |
| hard | 4 | 2 | 15.622 | 0.437 |
| monte | 2 | 1 | 2.211 | 0.400 |
| needle | 4 | 2 | 13.314 | 0.462 |
| ruler multi-needle | 3 | 1 | 4.972 | 0.375 |
| ruler topic-switch | 3 | 2 | 6.620 | 0.669 |
| ruler variable | 3 | 1 | 4.950 | 0.429 |

结论：

- extension 中有相当一部分没有改变最终 combo；
- 这是 speed 改进空间的后验上界，不是已验证 online rule；
- 下一步可做 calibrated skip-gate：预测哪些 low-margin block 即使 extension 后仍会保持 early combo；
- 这条线和 fused sparse probe 互补：skip-gate 减少 probe 次数，fused probe 降低单次 probe 成本。

### 已补：zero false-skip rule search

最新汇总见：`docs/pcic_skip_gate_rule_search_2026_06_29.md`

当前最简洁候选规则：

```text
anchor_hit and sentinel_horizon_gain_ratio <= 0
```

后验结果：

| rule | selected blocks | saved seconds | saved fraction | false skip |
| --- | ---: | ---: | ---: | ---: |
| `anchor_hit_and_ratio_le_0` | 6 | 29.856 | 0.285 | 0 |

含义：

- 这是一个保守 easy-regime skip candidate；
- 它没有碰 Hard-topic delayed-win blocks；
- 它主要跳过 Needle / RULER-style 中 validation anchor 已经胜出且 horizon gain 不支持换 combo 的 block；
- 仍需接入 runner 做在线验证，不能只凭后验写成最终方法。

### 已补：anchor nonpositive-gain skip-gate 在线负面消融

最新汇总见：`docs/pcic_skip_gate_anchor_gain_online_2026_06_29.md`

在线规则：

```text
if initial_selected_combo in validation_prior_anchors
and sentinel_horizon_gain_ratio <= 0:
    skip extension
```

结果：

| task | ΔPPL change | gate_s change | skipped |
| --- | ---: | ---: | ---: |
| hard | 0.000000 | -0.952 | 0 |
| needle | +0.000137 | +18.868 | 3 |
| ruler multi-needle | 0.000000 | +4.726 | 1 |
| ruler variable | +0.000026 | +12.672 | 2 |
| ruler topic-switch | +0.000163 | +24.987 | 3 |

结论：

- 后验 zero-false-skip 不等于在线有效；
- easy-regime 中提前固定 anchor 会轻微牺牲质量；
- corrected gate 下降，说明 skip 确实能减少候选 probe 成本；
- 但质量/选择不够稳；
- 该规则不进入主方法，默认关闭；
- 速度主线应继续 fused/sparse probe，同时 skip-gate 需要更强校准后再考虑。

计时口径修正见：`docs/pcic_corrected_gate_skipanchor_gain_2026_06_29.md`

### P0：证明不是固定策略能替代

做三组对比：

1. best fixed combo；
2. online PCIC；
3. oracle blockwise best。

如果 online PCIC 明显接近 oracle，并优于 best fixed，创新性会强很多。

### P1：证明 horizon rescue 必要

固定比较：

1. s32；
2. s64；
3. top2 cascade；
4. no rescue；
5. no risk memory。

重点展示 Monte s32 失败、top2 cascade 恢复的现象。

### P2：证明选择开销可控

继续做 fused/tensorized probe：

1. candidate-level fused loss probe；
2. tensorized grouped attention / mask construction；
3. 不复制完整 KV cache 的候选维；
4. 只扩展 top-k；
5. 最终报告 real wall-clock。

### P3：标准任务验证

先做小规模 smoke：

1. LongBench 子集；
2. RULER / Needle；
3. WikiText/PG19 PPL；
4. Qwen3-0.6B + 至少一个 LLaMA/Qwen2.5 系模型。

## 参考相关工作

- SparQ Attention: https://proceedings.mlr.press/v235/ribar24a.html
- Quest: https://arxiv.org/abs/2406.10774
- H2O: https://arxiv.org/abs/2306.14048
- StreamingLLM: https://arxiv.org/abs/2309.17453
- Landmark Attention: https://arxiv.org/abs/2305.16300
- SnapKV: https://arxiv.org/abs/2404.14469
- PyramidKV: https://arxiv.org/abs/2406.02069
- Ada-KV: https://arxiv.org/html/2407.11550
- Loki: https://arxiv.org/html/2406.02542
- MInference: https://arxiv.org/abs/2407.02490
- DuoAttention: https://arxiv.org/abs/2410.10819
- Scissorhands: https://proceedings.neurips.cc/paper_files/paper/2023/hash/a452a7c6c463e4ae8fbdc614c6e983e6-Abstract-Conference.html
- Keyformer: https://arxiv.org/abs/2403.09054
