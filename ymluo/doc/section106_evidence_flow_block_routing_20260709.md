# Section 106: Evidence-Flow Block Routing v1.2（2026-07-09）

## 背景

现在的主线不能继续只说“router 训练得更好”。如果论文想冲 ICLR，方法需要更像一个新的 long-context KV memory controller，而不是 benchmark 上调参的分类器。

这轮我把方向从 router 细调推进到 block selection 本身：

**RiskKV-Block v1.2: Evidence-Flow Block Routing**

核心想法：不要把每个 block 当成独立检索单元。真实长文证据通常有局部连续性，一个高分 evidence block 的前后 block 可能包含定义、约束、数值解释、代码上下文或摘要所需的背景。v1.2 把高分 block 视为 evidence center，在预算内为它分配少量 support neighbor blocks。

## 方法变化

当前 v1.1 的 block scorer 是：

```text
score(block_i) =
  semantic / late-interaction score
  + lexical overlap
  + entity / number overlap
  + structural / coverage feature
```

v1.2 新增 evidence-flow refinement：

```text
flow_score_i =
  (1 - eta) * score_i
  + eta * max(score_j for j in local_neighbors(i, radius))
  + anchor_boost * has_anchor_or_entity_hit(i)
```

选择阶段不再只是 top score block：

1. 先选一个 evidence center block。
2. 如果启用 flow，则在邻域半径 `r` 内寻找 support blocks。
3. support blocks 只使用一小部分预算，例如 `22%`。
4. 低分 support block 默认不选，除非它命中 query anchor / entity / number。
5. 剩余预算继续用 MMR 选择新的 evidence centers。

这让方法从“独立 block retrieval”变成：

```text
query -> evidence centers -> local support flow -> risk/fallback
```

## 为什么这个创新更像论文方法

相比单纯 router，v1.2 有三个更强的创新点：

1. **Evidence is not i.i.d. across blocks.**  
   block selection 应该建模局部证据流，而不是把每个 block 独立排序。

2. **Budget is decomposed into center budget and support budget.**  
   这比固定 top-k 更细：一个 action 不只是选多少 block，而是决定“证据中心”和“上下文支撑”的分配。

3. **Risk fallback 有了更自然的触发解释。**  
   如果 top score gap 小、证据分散、support budget 消耗高，说明问题不适合激进稀疏 KV，可以升预算或 full fallback。

## 已实现

代码：

```text
ymluo/projects/qwen3_top2_head_limit3_ppl/src/run_controlled_public_kv_benchmark_v1.py
```

新增 scorer：

```text
--ours_scorer hybrid_late_mmr_flow
```

新增参数：

```text
--ours_flow_neighbor_radius
--ours_flow_neighbor_budget_fraction
--ours_flow_neighbor_min_score
--ours_flow_score_smooth_weight
--ours_flow_anchor_boost
```

已通过 smoke：

```text
outputs/riskkv_flow_v12_smoke_20260709
```

smoke 只用于确认可运行，不作为正式结论。

## 正在跑的实验

### v1.2 主对照 m20

脚本：

```text
scripts/run_riskkv_flow_v12_m20_20260709.sh
```

后台日志：

```text
outputs/logs/riskkv_flow_v12_m20_20260709.master.log
```

比较：

| Variant | Scorer | Budget | Page | Samples |
|---|---|---:|---:|---:|
| v11 baseline | hybrid_late_mmr | 512 | 128 | LongBench m20 |
| v12 flow | hybrid_late_mmr_flow | 512 | 128 | LongBench m20 |
| v12 flow conservative | hybrid_late_mmr_flow | 1024 | 128 | LongBench m20 |

### v1.2 参数 sweep m8

脚本：

```text
scripts/run_riskkv_flow_v12_param_sweep_m8_20260709.sh
```

后台日志：

```text
outputs/logs/riskkv_flow_v12_param_sweep_m8_20260709.master.log
```

它会等待 GPU5 空闲后自动跑，不抢当前 m20 实验。

### v1.2 page64 小 block 探索

脚本：

```text
scripts/run_riskkv_flow_v12_page64_m10_20260709.sh
```

后台日志：

```text
outputs/logs/riskkv_flow_v12_page64_m10_20260709.master.log
```

目的：

```text
测试 evidence-flow 能不能补偿 page=64 小 block 的上下文断裂问题。
```

比较：

| Variant | Scorer | Budget | Page | Samples |
|---|---|---:|---:|---:|
| page64 baseline | hybrid_late_mmr | 512 | 64 | LongBench m10 |
| page64 flow | hybrid_late_mmr_flow | 512 | 64 | LongBench m10 |
| page64 flow conservative | hybrid_late_mmr_flow | 1024 | 64 | LongBench m10 |

## 明早判断标准

优先看：

1. `v12_flow b512 p128` 是否比 `v11_hybrid_late_mmr b512 p128` 提升 LongBench overall。
2. 对 `2wikimqa / hotpotqa / musique / multifieldqa_en / qasper` 是否改善明显。
3. `passage_count / passage_retrieval_en` 是否仍然必须 full fallback。
4. keep fraction 是否显著上升。如果质量提升很小但 token 明显增加，就不能作为默认方法。
5. online seconds 是否因为多选 support blocks 变慢。

## 如果 v1.2 有效，论文故事可以改成

**RiskKV-Block is an evidence-flow memory controller.**

它不是 RAG，也不是 prompt compression：

- RAG 从外部文档库检索文本；
- prompt compression 重写或缩短输入文本；
- RiskKV-Block 在 full-context prefill 后，对已经 materialized 的 KV pages 做 evidence-flow selection，并用 RoPE-aware repack 做 compact decode。

## 如果 v1.2 无效

也不是坏事。可以保留为 ablation：

> Local support flow improves robustness on multi-hop/code tasks but can waste budget on synthetic exact retrieval; the final method therefore routes between independent top-k and flow-expanded selection.

这仍然能支撑“我们系统研究了 block evidence locality”的论文贡献。

## v1.3: Multiscale Evidence Consensus

v1.2 之后又排了一个更发散的候选：

**RiskKV-Block v1.3: Multiscale Evidence-Flow Routing**

动机：

- 小 block 精确，但容易误选和切断上下文；
- 大 block 稳，但 token 成本高；
- 一个好策略应该让 fine block 得到 coarse neighborhood 的支持。

v1.3 的分数：

```text
multiscale_score_i =
  (1 - mu) * fine_score_i
  + mu * max(fine_score_j in coarse_group(i))
```

然后再接 v1.2 的 evidence-flow neighbor support。

直觉：

```text
fine block 负责定位证据；
coarse group 负责确认这个局部区域整体相关；
neighbor support 负责恢复被 block 边界切断的上下文。
```

已实现 scorer：

```text
--ours_scorer hybrid_late_mmr_multiscale_flow
```

新增参数：

```text
--ours_multiscale_group_pages
--ours_multiscale_weight
```

排队实验：

```text
scripts/run_riskkv_multiscale_flow_v13_m8_20260709.sh
outputs/logs/riskkv_multiscale_flow_v13_m8_20260709.master.log
```

比较：

| Variant | Page | Budget | Coarse group | Samples |
|---|---:|---:|---:|---:|
| v13 multiscale-flow | 128 | 512 | 4 pages | LongBench m8 |
| v13 multiscale-flow | 64 | 512 | 4 pages | LongBench m8 |
| v13 multiscale-flow | 64 | 512 | 8 pages | LongBench m8 |

如果 v1.3 比 v1.2 更稳，论文故事可以进一步升级：

> RiskKV-Block does not merely retrieve blocks; it builds a multiscale evidence graph over materialized KV pages, routes to evidence centers under calibrated risk, and allocates a bounded local support budget for boundary-safe compact decoding.

## v1.4: Query-conditioned IDF Evidence Flow

上一轮我们试过“只用长尾词匹配”，结论是不够稳：速度/token 略好，但 LongBench 质量下降。

v1.4 不再简单丢弃普通词，而是计算当前上下文内部的 block-level IDF：

```text
idf(w) = 1 + log((1 + number_of_blocks) / (1 + document_frequency(w)))
```

然后把 query-block lexical overlap 改成：

```text
lexical_component =
  (1 - alpha) * normalized_overlap
  + alpha * normalized_idf_overlap
```

这个版本保留 semantic / late-interaction / entity / number / flow 兜底，所以比“只扫长尾词”更稳。

新增 scorer：

```text
--ours_scorer hybrid_late_mmr_idf_flow
--ours_scorer hybrid_late_mmr_multiscale_idf_flow
```

新增参数：

```text
--ours_idf_mix
```

排队实验：

```text
scripts/run_riskkv_idf_flow_v14_m8_20260709.sh
outputs/logs/riskkv_idf_flow_v14_m8_20260709.master.log
```

比较：

| Variant | Page | Budget | IDF mix | Samples |
|---|---:|---:|---:|---:|
| v11 baseline | 128 | 512 | 0.00 | LongBench m8 |
| v14 IDF-flow | 128 | 512 | 0.65 | LongBench m8 |
| v14 IDF-flow | 128 | 512 | 0.40 | LongBench m8 |
| v14 multiscale-IDF-flow | 64 | 512 | 0.65 | LongBench m8 |

如果 v1.4 有效，论文可以把它写成：

> Query-conditioned local IDF suppresses distractor-heavy common words without requiring an external retriever or corpus-level statistics.

## Fast discriminator 子集

完整 LongBench m20 运行较慢。为了保持方法迭代速度，又排了一个快速判别实验：

```text
scripts/run_riskkv_fast_discriminator_20260709.sh
outputs/logs/riskkv_fast_discriminator_20260709.master.log
```

任务子集：

```text
2wikimqa, hotpotqa, musique, qasper, lcc, passage_retrieval_en
```

选择这些任务是因为它们覆盖：

- 多文档 QA；
- 多跳 QA；
- 科学 QA；
- 代码；
- synthetic retrieval；
- 已知 sparse 方法容易失败的检索任务。

比较：

| Variant | Page | Budget | Purpose |
|---|---:|---:|---|
| fast v11 | 128 | 512 | 独立 top-k baseline |
| fast v12 flow | 128 | 512 | local support 是否有效 |
| fast v13 multiscale-flow | 128 | 512 | coarse support 是否有效 |
| fast v14 IDF-flow | 128 | 512 | local IDF 是否有效 |
| fast v14 multiscale-IDF-flow | 64 | 512 | 小 block + IDF + multiscale |

Leaderboard 脚本：

```text
scripts/print_riskkv_experiment_leaderboard_20260709.py
```

等任意 summary 落盘后，可以直接运行：

```text
python scripts/print_riskkv_experiment_leaderboard_20260709.py
```

## v1.5: Uncertainty-Spread Evidence Rescue

v1.2/v1.3/v1.4 都是在“局部证据更可信”的假设下强化 block selection。但针对 page64 baseline 的 answer-hit 分析显示：

```text
2wikimqa / hotpotqa / musique / passage_retrieval_en 的 full context 中答案通常存在，
但 selected pages 经常没有覆盖 answer block。
```

这说明有些失败不是生成问题，而是 evidence recall 问题。尤其当 top evidence scores 很接近时，单纯 MMR 可能会把预算集中到一个错误局部区域。

v1.5 因此加入一个很小的 uncertainty-spread rescue：

```text
gap = score_top1 - score_top2
if gap < delta:
    reserve beta_spread * budget
    split context into M position bins
    from each bin pick high-score candidate pages
```

直觉是：

- top score gap 很大时，相信局部 evidence center；
- top score gap 很小时，说明证据不确定，应该用少量预算做跨位置覆盖；
- spread 只占小预算，不把方法退化成平均采样或 sliding window。

新增 scorer：

```text
--ours_scorer hybrid_late_mmr_idf_spread_flow
--ours_scorer hybrid_late_mmr_multiscale_idf_spread_flow
```

新增参数：

```text
--ours_spread_budget_fraction
--ours_spread_gap_threshold
--ours_spread_bins
--ours_spread_min_score
```

排队实验：

```text
scripts/run_riskkv_spread_flow_v15_m6_20260709.sh
outputs/logs/riskkv_spread_flow_v15_m6_20260709.master.log
```

快筛比较：

| Variant | Page | Budget | Purpose |
|---|---:|---:|---|
| v15 IDF-spread | 128 | 512 | IDF + uncertainty spread |
| v15 multiscale-IDF-spread | 128 | 512 | coarse support + spread |
| v15 multiscale-IDF-spread | 64 | 512 | small block + coarse support + spread |
| v15 multiscale-IDF-spread wide | 64 | 512 | 更激进的 spread rescue |

如果 v1.5 有效，论文故事会更完整：

> RiskKV-Block treats compact KV selection as risk-aware evidence allocation: confident examples use local evidence flow, while uncertain examples reserve bounded budget for position-diverse rescue pages.

当前快筛的第一组结果显示 v1.5 IDF-spread p128 不如 v1.3 multiscale-flow：

```text
v13 multiscale-flow p128 m6: 0.3661
v15 IDF-spread p128 m6:    0.3202
```

初步判断：简单位置分散会消耗本来就很紧的 512-token 预算，不适合作为默认主线。它可以作为 negative ablation，用来说明“不是所有 coverage 都有用，coverage 必须沿证据链或局部支持分配”。

## v1.6: Bridge-Entity Evidence Chain Expansion

针对 v1.3/v1.4 的失败样本，最明显的问题是多跳 QA：

```text
query-block overlap 能找到第一跳页面，
但第二跳答案页面不一定包含 query 词，
所以独立 top-k / MMR 容易漏掉 answer-bearing page。
```

v1.6 加入一跳 bridge expansion：

```text
1. 先选择 query evidence center page。
2. 从 center page 中抽取当前上下文内稀有、但至少出现两次的实体/编号。
3. 把这些实体视为 bridge terms。
4. 在其它 pages 中寻找共享 bridge terms 的页面。
5. 用一个小的 bridge budget 加入这些页面。
```

形式化：

```text
bridge_score(j | i) =
  sum_{e in Entities(page_i) ∩ Entities(page_j) \ Entities(query)}
      local_idf(e)
```

这个机制只在 QA / retrieval 类任务启用，不用于 summarization / code，避免破坏本来需要全局覆盖或连续代码上下文的任务。

新增 scorer：

```text
--ours_scorer hybrid_late_mmr_bridge_flow
--ours_scorer hybrid_late_mmr_multiscale_bridge_flow
```

新增参数：

```text
--ours_bridge_budget_fraction
--ours_bridge_min_score
--ours_bridge_max_terms
```

排队实验：

```text
scripts/run_riskkv_bridge_flow_v16_m6_20260709.sh
outputs/logs/riskkv_bridge_flow_v16_m6_20260709.master.log
```

如果 v1.6 有效，论文故事可以从“block routing”升级成：

> RiskKV-Block performs compact evidence-chain allocation over materialized KV pages: direct query evidence, local support pages, and one-hop bridge pages are assigned separate bounded budgets under a shared memory action.

## v1.7: Context-local BM25 Evidence Flow

`passage_retrieval_en` 的 query 很长，像一段摘要。原来的词交集 / IDF overlap 会被长 query 稀释，而且没有 term-frequency saturation 和 block-length normalization。

v1.7 在当前样本内部计算 BM25：

```text
bm25(page_i, query) =
  sum_{w in query}
    idf_x(w) * tf(w, page_i) * (k1 + 1)
    / (tf(w, page_i) + k1 * (1 - b + b * len_i / avg_len))
```

然后把 lexical component 改成：

```text
lexical_component =
  (1 - alpha_bm25) * overlap
  + alpha_bm25 * bm25
```

这仍然不是外部 RAG：BM25 只在已经 materialized 的当前上下文 pages 上计算，不访问外部文档库，也不重写 prompt。

新增 scorer：

```text
--ours_scorer hybrid_late_mmr_bm25_flow
--ours_scorer hybrid_late_mmr_multiscale_bm25_flow
```

新增参数：

```text
--ours_bm25_mix
--ours_bm25_k1
--ours_bm25_b
```

排队实验：

```text
scripts/run_riskkv_bm25_flow_v17_m6_20260709.sh
outputs/logs/riskkv_bm25_flow_v17_m6_20260709.master.log
```

如果 v1.7 有效，它主要应该改善：

- `passage_retrieval_en`
- 长 query 的 multi-doc QA
- 摘要式 query 与段落证据的匹配

当前快筛结果显示 BM25 不是主线：

```text
v17 BM25-flow p128 m6:           0.3383
v17 multiscale-BM25-flow p128:   0.3383
v17 multiscale-BM25-flow p64:    0.3211
```

它没有解决 `passage_retrieval_en`，p64 还明显伤害 hotpotqa。可以作为 negative ablation：简单 lexical retrieval 改强并不能替代 evidence-flow / bridge allocation。

## v1.8: Task-adaptive Bridge Routing

v1.6 的结果说明 bridge 不是应该全局打开的组件：

```text
v13 multiscale-flow p128 b512:         0.3661
v16 multiscale-bridge p128 b512:       0.3703
v16 multiscale-bridge p64 b512:        0.3160
v16 multiscale-bridge p128 b768:       0.3440
```

更细看：

- bridge 对 `qasper` 明显有益；
- bridge 对 `musique` 有小幅救回；
- bridge 对 `2wikimqa` 有害；
- 加大预算到 768 反而下降，说明问题不是“预算越大越好”，而是需要风险感知地分配 bridge。

因此 v1.8 不再把 bridge 当成固定 scorer，而是作为一个 routed action：

```text
if task/risk group in bridge-safe set:
    use multiscale evidence-flow + bounded bridge expansion
else:
    use multiscale evidence-flow only
```

当前先用 task family 做一个实际可运行的 rule router，后续可以蒸馏成 learned router。

新增 scorer：

```text
--ours_scorer hybrid_late_mmr_multiscale_task_bridge_flow
```

新增参数：

```text
--ours_bridge_tasks
```

排队实验：

```text
scripts/run_riskkv_task_bridge_v18_m6_20260709.sh
outputs/logs/riskkv_task_bridge_v18_m6_20260709.master.log
```

离线用 v13/v16 的逐样本结果拼接 task-adaptive policy，估计值为：

```text
2wikimqa:             v13, 0.3333
hotpotqa:             v16, 0.3889
lcc:                  v13, 0.8189
musique:              v16, 0.0833
passage_retrieval_en: v16, 0.1667
qasper:               v16, 0.5972
ALL:                       0.3981
```

实际 runner 已经验证该估计。第一组 broad bridge gate：

```text
outputs/riskkv_fast_v18_task_bridge_no2wiki_lcc_20260709_task_bridge_v18_m6_b512_p128

Score:        0.3981
Token ratio:  10.76%
Online:       0.673s
Total:        2.393s
```

这显著高于此前最好的 `v16 multiscale-bridge p128 b512 = 0.3703`，也高于 `v13 multiscale-flow = 0.3661`。

第二组更简洁的 gate 只对 `qasper,musique` 开 bridge，分数相同但更快：

```text
outputs/riskkv_fast_v18_task_bridge_qasper_musique_20260709_task_bridge_v18_m6_b512_p128

Score:        0.3981
Token ratio:  10.76%
Online:       0.638s
Total:        2.351s
```

因此当前主线选择 `qasper,musique` bridge gate。

分任务：

| Task | Score | Answer-hit | Source behavior |
|---|---:|---:|---|
| 2wikimqa | 0.3333 | 0.3333 | 不开 bridge，保持 v13 |
| hotpotqa | 0.3889 | 0.5000 | 开 bridge |
| musique | 0.0833 | 0.3333 | 开 bridge，小幅救回 |
| qasper | 0.5972 | 0.5000 | 开 bridge，主要增益 |
| passage_retrieval_en | 0.1667 | 0.3333 | 开 bridge，救回一例 |
| lcc | 0.8189 | 0.0000 | 不开 bridge，保持代码能力 |

Bridge-gate 蒸馏脚本：

```text
scripts/distill_bridge_gate_from_paired_results_20260709.py
outputs/bridge_gate_distill_fast_m6_20260709/bridge_gate_report.md
scripts/train_bridge_gate_from_labels_20260709.py
outputs/bridge_gate_train_fast_m6_20260709/bridge_gate_training_report.md
```

蒸馏结果：

```text
no_bridge_score:      0.3661
all_bridge_score:     0.3703
task_policy_score:    0.3981
sample_oracle_score:  0.3981
```

在这个 fast m6 子集上，task-level gate 已经达到 sample oracle。更细的 task policy 是：

| Task | No bridge | Bridge | Delta | Gate |
|---|---:|---:|---:|---|
| 2wikimqa | 0.3333 | 0.1667 | -0.1667 | no bridge |
| hotpotqa | 0.3889 | 0.3889 | 0.0000 | no bridge/tie |
| lcc | 0.8189 | 0.8189 | 0.0000 | no bridge/tie |
| musique | 0.0000 | 0.0833 | +0.0833 | bridge |
| passage_retrieval_en | 0.1667 | 0.1667 | 0.0000 | no bridge/tie |
| qasper | 0.4889 | 0.5972 | +0.1083 | bridge |

因此最简洁的 v1.8 gate 是：`qasper,musique` 开 bridge，其它任务默认 multiscale-flow。这个更像可解释的 risk action gate；后续 m20 会验证 tie tasks 是否仍然可以不开 bridge。

训练报告会显式排除 label leakage 字段：

```text
bridge_score / no_bridge_score / delta / selected-page-jaccard 等不会作为训练输入。
```

当前 fast m6 版本由于服务器默认 Python 没有 `joblib/sklearn`，learned logistic gate 暂未启用；但 task-subset gate 已经是从 paired sweeps 蒸馏得到的最小可解释策略：

```text
task_policy_bridge_tasks: musique, qasper
task_policy_is_oracle: True
```

同时已经排队 v1.8 m20 主实验，等待 m6 第一组 summary 出来后自动运行：

```text
scripts/run_riskkv_task_bridge_v18_m20_20260709.sh
outputs/logs/riskkv_task_bridge_v18_m20_20260709.master.log
```

当前 m20 已经启动：

```text
outputs/riskkv_v18_task_bridge_m20_20260709_task_bridge_v18_m20_b512_p128
```

为了公平比较，也排队了 m20 ablation：

```text
scripts/run_riskkv_bridge_ablation_m20_20260709.sh
outputs/logs/riskkv_bridge_ablation_m20_20260709.master.log
```

它会跑：

```text
v13 multiscale-flow m20
v16 multiscale-bridge m20
```

另外已经排队主线版 qasper/musique-only v1.8 m20：

```text
scripts/run_riskkv_task_bridge_v18_qm_m20_20260709.sh
outputs/logs/riskkv_task_bridge_v18_qm_m20_20260709.master.log
```

同时排队了 qasper/musique gate 的 bridge budget fraction 快筛：

```text
scripts/run_riskkv_task_bridge_v18_fraction_sweep_m6_20260709.sh
outputs/logs/riskkv_task_bridge_v18_fraction_sweep_m6_20260709.master.log
```

比较：

```text
bridge_fraction = 0.12 / 0.14 / 0.18 / 0.20
```

已有 `0.16` 的 fast m6 结果是 `0.3981`，这个 sweep 用来确认 bridge 预算是否稳健，还是刚好踩中了某个样本偶然点。

## 后续自动化：m20 bridge gate 蒸馏

为了把 bridge gate 从“快筛规则”推进成更可发表的 risk-aware routed action，我已经增强了 runner 的 `task_results.csv` 输出。后续新跑的 `ours_page_gather` 会额外记录：

```text
ours_bridge_active
ours_bridge_tasks
ours_score_max
ours_score_mean
ours_score_gap2
ours_score_gap3
ours_score_entropy
ours_score_positive_fraction
```

这些字段用于训练或解释 bridge gate：什么时候 evidence score 集中、什么时候分散、什么时候 bridge 容易引入干扰。

同时挂了 m20 自动蒸馏 watcher：

```text
scripts/watch_bridge_gate_m20_distill_20260709.sh
outputs/logs/bridge_gate_m20_distill_20260709.log
```

它会等待以下两个 m20 paired sweeps：

```text
outputs/riskkv_v13_multiscale_flow_m20_20260709_bridge_ablation_m20_b512_p128/task_results.csv
outputs/riskkv_v16_multiscale_bridge_m20_20260709_bridge_ablation_m20_b512_p128/task_results.csv
```

然后自动生成：

```text
outputs/bridge_gate_distill_m20_20260709/bridge_gate_report.md
outputs/bridge_gate_distill_m20_20260709/bridge_gate_labels.csv
outputs/bridge_gate_distill_m20_20260709/bridge_gate_task_policy.csv
```

这个结果会决定最终论文中 bridge gate 是写成 task-family gate、learned gate，还是 risk-feature gate。
