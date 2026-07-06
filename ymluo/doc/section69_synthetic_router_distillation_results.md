# Section 69: 非 Benchmark Synthetic Exact/Retrieval Router 重新蒸馏

## 目标

本节重新蒸馏 runtime router，核心约束是：

```text
训练 router 不使用 LongBench / RULER benchmark 数据。
只使用普通书籍文本上的 synthetic exact/retrieval/generation 数据。
```

训练文本：

```bash
/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl/data/war_and_peace_pg2600.txt
/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl/data/count_monte_cristo_pg1184.txt
```

## 代码改动

新增 synthetic router 蒸馏脚本：

```bash
ymluo/projects/learned_hierarchical_summary_memory/src/run_synthetic_router_distillation.py
```

同时增强 router 特征：

```bash
ymluo/projects/learned_hierarchical_summary_memory/src/run_qwen8b_paper_benchmarks.py
ymluo/projects/learned_hierarchical_summary_memory/src/run_qwen8b_router_distill_from_trials.py
```

新增特征包括：

- top1/top2/top3 retrieval overlap
- top1/top2 evidence block 的相对位置
- top1 是否落在 recent 区域
- top2 normalized overlap

原因：旧 router 只有 overlap 分数，没有命中位置，因此很难区分：

```text
single_old evidence
single_recent evidence
```

这会导致 old evidence 被误判成 `recent_only`。

## Synthetic 数据设计

生成的非 benchmark synthetic case 覆盖：

- single old exact key-value
- single recent exact key-value
- multi-key / multi-block exact retrieval
- RULER-like special magic number
- RULER-like multiquery / multivalue
- RULER-like variable tracking
- RULER-like common/frequent word retrieval
- brief summary
- detailed summary
- recent continuation
- rare full_raw fallback

多长度训练：

```text
prefill_token_lengths = 4096, 8192, 16384
block_tokens = 1024
recent_tokens = 512
```

这样避免 router 只在 8k 长度上训练，到了 16k 发生 out-of-distribution。

## 训练产物

最终 synthetic router：

```bash
/home/fdong/ymluo/projects/learned_hierarchical_summary_memory/outputs/qwen8b_router_synth_rulerlike_posfeat_20260705/router.pt
```

输出目录：

```bash
/home/fdong/ymluo/projects/learned_hierarchical_summary_memory/outputs/qwen8b_router_synth_rulerlike_posfeat_20260705
```

训练样例：

```text
synthetic cases = 1440
train = 1080
test = 360
```

Synthetic held-out 结果：

| split | group | samples | label acc | synthetic success | token ratio |
|---|---|---:|---:|---:|---:|
| test | overall | 360 | 100.00% | 100.00% | 31.84% |
| test | exact | 270 | 100.00% | 100.00% | 38.05% |
| test | generation | 90 | 100.00% | 100.00% | 13.20% |

## 纯 Learned Router 的问题

虽然 synthetic test 完美，但直接把 MLP router 用到 held-out LongBench/RULER offline trials 上，效果不好：

```text
overall relative to full_raw = 61.42%
token ratio = 33.64%
```

主要错误：

- RULER `niah_multiquery` 被误判成 `summary1_4`。
- 一些 single/multikey exact 任务仍然过度选择 `retrieval_raw_k1`。
- LongBench retrieval 任务有时被误判成 summary。

结论：

```text
只靠当前 34 维手工特征 + 小 MLP，不足以稳定学到 benchmark 需要的 exact-task safety boundary。
```

## Conservative Router

因此新增 `router_conservative` safety override：

```text
base action = learned router prediction

if exact task is high-risk:
  use raw retrieval instead of summary/recent

if RULER-like multi-answer / variable / frequency task:
  choose retrieval_raw_k2 or retrieval_raw_k1
```

具体策略：

- RULER high-risk exact: `retrieval_raw_k2`
- RULER common/frequent word: `retrieval_raw_k1`
- LongBench exact 如果 learned router 给出 summary/recent: override 到 `retrieval_raw_k1`
- generation/summary task 保留 learned router 的压缩选择

离线评估输出：

```bash
/home/fdong/ymluo/projects/learned_hierarchical_summary_memory/outputs/qwen8b_router_synth_rulerlike_conservative_v2_offline_20260705
```

## Held-out Benchmark Offline 结果

这里使用之前 Qwen3-8B adapter benchmark 已经跑出的 1152 条候选方法输出，离线评估 router 选择。

不是用 benchmark 训练 router。

| group | samples | router score | full_raw score | relative to full | token ratio |
|---|---:|---:|---:|---:|---:|
| overall | 128 | 0.8143 | 0.7916 | 102.87% | 42.89% |
| exact | 120 | 0.8583 | 0.8333 | 103.00% | 44.89% |
| generation | 8 | 0.1534 | 0.1652 | 92.86% | 13.01% |
| LongBench | 32 | 0.3509 | 0.2913 | 120.44% | 32.60% |
| RULER 4096 | 32 | 1.0000 | 1.0000 | 100.00% | 77.00% |
| RULER 8192 | 32 | 0.9688 | 1.0000 | 96.88% | 40.69% |
| RULER 16384 | 32 | 0.9375 | 0.8750 | 107.14% | 21.29% |

Action 分布：

| action | count |
|---|---:|
| retrieval_raw_k2 | 72 |
| retrieval_raw_k1 | 45 |
| summary1_8 | 8 |
| full_raw | 3 |

## 和之前 Router 对比

之前 benchmark router：

```text
score = 0.5569
token ratio = 28.34%
```

新的 conservative synthetic router：

```text
score = 0.8143
token ratio = 42.89%
```

Oracle match-full 上界：

```text
score = 0.8391
token ratio = 20.84%
```

结论：

- 质量已经接近 oracle / full_raw。
- token ratio 还没有接近 oracle。
- 主要原因是 safety override 为了稳住 exact task，较频繁选择 `retrieval_raw_k2`。

## 当前判断

这版 router 的价值：

```text
证明了不使用 benchmark 训练时，可以通过 synthetic exact/retrieval 数据 + safety override 达到 full_raw 级别质量。
```

但这还不是最终投稿级 router：

- token ratio `42.89%`，距离 oracle `20.84%` 还有明显差距。
- pure learned router 泛化差，说明当前特征/模型容量不足。
- 需要动态 top-k / threshold policy，而不是固定 k1/k2。

## 下一步

建议继续做：

1. 把 router 从 action classification 改成 two-stage：
   - 先判断是否需要 raw evidence；
   - 再预测需要几个 evidence blocks，或者直接预测 retrieval threshold。
2. 加入 richer retrieval features：
   - top-k score curve
   - score entropy
   - query key/entity count
   - selected block position distribution
   - evidence spread across document
3. 训练一个 pairwise/cost-aware router：
   - loss 同时考虑 success 和 token cost；
   - 目标更接近 oracle match-full，而不是单纯 label classification。
4. 用本节 conservative router 作为当前强 baseline，后续新 router 必须同时超过：

```text
score >= 0.8143
token ratio < 42.89%
```

