# Section 76: Qwen3-8B Recent-plus Benchmark 和 Router

## 目标

本节按新的问题定义做一版 Qwen3-8B 小规模正式测试：

```text
recent raw 固定保留；
router 只选择 old context 的回忆粒度；
彻底去掉 recent_only label。
```

## Benchmark 设置

模型：

```bash
/home/fdong/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218
```

Adapter：

```bash
/home/fdong/ymluo/projects/learned_hierarchical_summary_memory/outputs/qwen8b_lora_4k_1ksteps_no_bench_20260705/adapter
```

输出目录：

```bash
/home/fdong/ymluo/projects/learned_hierarchical_summary_memory/outputs/qwen8b_recent_plus_bench_small_20260705
```

覆盖任务：

```text
LongBench:
  hotpotqa, 2wikimqa, musique, passage_retrieval_en,
  passage_count, qasper, gov_report, multi_news

RULER:
  niah_single_1, niah_single_2, niah_multikey_1,
  niah_multiquery, niah_multivalue, vt, cwe, fwe

RULER lengths:
  4096, 8192, 16384
```

每个 task 取 1 个样例：

```text
32 cases total
```

方法：

```text
full_raw
recent_plus_summary1_8
recent_plus_summary1_4
recent_plus_summary1_2
recent_plus_static_hier
recent_plus_retrieval_raw_k1
recent_plus_retrieval_raw_k2
recent_plus_retrieval_raw_k3
recent_plus_retrieval_raw_k4
recent_plus_retrieval_raw_k8
```

运行时间：

```text
约 29 分钟
```

## 方法整体结果

| method | score | token ratio | speedup |
|---|---:|---:|---:|
| full_raw | 0.7918 | 100.00% | 1.00x |
| recent_plus_summary1_8 | 0.4769 | 16.93% | 1.57x |
| recent_plus_summary1_4 | 0.5092 | 27.55% | 1.46x |
| recent_plus_summary1_2 | 0.4806 | 48.71% | 1.28x |
| recent_plus_static_hier | 0.3826 | 13.43% | 1.62x |
| recent_plus_retrieval_raw_k1 | 0.6666 | 24.09% | 1.50x |
| recent_plus_retrieval_raw_k2 | 0.8840 | 33.00% | 1.42x |
| recent_plus_retrieval_raw_k3 | 0.8834 | 38.54% | 1.37x |
| recent_plus_retrieval_raw_k4 | 0.7900 | 43.65% | 1.33x |
| recent_plus_retrieval_raw_k8 | 0.8214 | 59.31% | 1.20x |

关键观察：

```text
recent_plus_retrieval_raw_k2:
  score = 0.8840
  relative = 111.65% full_raw
  token ratio = 33.00%

recent_plus_retrieval_raw_k3:
  score = 0.8834
  relative = 111.57% full_raw
  token ratio = 38.54%
```

这说明：

```text
固定 recent 后，old retrieval k2/k3 是非常强的 baseline。
```

## Recent-plus Oracle

### match-full oracle

定义：

```text
选择达到 full_raw score 的最便宜 recent_plus action。
summary task 使用 full_score - 0.03 slack。
```

结果：

```text
score = 0.8538
full_raw score = 0.7918
relative = 107.82%
token ratio = 22.06%
```

action 分布：

| action | count |
|---|---:|
| recent_plus_summary1_8 | 15 |
| recent_plus_static_hier | 7 |
| recent_plus_retrieval_raw_k1 | 6 |
| recent_plus_retrieval_raw_k2 | 3 |
| recent_plus_summary1_4 | 1 |

按 benchmark：

| group | score | full score | relative | token ratio |
|---|---:|---:|---:|---:|
| LongBench | 0.4150 | 0.2924 | 141.95% | 19.14% |
| RULER 4096 | 1.0000 | 1.0000 | 100.00% | 33.50% |
| RULER 8192 | 1.0000 | 1.0000 | 100.00% | 22.38% |
| RULER 16384 | 1.0000 | 0.8750 | 114.29% | 13.22% |

### best-score oracle

定义：

```text
选择 score 最高的 action；
如果多个 action 同分，选择 token ratio 最低的 action。
```

结果：

```text
score = 0.8869
full_raw score = 0.7918
relative = 112.00%
token ratio = 24.20%
```

action 分布：

| action | count |
|---|---:|
| recent_plus_summary1_8 | 14 |
| recent_plus_static_hier | 6 |
| recent_plus_retrieval_raw_k1 | 6 |
| recent_plus_retrieval_raw_k2 | 3 |
| recent_plus_summary1_2 | 2 |
| recent_plus_summary1_4 | 1 |

这个结果很重要：

```text
去掉 recent_only 后，oracle 仍然很强。
```

而且 token ratio 仍然在 22%-24% 左右，没有因为 recent 固定保留而完全失去压缩优势。

## Router 重新训练

输出目录：

```bash
/home/fdong/ymluo/projects/learned_hierarchical_summary_memory/outputs/qwen8b_recent_plus_router_small_20260705
```

Router checkpoint：

```bash
/home/fdong/ymluo/projects/learned_hierarchical_summary_memory/outputs/qwen8b_recent_plus_router_small_20260705/router.pt
```

训练候选方法：

```text
full_raw
recent_plus_summary1_8
recent_plus_summary1_4
recent_plus_summary1_2
recent_plus_static_hier
recent_plus_retrieval_raw_k1
recent_plus_retrieval_raw_k2
recent_plus_retrieval_raw_k3
recent_plus_retrieval_raw_k4
recent_plus_retrieval_raw_k8
```

注意：

```text
candidate_methods 中没有 recent_only。
```

Oracle label 分布：

| label | count |
|---|---:|
| recent_plus_summary1_8 | 15 |
| recent_plus_retrieval_raw_k1 | 7 |
| recent_plus_static_hier | 6 |
| recent_plus_retrieval_raw_k2 | 3 |
| recent_plus_summary1_4 | 1 |

Router split：

```text
train = 22
test = 10
```

### Router 结果

| split | group | samples | label acc | routed success | score | token ratio |
|---|---|---:|---:|---:|---:|---:|
| train | overall | 22 | 100.00% | 100.00% | 0.9236 | 24.22% |
| test | overall | 10 | 50.00% | 70.00% | 0.7000 | 23.34% |
| test | LongBench | 5 | 20.00% | 60.00% | 0.6000 | 30.66% |
| test | RULER 4096 | 1 | 100.00% | 100.00% | 1.0000 | 20.19% |
| test | RULER 8192 | 2 | 50.00% | 50.00% | 0.5000 | 15.74% |
| test | RULER 16384 | 2 | 100.00% | 100.00% | 1.0000 | 14.22% |

解释：

```text
这版 router 训练集太小，不能作为最终 router 结论。
```

但它验证了两件事：

1. `recent_only` 已经彻底从 label space 中移除。
2. 新 action space 的 oracle 本身仍然强。

## 关键发现

### 1. recent 固定保留后，oracle 仍然很强

旧担心是：

```text
recent 必选会显著拉高 token ratio。
```

但小跑结果显示：

```text
match-full oracle:
  107.82% full_raw
  22.06% token

best-score oracle:
  112.00% full_raw
  24.20% token
```

这个结果非常适合论文叙事：

```text
保留 recent 不会破坏压缩收益；
它让方法定义更合理、更接近真实生成。
```

### 2. k2/k3 是强 baseline

固定方法里：

```text
recent_plus_retrieval_raw_k2:
  111.65% full_raw / 33.00% token

recent_plus_retrieval_raw_k3:
  111.57% full_raw / 38.54% token
```

这说明在 recent-plus 设定下，简单固定 k2 已经很强。

后续 router 的目标应该是：

```text
接近 k2 的质量；
但在简单任务上切到 summary/static/k1，进一步省 token。
```

### 3. Oracle 更偏 summary/static，而不只是 retrieval

Oracle 选择最多的是：

```text
recent_plus_summary1_8
```

这说明：

```text
很多任务不需要 raw old evidence；
recent + coarse old summary 就够。
```

这和你的“根据任务难度选择不同粒度”一致。

## 当前结论

这个方向值得继续扩大。

当前最重要的结果是：

```text
recent-plus oracle:
  107.82% - 112.00% full_raw
  22.06% - 24.20% token
```

这比旧 action space 更干净，也不再依赖 `recent_only`。

但 router 还不能下结论，因为：

```text
只有 32 个 cases；
test 只有 10 个 samples；
label 分布不均衡。
```

## 下一步

建议下一步正式扩大到：

```text
max_examples_per_task = 4 或 6
same recent_plus action space
Qwen3-8B + adapter
```

然后重新训练：

```text
recent-required old-memory router
```

并报告：

```text
1. full_raw baseline
2. fixed recent_plus_retrieval_raw_k2
3. recent-plus oracle
4. learned recent-plus router
5. token ratio / speedup
6. action distribution
```

短期可写进论文的 baseline：

```text
recent_plus_retrieval_raw_k2:
  111.65% full_raw
  33.00% token
  1.42x speedup
```

这个 baseline 已经比之前很多 runtime router 更干净、更强。
