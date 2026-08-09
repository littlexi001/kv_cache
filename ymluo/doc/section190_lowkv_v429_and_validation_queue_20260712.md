# Section 190: Low-KV Overnight Follow-up, v429, and Validation Queue (2026-07-12)

## 已确认最强可用结果

当前已经完成的 LongBench M100 结果里，两个最值得保留的实际可用点是：

| 方法 | LongBench M100 score | 相对 full M100 | KV ratio | Online speed |
|---|---:|---:|---:|---:|
| v427 Source-Preserving Frontier Routing | 0.3774 | 103.18% | 5.09% | 7.68x |
| v428 v427 + RepoBench source | 0.3813 | 104.24% | 6.68% | 6.81x |

v427 是当前更干净的主方法候选：低 KV、更快、效果已经超过 full M100 baseline。
v428 是质量更强的 Pareto 点：RepoBench 质量提升明显，但 KV 和 online latency 都更高。

## 核心现象

M100 分任务对比显示：不同任务的最好 frontier 不同，直接把参数 overlay 到另一个 base 会破坏 reference/fallback 语义；但用 `__task_sources` 复制完整任务片段，可以保留源 policy 的动作语义。

因此当前最有价值的现象不是“继续扫参数”，而是：

1. fast base 适合稳定低 KV 任务；
2. frontier router 对 QA/code/global summary 的收益不同；
3. source-preserving composition 可以把不同 frontier 的优势组合起来，同时保留 fallback 语义。

## 新候选 v429

v429 使用 v417 fast base，然后按 M100 已完成结果把明显更强的任务切到对应源 policy：

| 任务 | 源 policy |
|---|---|
| qasper, lcc, gov_report, multi_news, triviaqa, repobench-p | v421 |
| hotpotqa, qmsum, samsum, narrativeqa | v397 |
| multifieldqa_en | v424 |
| musique | v413 |
| 其他稳定任务 | v417 base |

基于已完成 M100 的离线组合估计：

| 方法 | 预计 score | 预计 KV ratio | 预计 speed |
|---|---:|---:|---:|
| v429 source-best-frontiers | 0.3920 | 8.71% | 4.35x |

注意：这是由已完成任务结果拼接估算出来的上界式预测，但不是 oracle label，因为每个源 policy 都是实际可运行 policy。真实 M100 已经在后台运行。

## 后台验证队列

服务器上已经启动以下后台实验：

| 实验 | 输出目录 | 目的 |
|---|---|---|
| v427 LongBench M200 | `outputs/riskkv_v19_v427_v417_source_v421_winners_20260712_v427_m200_validate_m200_bDyn_pDyn` | 检查 v427 是否在更大样本上仍稳定 |
| v428 LongBench M200 | `outputs/riskkv_v19_v428_v427_plus_repobench_20260712_v428_m200_validate_m200_bDyn_pDyn` | 检查高质量 Pareto 点稳定性 |
| full_kv LongBench M200 | `outputs/riskkv_full_kv_longbench_m200_20260712` | 给 M200 提供同规模 full baseline |
| v427 RULER M50 b384 | `outputs/riskkv_v427_ruler_m50_b384_20260712` | 检查低预算配置在 4k/8k/16k 合成检索上的表现 |
| full_kv RULER M50 | `outputs/riskkv_full_kv_ruler_m50_20260712` | 给 RULER 提供同规模 full baseline |
| v429 LongBench M100 | `outputs/riskkv_v19_v429_source_best_frontiers_20260712_v429_m100_m100_bDyn_pDyn` | 验证 source-best-frontiers 是否兑现组合收益 |

## 汇总脚本

LongBench M200:

```bash
cd /home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl
/home/fdong/miniconda3/envs/moe/bin/python scripts/summarize_v427_v428_m200_validation_20260712.py
```

RULER:

```bash
cd /home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl
/home/fdong/miniconda3/envs/moe/bin/python scripts/summarize_v427_ruler_validation_20260712.py
```

v429 结果可直接读取：

```bash
cd /home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl
/home/fdong/miniconda3/envs/moe/bin/python scripts/summarize_v427_v428_m200_validation_20260712.py
```

或者单独聚合 `outputs/riskkv_v19_v429_source_best_frontiers_20260712_v429_m100_m100_bDyn_pDyn/task_results.csv`。

## 论文判断

现在比较像论文主线的故事是 Source-Preserving Frontier Routing：

1. 先发现不同任务需要不同 KV frontier；
2. 再证明 naive overlay 会破坏 fallback/reference 语义；
3. 最后提出 source-preserving composition，把每个任务的安全 frontier 作为可组合动作源；
4. 在 LongBench 上达到约 5%-7% KV、6.8x-7.7x online speed，并保持或超过 full baseline；
5. v429 如果兑现，将形成 8%-9% KV、4x+ speed、显著超过 full baseline 的质量优先点。

下一步最关键不是再堆单点分数，而是补充：

1. LongBench M200/full-scale 稳定性；
2. RULER 4k/8k/16k；
3. 多模型验证；
4. ablation：source-preserving vs naive overlay vs single frontier vs fixed budget；
5. 方法公式和 router/source-selection 的泛化版本。
