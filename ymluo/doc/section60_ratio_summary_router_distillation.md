# Section 60: ratio summary oracle 标签与 runtime router 蒸馏

## 目标

本节基于上一轮 `Qwen3-0.6B` 的 ratio-summary 实验结果，直接生成 oracle policy label，并蒸馏成推理时可以使用的小 router。

本次重点验证：

- `sum1/8`、`sum1/4`、`sum1/2` 是否应该作为可选动作，而不是固定拼接。
- oracle 是否会在不同任务上自动选择不同粒度的 summary 或 raw retrieval。
- 蒸馏出来的小 router 是否能在 runtime benchmark 路径里正常加载并实际路由。

## 输入实验结果

ratio-summary 原始结果目录：

```bash
/home/fdong/ymluo/projects/learned_hierarchical_summary_memory/outputs/qwen06b_ratio_summary_bad_tasks_20260704
```

使用的模型：

```bash
/home/fdong/hrj/prove/Qwen3-0.6B
```

候选动作：

```text
full_raw
summary1_8
summary1_4
summary1_2
summary1000
static_hier
retrieval_raw_k2
```

共 100 个 case，每个 case 都有 7 个候选方法的结果。

## oracle label 生成规则

对每个 case，先看所有候选动作的分数。

对于 exact-match 类任务：

- 如果存在满分动作，则 oracle 从满分动作里选择 token 开销最小的动作。
- 如果没有满分动作，则选择达到该 case 最高分的动作，并在其中选 token 开销最小者。

对于摘要类任务：

- 允许与 `full_raw` 有一个很小的 Rouge-L slack。
- 在分数接近 `full_raw` 的候选动作中选择 token 开销最小者。

因此 oracle 的含义不是“准确率最高的动作”，而是：

```text
在保持该样例可接受分数的前提下，选择最省 token 的 memory policy。
```

## oracle 标签分布

100 个 case 的 oracle label 分布如下：

| oracle 动作 | 数量 | 比例 |
|---|---:|---:|
| retrieval_raw_k2 | 34 | 34% |
| summary1_8 | 30 | 30% |
| static_hier | 20 | 20% |
| full_raw | 8 | 8% |
| summary1000 | 4 | 4% |
| summary1_2 | 2 | 2% |
| summary1_4 | 2 | 2% |

这个结果支持目前的设想：不同任务确实需要不同策略。`summary1_8` 和 `retrieval_raw_k2` 都被大量选择，说明“极低成本摘要”和“少量 raw retrieval”都有用；同时仍然有 8% 的 case 需要回退到 `full_raw`。

## router 蒸馏

蒸馏脚本：

```bash
ymluo/projects/learned_hierarchical_summary_memory/src/run_qwen8b_router_distill_from_trials.py
```

训练命令：

```bash
python ymluo/projects/learned_hierarchical_summary_memory/src/run_qwen8b_router_distill_from_trials.py \
  --benchmark_output_dir ymluo/projects/learned_hierarchical_summary_memory/outputs/qwen06b_ratio_summary_bad_tasks_20260704 \
  --output_dir ymluo/projects/learned_hierarchical_summary_memory/outputs/qwen06b_ratio_summary_router_distill_20260704 \
  --candidate_methods full_raw,summary1_8,summary1_4,summary1_2,summary1000,static_hier,retrieval_raw_k2 \
  --hidden_dim 48 \
  --epochs 1500 \
  --lr 0.002 \
  --weight_decay 0.0001 \
  --test_fraction 0.35 \
  --seed 2026070405
```

router checkpoint：

```bash
/home/fdong/ymluo/projects/learned_hierarchical_summary_memory/outputs/qwen06b_ratio_summary_router_distill_20260704/router.pt
```

离线蒸馏评估：

| split | samples | label acc | routed success | avg score | avg token ratio |
|---|---:|---:|---:|---:|---:|
| train | 65 | 93.85% | 83.08% | 83.08% | 32.08% |
| test | 35 | 51.43% | 54.29% | 54.29% | 33.16% |

测试集 label accuracy 不高，说明当前数据量太小，router 还不是一个稳定泛化模型；但 routed success 与 token ratio 已经说明它能学到一部分任务难度和策略选择信号。

## runtime 回放

runtime benchmark 输出目录：

```bash
/home/fdong/ymluo/projects/learned_hierarchical_summary_memory/outputs/qwen06b_ratio_summary_runtime_router_20260704
```

命令核心参数：

```bash
--methods router
--router_path ymluo/projects/learned_hierarchical_summary_memory/outputs/qwen06b_ratio_summary_router_distill_20260704/router.pt
```

真实 runtime 结果需要和上一轮 `full_raw` 按 case 合并后计算 token ratio，因为单独跑 `router` 时输出目录里没有同批 `full_raw` 基线。

合并后的总体结果：

| 方法 | score | token ratio vs full_raw | 实测生成时间 speedup |
|---|---:|---:|---:|
| full_raw baseline | 65.00% | 100.00% | 1.00x |
| distilled runtime router | 73.00% | 32.46% | 1.19x |

runtime router 的动作分布：

| routed action | 数量 | 比例 |
|---|---:|---:|
| summary1_8 | 30 | 30% |
| retrieval_raw_k2 | 25 | 25% |
| static_hier | 24 | 24% |
| summary1000 | 8 | 8% |
| full_raw | 8 | 8% |
| summary1_4 | 4 | 4% |
| summary1_2 | 1 | 1% |

按 benchmark 分组：

| benchmark | samples | score | token ratio | speedup |
|---|---:|---:|---:|---:|
| LongBench 子集 | 20 | 35.00% | 38.15% | 1.16x |
| RULER 8k | 40 | 82.50% | 32.19% | 1.15x |
| RULER 16k | 40 | 82.50% | 29.88% | 1.24x |

## 结论

这次结果比固定 ratio summary 更好。固定 `summary1_8`、`summary1_4`、`summary1_2` 的整体分数分别只有 28%、30%、35%，但 router 在同一组任务上达到 73%，同时平均只用 32.46% 的输入 token。

这说明当前方向更合理：

```text
不要固定使用某个 summary 粒度，也不要简单拼接所有 summary；
应该把 summary 粒度、raw retrieval、full raw fallback 都放进 action space，
由 query-aware router 按任务难度选择。
```

当前主要局限：

- 数据量只有 100 个 case，test label accuracy 仍然偏低。
- router 是在同一组 benchmark 结果上蒸馏，尚不能说明跨任务泛化。
- 速度提升只有 1.19x，低于 token 降幅，因为当前实现仍是 prompt-level 压缩，没有 CUDA/KV cache kernel 级优化。
- `summary1_8` 的 oracle 比例很高，但在部分多 needle 任务上容易丢关键信息，后续需要更强 summarizer 或 raw fallback。

下一步更适合做：

1. 用 Qwen3-8B 的 LongBench/RULER ratio-summary 结果重新生成 oracle。
2. 扩大每个任务的样例数，避免 0.6B 小样本 router 过拟合。
3. 把 router action space 改成更明确的分层动作：`summary_ratio_only`、`retrieval_raw_k`、`summary_plus_recent_raw`、`full_raw_fallback`。
4. 在最终论文实验里同时报告 oracle、distilled router、fixed summary、fixed retrieval 和 full raw。
