# Section 77: Qwen3-8B Recent-plus 多卡小规模扩展实验

## 目标

本节把 Section 76 的 recent-plus 设置从每个任务 1 个样例扩展到每个任务 4 个样例，并用多卡并行跑不同 shard。

核心策略不变：

```text
recent raw 固定保留。
router 只选择 old context 的记忆粒度或者检索预算。
去掉 recent_only label。
```

## 多卡运行设置

服务器：

```bash
fdong@10.176.37.31
```

模型：

```bash
/home/fdong/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218
```

LoRA adapter：

```bash
/home/fdong/ymluo/projects/learned_hierarchical_summary_memory/outputs/qwen8b_lora_4k_1ksteps_no_bench_20260705/adapter
```

输出目录：

```bash
/home/fdong/ymluo/projects/learned_hierarchical_summary_memory/outputs/qwen8b_recent_plus_bench_m4_parallel_20260706
```

合并后的结果目录：

```bash
/home/fdong/ymluo/projects/learned_hierarchical_summary_memory/outputs/qwen8b_recent_plus_bench_m4_parallel_20260706/merged
```

分片：

| shard | GPU | 内容 | cases |
|---|---:|---|---:|
| longbench_exact | 4 | hotpotqa, 2wikimqa, musique, passage_retrieval_en, passage_count, qasper | 24 |
| longbench_summary | 5 | gov_report, multi_news | 8 |
| ruler_4k8k | 6 | 8 个 RULER task, 4096 和 8192 | 64 |
| ruler_16k | 7 | 8 个 RULER task, 16384 | 32 |

合并后：

```text
cases = 128
trials = 1280
methods = 10
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

## 整体结果

| method | score | relative | token ratio | speedup |
|---|---:|---:|---:|---:|
| recent_plus_retrieval_raw_k3 | 0.8213 | 103.90% | 38.50% | 1.35x |
| recent_plus_retrieval_raw_k4 | 0.8057 | 101.93% | 43.46% | 1.30x |
| recent_plus_retrieval_raw_k2 | 0.7981 | 100.97% | 32.77% | 1.40x |
| recent_plus_retrieval_raw_k8 | 0.7980 | 100.95% | 58.12% | 1.19x |
| full_raw | 0.7905 | 100.00% | 100.00% | 1.00x |
| recent_plus_retrieval_raw_k1 | 0.6113 | 77.33% | 23.96% | 1.49x |
| recent_plus_summary1_4 | 0.4620 | 58.45% | 27.42% | 1.45x |
| recent_plus_summary1_2 | 0.4475 | 56.62% | 48.50% | 1.27x |
| recent_plus_summary1_8 | 0.4074 | 51.54% | 16.82% | 1.55x |
| recent_plus_static_hier | 0.3993 | 50.52% | 13.46% | 1.60x |

观察：

```text
固定 recent 后，old context retrieval k2/k3/k4 是当前最强的单一策略。
k3 用约 38.5% token 达到 103.9% full_raw score。
k2 用约 32.8% token 基本达到 full_raw。
```

这里的 speedup 是当前脚本测到的生成调用耗时比，不等价于 CUDA kernel 级最优加速。论文里仍然应该单独报告 attention/KV subsystem benchmark。

## 按 benchmark 的结果

| benchmark | full_raw | k1 | k2 | k3 | k4 | k8 | summary1_8 | summary1_4 | summary1_2 | static_hier |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| LongBench | 0.2868 | 0.2576 | 0.2550 | 0.3164 | 0.2229 | 0.1920 | 0.1920 | 0.1918 | 0.1964 | 0.1285 |
| RULER 4096 | 1.0000 | 0.9062 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.5938 | 0.6562 | 0.6562 | 0.6250 |
| RULER 8192 | 1.0000 | 0.7188 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.3750 | 0.5000 | 0.4688 | 0.4688 |
| RULER 16384 | 0.8750 | 0.5625 | 0.9375 | 0.9688 | 1.0000 | 1.0000 | 0.4688 | 0.5000 | 0.4688 | 0.3750 |

解释：

```text
RULER 上 retrieval raw 特别强，尤其 8k/16k。
LongBench 上 full_raw 本身只有 0.2868，说明这版生成设置、样例数量、评测口径都还比较粗。
LongBench 不能只看当前小跑结果，需要单独扩大样例并区分 exact QA 和 generation。
```

## Oracle 上界

match-full oracle：

```text
score = 0.8225
relative = 104.05% full_raw
token ratio = 22.83%
```

action 分布：

| action | count |
|---|---:|
| recent_plus_summary1_8 | 46 |
| recent_plus_static_hier | 41 |
| recent_plus_retrieval_raw_k1 | 17 |
| recent_plus_retrieval_raw_k2 | 14 |
| recent_plus_summary1_4 | 6 |
| recent_plus_summary1_2 | 2 |
| recent_plus_retrieval_raw_k4 | 1 |
| recent_plus_retrieval_raw_k3 | 1 |

best-score oracle：

```text
score = 0.8465
relative = 107.09% full_raw
token ratio = 25.31%
```

action 分布：

| action | count |
|---|---:|
| recent_plus_summary1_8 | 43 |
| recent_plus_static_hier | 37 |
| recent_plus_retrieval_raw_k1 | 20 |
| recent_plus_retrieval_raw_k2 | 15 |
| recent_plus_summary1_4 | 7 |
| recent_plus_summary1_2 | 3 |
| recent_plus_retrieval_raw_k8 | 1 |
| recent_plus_retrieval_raw_k4 | 1 |
| recent_plus_retrieval_raw_k3 | 1 |

关键结论：

```text
oracle 仍然显示这个路线有上界：约 23%-25% active token 可以达到或者超过 full_raw。
但是可部署 router 还没有达到 oracle 水平。
```

## Router 蒸馏结果

router 输出目录：

```bash
/home/fdong/ymluo/projects/learned_hierarchical_summary_memory/outputs/qwen8b_recent_plus_router_m4_20260706
```

router checkpoint：

```bash
/home/fdong/ymluo/projects/learned_hierarchical_summary_memory/outputs/qwen8b_recent_plus_router_m4_20260706/router.pt
```

训练数据：

```text
examples = 128
train = 83
test = 45
epochs = 1000
hidden_dim = 64
```

结果：

| split | group | samples | label acc | routed success | score | token ratio |
|---|---|---:|---:|---:|---:|---:|
| train | overall | 83 | 97.59% | 92.77% | 0.8661 | 23.60% |
| test | overall | 45 | 60.00% | 68.89% | 0.6518 | 22.97% |
| test | LongBench | 9 | 55.56% | 22.22% | 0.0369 | 23.22% |
| test | RULER 4096 | 12 | 75.00% | 91.67% | 0.9167 | 29.81% |
| test | RULER 8192 | 14 | 50.00% | 78.57% | 0.7857 | 25.06% |
| test | RULER 16384 | 10 | 60.00% | 70.00% | 0.7000 | 11.61% |

test split 的动作选择：

| predicted action | count | rate |
|---|---:|---:|
| recent_plus_summary1_8 | 19 | 42.22% |
| recent_plus_static_hier | 13 | 28.89% |
| recent_plus_retrieval_raw_k1 | 4 | 8.89% |
| recent_plus_retrieval_raw_k2 | 4 | 8.89% |
| recent_plus_summary1_2 | 2 | 4.44% |
| recent_plus_summary1_4 | 2 | 4.44% |
| recent_plus_retrieval_raw_k4 | 1 | 2.22% |

解释：

```text
router 目前过于偏向便宜 summary/static action。
这在 RULER 的简单样例上还可以，但对 LongBench exact QA 容易漏证据。
所以现在不能把 oracle 的 22%-25% token 上界直接当成可部署结果。
```

## 当前结论

1. recent raw 必选这个设计是对的，明显比 recent_only label 更符合生成和检索需求。
2. 在 RULER 上，recent_plus_retrieval_raw_k2/k3/k4 已经是很强的固定策略。
3. oracle 上界足够强，支持继续做 router。
4. 当前 router 训练样例太少，test 只有 45 个 case，尤其 LongBench test 只有 9 个 case，结论不稳定。
5. 下一步重点不是再堆一个复杂 router，而是扩大非 benchmark 训练数据和校准 label，让 router 学会什么时候必须选 raw retrieval 或 full old raw。

## 下一步建议

优先级最高：

```text
用非 benchmark synthetic exact/retrieval 数据扩充 router 训练集。
加入多证据、多 block、不确定证据数量、全文统计、全文摘要四类任务。
训练时显式加入长度特征、retriever 分数 gap、top-k coverage、query 类型特征。
```

然后再做：

```text
1. 用全部非 benchmark 训练数据训练 deployment router。
2. 在 LongBench/RULER 上只做测试，不用 benchmark labels 训练。
3. 分开报告：
   - fixed policy: k2/k3/k4
   - oracle upper bound
   - learned runtime router
   - attention/KV subsystem speed
   - end-to-end generation speed
```

当前最需要补的是 router 泛化，而不是 oracle 上界。
