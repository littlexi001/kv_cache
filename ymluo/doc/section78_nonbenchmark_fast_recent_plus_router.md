# Section 78: 非 Benchmark Synthetic 训练 Recent-plus Router

## 目标

本节目标是训练一个真正可推理时使用的 recent-plus router，并且训练数据不使用 LongBench/RULER benchmark label。

训练数据只来自非 benchmark 文本：

```text
War and Peace
The Count of Monte Cristo
```

做法是在真实书本文本里插入 synthetic evidence record，然后构造 exact/retrieval/summary/recent generation/full-context 等任务。LongBench/RULER 只作为 heldout offline evaluation。

## 代码改动

新增 fast router 训练脚本：

```bash
/home/fdong/ymluo/projects/learned_hierarchical_summary_memory/src/run_fast_recent_plus_router_training.py
```

新增对比脚本：

```bash
/home/fdong/ymluo/projects/learned_hierarchical_summary_memory/scripts/compare_fast_router_sweep.py
```

同时修正了两个实现细节：

```text
1. synthetic success 判断现在正确处理 recent_plus_*：
   recent raw 固定保留，因此 recent evidence / recent generation 不能被误判失败。

2. safety override 支持 recent_plus_* action：
   exact 任务如果误选 summary/static，可以升级到 recent_plus_retrieval_raw_k*。
```

## Synthetic 数据

训练使用的 synthetic case 不来自 benchmark。

覆盖类型：

```text
magic_single_old
magic_single_recent
magic_multiquery
magic_multivalue
single_old
two_old
three_old
four_old
natural_single_old
natural_two_old
natural_three_old
cwe_k1
fwe_k1
vt_k2
summary_brief
summary_detailed
recent_generation
full_context
```

其中 `natural_*` 是本次新增的自然问答风格 synthetic exact QA，用来减少 router 把 LongBench exact QA 错判为 summary 的问题。

输入长度：

```text
4096, 8192, 16384, 20000 tokens
```

每个文本源、每个长度约 260 个 synthetic case。

best router 的训练规模：

```text
examples = 2080
train = 1560
test = 520
```

训练 label 分布：

| label | count |
|---|---:|
| full_raw | 80 |
| recent_plus_retrieval_raw_k2 | 752 |
| recent_plus_retrieval_raw_k3 | 566 |
| recent_plus_retrieval_raw_k4 | 186 |
| recent_plus_summary1_4 | 80 |
| recent_plus_summary1_8 | 416 |

注意：这里没有使用 LongBench/RULER 的 oracle label 训练。

## 训练过的 Router

本次跑了两轮。

第一轮：

```bash
/home/fdong/ymluo/projects/learned_hierarchical_summary_memory/outputs/fast_recent_plus_router_sweep_20260706
```

第二轮加入 natural QA synthetic：

```bash
/home/fdong/ymluo/projects/learned_hierarchical_summary_memory/outputs/fast_recent_plus_router_sweep_v2_20260706
```

策略：

```text
budget
balanced
conservative
```

最终选择：

```bash
/home/fdong/ymluo/projects/learned_hierarchical_summary_memory/outputs/recent_plus_router_best_20260706/router.pt
```

这个 checkpoint 来自：

```bash
/home/fdong/ymluo/projects/learned_hierarchical_summary_memory/outputs/fast_recent_plus_router_sweep_v2_20260706/balanced/router.pt
```

## Heldout Benchmark 结果

评估数据：

```bash
/home/fdong/ymluo/projects/learned_hierarchical_summary_memory/outputs/qwen8b_recent_plus_bench_m4_parallel_20260706/merged
```

这个 heldout benchmark 包含：

```text
LongBench: 32 cases
RULER 4k: 32 cases
RULER 8k: 32 cases
RULER 16k: 32 cases
total: 128 cases
```

best router 结果：

| group | score | full_raw | relative | token ratio |
|---|---:|---:|---:|---:|
| overall | 0.7980 | 0.7905 | 100.95% | 41.09% |
| exact | 0.8417 | 0.8333 | 101.00% | 42.00% |
| generation | 0.1431 | 0.1473 | 97.11% | 27.49% |
| LongBench | 0.2545 | 0.2868 | 88.73% | 26.93% |
| RULER 4096 | 1.0000 | 1.0000 | 100.00% | 73.32% |
| RULER 8192 | 1.0000 | 1.0000 | 100.00% | 42.45% |
| RULER 16384 | 0.9375 | 0.8750 | 107.14% | 21.65% |

best router 的 heldout action 分布：

| action | count | rate |
|---|---:|---:|
| recent_plus_retrieval_raw_k2 | 77 | 60.16% |
| recent_plus_retrieval_raw_k3 | 28 | 21.88% |
| recent_plus_retrieval_raw_k4 | 2 | 1.56% |
| recent_plus_summary1_4 | 1 | 0.78% |
| recent_plus_summary1_8 | 20 | 15.62% |

## 对比结果

| run | train policy | eval policy | relative | token ratio |
|---|---|---|---:|---:|
| v1 | balanced | raw router | 96.01% | 36.99% |
| v1 | balanced | conservative safety | 99.97% | 40.37% |
| v1 | budget | conservative safety | 100.99% | 40.87% |
| v1 | conservative | conservative safety | 102.92% | 44.03% |
| v2 | balanced | raw router | 100.95% | 41.09% |
| v2 | budget | conservative safety | 100.95% | 39.73% |
| v2 | conservative | conservative safety | 102.92% | 43.84% |
| oracle | match-full | oracle | 105.96% | 23.90% |

我选择 v2 balanced raw router 作为当前 best，原因是：

```text
1. 不依赖 safety rule 才达到 100.95% full_raw。
2. RULER 4k/8k/16k 表现稳定。
3. token ratio 虽然是 41.09%，但质量比 30% 左右的 learned router 明显稳。
```

如果追求更低 token，可以临时使用：

```bash
/home/fdong/ymluo/projects/learned_hierarchical_summary_memory/outputs/fast_recent_plus_router_sweep_v2_20260706/budget/router.pt
```

并配合 conservative safety，得到：

```text
relative = 100.95% full_raw
token ratio = 39.73%
```

但这个结果更多依赖 safety override，不如 v2 balanced raw router 干净。

## 目前短板

最大短板仍然是 LongBench：

```text
LongBench relative = 88.73%
token ratio = 26.93%
```

原因不是 router 没学到 RULER，而是 LongBench 的自然问答、摘要生成、评测噪声和小样例更复杂。虽然这次加入 natural QA synthetic 后 overall 从 96.01% 提升到 100.95%，但 LongBench 单独看仍然没有达到 full_raw。

这说明下一步应该继续增强非 benchmark 的自然问答 synthetic，而不是用 LongBench oracle label 直接训练。

## 当前结论

当前已经有一个可用 router：

```text
Qwen3-8B recent-plus router
heldout benchmark relative = 100.95% full_raw
active token ratio = 41.09%
```

它还没达到 oracle 的：

```text
105.96% full_raw
23.90% token
```

但已经比之前 benchmark-only 小数据 router 稳得多。下一步最有价值的是继续缩小 `41.09% -> 25%-35%` 的 token gap，同时保持 `95%-100% full_raw`。

## 下一步

建议继续做三件事：

```text
1. 扩充非 benchmark natural QA：
   多跳问答、实体关系、跨段聚合、多个答案数量不确定。

2. 给 router 加 risk calibration：
   如果 query 像自然 QA 但 retriever gap 小，就选 k3/k4；
   如果 query 是摘要/续写，就选 summary1_8/summary1_4。

3. 在不使用 benchmark label 训练的前提下，
   继续用 LongBench/RULER 做 heldout evaluation。
```
