# Section 67: Qwen3-8B LoRA Adapter + Router 的 ICML 前置 Benchmark 结果

## 评测目标

本节评测刚训练完成的 Qwen3-8B 专用 LoRA adapter 和 runtime router。

重要约束：

```text
adapter 训练和 router 蒸馏没有使用 LongBench / RULER 数据。
本节只把 LongBench / RULER 作为 held-out benchmark 测试。
```

## 模型与产物

Base model：

```bash
/home/fdong/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218
```

LoRA adapter：

```bash
/home/fdong/ymluo/projects/learned_hierarchical_summary_memory/outputs/qwen8b_lora_4k_1ksteps_no_bench_20260705/adapter
```

Router：

```bash
/home/fdong/ymluo/projects/learned_hierarchical_summary_memory/outputs/qwen8b_router_distill_no_bench_20260705/router.pt
```

## 评测配置

adapter + router 多方法评测输出：

```bash
/home/fdong/ymluo/projects/learned_hierarchical_summary_memory/outputs/qwen8b_icml_bench_adapter_router_20260705
```

base full_raw baseline 输出：

```bash
/home/fdong/ymluo/projects/learned_hierarchical_summary_memory/outputs/qwen8b_icml_bench_base_fullraw_20260705
```

任务：

- LongBench: `hotpotqa, 2wikimqa, musique, passage_retrieval_en, passage_count, qasper, gov_report, multi_news`
- RULER: `niah_single_1, niah_single_2, niah_multikey_1, niah_multiquery, niah_multivalue, vt, cwe, fwe`
- RULER context length: `4096, 8192, 16384`
- 每个任务 4 个样例。

方法：

```text
full_raw
recent_only
static_hier
summary1_8
summary1_4
summary1_2
retrieval_raw_k1
retrieval_raw_k2
router
```

注意：这里使用的是当前脚本内的轻量 evaluator：

- QA / retrieval / RULER 使用 exact match。
- `gov_report` / `multi_news` 使用 Rouge-L。

因此本节结果适合做研发筛选和消融，不等价于最终论文的官方 LongBench scorer 结果。

## 运行耗时

base full_raw：

```text
128 cases x 1 method
约 13 分钟
```

adapter + router 消融：

```text
128 cases x 9 methods = 1152 trials
约 1 小时 42 分钟
```

## 总体结果

| method | samples | score | exact | token ratio | avg seconds | avg prompt tokens |
|---|---:|---:|---:|---:|---:|---:|
| adapted full_raw | 128 | 0.7916 | 0.7812 | 100.00% | 7.559 | 9890.6 |
| recent_only | 128 | 0.3371 | 0.3281 | 8.26% | 4.352 | 551.7 |
| static_hier | 128 | 0.4155 | 0.4062 | 13.69% | 4.500 | 997.7 |
| summary1_8 | 128 | 0.3299 | 0.3203 | 12.09% | 4.562 | 1205.2 |
| summary1_4 | 128 | 0.3294 | 0.3203 | 22.92% | 4.919 | 2311.6 |
| summary1_2 | 128 | 0.3620 | 0.3516 | 44.49% | 5.675 | 4513.7 |
| retrieval_raw_k1 | 128 | 0.6198 | 0.6094 | 37.08% | 4.994 | 2551.6 |
| retrieval_raw_k2 | 128 | 0.7834 | 0.7734 | 49.50% | 5.281 | 3424.7 |
| router | 128 | 0.5569 | 0.5469 | 28.34% | 4.913 | 2285.4 |

base Qwen3-8B full_raw baseline：

| method | samples | score | exact | token ratio | avg seconds |
|---|---:|---:|---:|---:|---:|
| base full_raw | 128 | 0.8148 | 0.8047 | 100.00% | 5.916 |

观察：

- 当前 adapter full_raw 比 base full_raw 略低：`0.7916 vs 0.8148`。
- `retrieval_raw_k2` 最接近 adapted full_raw：`0.7834 / 0.7916 = 98.97%`，token ratio `49.50%`。
- 当前 router 明显更省 token：`28.34%`，但 score 只有 `0.5569`，主要被 RULER exact 任务拖低。
- PEFT LoRA 未 merge 时有额外推理开销，因此本节的 wall-clock speed 不能作为 CUDA/kernel 优化后的最终速度结论。

## LongBench 结果

| method | samples | score | exact | token ratio | avg seconds |
|---|---:|---:|---:|---:|---:|
| adapted full_raw | 32 | 0.2913 | 0.2500 | 100.00% | 9.567 |
| recent_only | 32 | 0.1920 | 0.1562 | 8.99% | 5.957 |
| static_hier | 32 | 0.1619 | 0.1250 | 14.64% | 6.085 |
| summary1_8 | 32 | 0.1946 | 0.1562 | 13.08% | 6.218 |
| summary1_4 | 32 | 0.1613 | 0.1250 | 25.34% | 6.638 |
| summary1_2 | 32 | 0.1980 | 0.1562 | 50.11% | 7.547 |
| retrieval_raw_k1 | 32 | 0.3542 | 0.3125 | 40.30% | 6.576 |
| retrieval_raw_k2 | 32 | 0.2274 | 0.1875 | 54.80% | 6.895 |
| router | 32 | 0.3528 | 0.3125 | 22.54% | 6.445 |

base full_raw LongBench：

```text
score = 0.2906
exact = 0.2500
```

观察：

- LongBench 上 router 与 `retrieval_raw_k1` 最好，且都高于 full_raw。
- 这不一定说明模型真的更强，可能是 raw block retrieval 带来了去噪效果。
- LongBench 分数总体偏低，需要使用官方 scorer 和更多样例复核。

## RULER 总体结果

| method | samples | exact/score | token ratio | avg seconds |
|---|---:|---:|---:|---:|
| adapted full_raw | 96 | 0.9583 | 100.00% | 6.882 |
| recent_only | 96 | 0.3854 | 8.01% | 3.817 |
| static_hier | 96 | 0.5000 | 13.37% | 3.971 |
| summary1_8 | 96 | 0.3750 | 11.76% | 4.010 |
| summary1_4 | 96 | 0.3854 | 22.12% | 4.346 |
| summary1_2 | 96 | 0.4167 | 42.62% | 5.051 |
| retrieval_raw_k1 | 96 | 0.7083 | 36.01% | 4.467 |
| retrieval_raw_k2 | 96 | 0.9688 | 47.73% | 4.743 |
| router | 96 | 0.6250 | 30.28% | 4.403 |

base full_raw RULER：

```text
score = 0.9896
exact = 0.9896
```

观察：

- RULER 是当前方法的关键风险点。
- `retrieval_raw_k2` 很强：比 adapted full_raw 还略高，达到 `0.9688`，token ratio `47.73%`。
- 当前 router 只有 `0.6250`，说明 router 对 exact retrieval 难度估计不稳。
- 仅用 summary memory 不能处理多数精确回忆任务。

## RULER 按长度拆分

| length | method | score | token ratio |
|---|---|---:|---:|
| 4096 | full_raw | 1.0000 | 100.00% |
| 4096 | retrieval_raw_k2 | 1.0000 | 79.48% |
| 4096 | router | 0.6562 | 42.86% |
| 4096 | static_hier | 0.7188 | 20.85% |
| 8192 | full_raw | 1.0000 | 100.00% |
| 8192 | retrieval_raw_k2 | 0.9688 | 41.86% |
| 8192 | router | 0.6562 | 31.52% |
| 8192 | static_hier | 0.4062 | 12.26% |
| 16384 | full_raw | 0.8750 | 100.00% |
| 16384 | retrieval_raw_k2 | 0.9375 | 21.86% |
| 16384 | router | 0.5625 | 16.45% |
| 16384 | static_hier | 0.3750 | 7.01% |

观察：

- `retrieval_raw_k2` 在 16k 上 token ratio 只有 `21.86%`，score `0.9375`，这是目前最有论文价值的信号。
- router 经常选择 `retrieval_raw_k1`，对多 key / 多 value / vt 这类任务不够保守。

## Router 行为分布

router 在 128 个 case 上的选择：

| action | count | ratio |
|---|---:|---:|
| retrieval_raw_k1 | 89 | 69.53% |
| recent_only | 26 | 20.31% |
| full_raw | 9 | 7.03% |
| retrieval_raw_k2 | 4 | 3.12% |

结论：

- 当前 router 太偏向 `retrieval_raw_k1`。
- 对 RULER exact、多答案、多 block 任务，应该更频繁选择 `retrieval_raw_k2` 或动态 top-k。
- 这个 router 不能作为最终投稿版本。

## 对 ICML 投稿的判断

当前结果有一个强信号：

```text
retrieval_raw_k2:
overall 98.97% adapted full_raw score
49.50% active tokens

RULER:
101.09% adapted full_raw score
47.73% active tokens

RULER 16k:
107.14% adapted full_raw score
21.86% active tokens
```

但当前结果还不能直接作为 ICML 主结果：

- router 没有达到 `95%+ full_raw performance`，overall 只有 `70.35%` adapted full_raw。
- summary-only 方法在 exact retrieval 上明显不够。
- LongBench 只跑了每任务 4 个样例，并且使用轻量 evaluator。
- LoRA adapter 对 full_raw 略有损伤，需要检查是否因为训练分布窄或 adapter 未 merge。
- 当前 wall-clock 速度包含 PEFT 未 merge 和 Python prompt construction 开销，不能直接等价于 kernel-level speedup。

## 下一步

为了形成可投稿主结果，建议优先做三件事：

1. 重新蒸馏 router：增加非 benchmark synthetic exact/retrieval 数据，特别是多 key、多 value、多 block、unknown k 的样例；label space 改为动态 top-k 或 threshold policy，而不是固定 k1/k2。
2. 用官方 scorer 跑完整 LongBench/RULER：至少每任务更多样例，保留 `base full_raw`、`adapted full_raw`、`retrieval_raw_k2`、`router`、`static_hier`、`summary1_8/1_4/1_2`。
3. 做 adapter 复核：merge LoRA 后重新测速度；同时检查 adapter 是否降低 full_raw benchmark 分数。

