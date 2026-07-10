# Section 101: Block Size 与 TopK 联合 Sweep

## 目标

前面 `block=512` 的结果说明，缩小 block 可以显著降低 active KV，同时保持质量。因此这里进一步系统扫：

```text
block size: 256, 512, 1024, 2048
recent raw: 512 tokens
topK blocks: 1, 2, 3, 4, 6, 8
model: Qwen3-8B + qwen8b LoRA adapter
benchmark:
  LongBench 8 tasks x 3
  RULER 4k/8k/16k, 8 tasks x 3
```

记录指标：

```text
score: benchmark answer quality
active token ratio: prompt/KV token ratio vs full_raw
e2e speedup: 当前 HF generate 端到端时间加速
attention upper: 1 / active token ratio，近似 attention 子模块理论加速上限
```

输出：

```bash
/home/fdong/ymluo/projects/learned_hierarchical_summary_memory/outputs/blocksize_topk_sweep_m3_summary_20260707.txt
```

## 整体结果

### LongBench

| block | best topK | score | active token | e2e speed | attention upper |
|---:|---:|---:|---:|---:|---:|
| 256 | top4 | 0.2883 | 12.81% | 1.61x | 7.8x |
| 512 | top4 | 0.3262 | 19.93% | 1.54x | 5.0x |
| 1024 | top2 | 0.2878 | 21.56% | 1.53x | 4.6x |
| 2048 | top1/top3 | 0.2457 | 24.50%+ | 1.45x | 4.1x |

解释：

```text
LongBench 质量最高是 block512 top4。
但如果目标是刚超过 full_raw score，则 block256 top4 已经够用，并且只用 12.81% active token。
LongBench 不是 block 越大越好；topK 过大还会引入噪声。
```

### RULER 4k

| block | min topK for 1.0 score | active token | e2e speed | attention upper |
|---:|---:|---:|---:|---:|
| 256 | top3 | 30.63% | 1.33x | 3.3x |
| 512 | top2 | 37.69% | 1.19x | 2.7x |
| 1024 | top2 | 60.05% | 1.11x | 1.7x |
| 2048 | top2 | 79.05% | 1.04x | 1.3x |

结论：

```text
RULER 4k 最适合 block256。
虽然 topK 需要到 3，但总 token 仍显著低于 block512/1024。
```

### RULER 8k

| block | min topK for 1.0 score | active token | e2e speed | attention upper |
|---:|---:|---:|---:|---:|
| 256 | top3 | 15.27% | 1.70x | 6.5x |
| 512 | top3 | 22.70% | 1.51x | 4.4x |
| 1024 | top3 | 37.19% | 1.43x | 2.7x |
| 2048 | top3 | 62.54% | 1.20x | 1.6x |

结论：

```text
RULER 8k 明显选择 block256 top3。
这是目前最干净的质量-成本点：100% score，15.27% active token。
```

### RULER 16k

| block | best safe topK | score | active token | e2e speed | attention upper |
|---:|---:|---:|---:|---:|---:|
| 256 | top3/top4 | 0.9583 | 7.47%-8.32% | ~2.0x | 12.0x-13.4x |
| 512 | top3 | 1.0000 | 10.70% | 1.96x | 9.3x |
| 1024 | top4 | 1.0000 | 21.05% | 1.75x | 4.8x |
| 2048 | top4 | 1.0000 | 35.84% | 1.43x | 2.8x |

解释：

```text
如果只看平均超过 full_raw，block256 top2 只用 6.28% token 就达到 full_raw 平均分 0.875。
但如果要求整体 1.0 score，block512 top3 是更稳的点。
per-task 看，绝大多数 RULER16k 任务 block256 足够，少数任务需要 block512。
```

## Per-task 结论

### LongBench

| task | 推荐 block/topK | active token | 备注 |
|---|---:|---:|---|
| hotpotqa | block256 top4 | 10.5% | 多跳证据，需要更多 block |
| passage_retrieval_en | block1024 top1 | 13.9% | 单证据检索，大 block top1 更稳 |
| 2wikimqa | block256 top2 | 15.3% | 小 block 多证据有效 |
| multi_news | block256 top1 | 39.8% | summary 任务，仍不理想 |
| gov_report | 无法达到 full_raw | - | 需要 teacher/generative summary |
| qasper | full_raw 本身低 | - | 需要更强 evidence composer |
| passage_count | full_raw 本身低 | - | 需要全局 count/aggregation 策略 |

LongBench 说明：

```text
block size 不是唯一瓶颈。
检索类任务可以靠 block/topK 解决；
summary、count、scientific QA 需要专门策略。
```

### RULER

| group | 简单任务 | 多证据/困难任务 | 建议 |
|---|---|---|---|
| RULER 4k | block256 top1 | block256 top2/top3 | 默认 block256 |
| RULER 8k | block256 top1/top2 | block256 top3 | 默认 block256 |
| RULER 16k | block256 top1/top2 | block512 top2/top3 for safety | 默认 block256，低置信度回退 block512 |

## 当前判断

最重要的结论：

```text
block=256 是 RULER 上最好的默认 block size。
block=512 是长上下文 hard case 的安全回退。
block=1024/2048 通常 token 太粗，除少数 LongBench 单证据任务外，不适合作为默认。
```

如果要写成方法，可以叫：

```text
variable block-size, variable topK routing
```

推荐 action space：

```text
block256_top1
block256_top2
block256_top3
block256_top4
block512_top2
block512_top3
block1024_top1
summary / full_old fallback
```

不建议让 router 选择太多动作，例如 block2048 top8。这些点成本高、收益小，会增加训练难度。

## 对速度的解释

当前 HF 端到端速度没有达到 attention upper：

```text
RULER 16k block512 top3:
  active token = 10.70%
  attention upper = 9.3x
  observed e2e speedup = 1.96x

RULER 8k block256 top3:
  active token = 15.27%
  attention upper = 6.5x
  observed e2e speedup = 1.70x
```

这说明：

```text
1. 质量/token trade-off 已经很好。
2. 真正的大速度需要把 page scoring、topK、gather/repack 和 attention 放进高效 kernel/serving runtime。
3. 论文里应同时报告 active KV ratio、attention subsystem upper bound、当前 HF e2e speed。
```

## 下一步

下一步应该训练一个 block-size-aware router：

```text
input:
  query features
  context length
  task type
  page score gap
  page score entropy/dispersion
  router confidence

output:
  block size + topK + fallback action

label:
  从本次 sweep 生成 oracle label：
  在达到 quality threshold 的候选中选择 active token 最低的 action。
```

建议先训练两个版本：

```text
cost_strict:
  score >= 0.95 * full_raw 时选择最低 token action

quality_strict:
  score >= full_raw 或 task score 达到 1.0 时选择最低 token action
```

然后用 Qwen3-8B 跑 m10/m20，验证是否能稳定达到：

```text
RULER: 100% score, 10%-30% active token
LongBench retrieval: >= full_raw, 10%-25% active token
LongBench summary/count: 走专门 fallback，不强行 sparse span
```
