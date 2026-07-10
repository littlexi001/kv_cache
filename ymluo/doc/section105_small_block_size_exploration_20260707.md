# Section 105: 32/64/128 small-block 探索

日期：2026-07-07

## 目的

上一轮 `block=256/512/1024/2048` 的结果显示：

- `block=256` 在 RULER 4k/8k 上 token ratio 很低，质量也比较稳。
- `block=512` 在 LongBench 和 RULER16k 上更保守、更稳。
- `block=1024/2048` 大多太粗，token ratio 偏高。

这次继续向更小 block 探索：

```text
block = 32 / 64 / 128
topK = 1 / 2 / 3 / 4 / 6 / 8 / 12 / 16 / 24 / 32
```

核心问题：

1. 更小 block 是否能进一步降低 active KV？
2. 是否会因为 evidence 被切得太碎而明显降低召回质量？
3. router 是否能学会什么时候使用 32/64/128？

## 实验设置

模型：

```text
Qwen3-8B + 之前训练的 LoRA adapter
```

数据：

```text
LongBench: hotpotqa, 2wikimqa, musique, passage_retrieval_en, passage_count, qasper, gov_report, multi_news
RULER: 4k / 8k / 16k
每个 task 取 3 个样例
```

方法：

```text
full_raw
recent_plus_span_top{K}_b0_a0
```

其中 `recent raw` 固定保留，old context 按 block size 切块后做 evidence span retrieval。

输出：

```text
/home/fdong/ymluo/projects/learned_hierarchical_summary_memory/outputs/smallblock_topk_sweep_m3_summary_20260707.txt
```

## Overall frontier

### LongBench

| block | best score | best K | token ratio | e2e speedup |
|---:|---:|---:|---:|---:|
| 32 | 0.2446 | 32 | 12.96% | 1.61x |
| 64 | 0.2441 | 24/32 | 15.54%-18.58% | 1.55x-1.58x |
| 128 | 0.3294 | 12 | 15.54% | 1.58x |
| 256 | 0.2883 | 4 | 12.81% | 1.61x |
| 512 | 0.3262 | 4 | 19.93% | 1.54x |

结论：LongBench 上 `block=128` 是新的强候选。它比 `block=512 top4` token 更低，并且 score 略高；比 `block=256 top4` score 更高，但 token 稍高。

### RULER 4k

| block | 达到 100% 的最低组合 | token ratio |
|---:|---|---:|
| 32 | 未达到 100% | - |
| 64 | 未达到 100% | - |
| 128 | 未达到 100% | - |
| 256 | top3 | 30.63% |
| 512 | top2 | 37.69% |

结论：4k 短上下文不能过度激进。`block=128 top3` 可到 95.83%，token 22.82%，但要 100% 仍然需要 `block=256 top3`。

### RULER 8k

| block | 达到 100% 的最低组合 | token ratio |
|---:|---|---:|
| 32 | 未达到 100% | - |
| 64 | 未达到 100% | - |
| 128 | 未达到 100% | - |
| 256 | top3 | 15.27% |
| 512 | top3 | 22.70% |

结论：8k 上 `block=128 top3` 有 95.83%，token 11.17%；如果允许 95% full，这个组合非常好。如果要 100%，仍然需要 `block=256 top3`。

### RULER 16k

| block | score | 推荐组合 | token ratio |
|---:|---:|---|---:|
| 32 | 0.8333 | top16/top24/top32 | 5.69%-5.70% |
| 64 | 0.9167 | top16/top24/top32 | 7.18%-7.20% |
| 128 | 0.9167 | top16/top24/top32 | 10.18% |
| 256 | 0.9583 | top3 | 7.47% |
| 512 | 1.0000 | top3 | 10.70% |

结论：16k 上小 block 很有用，但不是越小越好。`block=64 top4` 已经能达到 full_raw 的 0.875，token 只有 4.81%；如果要超过 full_raw，`block=64 top16` 或 `block=128 top16` 可以到 0.9167，但要 100% 还是 `block=512 top3` 最稳。

## 新版 small-block router

我把 `32/64/128/256/512/1024/2048` 的 sweep label 合并，重新蒸馏了一版 router。

输出：

```text
/home/fdong/ymluo/projects/learned_hierarchical_summary_memory/outputs/smallblock_router_from_sweeps_m3_20260707/router.pt
```

训练结果：

| split | samples | label accuracy | avg score | avg token ratio |
|---|---:|---:|---:|---:|
| train | 82 | 90.24% | 0.8767 | 11.01% |
| test | 14 | 50.00% | 0.7857 | 9.61% |

label accuracy 不高，原因是 action space 变大后，很多 label 是“同样正确但 token 差一点”的近邻类别；这个指标不能单独说明 router 不好。更关键是实际生成结果。

## 实际 Qwen3-8B m3 生成评测

输出：

```text
/home/fdong/ymluo/projects/learned_hierarchical_summary_memory/outputs/qwen8b_smallblock_router_m3_20260707
```

整体结果：

| method | score | token ratio | speedup |
|---|---:|---:|---:|
| full_raw | 0.8116 | 100.00% | 1.00x |
| fixed block512 top3 | 0.8107 | 18.22% | 1.67x |
| previous blocksize router | 0.8535 | 11.04% | 1.76x |
| small-block router | 0.8634 | 7.82% | 1.80x |

分组结果：

| benchmark | full_raw | small-block router | relative | token ratio |
|---|---:|---:|---:|---:|
| LongBench | 0.2465 | 0.4536 | 184.03% | 12.15% |
| RULER 4k | 1.0000 | 1.0000 | 100.00% | 17.72% |
| RULER 8k | 1.0000 | 1.0000 | 100.00% | 8.80% |
| RULER 16k | 1.0000 | 1.0000 | 100.00% | 4.56% |

router action 分布：

| action | count |
|---|---:|
| `recent_plus_b32_span_top1_b0_a0` | 18 |
| `recent_plus_b32_span_top2_b0_a0` | 15 |
| `recent_plus_b32_span_top32_b0_a0` | 7 |
| `recent_plus_b64_span_top2_b0_a0` | 7 |
| `recent_plus_b64_span_top1_b0_a0` | 7 |
| `recent_plus_b32_span_top3_b0_a0` | 6 |
| `recent_plus_b128_span_top3_b0_a0` | 5 |
| 其它 `b32/b64/b128/b256/b512/b1024/b2048` action | 31 |

## 结论

这轮结果明显支持继续纳入小 block：

- `small-block router` 把 token ratio 从上一版 11.04% 进一步降到 7.82%。
- score 从 0.8535 提到 0.8634。
- RULER 16k 能在 4.56% active token 下保持 100%。
- LongBench 这组 m3 样本上也从 0.4140 提升到 0.4536。

但也有风险：

- 单独固定 `block=32` 并不稳，很多任务需要 top16/top32 才能恢复质量。
- `block=32/64` 会增加 page 数量，对真实 KV-native 实现里的 page scoring/topK/gather 开销提出更高要求。
- 当前 m3 样本小，而且 router 仍从 benchmark sweep 蒸馏，最终 paper 需要非 benchmark 训练 + held-out benchmark 测试。

## 推荐下一步

1. action space 不要简单全量保留，可以先收敛为：

```text
b32: top1/top2/top3/top12/top16/top32
b64: top1/top2/top3/top4/top16
b128: top2/top3/top6/top12
b256: top2/top3/top4/top8
b512: top1/top2/top3
b1024/b2048: top1 only
```

2. 用非 benchmark synthetic exact/retrieval + book QA 重新训练 router。
3. 做 held-out LongBench/RULER m10/m50，验证 small-block router 是否仍然稳定。
4. KV-native benchmark 里必须单独统计 small block 增加的 page scoring 和 gather 开销，因为 page 数明显变多。
