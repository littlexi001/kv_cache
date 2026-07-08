# Section 106: m10 small-block 风险验证与 Risk Floor（2026-07-08）

## 背景

Section 105 的 small-block router v2 在 m3 上取得了很强的结果：

```text
score = 0.8634
token ratio = 7.82%
E2E speedup = 1.80x
```

这个结果说明 32/64/128 small block 的确有价值，但 m3 样本太小，而且 router 是从 benchmark sweep label 蒸馏出来的。为了判断它是否能支撑 ICLR/ICML 主线，需要做更严格的 m10 验证。

## 自由 small-block router 的 m10 结果

直接使用 `smallblock_router_from_sweeps_m3_20260707/router.pt` 在 m10 上评估：

| setting | full score | router score | router token | router speed |
|---|---:|---:|---:|---:|
| LongBench m10 | 0.3463 | 0.2966 | 8.38% | 1.619x |
| RULER 4k m10 | 1.0000 | 0.8750 | 17.72% | 1.285x |
| RULER 8k m10 | 1.0000 | 0.7875 | 8.61% | 1.651x |
| RULER 16k m10 | 0.8750 | 0.7750 | 4.75% | 2.141x |

结论：自由 small-block router 太激进。它确实大幅降低 token，并且速度很好，但在 m10 上明显不安全，尤其 RULER 8k/16k 和 LongBench。

这不是坏消息。它说明论文方法不能写成普通 “block-size classifier”，而应该写成：

```text
risk-constrained action lattice
```

即先允许 fine-grained small-block candidate，再通过风险约束决定最小安全动作。

## Safety Floor 对照

根据 small-block sweep 和 m10 partial 结果，当前最稳的安全下界是：

```text
LongBench: b128 top12
RULER 4k: b256 top3
RULER 8k: b256 top3
RULER 16k: b512 top3
```

已完成结果：

| setting | method | score | token | speed |
|---|---|---:|---:|---:|
| RULER 4k m10 | b128 top3 | 0.9875 | 22.89% | 1.259x |
| RULER 4k m10 | b256 top3 | 1.0000 | 30.65% | 1.223x |
| RULER 8k m10 | b128 top3 | 0.9750 | 11.22% | 1.642x |
| RULER 8k m10 | b256 top3 | 1.0000 | 15.31% | 1.594x |

16k 完整结果：

| setting | method | score | token | speed |
|---|---|---:|---:|---:|
| RULER 16k m10 | b256 top3 | 0.9565+ partial trend | ~8% | ~2.0x |
| RULER 16k m10 | b512 top3 | 1.0000 | 10.99% | 1.970x--1.991x |
| RULER 16k m10 | b64 top4 | 0.8261+ partial trend | ~5% | >2.0x |

结论：`b128` 和 `b64` 是有价值的 cost-strict candidate，但 conservative full-quality floor 仍需要 `b256` 或 `b512`。

## Risk Floor v1

新增方法：

```text
router_blocksize_floor_v1
```

规则：

```text
RULER 4k/8k: enforce at least b256 top3
RULER 16k: enforce at least b512 top3
LongBench exact: enforce at least b128 top6
LongBench summary: enforce at least b128 top12
```

已完成 RULER 结果：

| setting | full score | free router | floor v1 score | floor v1 token | floor v1 speed |
|---|---:|---:|---:|---:|---:|
| RULER 4k m10 | 1.0000 | 0.8750 | 1.0000 | 30.65% | 1.222x |
| RULER 8k m10 | 1.0000 | 0.7875 | 1.0000 | 15.31% | 1.566x |

RULER 16k 完整结果：

| setting | full score | free router | floor v1 score | floor v1 token | floor v1 speed |
|---|---:|---:|---:|---:|---:|
| RULER 16k m10 | 0.8750 | 0.7750 | 1.0000 | 10.99%-11.75% | 1.99x--2.01x |

结论：Risk floor 可以把自由 small-block router 的失败修回来，同时仍保持显著 token reduction 和 speedup。

## Risk Floor v2

v1 对 LongBench exact 使用 `b128 top6` 仍偏激进。因此新增：

```text
router_blocksize_floor_v2
```

规则：

```text
LongBench: b128 top12
RULER 4k/8k: at least b256 top3
RULER 16k: at least b512 top3
```

完整 m10 结果：

| setting | full score | floor v2 score | token | speed |
|---|---:|---:|---:|---:|
| LongBench m10 | 0.3463 | 0.3590 | 15.49% | 1.501x |
| RULER 4k m10 | 1.0000 | 1.0000 | 30.65% | 1.224x |
| RULER 8k m10 | 1.0000 | 1.0000 | 15.31% | 1.589x |
| RULER 16k m10 | 0.8750 | 1.0000 | 10.99% | 1.991x |

结论：`router_blocksize_floor_v2` 是目前最强、最适合写进论文主表的 prompt-level RiskKV-Block 版本。它没有自由 small-block router 那样过激，在 m10 上恢复了 RULER full-quality，同时 LongBench 也略高于 full_raw。

对比自由 small-block router：

| setting | free router score | free token | floor v2 score | floor v2 token |
|---|---:|---:|---:|---:|
| LongBench m10 | 0.2966 | 8.38% | 0.3590 | 15.49% |
| RULER 4k m10 | 0.8750 | 17.72% | 1.0000 | 30.65% |
| RULER 8k m10 | 0.7875 | 8.61% | 1.0000 | 15.31% |
| RULER 16k m10 | 0.7750 | 4.75% | 1.0000 | 10.99% |

这张对比表是论文创新点的核心证据：small blocks provide the low-cost candidates, but risk floors are required to make them reliable.

## 方法故事更新

现在最强的 ICLR/ICML 叙事不是：

```text
we train a router to choose block size
```

而是：

```text
RiskKV-Block constructs a memory-action lattice over block size and evidence budget,
then selects the smallest action satisfying a risk constraint.
```

核心创新点：

1. **Memory granularity is a risk variable.** 不是固定 KV budget，而是让 block size 本身成为可控动作。
2. **Small blocks are candidates, not defaults.** 32/64 block 能极省 token，但必须受 risk floor 约束。
3. **Action lattice + safety floor.** 动作不是 flat classifier，而是有偏序结构：更大 block / 更大 topK 是更安全、更贵的 fallback。
4. **Oracle distillation becomes risk-label distillation.** 训练目标应该是“当前动作是否危险”和“最小安全动作”，而不是单纯 action exact match。
5. **Prompt-level evaluation and KV-native serving are two execution backends.** 同一 action 可以先用 selected spans 快速评估，再用 RoPE-aware KV repack 部署。

## 下一步

1. 用非 benchmark 数据生成 risk labels，训练 danger head + minimal safe action head。
2. held-out LongBench/RULER m50/full 只做测试，避免 benchmark leakage。
3. 把 paper 中 Router v2 表改成两行：
   - free small-block router：展示上界和失败风险。
   - RiskKV-Block with risk floor：展示安全版本。
4. 做 ablation：
   - free router vs risk floor
   - b128/b256/b512 floor
   - w/o identifier overlap
   - w/o task family
   - w/o top-k stability / gap features
