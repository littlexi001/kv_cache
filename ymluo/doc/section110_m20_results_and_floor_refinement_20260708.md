# Section 110: m20 结果、失败分析与 Floor Refinement（2026-07-08）

## 背景

Section 109 启动了两组 m20：

```text
1. main:
   full_raw, router_blocksize, router_blocksize_calibrated

2. ablation:
   full_raw,
   blocksize_calibrated_floor_only,
   b128 top12,
   b256 top3,
   b512 top3
```

运行已完成，每组 160 samples。

重要修正：

```text
block size 不是严格单调安全变量。
```

原因是改变 block size 会改变分块和 evidence scorer 排序；
因此 `b512 top3` 不一定支配 `b256 top3`，`b256 top3` 也不一定支配 `b128 top12`。

所以正式 calibrated method 应该选择完整 action：

```text
a_final = phi(g)
```

而不是：

```text
a_final = max_lattice(a_raw, phi(g))
```

代码已经更新：

```text
router_blocksize_floor_v2 = legacy monotone floor
router_blocksize_calibrated / riskkv_block_calibrated = complete-action calibrated floor
blocksize_calibrated_floor_only = calibrated action only
```

## m20 Main 结果

注意：这次 main 里的 `router_blocksize_calibrated` 是旧 monotone/lattice floor 版本启动后跑完的结果，
不能作为最终正式 calibrated method 使用；正式 complete-action 结果应看 ablation 的 `blocksize_calibrated_floor_only`。

| setting | method | score | token | speed |
|---|---|---:|---:|---:|
| LongBench m20 | full_raw | 0.3596 | 100.00% | 1.000x |
| LongBench m20 | free router | 0.2579 | 8.18% | 1.630x |
| LongBench m20 | legacy lattice floor | 0.3652 | 15.07% | 1.560x |
| RULER 4k m20 | full_raw | 1.0000 | 100.00% | 1.000x |
| RULER 4k m20 | free router | 0.8063 | 17.34% | 1.281x |
| RULER 4k m20 | legacy lattice floor | 0.9938 | 30.48% | 1.223x |
| RULER 8k m20 | full_raw | 1.0000 | 100.00% | 1.000x |
| RULER 8k m20 | free router | 0.7500 | 8.52% | 1.638x |
| RULER 8k m20 | legacy lattice floor | 0.9812 | 15.21% | 1.563x |
| RULER 16k m20 | full_raw | 0.8688 | 100.00% | 1.000x |
| RULER 16k m20 | free router | 0.7625 | 4.81% | 2.162x |
| RULER 16k m20 | legacy lattice floor | 1.0000 | 10.92% | 1.992x |

结论：

```text
free small-block router 在 m20 上明显不安全；
legacy floor 大幅修复，但 RULER4k/8k 仍有少量失败。
```

## m20 Complete-Action Calibration / Ablation

`blocksize_calibrated_floor_only` 是当前正确的 complete-action calibrated method。

| setting | method | score | token | speed |
|---|---|---:|---:|---:|
| LongBench m20 | full_raw | 0.3596 | 100.00% | 1.000x |
| LongBench m20 | calibrated b128 top12 | 0.3652 | 15.07% | 1.555x |
| RULER 4k m20 | full_raw | 1.0000 | 100.00% | 1.000x |
| RULER 4k m20 | calibrated b256 top3 | 0.9938 | 30.48% | 1.224x |
| RULER 8k m20 | full_raw | 1.0000 | 100.00% | 1.000x |
| RULER 8k m20 | calibrated b256 top3 | 0.9812 | 15.21% | 1.589x |
| RULER 16k m20 | full_raw | 0.8688 | 100.00% | 1.000x |
| RULER 16k m20 | calibrated b512 top3 | 1.0000 | 10.92% | 1.995x |

聚合平均（四组等权）：

```text
full_raw score ~= 0.8071
complete-action calibrated score ~= 0.8351
average token ~= 17.92%
```

这个结果仍然很强：

```text
1. 平均分高于 full_raw；
2. token 约 18%；
3. speed 约 1.59x；
4. LongBench 和 16k 都优于 full_raw。
```

但 RULER4k/8k 没有达到 100%，因此不能直接声称 full-quality on all RULER m20。

## Fixed Action Ablation

| setting | b128 top12 | b256 top3 | b512 top3 |
|---|---:|---:|---:|
| LongBench m20 | 0.3652 / 15.07% | 0.2905 / 10.91% | 0.2962 / 16.19% |
| RULER 4k m20 | 0.9812 / 27.39% | 0.9938 / 30.48% | 0.9938 / 42.66% |
| RULER 8k m20 | 0.9625 / 16.87% | 0.9812 / 15.21% | 1.0000 / 22.32% |
| RULER 16k m20 | 0.9437 / 9.65% | 0.9750 / 7.47% | 1.0000 / 10.92% |

关键结论：

```text
1. LongBench 明显偏好 b128 top12。
2. RULER8k/16k 如果要 100%，b512 top3 更稳。
3. RULER4k 的 b256/b512 top3 都只到 0.9938，说明需要 topK refinement。
4. 不存在一个固定 block/topK 能同时最优。
```

这支持论文创新点：

```text
RiskKV-Block is a calibrated complete-action policy,
not a fixed block-size heuristic.
```

## 失败点

主实验中 legacy calibrated floor 的失败：

```text
RULER4k:
  niah_single_1 case 11
  action = b256 top3
  prediction = 1234567890
  answer = 5107245

RULER8k:
  niah_single_2 case 11
  action = b256 top3
  prediction = 40
  answer = 5443951

  vt case 14
  action = b256 top3
  prediction contains unrelated variable sequence
  answers = GKRCZ, BNNZO, KMQZG, AUITQ, UTIBO

  vt case 17
  action = b256 top3
  prediction contains unrelated variable sequence
  answers = BUJJO, PRYJT, ONKQG, CTMKO, GGLZH
```

Interpretation：

```text
1. top3 对少数 single-needle / vt cases 仍然不够。
2. 8k 的 b512 top3 已经达到 100%，但 token 从 15.21% 增到 22.32%。
3. 4k 需要测试 b256 top4/top8 或 b512 top4/top8。
```

## 下一步 Floor Refinement

已新增脚本：

```text
ymluo/projects/learned_hierarchical_summary_memory/scripts/run_qwen8b_blocksize_floor_refinement_m20_20260708.sh
```

候选方法：

```text
full_raw
b128 top12
b128 top16
b256 top3
b256 top4
b256 top8
b256 top12
b512 top3
b512 top4
b512 top8
```

目标只跑：

```text
RULER4k
RULER8k
```

判定标准：

```text
1. 找到 RULER4k m20 score=1.0 的最低 token action；
2. 找到 RULER8k m20 score=1.0 的最低 token action；
3. 如果 b256 top4/top8 解决 8k，就不用升到 b512；
4. 如果 4k 仍然失败，需要按 task type calibration。
```

## 对论文当前判断

现在 ICLR/ICML 叙事更清楚，但也更诚实：

```text
旧说法：block size 越大越安全，所以做 floor。
新说法：block size 改变 evidence discretization，因此安全性不是单调的；
       我们校准的是完整 action。
```

这个新说法更有创新性，也更符合实验事实。

当前最强可写主结果仍然是：

```text
complete-action calibrated RiskKV-Block
```

但在最终投稿前，需要完成 RULER4k/8k floor refinement，让 RULER m20/full 更接近 100%。
