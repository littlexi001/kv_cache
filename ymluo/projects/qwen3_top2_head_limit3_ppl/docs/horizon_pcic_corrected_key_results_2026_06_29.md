# Horizon-PCIC Corrected Key Results（2026-06-29）

## 目的

旧 `gate_s` 对 extended cascade 存在低估：它只统计进入 extension 的候选，漏掉了没有进入 extension 的 early probe 候选。本文统一使用 corrected gate：

```text
corrected_gate =
    sum(initial_candidate_seconds except selected)
  + sum(extension_candidate_seconds except selected)
```

原始表：`docs/pcic_corrected_gate_core_results_2026_06_29.csv`

## 核心结果表

| case | avg_delta_ppl | old gate_s | corrected gate_s | corrected method/base | extended | combos |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| hard top2 | 0.004371 | 27.085 | 32.530 | 2.972 | 1 | `0,7/2,0,7,12/0,6/0,13` |
| hard conditional | -0.049633 | 53.648 | 89.624 | 6.227 | 4 | `0,6/2,0,7,12/0,6/0,6` |
| hard b8 top2 | 0.006074 | 49.737 | 75.197 | 3.263 | 5 | `0,7/2,0,7,12/0,6/0,13/2,0,7,12/0,13/7,13/0,13` |
| hard b8 conditional | -0.040598 | 107.643 | 179.677 | 6.248 | 8 | `0,6/2,0,7,12/0,6/0,6/7,13/0,6/7,13/2,0` |
| war top2 | -2.135311 | 6.544 | 6.544 | 2.596 | 0 | `0,7/0,7` |
| war conditional | -2.135311 | 6.659 | 6.659 | 2.603 | 0 | `0,7/0,7` |
| monte top2 | -0.219215 | 6.573 | 9.913 | 3.388 | 2 | `2,7/2,0` |
| monte conditional | -0.219215 | 6.641 | 10.011 | 3.392 | 2 | `2,7/2,0` |
| needle top2 | 0.000118 | 13.206 | 37.263 | 3.262 | 4 | `2,0,7,12/2,0/2,0/2,0` |
| needle conditional | -0.000166 | 39.909 | 81.705 | 5.869 | 4 | `2,0/2,0/2,0,7,12/2,0` |
| ruler multi-needle top2 | 0.000074 | 11.081 | 28.577 | 4.051 | 3 | `2,0/2,0/2,0` |
| ruler multi-needle conditional | 0.000074 | 16.650 | 42.868 | 5.534 | 3 | `2,0/2,0/2,0` |
| ruler variable top2 | 0.017397 | 11.857 | 24.768 | 3.699 | 2 | `2,0/2,0,7,12/2,0` |
| ruler variable conditional | -0.000564 | 13.249 | 41.103 | 5.388 | 3 | `0,13/2,0/2,0` |
| ruler topic-switch top2 | 0.000302 | 6.530 | 26.031 | 3.809 | 3 | `2,0/2,0/2,0` |
| ruler topic-switch conditional | 0.000139 | 9.931 | 39.543 | 5.209 | 3 | `0,13/2,0/2,0` |

## 质量结论

- Hard-topic：conditional anchor rescue 把 avg ΔPPL 从 `0.004371` 改善到 `-0.049633`。
- Hard-topic b8：conditional rescue 仍有收益，`0.006074 -> -0.040598`。
- War / Monte：conditional 不破坏已有 top2 质量。
- Needle / RULER-style：conditional 在 variable binding 和 topic-switch 上改善 PPL drift；multi-needle 持平。

## 速度结论

- corrected gate 后，conditional rescue 的真实候选 probe 成本比旧表更高。
- 当前方法的 paper claim 不能写成“已经快于 baseline”。
- 可写成：quality gain 明确，系统速度仍依赖 fused/sparse candidate probe 或更强 calibrated skip-gate。

## 当前主线表述

```text
Pairwise-CIC
+ online blockwise policy selection
+ conditional validation-prior horizon-anchor rescue gate
```

该主线的创新性仍主要来自“在线反事实策略选择 + horizon rescue 仲裁”，不是来自某个固定 sparse attention pattern。
