# Horizon-PCIC 关键结果汇总（2026-06-29）

本文件由 `scripts/summarize_horizon_pcic_results.py` 从远端 `outputs/*/pcic_r_blockwise_results.csv` 自动生成。

## 主结果表

| run | blocks | avg_delta_ppl | selected/base | serial_total/base | batched_proxy/base | gate_s | extended | early | avg_ext_cands | combos |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Hard-topic eval64 raw s64 | 4 | -0.076940 | 1.042 | 8.311 | n/a | 62.010 | 0 | 0 | 0.00 | `0,7;0,6;0,13;2,0,7,12` |
| Hard-topic eval64 top2 cascade | 4 | -0.076940 | 1.043 | 4.028 | 1.054 | 25.392 | 3 | 1 | 3.67 | `0,7;0,6;0,13;2,0,7,12` |
| Hard-topic eval128 raw s64 | 4 | 0.004371 | 1.043 | 4.689 | n/a | 62.204 | 0 | 0 | 0.00 | `0,7;2,0,7,12;0,6;0,13` |
| Hard-topic eval128 top2 cascade | 4 | 0.004371 | 1.037 | 2.648 | 1.039 | 27.085 | 1 | 3 | 3.00 | `0,7;2,0,7,12;0,6;0,13` |
| War raw s64 | 2 | -2.135311 | 1.039 | 4.157 | n/a | 13.120 | 0 | 0 | 0.00 | `0,7;0,7` |
| War top2 cascade | 2 | -2.135311 | 1.038 | 2.596 | 1.039 | 6.544 | 0 | 2 | 0.00 | `0,7;0,7` |
| Monte raw s64 | 2 | -0.219215 | 1.037 | 4.177 | n/a | 13.098 | 0 | 0 | 0.00 | `2,7;2,0` |
| Monte top2 cascade | 2 | -0.219215 | 1.037 | 2.596 | 1.054 | 6.573 | 2 | 0 | 2.50 | `2,7;2,0` |

## Gate 成本下降

| dataset | raw_s64_gate_s | top2_gate_s | reduction | quality_same |
| --- | ---: | ---: | ---: | --- |
| Hard-topic eval64 | 62.010 | 25.392 | 59.1% | True |
| Hard-topic eval128 | 62.204 | 27.085 | 56.5% | True |
| War | 13.120 | 6.544 | 50.1% | True |
| Monte | 13.098 | 6.573 | 49.8% | True |

## 质量参考 baseline

| dataset | baseline | avg_delta_ppl |
| --- | --- | ---: |
| Hard-topic eval64 | none | 0.030228 |
| Hard-topic eval64 | conffast_s8 | 0.003316 |
| Hard-topic eval64 | static_0,6 | -0.012719 |
| Hard-topic eval128 | none | 0.009629 |
| Hard-topic eval128 | conffast_s8 | 0.038679 |
| Hard-topic eval128 | static_0,6 | -0.020744 |
| War | old_no_rescue | -0.687288 |
| Monte | old_no_rescue | -0.108427 |

## 当前可写进 paper 的结论

1. `top2 cascade` 在 Hard-topic、War、Monte 上保持 `raw_s64` 的质量。
2. `top2 cascade` 将 gate 串行成本降低约 50%–59%，但仍未解决全部端到端速度问题。
3. 方法主线应表述为：`Pairwise-CIC + online blockwise policy selection + horizon-aware top-k cascade rescue gate`。
4. 创新点不是固定稀疏注意力规则，而是在线反事实候选评估预算分配。

