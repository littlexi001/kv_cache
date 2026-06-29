# PCIC Paper Readiness Gate（2026-06-29）

目的：把当前 paper 主线转成可重复检查的 readiness gate。每次补实验后重跑该脚本，即可看到哪些 claim 已经能写，哪些仍缺证据。

## 总览

- pass: 6
- partial: 0
- missing: 3

当前判断：方法创新性主线已经成型，但还不能标记为 ICML-ready；主要缺口是严格组件消融、正式 benchmark、真实速度。

## Gate Table

| gate | status | evidence | next action |
| --- | --- | --- | --- |
| `method_definition` | `pass` | Method spec contains formal problem, Pairwise-CIC, rescue gate, and Algorithm 1. | Keep as paper Method section seed. |
| `fixed_online_oracle` | `pass` | Mainline table shows conditional rescue reaches oracle on Hard-topic and online beats best fixed on Monte. | Replicate on formal LongBench/RULER subset. |
| `blockwise_dynamic_trace` | `pass` | Trace table shows non-trivial combo switches across hard-topic and RULER-style variable/topic cases. | Turn trace into paper figure with block text/task positions. |
| `corrected_speed_accounting` | `pass` | Corrected gate document prevents overclaiming speed and records conservative method/base ratios. | Implement fused/sparse candidate probe before claiming baseline speed. |
| `component_claim_boundary` | `pass` | Component matrix separates supported claims from missing direct ablations. | Update after strict ablation suite finishes. |
| `strict_component_ablation` | `missing` | strict ok=0, strict missing=8, historical baseline rows=6. | Run ONLY_CASES P0 first, then P1/P2 if P0 supports claim. |
| `rescue_quality_case` | `pass` | Hard main_cond_rescue improves over no_validation_anchor_top2 by -0.054004 ΔPPL. | Add memory_only_no_rescue comparison to isolate rescue gate itself. |
| `formal_benchmark` | `missing` | Current RULER-style results are synthetic/offline smoke, not formal RULER/LongBench. | Run formal or locally cached benchmark subset without external downloads. |
| `real_speed` | `missing` | Corrected gate shows method cost remains above baseline; fused/sparse candidate probe is not done. | Implement fused/sparse probe or report speed as limitation. |

## 结论

- 可以继续沿 `Pairwise-CIC + online blockwise selection + rescue gate` 主线推进。
- 当前最强、最安全的论文 claim 是：online counterfactual policy selection 能修复固定策略 / short-horizon 的失败。
- 暂时不能强 claim：端到端快于 baseline、Pairwise/memory 在所有任务上不可替代、正式 benchmark 已充分验证。

CSV：`docs/pcic_paper_readiness_gate_2026_06_29.csv`
