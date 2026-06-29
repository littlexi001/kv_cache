# PCIC Component Evidence Matrix（2026-06-29）

目的：把 paper 的三个核心贡献拆成可审稿的 evidence / boundary / missing ablation，避免把尚未证明的内容写成强 claim。

## 总表

| component | status | positive evidence | boundary / missing | source |
| --- | --- | --- | --- | --- |
| Online blockwise selection | `supported_but_needs_standard_benchmark` | Monte cond/online vs best fixed ΔPPL -0.210210；Hard-topic eval128 cond vs best fixed ΔPPL -0.028889。 | War 是 easy regime，fixed=online=oracle，说明动态选择收益依赖非平稳文本。 | `docs/pcic_mainline_fixed_online_rescue_2026_06_29.md` |
| Conditional horizon rescue gate | `quality_supported_speed_not_solved` | Hard-topic eval128 top2 ΔPPL 0.004371 -> cond -0.049633；cond-oracle gap 0.000000。 | corrected gate_s 从 top2 32.530 增至 cond 89.624，速度仍是瓶颈。 | `docs/horizon_pcic_corrected_key_results_2026_06_29.md` |
| Validation-prior anchor | `supported_with_threshold_risk` | margin=0.012: Hard-topic ΔPPL -0.049633；War ΔPPL -2.135311；Monte ΔPPL -0.219215。 | 阈值 0.012 仍是经验选择，需要标准验证集或自适应 margin 规则。 | `docs/pcic_conditional_auto_anchor_margin_grid_2026_06_29.md` |
| RULER-style variable rescue | `mechanism_supported_not_formal_benchmark` | RULER-style variable top2 ΔPPL 0.017397 -> cond -0.000564。 | 这是 synthetic/offline smoke，不是正式 RULER；只能作为机制证据。 | `docs/pcic_ruler_style_smoke_2026_06_29.md` |
| Skip/early-exit heuristics | `negative_ablation_guides_future_speed` | skip rule 在 corrected gate 上能省成本，如 needle 81.705 -> 67.423。 | needle ΔPPL 退化 0.000137；因此默认关闭，不作为主方法。 | `docs/pcic_corrected_gate_skipanchor_gain_2026_06_29.md` |
| Pairwise-CIC / risk memory ablation | `missing_direct_ablation` | 当前 Method spec 已定义；已有 fixed/online/oracle 与 trace 间接支持 policy selection。 | 缺少直接 no-pairwise、no-memory 消融；这是 ICML 证据链的 P0 缺口。 | `docs/horizon_pcic_method_spec_2026_06_29.md` |

## 论文写法建议

- 可以强写：`online blockwise policy selection`、`conditional horizon rescue gate`、`fixed policy 不足以覆盖非平稳文本`。
- 可以作为机制证据写：RULER-style variable binding smoke、blockwise policy trace、delayed-win case study。
- 必须保守写：速度。corrected gate 后 conditional rescue 仍明显慢，当前不能声称端到端快于 baseline。
- 不能强写：Pairwise-CIC/risk memory 的直接必要性，直到补齐 no-pairwise/no-memory 消融。

## 下一步最小实验矩阵

可执行脚本：

- `scripts/run_pcic_minimal_component_ablation_suite.sh`
- `scripts/summarize_pcic_minimal_component_ablation.py`
- 当前占位结果表：`docs/pcic_minimal_component_ablation_2026_06_29.md`
- 运行 runbook：`docs/pcic_minimal_component_ablation_runbook_2026_06_29.md`
- Paper readiness gate：`docs/pcic_paper_readiness_gate_2026_06_29.md`

说明：占位结果表会自动复用已有 corrected core CSV 填充 `no_validation_anchor_top2` 与 `main_cond_rescue`，严格消融仍显示为 `missing`，直到实际运行 suite。

| ablation | 目的 | 最小数据 | 成功标准 |
| --- | --- | --- | --- |
| no-rescue | 证明 rescue gate 必要 | Hard-topic eval128 + RULER variable | `memory_only_no_rescue` 明显差于 `main_cond_rescue`，且 `no_validation_anchor_top2` 暴露 short-horizon/anchor failure |
| no-memory | 证明 historical prior 必要 | Monte + hard-topic b8 | 设置 `--risk_memory_use_history false` 后 block trace 更不稳或 PPL drift 变差 |
| no-pairwise | 证明 Pairwise-CIC 不是普通 ranking | Monte + RULER variable | 设置 `--pairwise_candidate_probe false` 后更接近 memory/fixed 或 short-horizon failure |
| fixed-best | 证明不是固定 combo | 正式 LongBench/RULER subset | online 接近 oracle 且优于 best fixed |
| fused-probe | 证明系统可行 | hard/war/monte | corrected gate 或 tokens/s 接近 baseline |

## 当前结论

Horizon-PCIC 的 paper 主线已经具备方法创新性雏形：它把 KV compression 从固定稀疏规则提升为在线反事实策略选择。
但 ICML 级投稿还缺两个硬证据：

1. `no-pairwise / no-memory` 直接消融；
2. 正式 benchmark 与真实速度/fused probe。

CSV：`docs/pcic_component_evidence_matrix_2026_06_29.csv`
