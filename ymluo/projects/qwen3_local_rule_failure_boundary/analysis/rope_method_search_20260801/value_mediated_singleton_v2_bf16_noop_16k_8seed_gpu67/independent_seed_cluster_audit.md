# Value-mediated no-op：独立 seed-cluster 审计

独立统计单位是 **seed**。Layer、head、token 行仅在 seed 内聚合，绝不作为独立样本 bootstrap。
本报告只读取 `merged/case_rows.csv`、`merged/value_samples.csv` 和 raw result；
没有拼接重复的 `first_order_prediction_summary` / `singleton_prediction_summary`，也没有使用 `value_sample_summary` 中重复的 `all` alias。

- Bootstrap：percentile 95% CI，50000 次，固定 RNG seed 20260801。
- Seeds：[0, 1, 2, 3, 4, 5, 6, 7]；实际长度：[16384]。
- Raw/case 审计全部通过：True。

## 一阶预测闭合

| 范围 | events | Pearson | 95% CI | Spearman | 95% CI | sign accuracy | 95% CI |
|---|---:|---:|---:|---:|---:|---:|---:|
| all_target | 32 | 0.866119 | [0.748752, 0.935032] | 0.820015 | [0.704142, 0.89286] | 0.6875 | [0.59375, 0.8125] |
| evidence_only_target | 16 | 0.9727 | [0.946473, 0.988145] | 0.976471 | [0.91018, 1] | 0.9375 | [0.8125, 1] |
| gold_target | 8 | 0.95976 | [0.744833, 0.994454] | 0.904762 | [0.392405, 1] | 0.875 | [0.625, 1] |
| conflict_target | 8 | 0.943465 | [0.819335, 0.98991] | 0.97619 | [0.72973, 1] | 1 | [1, 1] |
| non_evidence_target | 16 | 0.469374 | [-0.182605, 0.708435] | 0.217647 | [-0.305423, 0.611276] | 0.4375 | [0.1875, 0.6875] |
| random_evidence | 16 | 0.0514889 | [-0.604931, 0.508203] | -0.0470588 | [-0.654545, 0.459701] | 0.5625 | [0.375, 0.75] |

Evidence-only 回归：

- slope 1.15995, CI [1.03112, 1.32129]；
- intercept 0.00829464, CI [-0.0118787, 0.0285442]；
- R² 0.946145, CI [0.895811, 0.97643]。

## Seed-macro Gold − Conflict

| 指标 | 差值 | seed-cluster 95% CI |
|---|---:|---:|
| dm_dscore | 0.00128447 | [0.000814679, 0.00186148] |
| direct_ov_centered_margin_derivative | 0.0263228 | [0.0218541, 0.0301782] |
| attention_probability | 0.000504166 | [-0.0016389, 0.00275264] |
| suppression_gap | 0.09392 | [-0.0257063, 0.231853] |

## Evidence target − matched random

| 指标 | 配对差 | seed-cluster 95% CI |
|---|---:|---:|
| sign_accuracy | 0.375 | [0.125, 0.625] |
| symmetric_closure_error | -0.785169 | [-0.876214, -0.661341] |
| absolute_closure_error | -0.00655699 | [-0.0275446, 0.0113627] |
| absolute_actual_delta_margin | 0.0977892 | [0.0397668, 0.147574] |
| delta_gold_nll | -0.00569396 | [-0.042475, 0.0222727] |

## 审计限制

- Instrumented 与 no-op margin/NLL 严格一致：True / True。
- Custom no-op 相对 native 的最大绝对 pair-margin drift：0.238789；top-1 agreement：1。
- Candidate ranking 使用 oracle answer gradient；这里只能作为机制诊断，不能称可部署 selector。
- Evidence target 与 random 没有匹配 decisive-token 身份；相关差异不能解释为公平检索基线优势。
- 本批实验只有 Qwen3-8B、上下文长度 16,384 tokens、score lift [0.25] 和 8 个合成 seeds，不能外推到其他长度、模型或真实任务。
- 当前 `merged/merge_config.json` 是 shard-derived schema，且与实际 CSV/raw 内容一致；`merge_config_legacy_incorrect.json` 仅为旧错误配置的归档，不参与本报告统计。
