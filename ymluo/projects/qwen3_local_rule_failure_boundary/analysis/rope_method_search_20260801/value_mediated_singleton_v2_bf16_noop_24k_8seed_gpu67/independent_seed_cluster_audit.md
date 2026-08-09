# Value-mediated no-op：独立 seed-cluster 审计

独立统计单位是 **seed**。Layer、head、token 行仅在 seed 内聚合，绝不作为独立样本 bootstrap。
本报告只读取 `merged/case_rows.csv`、`merged/value_samples.csv` 和 raw result；
没有拼接重复的 `first_order_prediction_summary` / `singleton_prediction_summary`，也没有使用 `value_sample_summary` 中重复的 `all` alias。

- Bootstrap：percentile 95% CI，50000 次，固定 RNG seed 20260801。
- Seeds：[0, 1, 2, 3, 4, 5, 6, 7]；实际长度：[24576]。
- Raw/case 审计全部通过：True。

## 一阶预测闭合

| 范围 | events | Pearson | 95% CI | Spearman | 95% CI | sign accuracy | 95% CI |
|---|---:|---:|---:|---:|---:|---:|---:|
| all_target | 32 | 0.788738 | [0.602106, 0.891611] | 0.586144 | [0.392936, 0.799041] | 0.59375 | [0.53125, 0.6875] |
| evidence_only_target | 16 | 0.936083 | [0.848731, 0.972466] | 0.9 | [0.756839, 0.991018] | 0.8125 | [0.625, 0.9375] |
| gold_target | 8 | 0.976286 | [0.889539, 0.99724] | 1 | [1, 1] | 0.875 | [0.625, 1] |
| conflict_target | 8 | 0.884679 | [0.193519, 0.995336] | 0.714286 | [-0.150685, 1] | 0.75 | [0.375, 1] |
| non_evidence_target | 16 | 0.217583 | [-0.432793, 0.775172] | 0.05 | [-0.581602, 0.605343] | 0.375 | [0.1875, 0.625] |
| random_evidence | 16 | -0.226263 | [-0.688122, 0.535078] | -0.0235294 | [-0.548961, 0.5727] | 0.625 | [0.375, 0.8125] |

Evidence-only 回归：

- slope 1.05527, CI [0.846491, 1.24007]；
- intercept -0.00839975, CI [-0.0279641, 0.0158046]；
- R² 0.876251, CI [0.720345, 0.94569]。

## Seed-macro Gold − Conflict

| 指标 | 差值 | seed-cluster 95% CI |
|---|---:|---:|
| dm_dscore | 0.00161679 | [0.00115953, 0.00216051] |
| direct_ov_centered_margin_derivative | 0.0269347 | [0.0235761, 0.0301035] |
| attention_probability | -0.00135538 | [-0.00330615, 0.000802424] |
| suppression_gap | 0.0421594 | [-0.0851678, 0.190029] |

## Evidence target − matched random

| 指标 | 配对差 | seed-cluster 95% CI |
|---|---:|---:|
| sign_accuracy | 0.1875 | [0, 0.4375] |
| symmetric_closure_error | -0.544146 | [-0.7233, -0.379145] |
| absolute_closure_error | 0.00688256 | [-0.0127753, 0.0268166] |
| absolute_actual_delta_margin | 0.0849732 | [0.0303791, 0.136843] |
| delta_gold_nll | -0.00324717 | [-0.0437339, 0.0368621] |

## 审计限制

- Instrumented 与 no-op margin/NLL 严格一致：True / True。
- Custom no-op 相对 native 的最大绝对 pair-margin drift：0.463829；top-1 agreement：1。
- Candidate ranking 使用 oracle answer gradient；这里只能作为机制诊断，不能称可部署 selector。
- Evidence target 与 random 没有匹配 decisive-token 身份；相关差异不能解释为公平检索基线优势。
- 本批实验只有 Qwen3-8B、上下文长度 24,576 tokens、score lift [0.25] 和 8 个合成 seeds，不能外推到其他长度、模型或真实任务。
- 当前 `merged/merge_config.json` 是 shard-derived schema，且与实际 CSV/raw 内容一致。
