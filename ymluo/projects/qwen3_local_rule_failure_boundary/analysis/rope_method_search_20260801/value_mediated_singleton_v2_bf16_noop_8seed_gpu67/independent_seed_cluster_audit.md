# Value-mediated no-op：独立 seed-cluster 审计

独立统计单位是 **seed**。Layer、head、token 行仅在 seed 内聚合，绝不作为独立样本 bootstrap。
本报告只读取 `merged/case_rows.csv`、`merged/value_samples.csv` 和 raw result；
没有拼接重复的 `first_order_prediction_summary` / `singleton_prediction_summary`，也没有使用 `value_sample_summary` 中重复的 `all` alias。

- Bootstrap：percentile 95% CI，50000 次，固定 RNG seed 20260801。
- Seeds：[0, 1, 2, 3, 4, 5, 6, 7]；实际长度：[8192]。
- Raw/case 审计全部通过：True。

## 一阶预测闭合

| 范围 | events | Pearson | 95% CI | Spearman | 95% CI | sign accuracy | 95% CI |
|---|---:|---:|---:|---:|---:|---:|---:|
| all_target | 32 | 0.83258 | [0.729593, 0.931551] | 0.795455 | [0.663723, 0.888807] | 0.78125 | [0.75, 0.84375] |
| evidence_only_target | 16 | 0.960261 | [0.938078, 0.992556] | 0.947059 | [0.849519, 1] | 0.9375 | [0.8125, 1] |
| gold_target | 8 | 0.987093 | [0.903042, 0.997697] | 1 | [1, 1] | 0.875 | [0.625, 1] |
| conflict_target | 8 | 0.968307 | [0.920746, 0.998357] | 0.952381 | [0.620253, 1] | 1 | [1, 1] |
| non_evidence_target | 16 | 0.222793 | [-0.179351, 0.543576] | 0.411765 | [-0.0933735, 0.776119] | 0.625 | [0.5, 0.8125] |
| random_evidence | 16 | 0.58246 | [-0.287666, 0.948048] | -0.0647059 | [-0.476331, 0.435821] | 0.4375 | [0.25, 0.625] |

Evidence-only 回归：

- slope 1.07464, CI [0.931791, 1.20863]；
- intercept 0.00323549, CI [-0.014203, 0.0315864]；
- R² 0.922102, CI [0.87999, 0.985167]。

## Seed-macro Gold − Conflict

| 指标 | 差值 | seed-cluster 95% CI |
|---|---:|---:|
| dm_dscore | 0.00124756 | [0.000829105, 0.00169364] |
| direct_ov_centered_margin_derivative | 0.0242694 | [0.0208041, 0.0272652] |
| attention_probability | -0.000398297 | [-0.00286585, 0.00203566] |
| suppression_gap | 0.150378 | [-0.049716, 0.351835] |

## Evidence target − matched random

| 指标 | 配对差 | seed-cluster 95% CI |
|---|---:|---:|
| sign_accuracy | 0.5 | [0.25, 0.75] |
| symmetric_closure_error | -0.699685 | [-0.832275, -0.544187] |
| absolute_closure_error | -0.00218926 | [-0.0386515, 0.0287372] |
| absolute_actual_delta_margin | 0.0959669 | [0.0378144, 0.163928] |
| delta_gold_nll | -0.0223925 | [-0.0965381, 0.0331535] |

## 审计限制

- Instrumented 与 no-op margin/NLL 严格一致：True / True。
- Custom no-op 相对 native 的最大绝对 pair-margin drift：0.283756；top-1 agreement：1。
- Candidate ranking 使用 oracle answer gradient；这里只能作为机制诊断，不能称可部署 selector。
- Evidence target 与 random 没有匹配 decisive-token 身份；相关差异不能解释为公平检索基线优势。
- 本批实验只有 Qwen3-8B、上下文长度 8,192 tokens、score lift [0.25] 和 8 个合成 seeds，不能外推到其他长度、模型或真实任务。
- 当前 `merged/merge_config.json` 是 shard-derived schema，且与实际 CSV/raw 内容一致；`merge_config_legacy_incorrect.json` 仅为旧错误配置的归档，不参与本报告统计。
