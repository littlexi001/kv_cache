# 阶段性结果

## 实验设置

本实验比较 `full_kv` 与冻结的
`qksieve_qmse_oas_requestlocal_valuesketch16_sorted_c64`。后者固定使用
request-local QK-balanced、query-weighted qMSE/OAS、240-bit Key 索引、最多
512 个阈值样本，以及 rank-16、block-256、INT4、`alpha=0.5` 的
ValueSketch。候选预算为
`min(N, 1280, max(256, ceil(0.06N)))`，没有 router、长度切换或 Full
fallback。

RULER 覆盖 13 个任务。4K/8K/16K/32K 每个 task-length 使用 10 条样本，
64K/128K 使用 5 条；最终证据必须包含 650 个严格配对、1,300 行和 78 个
完整 task-length 单元。本页记录运行中的固定快照，不替代最终汇总。

## 审计修正

第一轮 RULER 运行虽然生成了可评分的预测，但启动器没有打开
`--collect_attention_stats`。因此它不能证明量化检索、有效阈值样本数和
ValueSketch 在每条样本上实际执行。该轮 CSV 与日志被保留用于故障追踪，但所有
分数均从论文证据中排除；不能用“方法名正确”替代执行路径证据。

修正后的 v2 运行逐行要求：

- `executed_path` 等于冻结 Robust 方法；
- `packed_qmse_sample_count` 为正且不超过 512；
- `packed_qmse_value_sketch_executed=1`，rank/bits/alpha 分别为 16/4/0.5；
- `sampled_quantile_fallback=0`；
- Full 与 Robust 的 `task + sample_id` 严格配对。

截至本次记录，v2 已产生 1,008 行和 504 个严格配对，覆盖 13 个任务与
4K--32K 的 52 个 task-length 单元；504 条 Robust 行全部通过上述路径审计，
fallback 和路径错配均为 0。正式分数要等 64K/128K、650 个配对和 10,000 次
paired bootstrap 全部完成后再写入，运行中的局部均值不作为论文结果。

## 三模型 LongBench

冻结 Robust 已完成独立的三模型筛查。每个模型使用 LongBench 16 个英文任务、
每任务 10 条、offset 40，共 160 个严格 Full--Robust 配对；所有模型使用同一套
预算、量化、ValueSketch 和无 fallback 合同。

| 模型 | Full macro | Robust macro | 相对 Full | task-bootstrap 95% CI | 平均 active attention |
|---|---:|---:|---:|---:|---:|
| Llama-3.1-8B-Instruct | 0.426435 | 0.420810 | 98.681% | [96.393%, 100.504%] | 7.215% |
| Qwen3-4B-Instruct | 0.397178 | 0.398015 | 100.211% | [98.907%, 101.720%] | 7.178% |
| Mistral-7B-Instruct-v0.3 | 0.421645 | 0.415267 | 98.487% | [95.084%, 100.893%] | 6.835% |

三者均完成 160 个配对、16 个任务，fallback 为 0；有效 quantile 样本均值分别
为 508.8、508.8、510.4。汇总 JSON 的 SHA256 为
`9ceadaace51808989a222df6669ca5261e233c8ebd21f67b3f2bcd9851b8bf30`，并已通过
`validate_multimodel` 的冻结合同检查。

绝对分数下降最大的任务分别是 Llama 的 LCC（-0.064286）、Qwen 的 GovReport
（-0.014001）和 Mistral 的 HotpotQA（-0.100000）。由于每个任务只有 10 条，
三个置信区间均跨过 100%；该实验只支持“未观察到跨模型系统性崩溃”，不能证明
每个模型与 Full 等价，也不能替代 Llama 的完整 3,750 样本主表。

## 尚不能下的结论

修正后的正式 RULER 和冻结 Robust 的完整 3,750-pair Llama LongBench 仍在运行。
在其汇总器与总 evidence verifier 通过前，不能声称 4K--128K 全长度等价，也
不能把旧 reference profile 的 3,750-pair 结果写成部署主路径结果。
