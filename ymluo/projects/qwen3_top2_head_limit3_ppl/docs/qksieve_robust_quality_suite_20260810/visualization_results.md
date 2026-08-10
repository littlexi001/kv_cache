# 最终结果

## 实验设置

本实验比较 `full_kv` 与冻结的
`qksieve_qmse_oas_requestlocal_valuesketch16_sorted_c64`。后者固定使用
request-local QK-balanced、query-weighted qMSE/OAS、240-bit Key 索引、最多
512 个阈值样本，以及 rank-16、block-256、INT4、`alpha=0.5` 的
ValueSketch。候选预算为
`min(N, 1280, max(256, ceil(0.06N)))`，没有 router、长度切换或 Full
fallback。

RULER 覆盖 13 个任务。4K/8K/16K/32K 每个 task-length 使用 10 条样本，
64K/128K 使用 5 条；最终证据包含 650 个严格配对、1,300 行和 78 个完整
task-length 单元。

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

v2 最终包含 650 个严格配对。分布式合并器原子选择 624 个主机 pair 和 26 个
补充机 pair，跨主机拼接 pair 的数量为 0。所有 Robust 行通过路径审计，fallback
和路径错配均为 0。两台机器的模型权重与 tokenizer/config 逐文件哈希一致。
33 条重叠结果中只有 `qa_squad_65536_2` 的 Robust 文本不同，两侧得分均为 0，
因此不改变任何质量统计；具体预测哈希保存在 `merge_audit.json`。

## RULER 质量

首先看相对 Full 的质量保持率。100% 表示与 Full 宏平均相同；低于 100% 表示
下降。总体宏平均按 78 个 task-length 单元等权计算，区间通过 10,000 次
task-length bootstrap 得到。

| 长度 | Full | Robust | 相对 Full | 95% CI |
|---:|---:|---:|---:|---:|
| 4K | 0.883077 | 0.878974 | 99.535% | [98.848%, 100.000%] |
| 8K | 0.887564 | 0.887051 | 99.942% | [99.283%, 100.790%] |
| 16K | 0.878077 | 0.888974 | 101.241% | [99.063%, 105.349%] |
| 32K | 0.834231 | 0.831667 | 99.693% | [99.021%, 100.000%] |
| 64K | 0.809231 | 0.809231 | 100.000% | [100.000%, 100.000%] |
| 128K | 0.757179 | 0.770256 | 101.727% | [95.208%, 111.353%] |
| **总体** | **0.841560** | **0.844359** | **100.333%** | **[99.185%, 101.750%]** |

Robust 平均每个 head 使用 842.39 个 attention token，即历史的 4.971%。最差
单元是 `niah_multikey_2@128K`：Full 为 1.0，Robust 为 0.8；该长度每个单元
只有 5 条，因此单元方差较大。按六个长度平均，最差任务仍是
`niah_multikey_2`，保持率为 96.667%。这说明总体没有随长度增长的系统性崩溃，
但不支持“每个 task-length 单元都与 Full 等价”。

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

## 结论边界

正式 RULER、完整 3,750-pair Llama LongBench 和三模型 screen 已通过冻结合同
验证。它们支持“同一组超参数在 4K--128K、13 个 RULER 任务和三个模型上没有
观察到总体质量崩溃”。由于部分单元样本少、跨模型 screen 每任务只有 10 条，
不能声称逐任务、逐模型严格等价。RULER harness 的输出长度不同，其速度只作
诊断；论文系统速度必须引用固定生成步数的 MHA attention/decode 测试。

最终 RULER 汇总 SHA256 为
`bf945639ecd056b0dc0e7e6411be5c081a0fcaea7821caffa41028eae77c82b1`，合并 CSV
SHA256 为 `b27ef7fd63c3bf836b4fca457da92d288acb60fe4a0e88a4e03b13460b1353f8`。
