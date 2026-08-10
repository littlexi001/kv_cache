# QKSieve-Robust 冻结质量验证

## 研究问题

同一套不含 router、长度切换和 Full fallback 的 QKSieve-Robust 参数，能否在
4K--128K 的 RULER、完整 LongBench 和 Llama/Qwen/Mistral 三个模型上保持 Full
KV 的任务质量？可证伪标准是：完整配对后总体质量显著下降，或质量随长度增加
出现无法由 Full 基线和任务难度解释的系统性崩溃。

## 先验与数学模型

先验一是每个 attention head 的主要输出由少量高分 token 和未选 token 的聚合
Value 贡献决定。对 query `q`、历史 key/value `(k_i,v_i)`，Full attention 为：

`y = sum_i exp(q^T k_i) v_i / sum_i exp(q^T k_i)`。

QKSieve 用压缩代理分数选集合 `S`，在 `S` 上读取原始 FP16/BF16 K/V 并计算精确
attention；未选集合用 rank-16、block-256、INT4 ValueSketch 估计 softmax 分子和
分母。若代理排序保留主要 softmax mass，且 ValueSketch 的尾部估计误差有限，
则输出误差由“漏选质量”和“尾部估计误差”共同控制，而不是只由 top-k recall
决定。

先验二是这一机制应由当前请求的 Q/K 二阶矩决定，而不依赖任务标签。每层、每个
KV head 使用 request-local QK-balanced 坐标和 query-weighted qMSE/OAS，在固定
240 bit/token/head 下分配 8 个 16-D band 的位宽。token 预算固定为
`min(N,1280,max(256,ceil(0.06N)))`，阈值最多采样 512 个位置。

## 实现合同

- 冻结数值提交：`328e01718deebfdfc80dbd8e588a1a95a1832b59`；
- 审计实现提交：`f300fb280a597ceb124d454cdfc9a0a1665d6a04`；
- 240-bit Key 索引，rank-16/block-256/INT4 ValueSketch，`alpha=0.5`；
- 原始 K/V 只在选中位置参与精确 attention；
- 无 exact-QK rerank、recent/sink 保留、router、任务规则、长度切换或 Full fallback；
- 每条结果记录实际执行路径、quantile 样本数、ValueSketch 开关和候选预算。

## 实验设置

RULER 使用 Llama-3.1-8B-Instruct、13 个官方任务和 seed 42。4K/8K/16K/32K
每个 task-length 10 条，64K/128K 每个 task-length 5 条，共 650 个严格
Full--Robust pair。主指标是 78 个 task-length 单元的宏平均、相对 Full 和
10,000 次 paired bootstrap 95% CI。Prompt 审计要求全部不超过模型的 131,072
位置上限。

跨模型筛查使用 LongBench 16 个英文任务，每个模型固定 offset 40 后每任务取
10 条，共 160 pair；模型为 Llama-3.1-8B、Qwen3-4B 和 Mistral-7B。Llama 的
完整 LongBench 主表另含 3,750 pair，使用官方 middle truncation 和统一停止规则。

脚本与原始结果位于：

- `src/merge_qksieve_ruler_distributed_20260810.py`；
- `src/summarize_qksieve_robust_ruler_20260810.py`；
- `raw_results/ruler_v2_distributed_audited/`；
- `visualization_results.md`。

## RULER 结果

| 长度 | Full | Robust | 相对 Full | 95% CI |
|---:|---:|---:|---:|---:|
| 4K | 0.883077 | 0.878974 | 99.535% | [98.848%, 100.000%] |
| 8K | 0.887564 | 0.887051 | 99.942% | [99.283%, 100.790%] |
| 16K | 0.878077 | 0.888974 | 101.241% | [99.063%, 105.349%] |
| 32K | 0.834231 | 0.831667 | 99.693% | [99.021%, 100.000%] |
| 64K | 0.809231 | 0.809231 | 100.000% | [100.000%, 100.000%] |
| 128K | 0.757179 | 0.770256 | 101.727% | [95.208%, 111.353%] |
| **总体** | **0.841560** | **0.844359** | **100.333%** | **[99.185%, 101.750%]** |

平均 active attention 为 842.39 token/head，即历史的 4.971%。最差单元
`niah_multikey_2@128K` 从 1.0 降至 0.8；按长度平均后最差任务保持率为
96.667%。因此总体假设通过，但“每个单元都等价”的更强假设失败。

## LongBench 与跨模型结果

完整 Llama LongBench 的 Full/Robust macro 为 `0.459011/0.458692`，保持率
`99.930%`，95% CI `[99.538%,100.213%]`。160-pair 跨模型筛查的保持率为：

- Llama-3.1-8B：98.681%，CI `[96.393%,100.504%]`；
- Qwen3-4B：100.211%，CI `[98.907%,101.720%]`；
- Mistral-7B：98.487%，CI `[95.084%,100.893%]`。

三个区间均跨 100%，支持“没有观察到跨模型系统性崩溃”，不支持逐模型严格等价。

## 来源审计与失败解释

RULER 合并器从主机原子选择 624 个完整 pair，从补充机选择 26 个完整 pair；
跨主机拼接半对的数量为 0。两台机器的模型权重、tokenizer 和 config 哈希完全
一致。33 条重叠结果中只有一条 Robust 预测文本不同，但两侧官方得分均为 0；
这是跨驱动生成的轻微非确定性，不影响质量结论，具体预测哈希已记录。

短长度 RULER harness 的在线速度可能低于 Full，因为输出通常只有数个 token，
无法摊薄索引与 query 成本；该 harness 还允许两种方法生成不同 token 数。因此
它不用于系统速度主张，固定步数的 MHA attention/decode benchmark 才是速度证据。

## 结论与边界

冻结 QKSieve-Robust 在完整 RULER、完整 Llama LongBench 和三模型独立筛查上没有
出现总体质量崩溃，且所有结果都不依赖 fallback。证据覆盖 4K--128K，但不直接
外推到 256K/512K，也不证明每个任务、每个模型与 Full 严格等价。质量子问题已
稳定；论文总证据链当前只缺真实 H100 上的 64K/128K attention、稳态 decode 和
请求级速度复测。
