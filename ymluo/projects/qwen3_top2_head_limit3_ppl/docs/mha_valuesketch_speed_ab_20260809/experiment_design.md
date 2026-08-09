# MHA 下 ValueSketch 有无补偿速度对照：实验方案

## 研究问题

在候选 token 完全相同时，ValueSketch 补偿对 MHA attention 子系统和真实模型稳态 decode 的速度成本是多少？

## 条件

| 条件 | 候选选择 | 候选 K/V | 未选 Value 补偿 |
|---|---|---|---|
| Full | 不检索 | 全部 FP16 K/V | 不适用 |
| QKSieve-Fast | QK-balanced，最多 1,280/head | 原始 FP16 | 关闭 |
| QKSieve-Robust | 与 Fast 完全相同 | 原始 FP16 | rank-16、block-256、INT4 |

## Attention 子系统

- 硬件：单张 RTX 3090。
- 张量布局：batch 1，MHA 32Q/32KV，head dimension 128。
- 长度：8K、16K、32K、64K、128K。
- 随机种子：20260809、20260810、20260811。
- 每个种子：10 次 warmup、40 次测量；64K/128K 的脚本上限为 20 次测量。
- 计时：独立 CUDA events，分别测 Query 投影量化、selector 扫描、候选 attention、带补偿的合并 attention 和完整路径。
- 汇总：三个种子的中位数。
- 通过条件：所有长度和种子中，Fast 与 Robust 的阈值最大绝对差为 0、候选数最大差为 0、候选集合完全一致。

脚本：`src/benchmark_qksieve_fier_mha_speed_20260808.py`。

远端结果：`/home/fdong/qksieve_iclr2027/results/20260809_mha_valuesketch_attention_ab_v1/`。

## 真实模型稳态 Decode

- 模型：`Yarn-Llama-2-7b-128k`，FP16，标准 HF Llama，`trust_remote_code=0`。
- 长度：32K、64K、128K。
- 生成：greedy 64 token；前 16 token 作为 warmup，后 48 token求平均。
- 卡数：32K 两张、64K 三张、128K 八张 RTX 3090；同一长度内三个条件卡数相同。
- Full、Fast、Robust 均重新从相同文本构造历史 KV。
- 一次性 prefill、QK 索引和 ValueSketch 构建时间单独记录。

脚本：`src/run_qksieve_fier_autoregressive_speed_20260808.py` 和 `scripts/launch_qksieve_mha_valuesketch_decode_ab_strict_20260809.sh`。

远端结果：`/home/fdong/qksieve_iclr2027/results/20260809_qksieve_mha_valuesketch_decode_ab_strict_v1/`。

## 判定

- 通过：Robust 在 64K/128K 的 attention 和稳态 decode 都快于 Full，同时候选一致性检查通过。
- 失败：Robust 不快于 Full，或两条路径候选不一致。
- 证据不足：只得到 synthetic attention 或只得到延迟分解，没有真实模型 decode。

## 已知限制

真实 decode 当前为单个模型、单个文本、单个随机种子；它能回答实现速度，不足以给出跨模型置信区间。质量收益由独立 ValueSketch 去留消融报告，不从本速度实验外推。
