# QKSieve MHA 真实整模型 Decode 测速

## 结论

在原生 128K 的 MHA 模型上，QKSieve 的真实逐 token 稳态 decode 加速随上下文长度增长：64K 为 **2.725x**，128K 为 **3.898x**。这里测量的是完整 HuggingFace 模型 forward，包括 selector、候选 K/V 消费、MLP、通信和其余模型开销，不是 attention 微基准或延迟分解估计。

QKSieve 仍有一次性索引和首步准备成本。将这些真实开销计入后，128K 上生成 16、32、64 token 的在线加速分别为 **1.116x、1.736x、2.401x**；64K 上分别为 **0.753x、1.180x、1.647x**。因此该实现适合长上下文、多轮问答和 Agent 复用 KV 的场景，8K 短上下文没有速度收益。

## 实验设置

- 模型：`NousResearch/Yarn-Llama-2-7b-128k`。
- 结构：32 层，32 query heads，32 KV heads，head dimension 128，是真正的 MHA。
- 运行时：FP16，标准 HuggingFace Llama + PyTorch SDPA；Full KV 没有 GQA `repeat_kv`。
- GPU：RTX 3090。8K/16K/32K/64K/128K 分别使用 1/1/2/3/8 张卡；同一长度的 Full 与 QKSieve 使用相同卡数和模型切分方式。
- Prefill：双方均为 dense chunked prefill，不计入稳态 decode；精确 K/V 均常驻 GPU。
- 生成：真实 greedy argmax feedback。16K 以上生成 64 token，前 16 token 不进入稳态均值；8K smoke 生成 32 token，前 8 token不进入稳态均值。
- QKSieve：无 ValueSketch 补偿；候选预算为 `min(1280, max(256, ceil(0.06N)))`；quantile sample 为 512；attention split 为 8。

## 稳态 Decode

| 历史长度 | GPU | token/head | KV 比例 | Full ms/token | QKSieve ms/token | QKSieve tok/s | 加速 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 8K | 1 | 492 | 6.01% | 36.75 | 37.33 | 26.79 | 0.984x |
| 16K | 1 | 984 | 6.01% | 52.39 | 37.86 | 26.42 | 1.384x |
| 32K | 2 | 1,280 | 3.91% | 83.76 | 51.64 | 19.36 | 1.622x |
| 64K | 3 | 1,280 | 1.95% | 144.81 | 53.15 | 18.81 | 2.725x |
| 128K | 8 | 1,280 | 0.98% | 268.19 | 68.80 | 14.54 | **3.898x** |

稳态 decode 是 Agent 连续生成或多轮复用场景中最有代表性的口径。它包含每个 decode step 的全部 QKSieve 计算，但摊销掉了一次性的请求索引准备。

## 包含一次性开销的在线速度

| 历史长度 | 16-token speedup | 32-token speedup | 64-token speedup | 拟合回本点 |
|---:|---:|---:|---:|---:|
| 8K | 0.236x | 0.381x | - | 不回本 |
| 16K | 0.326x | 0.529x | 0.764x | 约 136 token |
| 32K | 0.473x | 0.732x | 1.008x | 约 63 token |
| 64K | 0.753x | 1.180x | 1.647x | 约 25 token |
| 128K | 1.116x | 1.736x | **2.401x** | 约 14 token |

这里的在线时间从 QKSieve 索引准备开始，到指定数量的生成 token 结束。QKSieve 报告的 QK factor prebuild 约为 0.74--0.78 秒，但第一次 sparse step 还有量化索引物化等惰性开销；因此回本点由实际累计时间拟合，而不是只用 prebuild 字段估计。

## Prefill 与完整请求

双方没有优化 prefill。128K 的 dense prefill 分别为 Full `234.19s`、QKSieve `234.50s`；即使后续生成 64 token 的在线部分达到 2.401x，`prefill + 64-token decode` 的完整新请求加速仍只有 **1.040x**。64K 对应为 **1.059x**。

这不否定 decode 加速，但限定了论文中的准确表述：主结果适用于 KV 已存在或可复用的长上下文 decode；不能把它写成“从零开始的短回答端到端接近 4x”。

## 结果边界

本轮确认了真实模型图上的速度，不承担质量结论。输入是固定重复 token stream、单 seed，生成序列的一致率不能替代 LongBench、RULER 或 PPL。论文正式表格还应补：多 seed 重复、H100 80GB 的单卡或固定并行配置，以及与 FIER 官方 kernel 在相同模型和相同候选预算下的 decode 对照。

原始 JSON 和汇总位于 `results/20260809_qksieve_mha_real_decode_yarn128k_v1/`。启动脚本为 `scripts/launch_qksieve_mha_real_decode_20260809.sh`。
