# QKSieve 与 FIER 的同条件检索对照

## 1. 为什么必须单独做这个对照

FIER 是 QKSieve 最接近的已发表方法：两者都扫描低比特 Key 索引，以近似 QK score 做 token-level top-k，再在选中的原始 K/V 上计算精确 attention。

FIER 论文定义的是：

- 沿序列方向每 32 个 token 分一组；
- 每个 channel 在组内做 1-bit RTN；
- 每组、每 channel 保存一对 FP16 scale/bias；
- 低比特 score 直接 top-k；
- 对选中的原始 K/V 做精确 attention。

论文未提供可找到的官方代码，因此本实验是严格按论文算法和公式实现的 audited reproduction，不应写成“运行了 FIER 官方实现”。论文来源：

- [FIER 论文](https://aclanthology.org/2025.findings-emnlp.515.pdf)
- [FIER arXiv](https://arxiv.org/abs/2508.08256)

## 2. 公平性

| 方法 | 索引大小 / token / KV head | 相对 FP16 K+V |
|---|---:|---:|
| FIER 1-bit、group=32 | 32 B | 6.250% |
| QKSieve 240-bit mixed index | 30 B | 5.859% |

FIER 的 32 B 来自每个 Key coefficient 的 1-bit code，以及每 32 个序列 token、每个 channel 共享的一对 FP16 scale/bias。QKSieve 索引略小，而不是通过使用更大索引取得优势。

两个方法使用：

- 相同的原始 Query、Key；
- 相同的每个 query head 独立 top-k；
- 相同的 active-token 数；
- 相同的精确 score 作为评价基准；
- 都不做 exact rerank、recent/sink 保护或 Full fallback。

## 3. 数据与独立性

- 模型：Qwen3-4B-Instruct。
- 历史长度：32K。
- 文本：sports、medicine。
- 层：全部 36 层。
- KV heads：每层 8 个。
- QKSieve 使用每层前 8 个 Query step 构造 Query covariance。
- 评价只使用后 8 个未参与校准的 Query step。
- 总计 18,432 个 held-out query-head 条件。

这不是 LongBench 任务分数，而是控制变量严格的索引质量实验。

## 4. 结果

| 方法 | Active KV | Top-k recall | Attention mass recall | 精确 top-1 recall | Score correlation |
|---|---:|---:|---:|---:|---:|
| FIER-g32 | 1% | 43.36% | 76.40% | 32.13% | 0.8575 |
| QKSieve | 1% | **73.44%** | **83.48%** | **67.10%** | **0.9683** |
| FIER-g32 | 2% | 47.90% | 81.61% | 32.13% | 0.8575 |
| QKSieve | 2% | **75.70%** | **87.30%** | **67.10%** | **0.9683** |
| FIER-g32 | 4% | 53.38% | 86.63% | 32.13% | 0.8575 |
| QKSieve | 4% | **78.50%** | **90.99%** | **67.10%** | **0.9683** |

QKSieve 相对 FIER 的 attention mass recall 增益：

| Active KV | 增益 | 95% cluster-bootstrap CI |
|---|---:|---:|
| 1% | +7.08 个百分点 | [6.17, 8.04] |
| 2% | +5.68 个百分点 | [4.94, 6.48] |
| 4% | +4.35 个百分点 | [3.79, 5.01] |

bootstrap 以“主题 × 层”为 cluster，共 72 个 cluster；三个预算下 QKSieve 在 72/72 个 cluster 上的平均 attention mass 都高于 FIER。

## 5. 最重要的解释

FIER 用同一种 1-bit 近似处理全部 128 个原始 Key 维度。它的优势是简单、索引规整，但会把大量 bit 花在 Query 不敏感的方向，同时让 Query 敏感方向也只有 1 bit。

QKSieve 先构造 QK-balanced 坐标，再让每层、每个 KV head 把有限 bit 自动分配到真正影响 QK score 的方向。因此：

- centered score RMSE：QKSieve 0.674，FIER 2.358；
- score correlation：QKSieve 0.968，FIER 0.858；
- 精确 top-1 token 召回：QKSieve 67.10%，FIER 32.13%。

这与理论中的最优 rank-$r$ score 子空间、QK-weighted quantization error 和排序 margin 分析一致。

## 6. 当前可以和不能写进论文的结论

可以写：

> 在近乎相同且略小的索引字节下，QKSieve 在全部层和 held-out Query 上显著提高 top-k 与 attention-mass recall；在 2% active KV 时，其 attention mass 已高于 FIER 的 4% operating point。

暂时不能写：

- QKSieve 已在 LongBench 端到端质量上击败 FIER；
- QKSieve 的真实 kernel 一定比 FIER Triton kernel 更快；
- 当前实现是 FIER 官方代码。

最终还需要：

1. 已实现 `fier_rtn1_g32_packed_fulltopk`：bit-plane 真正以 1 bit/coefficient 存储，每 32 个 token 共享逐通道 FP16 上下界，实际索引为 32 B/token/KV-head。
2. 已接入与 QKSieve 相同的动态 active-token 预算、`torch.topk` 和 exact sparse-attention kernel；旧的 `fier_rtn1_g32_fulltopk` 只保留为解压 FP16 的质量参考路径。
3. CPU 编码、增量更新、GQA score 与实际字节数测试已通过，远端 CUDA 扩展已编译。GPU 数值验证、LongBench/RULER 和正式速度尚未运行，不能提前填写结果。
4. 恢复 GPU 后需在同一 GPU 报告 index update、index scan、top-k、exact sparse attention 和 whole-model decode，并比较 equal-active-token、equal-index-byte 和 equal-quality 三个 operating point。

实现入口：

- `src/fier_rtn1_cuda_20260728.py`
- `src/validate_fier_rtn1_cuda_20260728.py`
- LongBench 方法名：`fier_rtn1_g32_packed_fulltopk`
- 受控 256-bit 消融：`qksieve_fullprompt_random_uniform1_fulltopk`、`qksieve_fullprompt_keypca_uniform1_fulltopk`、`qksieve_fullprompt_qkbalanced_uniform1_fulltopk`
- 受控 240-bit Key-MSE 消融：`qksieve_fullprompt_keypca_autokey_fulltopk` 与 `qksieve_fullprompt_qkbalanced_autokey_fulltopk`
- 三方法完整对照脚本：`scripts/launch_qksieve_fier_packed_longbench_5gpu_20260728.sh`
- 八方法 m20 因果消融脚本：`scripts/launch_qksieve_uniform1_ablation_longbench_m20_5gpu_20260728.sh`

原始结果：

`results/20260728_fier_qksieve_fair_retrieval_32k/`
