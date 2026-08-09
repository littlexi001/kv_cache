# QKSieve 64K 解码冗余操作裁剪结论

## 结论

在 Qwen3-4B-Instruct-2507、64K 历史、RTX 3090 上，当前可以安全冻结的裁剪只有一项：把每 head 的候选阈值样本从约 3,328 降到 512。保留全部 36 层 rank-16 ValueSketch、确定性 `sortcompact` 和自动 exact-attention 分块后，CUDA Graph 延迟为 23.877 ms/token；相对原生 GQA、零拷贝 Full KV 的 25.891 ms/token，公平加速为 1.084x。

关闭 ValueSketch 可把延迟降到 21.411 ms/token，即 1.209x，但四类自然文本的 Full top-1 一致率从 99.61% 降到 98.05%，不能作为保守主版本。

## 方法与假设

QKSieve 先用低比特 QK-balanced 索引估计分数，再选最多 1,280 token/head，从 GPU 常驻完整 K/V 执行 exact attention。ValueSketch 近似未选 token 的 softmax Value 尾部贡献。

本次假设是：分位点只需近似，因此阈值样本可缩小；部分层的 ValueSketch、候选排序或 exact-attention 分块可能是冗余操作。每个假设分别通过真实 Graph 延迟、PPL、top-1、KL、候选溢出和 Graph/Eager token 一致性检验。

## 主要证据

| 条件 | ms/token | 相对 Full | 四主题 top-1 | 判定 |
|---|---:|---:|---:|---|
| Full KV | 25.891 | 1.000x | 100.00% | 基线 |
| 全层 rank-16 sorted，512 样本 | **23.877** | **1.084x** | **99.61%** | 冻结 |
| NoVS，512 样本 | 21.411 | 1.209x | 98.05% | 质量不足 |
| 仅中后 24 层 ValueSketch | 22.733 | 1.139x | 98.83% | 质量不足 |
| 全层 rank-16 unsorted | 25.235 | 1.026x | 99.61% | 更慢 |

NoVS 的四主题几何 PPL为 6.0202，甚至低于 Full 的 6.1475，但 top-1 更差。这说明平均 NLL 会掩盖少量关键预测翻转，不能单独作为删除补偿的依据。

强制 exact-attention split 为 4/2/1 时，延迟分别为 21.830/24.881/33.730 ms/token，均不优于自动 split 的 21.411 ms/token。减少 kernel 工作损失了 GPU 并行度。删除候选排序也更慢，因为 `sortcompact` 实际上是融合候选整理与压缩的快路径。

旧 c64 eager 阶段 profiler 报告候选 sparse attention 28.04 ms、代理检索 7.63 ms、ValueSketch 索引 2.46 ms、query 准备 2.10 ms、Key append 1.55 ms。该测量包含逐阶段同步，不能与 CUDA Graph 总延迟相加；它只说明核心成本集中在候选 attention/尾部合并和代理检索。64K prefill 约 53.98 s、请求局部索引准备约 0.37 s、ValueSketch 预计算约 0.42 s，均为一次性成本，不在稳态 Graph token 延迟中。

随后使用 Nsight CUDA Graph node trace 直接分解当前 s512：模型 GEMM/GEMV 10.870 ms/token，其他模型 kernel 3.125 ms，候选 exact attention 与 tail merge 5.741 ms，代理扫描/阈值/压缩 4.554 ms，query 投影 0.200 ms，suffix 追加 0.045 ms，Graph/runtime 间隙 0.746 ms。Nsight wall 为 25.280 ms，干净 Graph 为 23.877 ms。结果表明 55.36% 已是模型底座，40.72% 是检索和候选 attention；外围操作不是主要瓶颈。

## 实现合同

主版本使用：

```text
variant = qksieve_qmse_oas_requestlocal_valuesketch16_sorted_s512_k1280
QKSIEVE_MIN_QUANTILE_TAIL_SAMPLES = 8
ValueSketch layers = 0..35
exact-attention split = auto
```

动态新增 suffix 必须追加为精确候选；候选溢出必须为 0；Graph 与 Eager 贪心 token 必须一致。实验入口为 `scripts/run_qksieve_growing_graph_128k_2gpu_smoke_20260808.sh`。

## 失败解释与下一步

外围操作裁剪已经接近收益上限：保守版本只比 Full 快 8.4%，激进全删补偿也只有 20.9%。下一步不应继续枚举“删哪个步骤”，而应对三个核心 kernel 做独立测量和融合：低比特代理扫描、候选 exact attention、ValueSketch 尾部合并。优先目标是让 proxy scan 直接产出 exact-attention 可消费的连续候选布局，并把 ValueSketch 分子/分母合并进同一次输出归约。

## 结论边界

结论只覆盖一个模型、一个长度、一种 GPU、一个速度窗口和四条 64-token 自然文本流。它证明了哪些局部删除在该设置下通过或失败，不证明完整 LongBench、RULER、多模型、8K–256K 或多 seed 的通用性。
