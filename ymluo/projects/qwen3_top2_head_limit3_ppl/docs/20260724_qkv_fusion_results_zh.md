# QK 与 Value Attention 融合算子结果

## 方法

新方法 `countcap_fullprompt_keypca_direct_qkvfused` 保持以下算法不变：

- Key-PCA + INT4 索引；
- 256 点 sampled-quantile；
- 约 6% 候选；
- 不做最终 top-2% 精确重排；
- 不回退 Full。

算子直接使用候选索引完成真实 QK、self score、softmax 和 Value
聚合，不再生成候选 score 中间张量。

## 微内核

| 长度 | staged QK + attention | QK+V fused | 加速 | 最大误差 |
|---:|---:|---:|---:|---:|
| 8K | 0.202 ms | 0.121 ms | 1.67x | 6.1e-5 |
| 16K | 0.288 ms | 0.262 ms | 1.10x | 6.1e-5 |

另测试了 warp 协作 QK。它在 8K 为 0.158 ms、16K 为 0.391 ms，
都慢于普通融合内核，因此不进入主方法。原因是候选并行度从约 128
降到 4，warp reduction 开销超过了连续访存的收益。

## 生成一致性

在 8K/16K、32/64 生成长度、每点三次重复的对照中，
QK+V fused 与上一版 direct fused 的生成 token 和完整预测 12/12
完全一致。

## 公平冷启动结果

为避免前一个稀疏方法预先建立 Key-PCA/INT4 状态，公平实验将
QK+V fused 放在 Full 前面独立运行。每点重复三次并报告中位数。

| Prompt | 生成 | Full online | QK+V fused online | Online speed | Total speed |
|---:|---:|---:|---:|---:|---:|
| 8K | 32 | 1.290 s | 2.560 s | 0.504x | 0.698x |
| 8K | 64 | 2.562 s | 3.966 s | 0.646x | 0.739x |
| 16K | 32 | 1.982 s | 2.542 s | 0.780x | 0.893x |
| 16K | 64 | 3.935 s | 4.064 s | 0.968x | 0.952x |

## 成本拟合

`Tonline = Tfixed + (G - 1) * Tstep`：

| Prompt | 方法 | Tfixed | Tstep | 交叉点 |
|---:|---|---:|---:|---:|
| 8K | Full | 0.057 s | 39.77 ms | - |
| 8K | QK+V fused | 1.198 s | 43.94 ms | 不存在 |
| 16K | Full | 0.090 s | 61.04 ms | - |
| 16K | QK+V fused | 1.068 s | 47.56 ms | 约 74 token |

## 已预建索引口径

如果前一个稀疏动作已经建立 Key-PCA/INT4 状态，QK+V fused 的
16K x 64 online 可达到 1.176x。该数字代表索引复用或提前建表场景，
不能作为单方法冷启动结果。

## 结论

QK+V 融合将 16K 的运行期单步时间从上一版约 53 ms 降到约
47.6 ms，已经明显低于 Full 的 61.0 ms。当前 16K 的主要问题从
单步 attention 变成约 1.07 s 的首次索引构建成本。

8K 的单步时间仍比 Full 高约 4.17 ms，因此仅移动或摊销建表成本
仍不能让 8K 反超。下一步应分别处理：

1. 在 prefill 期间增量建立或异步建立 PCA/INT4 索引，降低首次
   decode 固定成本；
2. 融合 sampled-quantile 扫描与候选消费，继续降低 8K 每步约
   4 ms 的差距。
