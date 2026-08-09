# 长上下文 KV 检索：并行发现、方法优化与当前结果

更新时间：2026-07-20

> 当日晚些时候已完成硬件感知 auto-split CUDA 优化和额外 128K 独立主题验证。最新可部署版本、完整速度和质量结果见 `20260720_overnight_numeric_and_cuda_optimization_zh.md`；本文保留低频双空间阶段的实验记录。

## 1. 当前结论

当前最稳妥的实用主方法仍是：

**compact PCA64 INT4 全局索引 -> 每个 query head 取 8% 候选 -> 原始 K 做 exact rerank -> 最终只对 2% K/V 做 attention。**

这一路径不使用任务标签、答案、oracle 或学习式 router。新探索出的低频双空间检索能改善部分长距离召回，但在 64K PPL 上没有稳定超过基础 PCA64，因此暂时作为机制消融和可选安全动作，不设为默认。

目前最重要的公平速度结果是：Qwen3-4B-Instruct、RTX 3090、batch 1、逐 token forward 下，64K 两个主题的基础 PCA64 整模型平均加速约 **1.52x**；128K 两主题、两卡均衡分层的平均整模型加速约 **1.73x**。去掉首步索引构建后，64K/128K 长生成 steady decode 分别约为 **1.95x/2.32x**。

## 2. 从并行研究中得到的可用发现

并行研究的核心发现是，长距离退化不只来自“内容不相似”，还来自 RoPE 位置相位和 softmax 分母：

1. pre-RoPE 的内容 Q/K 在长距离下仍可保持语义对齐；
2. post-RoPE QK 会随相对位置发生明显变化；
3. 上下文变长后，即使目标 logit 不变，也会因 softmax 分母增加而损失 attention mass；
4. 低 RoPE 频率变化较慢，适合作为主 PCA 索引漏召回时的互补信号。

这些发现不能直接替代现有方法，但给出了一个可检验的新方向：主索引继续近似真实 post-RoPE attention，第二索引只负责寻找被位置相位干扰的长距离语义候选。

## 3. 基础 PCA64 方法

### 3.1 索引表示

每个 KV head 使用 64 维 PCA 表示 K。64 维分为四个 16 维频带，每个 token 保存：

```text
32 B  INT4 PCA code
 2 B  FP16 base scale
 2 B  四个 4-bit 相对 scale exponent
-------------------------------------
36 B  / token / KV head
```

完整 FP16 K+V 为 `2 * 128 * 2 = 512 B`，因此纯索引占：

```text
36 / 512 = 7.03125%
```

这一表示比旧的单 scale INT4 多 2 B，但显著降低不同谱带之间的量化失真。

### 3.2 在线检索

```text
post-RoPE query
    -> PCA64 投影和 INT8 query 量化
    -> 扫描 compact PCA64 INT4 K 索引
    -> 每个 query head 取近似分数最高的 8% token
    -> 读取候选的原始 K，重新计算 exact QK
    -> 保留 exact top-2%
    -> 用对应原始 V 计算最终 attention
```

8% 是近似检索候选池，不是最终 attention 比例；最终每个 query head 只使用约 2% 历史 token。

## 4. 低频双空间分支

### 4.1 设计

第二索引对 post-RoPE K 做逆旋转，只取最低 16 个 RoPE 频率对，即 32 维：

```text
post-RoPE K -> inverse RoPE -> lowest-frequency 32D -> L2 normalize -> INT2
```

归一化后使用固定裁剪范围 `1.5 / sqrt(32)`，因此每个 token 不再保存独立 scale。索引只需：

```text
32 dimensions * 2 bits = 64 bits = 8 B / token / KV head
```

每四个生成 token 刷新一次低频候选，每次取全历史 0.5% token，与 PCA64 的 8% 候选做去重并集，最后仍用原始 K exact rerank 到 2%。

### 4.2 距离感知版本

真实 Q/K trace 显示，只扫描最老的 50% 历史，能在 sports 和 medicine 上分别保留约 72% 的低频召回增益。该版本保持同样 0.5% 救援数，但低频打分只覆盖最远半区。

| 方法 | Sports top-2% recall | Medicine top-2% recall | 平均输出相对 L2 |
|---|---:|---:|---:|
| PCA64 candidate 8% | 98.653% | 97.984% | 0.445% |
| 低频 INT2，全历史 | 98.859% | 98.301% | 0.363% |
| 低频 INT2，最老 50% | 98.801% | 98.211% | 0.395% |

最低频 32 维进一步压到 1-bit 后，最老 50% 版本只保留约 35%--42% 的 INT2 增益，因此不进入主实现，只保留为超低显存消融。

## 5. 独立主题质量验证

32K、每主题一个未参与方法设计的窗口、64 个评测 token：

| 方法 | 四主题平均 NLL | 相对 Full PPL 质量 | 结果趋势 |
|---|---:|---:|---|
| Full attention | 2.92719 | 100.00% | 基线 |
| PCA64 candidate 8% | 2.94026 | 98.70% | 当前实用主方法 |
| 全历史低频 INT2 救援 | 2.93861 | 98.86% | 3/4 主题改善，但收益很小 |
| 最老 50% 低频 INT2 救援 | 2.93950 | 98.78% | 以半扫描保留部分收益 |

四个独立主题为 `computer / space / politics / religion`。低频救援在后三者改善，在 computer 上退化，因此不能宣称单调提升。

64K、sports/medicine、每主题 32 个评测 token：

| 方法 | Sports NLL | Medicine NLL | 两主题平均 NLL |
|---|---:|---:|---:|
| Full attention | 2.04866 | 2.00765 | 2.02815 |
| PCA64 candidate 8% | **2.01654** | 2.00130 | **2.00892** |
| 全历史低频 INT2 救援 | 2.02334 | **1.99828** | 2.01081 |
| 最老 50% 低频 INT2 救援 | 2.02646 | 2.00416 | 2.01531 |

这组 64K 结果说明基础 PCA64 已经很好，额外低频候选会帮助 medicine，却会扰动 sports 的最终 top-2% 集合。当前证据不支持把双空间设成无条件默认路径。

## 6. 实际速度

### 6.1 每层 attention pipeline

这里计入索引扫描、top-k、候选 exact rerank 和稀疏 attention，不包括模型 MLP。

| 长度 | Full SDPA | PCA64 | 全历史低频 | 最老 50% 低频 | PCA64 加速 |
|---:|---:|---:|---:|---:|---:|
| 64K | 1.234 ms | 0.835 ms | 0.929 ms | 0.931 ms | 1.48x |
| 128K | 2.436 ms | 1.167 ms | 1.312 ms | 1.289 ms | 2.09x |

128K 时，低频扫描加 top-k 每四步均摊开销为：

```text
全历史低频 INT2: 0.090 ms / layer / token
最老 50% INT2:   0.060 ms / layer / token
```

### 6.2 整模型逐 token forward

公平比较要求 Full 和 sparse 都逐 token 运行。不能把 Full 的 32/64-token 并行评分时间与 sparse 的逐 token 时间比较。

64K sports/medicine 的配对结果：

| 主题 | Full | PCA64 | 整模型加速 |
|---|---:|---:|---:|
| Sports | 9.243 s | 6.053 s | 1.53x |
| Medicine | 9.250 s | 6.117 s | 1.51x |
| **合计** | **18.493 s** | **12.170 s** | **1.52x** |

另四个未见主题在 64K、约 31 个在线 step 下的结果为：

| 主题 | Full NLL | PCA64 NLL | Full 时间 | PCA64 时间 | 加速 |
|---|---:|---:|---:|---:|---:|
| Computer | 3.1163 | **3.0890** | 4.573 s | 3.621 s | 1.26x |
| Politics | **2.1174** | 2.1253 | 4.576 s | 3.632 s | 1.26x |
| Space | 3.9054 | **3.8696** | 4.587 s | 3.561 s | 1.29x |
| Religion | **2.7129** | 2.7535 | 4.603 s | 3.590 s | 1.28x |
| **合计/平均** | **2.9630** | **2.9593** | **18.339 s** | **14.403 s** | **1.27x** |

128K 单卡因完整 Qwen3-4B prefill KV 超出 24GB 3090，最终使用两卡 `device_map=balanced` 做相同 Full/PCA64 配对：

| 主题 | Full PPL | PCA64 PPL | Full 时间 | PCA64 时间 | 加速 |
|---|---:|---:|---:|---:|---:|
| Medicine | 6.363 | **6.290** | 8.751 s | 5.039 s | 1.74x |
| Politics | 10.814 | **9.986** | 8.752 s | 5.086 s | 1.72x |
| **合计** | -- | -- | **17.503 s** | **10.125 s** | **1.73x** |

32K 仍位于当前实现的速度交叉点附近，稀疏路径没有稳定快于 Full。部署策略应使用纯长度 gate：短上下文走 Full，64K 及以上再启用稀疏检索。

### 6.3 首步转换与 steady decode

Sparse 的首个在线 step 会建立 PCA basis 并把完整历史投影量化。将首步与后续 step 分开后：

| 长度 | Sparse 首步 | Sparse steady | Full 每步 | Steady 加速 |
|---:|---:|---:|---:|---:|
| 64K，单卡 | 1.278 s | 75.2 ms | 146.8 ms | **1.95x** |
| 128K，两卡分层 | 1.381 s | 121.8 ms | 282.3 ms | **2.32x** |

因此短生成看到的 1.27x--1.73x 主要受一次性转换摊销影响；生成 token 足够多时，速度逐渐接近 steady 值。64K 和 128K 使用不同设备映射，绝对毫秒数不能直接跨行比较，只能在每行内部与匹配 Full 比较。

### 6.4 扩展加载开销

原实现中的 `load_inline` 会让多个新进程重复进入 ninja 编译，并可能把约两分钟等待误计入 online 时间。现在扩展名按源码版本管理，进程优先直接加载同版本已编译 `.so`，仅在二进制不存在时编译。直接加载 smoke test 的完整进程墙钟时间为 3.37 s，不再触发两分钟重编译。

## 7. 被否定的方向

### 主候选每四步刷新

相邻 query 的 PCA64 candidate-8% 集合确实有约 76%--79% 重合，但复用四步后：

| 主题 | Refresh-1 recall | Refresh-4 recall | Refresh-1 L2 | Refresh-4 L2 |
|---|---:|---:|---:|---:|
| Sports | 98.65% | 91.60% | 0.27% | 3.15% |
| Medicine | 97.98% | 87.88% | 0.62% | 5.56% |

候选集合“多数稳定”不等于关键 top-2% token 稳定。该方向会让输出误差扩大约 9--12 倍，不能用于主方法。

### 其他 RoPE-free 表示

完整 pre-RoPE 内容空间、固定相对距离重编码、多尺度 canonical score 和 block prototype 均未稳定改善基础 PCA64。真正有互补性的只有最低频 32 维，但其端到端收益仍不稳定。

### 流式早冻结 PCA basis

为了隐藏首步索引转换，测试了只使用 prompt 前 2K/4K/8K token 确定 PCA basis，之后在 prefill 中流式量化 K。该方法失败：前 8K basis 与完整 basis 的平均主角余弦已接近 0.90，但 sports/medicine 的 candidate-8% recall 仍只有 83.70%/84.83%，明显低于完整上下文 basis 的 98.65%/97.98%。

这说明高维子空间的平均接近不足以保证 top-k 排序边界稳定。当前不能通过早冻结 basis 消除转换；后续若优化该开销，应优先并行化完整 conversion，而不是牺牲 basis 统计范围。

## 8. 当前方法定位

论文主方法建议冻结为：

```text
Compact spectral-scale PCA64 INT4 index
+ per-query-head candidate retrieval
+ exact rerank to top-2% attention
+ length-gated deployment
```

低频双空间建议作为机制章节和安全动作消融：它证明 RoPE 长距离相位失配可以由 pre-RoPE 低频内容空间部分修复，但现阶段不应包装成总能提升的默认模块。

当前可以支持的速度表述是：64K--128K 短生成整模型加速约 1.27x--1.73x；摊销一次性索引构建后的长生成 steady decode 约 1.95x--2.32x。更高速度需要继续优化索引转换和全局 top-k，而不能把 attention 理论上界直接当作整模型结果。
