## 1. 问题

当前方法用 PCA64 INT4 索引近似计算 query 与全部历史 key 的分数，再从中检索每个 query head 需要的少量 token。精确 attention 已经只读取约2%的历史连接，但 PCA 索引仍在每个 decode step 扫描全部 `N` 个历史 token。

因此，新的问题不是继续减少 attention token，而是：

> 能否避免在相邻 decode step 重复计算几乎相同的全局检索分数？

## 2. 观察

### 2.1 检索分数可以递推

同一个 key 索引在相邻 decode step 保持不变。令第 `t` 步的代理分数为 `s_t = K q_t`，则有：

`s_t = s_(t-1) + K (q_t - q_(t-1))`

这意味着检索分数不是每步都必须从零计算的无状态结果，而是可以增量更新的时序状态。

### 2.2 Query 变化集中在少数高方差谱维度

PCA64 的维度按特征值从小到大排列。真实32K query trace 显示，只用最高方差的16维更新 score cache，已经能恢复大部分相邻步分数变化；真正需要64维重算的只是少数漂移较大的 KV heads。

### 2.3 漂移可以在 PCA 度量下直接估计

对未参与16维增量更新的低方差维度，累计 query 残差 `R`。对 KV head `h`，定义谱风险：

`risk_h = mean_g sqrt( sum_j(lambda_hj * R_hgj^2) / sum_j(lambda_hj * q_hgj^2) )`

其中 `g` 是同一 GQA 组内的 query heads，`lambda` 是 PCA 特征值。这个风险只依赖当前 query、累计残差和已有 PCA 特征值，不需要任务标签、训练数据或 router。

## 3. 方法

每层维护两类状态：

1. 原有 PCA64 INT4 全局 K 索引；
2. FP16 代理分数缓存，形状为 `[batch, query_heads, history]`。

执行流程如下：

1. 第一个 decode step 用完整 PCA64 扫描初始化 score cache。
2. 后续每步计算 `delta_q = q_t - q_(t-1)`。
3. 所有 KV heads 只用最高方差16维更新历史 score cache。
4. 累计被省略的48维 query 残差，并计算每个 KV head 的谱风险。
5. 主配置刷新风险最高的25% KV heads：这些 heads 用64维重算，其他 heads 保持16维增量更新。代码按比例计算刷新数，不假设模型固定拥有8个 KV heads。
6. 新生成 token 的一个新分数直接原位写入 score cache，不复制整个历史分数张量。
7. 更新后的代理分数继续进入原有候选检索、exact rerank 和动态预算规划。

主配置是25%谱风险刷新；12.5%刷新作为更激进的速度档。两者都是固定比例而非模型相关阈值。

### 3.1 额外存储

在 GQA group size 为4、head dimension 为128时，FP16 score cache 相对完整 FP16 K/V 的比例为：

`4 / (2 * 128) = 1.5625%`

该缓存用于检索分数，不是精确 K/V。当前实验 harness 仍保留 HuggingFace 完整 DynamicCache，因此这里的1.5625%是算法新增状态，不能表述为已经实现的物理 GPU KV 占用。

## 4. 结果

### 4.1 离线检索质量

协议：Llama-3.1-8B-Instruct，sports/medicine 两个32K窗口，5层，每层16个连续 query steps。候选为8%，最终保留2%。

| 更新规则 | 64维刷新率 | 平均扫描维度 | Exact top-2%候选召回 | Attention mass |
|---|---:|---:|---:|---:|
| 每步完整 PCA64 | 100% | 64.00 | 98.83% | 86.18% |
| 谱风险 top1 | 12.50% | 22.00 | 93.88% | 85.95% |
| 谱风险 top2 | 25.00% | 28.00 | 95.96% | 86.07% |
| 谱风险 top3 | 37.50% | 34.00 | 97.07% | 86.11% |
| 固定每4步刷新 | 20.00% | 25.60 | 95.17% | 85.91% |

top2 用28维平均扫描量超过了固定周期刷新，并且不依赖绝对阈值。

### 4.2 两个模型的实际 PPL

协议：32K历史，5个主题，每主题128个 target tokens。表中的 PPL 是五主题几何平均，越低越好。

| 模型 | 方法 | PPL | 相对完整 PCA64 | Attention links | 定位阶段 exact QK | 64维刷新率 |
|---|---|---:|---:|---:|---:|---:|
| Llama-3.1-8B | 每步 PCA64 | 12.3009 | 100.00% | 2.184% | 2.980% | 100% |
| Llama-3.1-8B | 谱风险 top1 | 12.3332 | 99.74% | 2.081% | 2.812% | 12.73% |
| Llama-3.1-8B | 谱风险 top2 | **12.2796** | **100.17%** | 2.158% | 2.912% | 25.20% |
| Qwen3-4B | 每步 PCA64 | 14.9579 | 100.00% | 3.599% | 4.464% | 100% |
| Qwen3-4B | 谱风险 top1 | 15.0621 | 99.31% | 2.906% | 3.642% | 12.73% |
| Qwen3-4B | 谱风险 top2 | **14.7135** | **101.66%** | 3.135% | 3.906% | 25.20% |

top2 在两个模型上都没有出现 PPL 退化。PPL 的小幅改善应视为近似检索带来的扰动，不能在更大测试完成前宣称方法提高了模型能力。

### 4.3 CUDA 检索子系统速度

计时包含谱风险计算、top-k head 选择、INT4索引扫描、FP16 score cache 读取/累加/写回。硬件为 RTX 3090。

| Batch | 32K | 64K | 128K |
|---:|---:|---:|---:|
| 1 | 3.42x | 2.63x | 2.91x |
| 4 | 2.90x | 3.04x | 3.11x |
| 8 | 3.07x | 3.10x | **3.13x** |

第一版谱 gate 只启动一个 CUDA thread，串行计算全部 head 风险。Nsight 显示 gate 本身达到90.1微秒/层，比增量扫描更慢。将32个 query heads 并行计算风险、并行更新 anchor 后，top2扫描提升到上表结果。

纯16维增量 kernel 在32K/64K/128K分别达到约7.18x、5.25x、5.08x。top2仍低于纯16维更新，是因为25%的 KV heads 必须做64维刷新。

加入8%有序候选选择后，top2在32K/64K/128K的完整检索前端加速约为2.03x、1.50x、1.62x。排序会缩小扫描加速，但整体仍快于每步完整 PCA64。

### 4.4 当前端到端结果

计时在每个 token 的 model forward 前后执行 CUDA synchronize。索引扫描、谱 gate、候选排序、exact rerank、稀疏 attention、MLP和状态更新全部计入；诊断结果的 GPU 到 CPU 搬运放在计时区间之外。

| 模型与长度 | 方法 | 主题数/target | PPL几何平均 | Model forward | 相对速度 |
|---|---|---:|---:|---:|---:|
| Llama-3.1-8B，32K | 每步 PCA64 | 2×128 | 6.9884 | 82.29 s | 1.000x |
| Llama-3.1-8B，32K | 谱风险 top2 | 2×128 | **6.9718** | **79.28 s** | **1.038x** |
| Qwen3-4B，64K | 每步 PCA64 | 2×64 | 11.4882 | 89.51 s | 1.000x |
| Qwen3-4B，64K | 谱风险 top2 | 2×64 | **10.9251** | **86.95 s** | **1.029x** |

旧 harness 曾得到0.927x，原因不是模型 forward 变慢，而是 top2 额外记录7类 transport 指标，旧计时把每 token 数百次小 tensor 创建和 GPU 到 CPU 日志搬运也算入在线时间。将统一日志字段改为 Python 标量、把日志汇总移出 forward 计时，并把 previous projected query 改成原位更新后，32K和64K均实现了小幅正向端到端加速。

当前端到端收益仍远小于3x子系统收益，说明32K/64K下 attention 检索只占整个模型 forward 的一部分；不能用子系统加速替代端到端结果。

## 5. 已否定或暂不采用的分支

1. 每个 query 动态选择16个维度：维度 top-k 和非连续访存开销超过节省的乘法量。
2. 固定每2/4/8步刷新：会累积误差，实际 PPL 明显弱于谱风险刷新。
3. 固定绝对风险阈值：Llama 上有效，但同一阈值在 Qwen 上触发91.8%刷新，不具备跨模型可迁移性。
4. 相对风险 top1/top2切换：刷新率从25%降到20.3%时，召回也同步下降，没有形成明显新 Pareto 点。
5. Margin certificate：严格 Cauchy 上界只有约0.1%到0.25%的 query-head steps 能证明无需扫描，上界过松。
6. 把少数刷新 heads 单独组成 compact kernel：只有1到2行工作时 GPU 占用不足，比一次混合 launch 更慢。
7. 每步把 FP16 score cache 转为 FP32再做 top-k：两主题时间从86.35秒增至96.49秒，整段转换和分配成本高于排序收益。
8. 无序选8%再只排序候选：64K/128K微基准略有收益，32K更慢，尚未形成稳定替代方案。

## 6. 创新性判断

这条方法最核心的观点是：

> 全局 KV 检索分数应被视为随 decode 更新的时序状态，而不是每一步重新计算的无状态检索结果。

与每步重新计算低维 QK proxy 的方法相比，本文直接递推整段代理 score；与只复用候选 token 的时序方法相比，本文维护的是全局分数状态，并用 key covariance 定义的谱风险决定哪些 KV heads 必须刷新。

目前的故事可以收敛为三个贡献：

1. Score transport 恒等式及低秩增量更新；
2. 无训练、尺度无关的 per-KV-head 谱风险刷新；
3. 融合 INT4 delta update 与 score cache 的 CUDA 实现。

初步相关工作检索尚未发现完全相同的组合，但这不构成完整 novelty clearance。重点相关方向包括 [ShadowKV](https://arxiv.org/abs/2410.21465)、[Loki](https://arxiv.org/abs/2406.02542)、[SparQ Attention](https://arxiv.org/abs/2312.04985)、[Quest](https://arxiv.org/abs/2406.10774) 和 [LRQK](https://arxiv.org/abs/2510.23649)。

## 7. 下一步验证

1. 在完整 LongBench、RULER 32K/64K/128K 上做独立测试。
2. 增加更多窗口、seed和至少一个不同 KV-head 数模型，报告误差条和质量/吞吐 Pareto。
3. 融合或替换当前多 kernel 的有序候选选择，进一步缩小子系统与端到端收益差距。
4. 接入真正的 GPU hot KV + CPU full KV layout，测量物理 GPU KV 占用，而不是继续使用完整 DynamicCache。

当前结论是：算法核心已经形成，而且比继续训练 router 更简单、更可解释；端到端速度已经从负收益修正为32K/64K的小幅正收益。完整 benchmark、真实物理 KV layout 和更大规模端到端速度仍是论文主张成立前必须完成的工作。
