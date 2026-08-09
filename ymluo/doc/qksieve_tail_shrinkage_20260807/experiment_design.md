# 实验设计

## 研究问题

在候选集合、QK 索引和 exact sparse attention 完全不变时，能否用无训练的数值收缩系数替代固定 `tail alpha=1`，同时覆盖短 LongBench 与 32K–128K PPL？

## 条件定义

- `Full`：完整 FP16 KV attention。
- `alpha=0`：保留 ValueSketch 索引和相同 selector kernel，但不把尾部加入输出。
- `alpha=1`：当前冻结方法，完整加入 ValueSketch 尾部。
- `SURE`：使用估计噪声和校正能量计算正部 SURE 系数。
- `Ridge`：使用信号/信号加噪声比计算连续系数。
- `oracle alpha`：利用 Full 输出选择最优标量，仅作为上界。

## 实验一：严格短任务归因

- 模型：Llama-3.1-8B-Instruct。
- 数据：LongBench `lcc`、`multifieldqa_en`、`qmsum`，每任务前 5 条。
- 固定变量：prompt、候选、动态预算、c64、rank-16 INT4、随机种子和停止规则。
- 改变量：仅 `tail alpha`。
- 指标：任务分数、逐样本预测哈希、候选 overflow。
- 通过：alpha=0 相比 alpha=1 改善，且 debug-disable 与 alpha=0 预测逐字一致。

## 实验二：长度 PPL 对照

- 模型：Qwen3-4B-Instruct-2507。
- 文本：同一自然代码流、seed `20260843`。
- 长度：32K、64K、128K。
- 每个长度：16 个目标 token。
- 预算：每 head 最多 1280 个 exact token。
- 指标：NLL、几何 PPL、相对 Full 质量、实际 attention token 数。
- 通过：SURE/Ridge 跟随每个长度中更好的固定 alpha，而不是固定偏向 0 或 1。

## 实验三：真实 Q/K/V 机制审计

- 长度：32K、64K；初期每个长度 5 层、1 个 decode query。
- 层：`0,8,16,24,35`。
- 每层所有 KV head 和 GQA query head。
- 指标：selected mass、Value explained variance、tail effective tokens、真实 tail correction、估计噪声、固定/自适应/oracle 输出相对 L2。
- 通过：估计噪声与真实尾部误差正相关，自适应系数缩小固定 alpha 的失败尾部。

## 实验四：运行时实现

- 独立 CUDA Event 测量：retrieval、sparse attention、整模型 decode。
- 长度：32K、64K、128K。
- 冷启动和暖启动分开；不把 prefill 混入 steady decode。
- 通过：新增标量统计开销低于 2%，质量达到实验二标准。

## 已知限制

- 当前短任务只有 15 条，足够做归因，不足以支持最终 LongBench 主表。
- 单一自然代码流不能证明主题通用性；通过最小测试后需补体育、医学、代码、叙事等主题。
- 3090 双卡 128K 的 device-map 开销不代表 H100 单卡部署速度。

## 实验五：用条件残差换取更小候选预算

- 固定 trace：NarrativeQA 32K/64K/128K，LCC 64K，QMSum 64K。
- 固定 QK 索引：QK-balanced、qMSE 240 bit、proxy top-k、相同请求内校准。
- 当前参照：`top_k=1280`、rank-16 INT4 ValueSketch、无 block 条件残差。
- 测试预算：`top_k in {960, 768, 640}`。
- 测试方法：全局 ValueSketch、block-256 residual mean、block-256 conditional residual d8。
- 主指标：逐 head relative L2 的 mean/P90；次指标：经 `W_o` 合并后的逐层 relative L2。
- 通过：至少一个低预算方法在五个 trace 上的 mean 和 P90 都不高于当前 top-1280 参照，且没有单个 trace 退化超过 5%。
- 失败：低预算仅改善平均值但在任一任务或 128K 明显恶化；这说明 block 模型不能安全换取速度。
- 通过后下一步：只对最小通过预算做真实 PPL；PPL 通过后才实现融合 CUDA kernel。

## 实验六：降低全序列 ValueSketch rank

实验五若确认条件残差能覆盖全局低秩重建的系统误差，则进一步减少每 token 都必须读取的 ValueSketch 坐标。

- 固定参照：rank-16、top-1280、无条件残差。
- 第一轮配置：`rank8/top960`、`rank8/top768`、`rank4/top960`，均使用 block-256 conditional residual d8。
- 若 rank-8 只在 128K 失败，边界复核 `rank8/top1120`、`rank8/top1280` 和 `rank4/top1280`；这一步区分“rank 不足”和“rank 与候选同时缩小导致联合失败”。
- 数据和指标：与实验五完全相同，禁止换 trace 或只报告平均值。
- 通过：五个 trace 的 mean、P90 和 `W_o` 投影误差均不高于参照；最坏 `W_o` 比不超过 1.05。
- 速度含义：rank-8 将全序列 Value code 从 8 Byte/token/head 降为 4 Byte，rank-4 降为 2 Byte；只有质量通过后才允许声称潜在带宽收益。
- 失败：条件残差只能补均值偏差，无法补回较低 rank 遗失的 query-dependent Value 变化。

## 实验七：量化 block 条件统计

- 首选结构：rank-8/top-1120；备选结构：rank-4/top-1280。
- 模拟精度：FP16、对称 INT8、对称 INT4。每个 block 的残差均值和 Key 坐标均值独立定标；每个 KV head 的条件矩阵独立定标。
- 参照仍是原 rank-16/top-1280，不允许改成未量化的新结构。
- 通过：五个 trace 全通过，且最坏 `W_o` 比不超过 1.05。
- 存储：FP16 block 统计约 1.06 Byte/token/head；INT8 约 0.53 Byte；INT4 约 0.27 Byte，另有尺度元数据。
- 失败：量化破坏 block 偏差或 Key-Value 相关方向，必须保留更高精度，不能沿用未量化结论。
