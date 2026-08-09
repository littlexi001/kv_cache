# JointKV 真实推理接入设计

## 研究问题

此前 `JointKV-SieveCUDA` 的高加速结果使用随机生成的索引值，只能证明 CUDA
数据通路能够运行，不能证明真实 K/V 编码后仍有相同的质量和速度。本子问题检验：

> 从真实 post-RoPE K/V 构建紧凑索引后，是否能在 Hugging Face 自回归 decode
> 中同时保持质量并获得端到端 decode 加速？

可证伪条件是：真实候选无法保留关键注意力、完整模型 PPL 明显变差，或把全部
额外开销计入后 decode 不再加速。

## 数值假设

1. 每个 KV head 的 Key 可在 Query 二阶矩定义的坐标系中编码。
2. 64 个逐残差二值方向表示主要 QK 分数，48 个二值方向补充残差信息。
3. 64 类联合 K/V 聚类 ID 同时携带 Key 偏置和 Value 统计信息。
4. 真实高分 token 可能集中在少数连续区间，因此不能强制每 32 个 token 留固定数量。
5. 同一 KV head 下的不同 Query head 可能关注不同 token，因此候选必须按 Query head 独立产生。

前 3 项定义紧凑分数近似；后 2 项来自真实 Layer-0 失败诊断，并直接决定新的
全局 per-query selector。

## 持久索引

每个 token、每个 KV head 保存：

| 字段 | 逻辑位数 | CUDA 物理存储 |
|---|---:|---:|
| 主二值编码 | 64 | 8 Byte `int64` |
| 残差二值编码 | 48 | 8 Byte `int64`，高 16 bit 未使用 |
| Joint ID | 6 | 1 Byte `uint8` |
| Key/Value risk code | 8 | 1 Byte `uint8` |
| 合计 | 126 bit | 18 Byte |

当前 selector 未使用 risk code，但为与完整索引格式一致仍保留该字节。辅助索引
相当于 FP16 K+V 的 `18 / (2 * 128 * 2) = 3.516%`。真实 FP16 K/V 仍常驻 GPU，
因此这是 attention 计算与读带宽稀疏化，不是总 KV 存储压缩。

## 当前算法

输入为 dense prefill 得到的每层 post-RoPE K/V，以及当前 decode 的 post-RoPE Q。

1. 对 prefill K 生成 64-bit 主编码。
2. 用主编码残差和 V 分配 64 类 Joint ID。
3. 对去除 Joint Key 中心后的残差生成 48-bit 编码。
4. 对每个 Query head 独立把 Q 投影为 192 个 probe 值。
5. 在同一 CUDA kernel 内，从 probe 值生成 14 个 byte LUT，每个 LUT 有 256 项。
6. 从 512 个均匀位置估计分数阈值。
7. 全局扫描真实紧凑索引，把高于阈值的 token 压入该 Query head 的候选数组。
8. 把 prefill 之后已经生成的 token 作为 exact suffix 追加；当前 token 作为 exact self token。
9. 在候选的真实 FP16 K/V 上计算精确 QK、softmax 和 Value 加权和。

当前实现没有 Full fallback，也没有 tail correction。候选内 softmax 会重新归一化，
被省略 token 的合计 Value 贡献完全丢失；实验表明这是当前主要质量缺口。

## 实现约束

- selector 必须为 per-query 全局选择，不能退回共享 GQA 候选或固定 warp quota。
- workspace 溢出视为无效实验；不得把先写入的 token 当作合法 top-k。
- Full 与 Sparse 必须分别使用真实 HF forward，并在 `torch.no_grad()` 中运行。
- prefill、索引构建、稳态 decode 和一次性在线总时间必须分开报告。
- CUDA 扩展编译、模型加载和首次 JIT 不计入 decode；索引构建必须单独计入。
- 质量必须在连续多步 decode 上评估，不能只用单步 attention mass 或 8 个简单 token。

## 代码入口

- 真实索引：`src/jointkv_real_index_20260802.py`
- per-query selector 与融合 query/LUT kernel：`src/jointkv_global_threshold_cuda_20260802.py`
- 单层数值和阶段测速：`src/benchmark_jointkv_global_real_layer_cuda_20260802.py`
- 真实 28 层 HF decode：`src/run_jointkv_real_inference_20260802.py`
