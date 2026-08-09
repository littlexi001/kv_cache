# 实验设计

## 环境

- 服务器：`fdong@10.176.34.113`
- GPU：RTX 3090 24GB
- 模型：Qwen3-4B-Instruct-2507，FP16
- PyTorch：2.7.1+cu126
- 上下文：8K、16K、32K、64K、128K
- 每个长度：3 个独立进程重复
- 质量 probe：每次 64 个 teacher-forced token
- QKSieve：request-local QK-balanced、240 bit/token/head、sample target 64、最多 1,280 token/head
- Value 补偿：rank-16、block-256、INT4、W_o metric

## 计时协议

每个重复都记录：

1. `full_step_ms`：真实模型 Full attention 的稳态 decode 延迟。
2. `sparse_step_ms`：完整 QKSieve 在线路径的稳态延迟，包含方法引入的所有逐 token 开销。
3. `warm_fixed_s`：持久 KV 伴随索引已经存在时，每个新 query 的固定开销。
4. `resident_build_s`：构建并持久化 query-independent 索引的成本。
5. `cold_fixed_s = warm_fixed_s + resident_build_s`。
6. `prefill_s`：从原始文本构建完整 KV 的 dense prefill。

总 decode 加速分别按 `G={1,4,8,16,32,64,128,256,512}` 计算，不能只报告 steady speedup。

## 质量与等价性

本轮 PPL probe 只用于确认工程改造没有扰动数值方法，不能替代 LongBench/RULER。接受优化前必须满足：

- qMSE allocation 与 active packed-Key hash 36/36 层一致；
- fused Value append 在 FP16/BF16、多 seed、多 append 长度下逐位一致；
- 真实模型生成 token IDs 一致；
- steady decode 不因优化稳定回退超过 1%。

LongBench/RULER 后续使用同一实现，不更换预算、候选规则或 Value 补偿。任务实验报告分数、实际输出 token 数和包含索引的请求时延。

## 结果位置

- 最终长度扫描：远端 `results/qksieve_final_optimized_length_sweep_20260807`
- 本地汇总：`docs/qksieve_kernel_optimization_20260807/data/length_speed_surface.json`
- fused Value append 验证：远端 `results/qksieve_wometric_value_append_validation_20260807.json`
- 32K/64K/128K Value append A/B：远端 `results/qksieve_wometric_append_ab_{32k,64k,128k}_20260807`
- batched qMSE 微基准：远端 `results/qksieve_batched_qmse_allocation_32k_20260807.json`
- batched qMSE 32K A/B：远端 `results/qksieve_batched_qmse_ab_32k_20260807`
- stride-aware/tiled 连续张量微基准：远端 `experiments/frozen_c64_20260807/results/tiled_exact_32k128k_v1_20260807.json`
- 预分配 KV cache 消费阶段归因：远端 `experiments/frozen_c64_20260807/results/preallocated_consumer_attribution_v1_20260807.json`
- 32K 真实模型三路径 A/B：远端 `experiments/frozen_c64_20260807/results/tiled_workspace_ab_32k_smoke_20260807`
- 128K 双卡真实模型三路径 A/B：远端 `experiments/frozen_c64_20260807/results/tiled_workspace_ab_128k_dual_20260807`
- 被拒绝的 host metadata A/B：远端 `results/qksieve_hostmeta_ab_32k_20260807`

## 当前限制

- 64K 三次并行扫描存在明显运行时方差，正式论文数值需用独占 GPU、锁定时钟的重复实验替换。
- 8K/16K 的纯 QKSieve 在线路径仍不快于 dense SDPA。
- 128K 本轮质量是固定语料 PPL probe，不是跨数据集质量结论。
- 128K 三路径 A/B 因单卡 dense prefill OOM 使用两张 RTX 3090；这是双卡模型延迟，不能与单卡 H100 数字混用。
- 尚未在 H100 80GB 上按相同协议复测。

## Resident Key moment 验收

在同一 32K 前缀、同一目标 token 和同一 GPU 上顺序比较：

1. `off`：每个 query 重新从历史 K 计算二阶矩和 QK-balanced 因子；
2. `moments`：前缀生命周期内缓存 `key_mean/C_k`，每个 query 使用原求解器重算因子；
3. `factors`：现有的完整 Key 因子常驻路径，仅作速度上界对照。

`moments` 进入无损工程路径必须同时满足：36/36 层 bit allocation 和 active packed-Key hash 一致，候选计数与候选 ID 一致，目标 NLL 误差不超过 `1e-9`，subsequent-request fixed latency 至少降低 5%。`factors` 即使质量 probe 通过，也因改变浮点分解顺序而继续作为 Agent 可选项，不算严格等价主路径。
