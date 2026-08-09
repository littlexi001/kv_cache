# 确定性采样 QKSieve：实验设计

## 1. 研究问题

1. bitmask + 固定顺序归约能否在不改变候选集合的前提下消除 CUDA 原子操作带来的不确定性？
2. `c64` 分位采样能否降低固定样本数对长度和微小数值扰动的敏感性？
3. 新路径在 32K、64K、128K 的真实模型稳态 decode 中是否仍有净加速？

## 2. 条件定义

| 条件 | 候选规则 | 候选压缩 | Value 尾项 |
|---|---|---|---|
| Full Attention | 全部历史 token | 无 | 原始 FP16 V |
| Full-topk rank16 | 完整代理分数 + PyTorch top-k | top-k 输出 | rank-16 INT4 |
| Atomic sampled s1024 | 固定 1024 样本阈值 | 全局 atomicAdd | 原子浮点归约 rank-16 INT4 |
| Deterministic sampled s1024 | 固定 1024 样本阈值 | bitmask + 前缀扫描 | 固定树归约 rank-16 INT4 |
| Deterministic c32 | `m(N)=ceil_align(32/p)` | bitmask + 前缀扫描 | 固定树归约 rank-16 INT4 |
| Deterministic c64 | `m(N)=ceil_align(64/p)` | bitmask + 前缀扫描 | 固定树归约 rank-16 INT4 |

除候选阈值与压缩方式外，模型、文本、目标 token、QK 代理、Value rank、精确 attention 预算和原始 K/V 均保持一致。

## 3. 实验 A：扫描内核等价性和确定性

### 设置

- 模型张量形状：`B=1, QH=32, KVH=8, d=128`。
- dtype：Query/scale/Value 元数据为 BF16，packed Key/Value 为 uint8。
- 长度：32K、64K、128K。
- Value rank：16；Value block：256。
- 重复次数：主要检查 50 次，c64 补充检查 20 次。
- 随机种子：20260804。

### 指标

- `candidate_sets_equal`：每个 head 按下标排序后的集合是否与 atomic 版完全相同。
- `threshold_max_abs_error`：两版本阈值最大绝对误差。
- `deterministic_mismatch_runs`：相同输入重复时任一输出张量不同的运行次数。
- `tail_*_relative_error`：因求和顺序不同造成的尾项最大误差，除以 atomic 输出最大绝对值。
- `atomic_ms` / `deterministic_ms`：单层候选扫描、压缩和尾项统计的独立 CUDA Event 时间。
- `workspace_mib`：mask 和 block partial workspace，不含已有索引。

### 通过条件

- 候选集合相同；阈值误差为 0。
- 确定性版本 50 次 mismatch 为 0。
- 尾项相对误差小于 `1e-5`。
- 确定性开销不超过 atomic 版 15%。

### 脚本和结果

- 脚本：`src/validate_qksieve_valuesketch_deterministic_20260804.py`
- 结果：远端 `results/20260804_qksieve_valuesketch_deterministic_v1/`

## 4. 实验 B：跨进程真实模型重复性

### 设置

- 模型：Qwen3-4B-Instruct。
- 数据：20 Newsgroups 的 sports、medicine 文本，重复拼接到 32K。
- 评测：teacher-forced 32 token PPL。
- sports 使用相同 seed 在两张 RTX 3090 上独立运行。
- 对比固定 `s1024` 和 `c64`。

### 指标

- PPL 质量保持率：`exp(NLL_full-NLL_sparse)`，100% 表示与 Full 相同。
- top-1 agreement：稀疏与 Full 的 argmax 一致比例。
- `KL(full || sparse)`。
- 每 head 实际候选数 mean/min/max。
- 稳态 decode ms/token。

### 解释边界

扫描内核的逐 bit 确定性只保证“相同输入张量得到相同输出”。独立进程的 prefill、PCA/eigh 和模型 GEMM 仍可能产生微小浮点差异。因此本实验关注候选数和质量对这些扰动的敏感性，不要求整个模型跨 GPU 逐 bit 相同。

### 脚本和结果

- `scripts/launch_qksieve_sorted_sampledfused_eval32_3gpu_20260804.sh`
- `scripts/launch_qksieve_deterministic_c64_eval32_3gpu_20260804.sh`
- 结果：远端同名 `results/` 目录。

## 5. 实验 C：长度质量与速度

### 设置

- 模型：Llama-3.1-8B-Instruct。
- 长度：32K、64K、131040 历史 token；最后一个长度加 32 个目标 token 后不超过 131072。
- 文本：held-out `mixed_b` 主题流。
- 评测 token：32。
- 32K 用 1 张 RTX 3090；64K/128K 用 2 张 RTX 3090，Full 和稀疏在同一设备布局上比较。
- 方法：Full Attention、deterministic c32 和 deterministic c64 rank-16 ValueSketch。

### 速度口径

- `steady_sparse_seconds_per_step`：排除一次索引构建后的真实整模型 decode 时间。
- 稳态加速：`T_full_step / T_sparse_step`。
- `fixed_sparse_overhead_seconds`：请求内 PCA、量化 Key 索引和 ValueSketch 的一次性成本。
- break-even token 数：

$$
G_{BE}=\frac{T_{fixed}}{T_{full,step}-T_{sparse,step}}.
$$

### 通过条件

- 32K 质量保持率至少 99%，稳态至少 1.7x。
- 64K 质量保持率至少 99%，稳态至少 2.3x。
- 128K 质量保持率至少 99%，稳态至少 2.8x。
- top-1 agreement 为 100%，candidate overflow 为 0。

### 脚本和结果

- 最终脚本：`scripts/launch_qksieve_v40_c64_final_profile_5gpu_20260804.sh`
- 非 profile 结果：远端 `results/20260804_qksieve_deterministic_truec64_length_v40_4gpu_v1/`
- 32K 复测与分阶段 profile：远端 `results/20260804_qksieve_v40_c64_final_profile_5gpu_v1/`

## 6. 当前已知限制

- probe token 数仍为 32，不能替代完整 LongBench/RULER。
- 不同长度使用的 GPU 数不同；每个长度内部 Full 与稀疏公平，但不能直接把绝对 ms 跨长度解释为单卡 scaling。
- 当前保留原始 FP16 K/V；尚未测 CPU/offload 或只保留压缩索引的部署方式。
- 32K 以下的固定开销交叉点仍需单独扫描。
