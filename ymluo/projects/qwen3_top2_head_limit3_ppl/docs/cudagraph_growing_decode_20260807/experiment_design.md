# 实验设计

## 研究问题

1. 动态后缀候选是否与显式切片参考实现逐元素一致？
2. 一张 CUDA Graph 能否在有效长度变化时复用？
3. 整模型真实增长 decode 是否达到固定位置测速上界？
4. 加速是否随上下文长度稳定增加？

## 设置

- 模型：Qwen3-4B-Instruct-2507，FP16。
- GPU：NVIDIA RTX 3090，单卡。
- batch：1；greedy decode。
- 前缀长度：4K、8K、16K、32K、64K。
- QKSieve：request-local QK-balanced 坐标、低比特 packed index、sampled quantile、
  最多 1,280 个前缀 token/head、rank-16 INT4 ValueSketch，不回退 Full。
- 后缀：全部精确保留。
- 整模型 Graph 包含 K/V 写入、检索、attention、MLP、logits、argmax 和位置递增。
- 随机种子：20260861 至 20260866。

## 正确性条件

attention 层通过条件：

- 动态有效长度输出与显式 K/V 切片参考输出最大绝对误差为 0。
- 普通执行与 Graph replay 的候选数和候选 ID 完全相同。
- 测试后缀长度覆盖 0、1、7、31、127、255。

整模型通过条件：

- 普通增长执行与 Graph replay 的 greedy token 序列相同。
- 记录逐步 logits 最大绝对误差。
- 若底层 CUDA 归约本身非确定，则 Graph 差异不得导致 top-1 改变，并单独披露。

## 速度指标

- `wall_ms_per_token`：CPU 发起连续 Graph replay 后同步得到的平均时间。
- `cuda_ms_per_token`：CUDA event 测得的设备时间。
- 主加速比：最佳 Full 固定位置 Graph 延迟除以 QKSieve 真实增长 Graph 延迟。

固定位置 Full Graph 不包含增长后缀管理，因此是偏向 Full 的保守基线。动态 mask
Full Graph 会触发慢 SDPA kernel，只作为失败诊断，不进入主加速比。

## 结果位置

- attention 4K：远端 `results/dynamic_suffix_graph_4k_smoke_20260807/results.json`
- attention 64K：远端 `results/dynamic_suffix_graph_64k_20260807/results.json`
- 整模型 4K：远端 `results/model_growing_graph_qk_4k_smoke_r2_20260807/summary.json`
- 整模型 8K/16K/32K：远端对应 `model_growing_graph_qk_*_20260807/summary.json`
- 整模型 64K：远端 `results/model_growing_graph_qk_64k_20260807/summary.json`
- Full 动态 mask 诊断：远端 `results/model_growing_graph_both_64k_r2_20260807/summary.json`

## 失败判据

- 任一候选或输出不等价：动态后缀实现失败。
- Graph 需要按 token 重捕获：固定地址设计失败。
- 64K 增长延迟显著高于固定位置 QKSieve 上界：cache 写入或动态长度仍是瓶颈。
- 速度仅来自不公平 Full 实现：不能形成论文速度结论。
