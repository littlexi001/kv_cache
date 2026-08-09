# 条件矩残差修正：实验设计

## 研究问题

交错留出估计的解析 Wiener 系数，能否消除完整强度条件修正在少数请求上的过修正，同时保留其对层输出 KL/L2 的收益？

## 第一阶段：真实 Q/K/V 层输出

- 模型：Qwen3-4B-Instruct 与 Llama-3.1-8B-Instruct。
- 主题与长度：Qwen 体育/医学 32K、96K；Llama 宗教 4K、计算机 128K。
- 对比：残差均值、完整强度条件矩 d8/d16、Wiener 条件矩 d8/d16。
- 固定：相同 QK proxy、相同 top-k、相同 ValueSketch16 INT4、相同尾部分区。
- 主要指标：经过 `o_proj` 后的相对 L2，越低越好。
- 诊断：每个 KV head 的 `gamma`、留出集平方误差下降率、最差请求。

通过条件：Wiener d8 的宏平均和最差相对 L2 均不差于残差均值，并优于固定强度 d8；至少两个模型上方向一致。

失败条件：任一模型的最差误差明显增大，或 `gamma` 与留出收益无关。此时条件矩不能替代 QKSieve-Robust。

证据不足：样本数不足以覆盖至少两个模型、两个主题和三个长度区域，或 exact-QK oracle 已因固定 top-k 明显失败。

## 第二阶段：闭环质量

仅在第一阶段通过后执行。先使用 Qwen 4K/32K/96K 六个小样本，比较 Full、Robust、Wiener 条件矩；再用 Llama 独立样本。报告 PPL、NLL 差、KL、top-1 一致率、active token/head 与辅助索引比例。

## 第三阶段：系统实现

CUDA 融合顺序：packed score 扫描 -> 候选选择 -> 8 维加权矩 -> Value 尾部合并。先比较 Python 参考与 CUDA 输出，最大绝对误差和相对误差均通过既有 FP32 累加容差后，再独立测 attention 子系统和整模型稳态 decode。不得用延迟分解代替实测。

## 产物

- 分析脚本：`src/analyze_qksieve_tail_partition_calibration_20260803.py`
- 单测：`tests/test_qksieve_tail_partition_calibration.py`
- 第一阶段结果：远端 `results/20260804_conditional_wiener_generality_6gpu_v1`
- 后续闭环与 CUDA 结果将在通过前一阶段后创建独立目录。
