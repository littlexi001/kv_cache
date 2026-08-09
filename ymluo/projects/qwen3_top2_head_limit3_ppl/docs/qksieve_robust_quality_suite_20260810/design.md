# 设计

## 问题

冻结后的 QKSieve-Robust 已有 32K--128K PPL、MHA attention 与真实 decode
证据，但正式 RULER 和跨模型结果仍来自旧 selector、旧 `alpha=1.0` 或小规模
诊断。它们不能直接作为冻结主方法的论文结果。

## 假设

固定同一个数值契约后，若方法确实通用，则它应同时满足：

1. 在 RULER 的检索、变量跟踪、词抽取和 QA 任务上保持 Full KV；
2. 从 Llama 转移到 Qwen 和 Mistral 时不需要重新训练或调参；
3. 每条稀疏结果都真实执行 240-bit Key 索引、rank-16 INT4 ValueSketch、
   `alpha=0.5`、最多 512 个阈值样本，并且没有 Full fallback。

## 实现

`qksieve_robust_contract_20260810.py` 是机器可检查的冻结契约。正式汇总器会
同时检查配置采样数和运行时有效采样数，避免把配置的 1,280--6,656 错写成
真实扫描量；冻结运行时的上限始终为 512。

RULER 使用 13 个任务和 4K/8K/16K/32K/64K/128K 六个长度。跨模型
LongBench 使用同一批、预先固定 offset 的 160 个样本，模型为
Llama-3.1-8B-Instruct、Qwen3-4B-Instruct 和 Mistral-7B-Instruct。

## 决策规则

实验不用于修改预算、位宽、ValueSketch 或 `alpha`。若某模型或任务失败，
结果作为失败边界报告；只允许修复不改变冻结数值定义的兼容性或实现错误。
