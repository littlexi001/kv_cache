# 实验设计

## 目的

用最小但可解释的预训练实验筛选三种位置编码。首先检查代码和优化是否正常，再比较长距离检索外推。所有结论都必须相对原生 RoPE baseline 给出。

## 条件

| 名称 | 唯一变化 |
|---|---|
| `native` | 所有层、所有频率使用原生 RoPE |
| `deep_highfreq_drop` | L6-L11 的 F0-F7 不旋转 |
| `slow_rope` | 所有层、所有频率的相位乘 0.5 |
| `smooth_layer_frequency` | 按 `docs/design.md` 中的连续函数缩放相位 |

## 数据生成

- vocabulary：32,000 个离散 token。
- 每条训练序列长度：2,048。
- 每条序列包含 16 条 `FACT key VALUE value` 事实。
- 结尾包含 4 个查询，格式均为 `QUERY key ANSWER value`。
- 目标事实从全文不同位置均匀采样，其他事实构成干扰。
- filler 由可预测的局部模板构成，用于检查局部顺序建模。
- 数据在线生成。相同 `global_micro_step` 与 rank 在四个条件中产生完全相同的 token 序列。
- 四个答案位置的 loss 权重为 64；普通结构 token 权重为 1，稀疏 filler 监督权重为 0.25。这样检索答案约占总 loss 权重的四成，避免模型只学习格式而不学习检索。

## 训练

| 参数 | 数值 |
|---|---:|
| 训练 token | 20M/条件 |
| sequence length | 2,048 |
| GPUs | 2 x RTX 3090/条件 |
| micro batch | 2/GPU |
| gradient accumulation | 4 |
| global tokens/update | 32,768 |
| optimizer | AdamW |
| peak learning rate | 3e-4 |
| warmup | 50 updates |
| weight decay | 0.1 |
| precision | BF16 |
| seed | 20260807 |

这是筛选轮。若至少一个变体显示稳定优势，再把训练规模扩到 100M-500M token，并加入自然文本。

## 评测指标

### Gold answer accuracy

在 `ANSWER` 后预测的 top-1 token 是否等于真实 value。越高越好。

### Gold answer NLL

\[
\mathrm{NLL}_{gold}=-\log P(v_{gold}\mid context,query).
\]

越低越好。它比准确率更连续，可显示尚未跨越 top-1 边界的改进。

### 长度外推

在 512、1K、2K、4K、8K token 上各测试相同数量样本。训练只见过 2K；4K/8K 检查外推。

### 短程保持

比较 512/1K 的答案 NLL 和局部模板 loss。长程提高但短程明显退化的方案不通过。

### 训练稳定性与成本

保存 loss、gradient norm、tokens/s、最大显存和 wall time。若收益来自显著更大计算量，需要单独说明。

## 判定规则

对每个变体，相对 `native`：

- **通过筛选：** 4K 和 8K 的平均 Gold NLL 都降低，至少一个长度的准确率提高，并且 512/1K NLL 退化不超过 5%。
- **失败：** 4K/8K 均无改善，或短程 NLL 退化超过 10%。
- **证据不足：** 准确率方向不一致且 NLL 改变小于 2%，需要更多 seeds 或更多训练 token。

## 服务器分配

- `10.176.37.30`：GPU 4-5 跑 `deep_highfreq_drop`；GPU 6-7 跑 `slow_rope`。
- `10.176.37.31`：GPU 0-1 跑 `native`；GPU 2-3 跑 `smooth_layer_frequency`。

每个条件都使用 2 卡，避免 world size 不同造成训练动态不一致。

## 产物

- 训练入口：`src/train_synthetic_pe.py`
- 汇总入口：`src/summarize_results.py`
- 日志：`outputs/<variant>/train.jsonl`
- 评测：`outputs/<variant>/eval.jsonl`
- checkpoint：`outputs/<variant>/checkpoints/`
- 迭代记录：`notes/iteration_ledger.md`
