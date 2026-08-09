# 深层 head-group × RoPE 频率删除：问题与模型

## 可证伪假设

在 Qwen3-8B 的长上下文推理中，深层少数 GQA head-group 的少数 RoPE 频率对会因远距离相位失配而压低正确证据。只把这些位置旋转改成单位旋转、同时保留原始 Q/K 内容，可能在不重新训练的情况下提高 RULER-32K 分数。

若所有局部删除都不优于原生 RoPE，或仅在筛选样本上提高而在完整 26 条样本上退化，则当前假设或当前搜索粒度不成立。

## 为什么搜索单位是 head-group

Qwen3-8B 有 36 层、32 个 Query head、8 个 KV head，head dimension 为 128。每个 KV head 被连续 4 个 Query head 共享。因此，本实验把一个可解释的 head 单元定义为：

- KV head `g`；
- Query heads `4g, 4g+1, 4g+2, 4g+3`；
- `g = 0,...,7`。

对这个单元的 Q 与 K 同时执行相同的频率删除，避免只改 Q、不改共享 K 的不对称干预。

## “删除频率”的精确定义

Qwen3 的一个 head 有 64 个二维 RoPE 频率对。Transformers 使用 split-half 配对：频率 `i` 对应维度 `i` 与 `i+64`。

标准 RoPE 为

$$
q'_{l,h,i}=R(\phi_{t,i})q_{l,h,i},\qquad
k'_{l,g,i}=R(\phi_{p,i})k_{l,g,i}.
$$

若三元组 `(layer l, KV head-group g, frequency i)` 被选中，则改为

$$
q'_{l,h,i}=q_{l,h,i},\qquad
k'_{l,g,i}=k_{l,g,i},\qquad h\in\{4g,\ldots,4g+3\}.
$$

其他层、head 和频率完全保持原生 RoPE。这里不把 Q/K 数值置零，因此删除的是位置旋转，不是语义维度。

## 预期机制

单个频率对对 QK 的贡献为

$$
s_i(\Delta)=A_i\cos(\Delta\omega_i)+B_i\sin(\Delta\omega_i).
$$

删除该频率的旋转后，它变为距离无关的内容项

$$
s_i^{\mathrm{NoPE}}=A_i.
$$

若远距离时原生项小于内容项，即

$$
s_i(\Delta)<A_i,
$$

该删除可能提高证据 QK；反之则会破坏模型已学到的位置模式。因此预期只有少数深层 head-group 与频带有益，而不是删除越多越好。

## 声明边界

本实验是冻结权重的推理时干预。它能检验局部 RoPE 相位是否存在可利用的有害区域，但不能证明这些区域在训练后仍然最优，也不能把 26 条 RULER 样本上的提高直接外推到全部长上下文任务。

## 稳定方案阶段：连续频率缩放

二值删除是连续缩放的端点。对选定的 layer `l`、GQA head-group `g` 和频率 `i`，定义

$$
q'_{l,g,i}(p)=R(\alpha_{l,g,i}p\omega_i)q_{l,g,i},\qquad
k'_{l,g,i}(p)=R(\alpha_{l,g,i}p\omega_i)k_{l,g,i},
$$

其中 `0 <= alpha <= 1`：

- `alpha=1` 是原生 RoPE；
- `alpha=0` 删除位置旋转但保留 Q/K 内容；
- 中间值减慢旋转频率。

对应 QK 项仍为

$$
s_i(\Delta;\alpha)=A_i\cos(\alpha\Delta\omega_i)
+B_i\sin(\alpha\Delta\omega_i),
$$

所以它仍只依赖相对距离，不引入绝对位置交叉项，也不破坏 KV cache。稳定方案阶段比较 `F46`、`F47`、`F40–47` 频带及两处干预的多个 alpha，并在新 RULER seeds 上选择；seed 42 不再参与选择。
