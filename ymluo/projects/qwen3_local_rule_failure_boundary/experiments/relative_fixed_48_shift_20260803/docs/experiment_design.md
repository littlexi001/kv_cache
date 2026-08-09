# 实验协议

## 输入与参数

- 模型：Qwen3-8B，FP16，SDPA，`global_max_position=130000`、`original_max_position_embeddings=40960`，沿用旧实验的 YaRN/RoPE scaling，读取 greedy next-token distribution。
- 原样例：seed 0，英文单 token 链 `river -> window -> basket`，旧 body 长度 47，legacy full2 Query。
- 位移：48 token。
- 新增文本：由旧项目的确定性 plain-filler 生成器产生；选择一个恰好 48-token、起始带词界且末尾带换行的连续片段。三组中的两组加长条件复用完全相同的 token IDs。
- Query 位置：以 prompt 最后一个 token（`Answer: ` 后的空格）为实际 next-token Query。

## 位置不变量

对每个条件保存：四个证据 token 的绝对位置、Query 位置，以及 `Query position - evidence position`。

通过条件：

- `gap_plus_48` 的每个证据相对距离等于原始距离加 48；
- `co_shift_plus_48` 的每个证据相对距离与原始条件完全相等；
- 两个加长条件的 prompt token 数相同；
- 两个加长条件插入的 48 个 token IDs 完全相同。

任意一条不满足则实验标记为无效，不解释模型结果。

## 指标

- `P(basket)`、Gold NLL/PPL；
- 最强错误 token 及其概率；
- 输出 margin：`log P(basket) - log P(strongest wrong)`；
- 四个原子证据 token 的全模型平均 attention mass；
- L30--L33 的证据 attention mass；
- 逐层逐 head 的证据 post-RoPE QK logit 与 attention mass，保存为原始 JSON 供复核。

## 判读

- `gap_plus_48` 明显退化而 `co_shift_plus_48` 接近原始：支持相对距离是主要变量。
- 两种加长条件都退化：说明仅保持相对位置不足，因果前缀引起的内容状态变化不可忽略。
- `co_shift_plus_48` 比 `gap_plus_48` 好但未恢复原始：相对距离与前缀内容状态共同作用。
