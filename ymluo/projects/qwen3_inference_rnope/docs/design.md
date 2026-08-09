# 推理时 RNoPE：研究设计

## 可证伪问题

对一个只用 RoPE 训练完成的 Qwen3-8B checkpoint，不更新任何权重，只在推理时让固定层的 Q/K 跳过 RoPE，能否提高 RULER-32K 的长程检索分数？

## 物理先验与主张边界

RoPE 将相对距离写入 QK 分数；NoPE 层只按当前内容表征计算 QK，因此可能避免远程证据受到相位抑制。但是，Cohere 的 RNoPE 和 SmolLM3 都是在混合架构下训练得到的。Qwen3-8B 的各层从未适应 NoPE，直接在推理时删除旋转会产生训练—推理分布偏移。

因此，本实验只检验：**训练完成的纯 RoPE 模型能否零训练迁移到混合 RoPE–NoPE 推理。** 它不等价于复现训练得到的 RNoPE，也不能否定经过继续训练的混合架构。

## 数学干预

标准 RoPE 层计算

$$
s_l(t,p)=\frac{(R_t q_l)^\top(R_p k_l)}{\sqrt{d_h}}.
$$

被指定为 NoPE 的层改为

$$
s_l^{\mathrm{NoPE}}(t,p)=\frac{q_l^\top k_l}{\sqrt{d_h}},
$$

实现上把该层传入 RoPE 的 $\cos$ 全部替换为 1、$\sin$ 全部替换为 0。Q、K、V、投影矩阵、模型权重、causal mask 和 softmax 均不改变。

## 条件

- `native_rope`：36 层全部使用 checkpoint 原生 RoPE。
- `nope_every4_offset3`：层 3、7、11、…、35 跳过 RoPE；每组 3 个 RoPE 层后接 1 个 NoPE 层，是论文最终 1:3 排列的直接推理时版本。
- `nope_every4_offset0`：层 0、4、8、…、32 跳过 RoPE；用于排除“每第 4 层”的零基/一基编号歧义。
- `nope_alternating_odd`：奇数层跳过 RoPE；对应论文分析阶段的 1:1 交替结构。

## 输出

每个样本保存完整预测、RULER 官方分数、首答案 token 正确率与 NLL、完整 gold answer 的 teacher-forced NLL、运行时间、非有限值审计和实际 NoPE 层列表。

