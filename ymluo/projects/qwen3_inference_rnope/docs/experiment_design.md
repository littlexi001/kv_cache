# 推理时 RNoPE：实验协议

## 实验设置

- 模型：Qwen3-8B，NF4 权重、BF16 计算，权重完全冻结。
- 数据：RULER-32K，13 个任务，每任务 2 条，共 26 条；所有条件使用严格相同的 prompt。
- 解码：greedy，最多使用样本规定的生成长度，上限 128 token。
- 主指标：26 条样本的 RULER 官方分数均值，以及 13 个任务的宏平均。
- 辅助指标：首答案 token 准确率、首 token NLL、完整答案 NLL、相对 `native_rope` 的逐样本变化、改善/退化/不变数量。
- 随机性：模型为 eval 模式，greedy 解码；bootstrap 只用于报告逐样本差值的 95% 区间，固定 seed 20260804。
- 计算：服务器 8 张 RTX 3090；每卡一个 shard，每个 shard 处理 3–4 条样本及全部条件。

## 实现契约

输入是同一 Qwen3-8B checkpoint、同一 tokenizer 和同一组 RULER prompts。对每个样本和每个条件：

1. 激活该条件对应的 NoPE 层集合。
2. 从空 KV cache 开始，对完整 prompt 重新 prefill。
3. 在 NoPE 层用单位旋转，即 `cos=1, sin=0`；其他层使用原始 RoPE。
4. 记录问题末尾的首答案 token 分布。
5. greedy 生成完整答案并计算 RULER 官方分数。
6. 回退到 prompt cache，teacher-force 正确答案并计算平均 NLL。
7. 保存逐样本 JSONL；任何 NaN/Inf、层集合不一致或样本缺失都标为失败。

## Smoke 与扩展条件

先在 GPU0 上对 1 条已知 RULER 样本运行 `native_rope`、`native_replay` 和 `nope_every4_offset3`：

- `native_replay` 与 `native_rope` 的首 token logits 最大绝对误差必须小于 $10^{-4}$；
- 所有 logits、NLL 和分数必须为有限值；
- 每个条件必须写出一条完整结果。

Smoke 只检查实现，不按性能决定是否扩展。通过后，启动 8 卡正式实验。

## 支持、反对与证据不足

- 支持零训练迁移：`nope_every4_offset3` 的平均官方分数高于原生 RoPE，且配对区间主要位于 0 以上，首 token 和完整答案 NLL 不出现系统性退化。
- 反对零训练迁移：官方分数下降，或检索任务偶有改善但整体 NLL/生成显著恶化。
- 证据不足：差值接近 0、区间宽且改善/退化样本混杂；需要增加 seeds 或进行继续训练后再判断。

