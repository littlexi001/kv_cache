# Qwen3-8B RNoPE 继续预训练：实验协议

## 设置

- 基座：Qwen3-8B；NF4 加载，BF16 计算。
- 位置机制：36 层中第 3、7、11、…、35 层为 NoPE，其余层使用原生 RoPE。
- 可训练参数：LoRA rank 16，alpha 32；目标为 `q_proj/k_proj/v_proj/o_proj/gate_proj/up_proj/down_proj`。
- 目标：标准 causal next-token cross entropy。
- 数据：PG19 的 9,899,520 个真实文本 token；尾部 32 个不重叠序列只用于 validation。
- 训练量：约 10M token；8K 时 153 steps，4K 回退时 306 steps；8 卡、每卡 batch 1。
- 优化器：paged AdamW 8-bit；学习率 $2\times10^{-4}$；cosine schedule；5% warmup。
- 评测：训练前后 PG19 validation loss/PPL；训练后 RULER-32K 13 任务 × 2 条，比较适配器在 `native_rope` 和训练使用的 `nope_every4_offset3` 下的表现。零训练参照来自上一轮严格相同 RULER 样本。

## 执行契约

1. 使用 8 卡 `torchrun` 完成 1 step smoke，检查 DDP、显存、loss 和保存。
2. 若 8K smoke 成功，则跑 153 steps；若失败则自动以 4K 重试并跑 306 steps。
3. 保存训练曲线、pre/post validation 指标、checkpoint 和最终 adapter。
4. 训练成功后自动释放训练进程，再在 8 卡上分片运行 RULER-32K。
5. 每个阶段分别写入 `done`/`failed` 标记；最外层由 `launcher.done`/`launcher.failed` 表示完整管线状态。

## 判据

- 适配有效：NoPE 条件的 RULER 官方分数明显高于零训练的 13.65%，同时 validation PPL 不发生灾难性上升。
- 适配不足：loss 稳定下降但 RULER 仍接近零训练结果。
- 适配有害：validation PPL 或 RULER 进一步退化。
- 本轮证据不能回答：全参数继续预训练、更大训练 token 数或加入 document masking 后能否成功。

## 追加训练量扫描

第一轮10M token后，RNoPE 的 RULER-32K 从13.65%恢复到49.29%，但仍低于原生 RoPE 的85.19%。为区分“训练不足”和“当前方法上限”，从该 adapter继续训练约90M token，使累计训练量达到约100M。保持模型、4K序列、PG19 token stream、NoPE层和LoRA参数完全相同；学习率重启为 $5\times10^{-5}$。每追加约10M token保存一个 adapter，训练结束后对所有 checkpoint运行相同26条 RULER-32K。若分数持续上升并超过85.19%，支持训练不足解释；若较早形成平台，则当前数据、LoRA容量或层配置是更可能的瓶颈。
