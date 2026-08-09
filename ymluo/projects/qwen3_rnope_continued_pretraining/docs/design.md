# Qwen3-8B RNoPE 继续预训练：研究设计

## 可证伪问题

Qwen3-8B 在纯 RoPE checkpoint 上直接进行推理时 RNoPE 会显著退化。若固定层 3、7、11、…、35 为 NoPE，并继续做一小轮长文本 next-token training，模型能否适应新的位置机制并恢复长程检索？

## 干预

NoPE 层将 RoPE 的旋转替换为单位旋转：`cos=1, sin=0`。其余层、causal mask、attention softmax 和基础权重保持不变。由于 8×RTX 3090 不适合直接保存 8B 全参数 Adam 状态，本轮使用 NF4 基座和 LoRA 更新 Q/K/V/O 及 MLP 投影；它检验的是低成本适配能否恢复，而不是完整复现全参数 RNoPE 预训练。

## 数据和边界

训练数据是服务器已有的真实 PG19 长文本 token stream，不使用 RULER 标签。训练约 10M token，序列长度优先为 8K；若 8K 分布式 smoke OOM，则自动回退到 4K。测试使用未参与训练的 26 条 RULER-32K。

