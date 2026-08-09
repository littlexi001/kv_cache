# Qwen3-8B RNoPE 继续预训练：结果记录

第一轮使用4K序列训练10.027M token。PG19 validation PPL 从29.20降至13.32；RNoPE 的 RULER-32K 从零训练的13.65%提高到49.29%，但仍低于原生 RoPE 的85.19%。这说明继续训练能够适配部分结构变化，但当前训练量不足以达到 baseline。

追加实验从10M adapter继续训练至累计约100M token，并每约10M token保存和评测一个 checkpoint。待实验完成后在此记录训练量—RULER曲线、最高点和是否超过85.19%。
