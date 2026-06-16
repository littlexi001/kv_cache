# KV Bucket and Expert Bucket Co-compute: Results

## Current Evidence

1. Qwen3-0.6B 的实际 QK top-k 图具有高于距离与 attention-target popularity 匹配基线的局部两跳闭包。
2. 相似 query 的历史 retrieval set 比位置匹配随机 query 更相似。
3. `W_Q^T W_K` 本身高度非对称，因此闭包来自训练后数据流形、表示、RoPE 与 attention 计算的联合结果，而不是一个全局 PSD kernel。
4. 既有 pre-router cluster attention 在 synthetic 上可保持约 `93.9%` NTP，并只保留约 `25%-32%` 历史 KV。
5. 独立 K-space cluster index 已在真实预训练模型 inference 上表现出质量/候选比例 Pareto 收益。

## What Has Not Been Tested

1. 真实数据训练中的 shared KV/expert bucket；
2. head-level 相比 token-level 的受控收益；
3. center removal 对 learned bucket 的因果作用；
4. Shared bucket 相比 Separate buckets 的收益；
5. 真实系统延迟与显存访存收益。

因此当前结论仍是：几何基础存在，核心架构假设尚未验证。
