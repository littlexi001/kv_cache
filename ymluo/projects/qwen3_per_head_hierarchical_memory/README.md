# Qwen3 Per-head Hierarchical Memory

这个项目验证“每个 attention head 拥有独立多级记忆”的推理时方案。

- L0 / hot：实际暴露给稀疏 attention 的位置，每个 query head 硬上限 500 token。
- L1 / warm：每个 head 独立维护的候选位置，默认 4096 个。
- L2 / cold：完整历史的共享分块索引，默认 64 token/block、每次短列 64 blocks。
- 外部检索不读取 attention QK；完整 attention 只作为离线 Top-2% Oracle 评分器。

三种等容量策略：

1. `sink_recent_500`：sink + 最近 token。
2. `flat_function_500`：直接从完整历史检索。
3. `hier_function_500`：L2 block shortlist → 持久 L1 → L0 promotion。

置信度门控可只让指定功能的 head 使用远程检索，其余 head 自动退化为 recent-500：

```bash
cd /home/fdong/ymluo/projects/qwen3_per_head_hierarchical_memory
CUDA_VISIBLE_DEVICES=0 \
PROMOTION_POLICY=confidence_gated \
PROMOTION_CATEGORIES=semantic_evidence \
L0_RECENT_TOKENS=448 \
bash scripts/run_sparse_ppl_server.sh
```

当前结论：分层状态、448 个独立 head、L0≤500 和真实稀疏 attention 质量路径均已跑通。精确 Oracle Top-2% 在 512-token 长测上将 PPL 从 25.3298 降至 25.1366，但当前简单外部检索器还不能复现这项收益。详见 [实验报告](docs/results_20260716_zh.md)。

当前 PPL 实验仍物理保存完整 KV，只在 attention 计算前注入 per-head L0 mask，因此验证的是质量而非显存/速度。真正部署还需要 page/block KV compaction。
