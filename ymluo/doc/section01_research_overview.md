# Section 1 — 研究总览

> 最近同步日期：2026-05-23

## 研究主题

新增日期：2026-05-14

这个工作区围绕一个核心问题展开：**能否把长上下文 KV cache 看成一个可索引、可压缩、可检索的记忆系统，而不是每次 decode 都密集扫描的扁平 token 序列。**

当前主要方向包括：

- 在精确 attention 前先做 block/chunk 级候选召回。
- 对旧 KV memory 做学习式压缩，同时保留 recent tokens 和 anchor tokens。
- 用 attention energy 估计在有限 loss 影响下可以丢弃多少上下文。
- profile K-cache 的数值、相邻 token delta、范数、pairwise cosine，理解不同 layer/head 到底存了什么。
- 用可控 synthetic retrieval 任务验证 anchor-only KV、U-Net mask schedule 与 answer-only training 是否能学到可检索记忆。
- 用 interval subsequence next-token 任务检查 U-Net stride schedule 对局部/跨步序列规律的建模能力，并导出 attention scores 做层/head 诊断。

更完整的搜索系统类比和动机见：

```text
KVCache_Indexing_Knowledge_Retrieval_2026-05-09.md
```

下面的命令默认从仓库根目录 `kv_cache` 运行。如果已经 `cd ymluo`，去掉命令里的 `ymluo/` 前缀。

## 目录结构

新增日期：2026-05-14

| 路径 | 作用 |
| --- | --- |
| `KVCache_Indexing_Knowledge_Retrieval_2026-05-09.md` | 把搜索、向量索引、层级结构、知识图谱等思想映射到 KV-cache lookup 的研究笔记。 |
| `projects/qwen3_kcache_cosine_heatmap` | Qwen3-0.6B 的 K-cache token-token cosine 热力图分析。 |
| `projects/qwen3_kcache_norm_analysis` | Qwen3-0.6B 的 K-cache norm、attention energy、pruning loss/PPL 分析。 |
| `projects/qwen3_kcache_avg_topk` | 推理期 sparse decode 实验：用 K-cache block 平均向量做 top-k block 选择。 |
| `projects/qwen3_chunk_routing` | Qwen3-0.6B chunk attention 训练框架，包含 `baseline`、`oracle`、`router` 模式。 |
| `projects/qwen3_unet_synthetic_retrieval` | 可控 synthetic retrieval 评估与 answer-only 训练。 |
| `projects/qwen3_interval_subseq_retrieval` | arithmetic subsequence next-token 训练和 attention score dump。 |
| `projects/pyramid_kv_compression` | 继续预训练实验：用学习到的 summary 替换旧的中间层 KV blocks。 |
| `projects/qwen3_kcache_value_delta_analysis` | Qwen3-8B 的 K-cache 取值、范数、相邻 token delta 分布分析。 |
| `projects/qwen3_moe_attention_cluster` | MoE 专家选择性实验：用 attention 权重引导 router 做 token 聚类（合成数据）。 |
| `projects/qwen15_moe_real_attention_cluster` | 同上核心思想在真实 Qwen1.5-MoE + DCLM 文本上的迁移实验。 |
| `logs/` | 历史日志或 workspace snapshot。 |
| `utils/` | 共享工具目录。 |

每个项目都有自己的 README 和 scripts；这些 section 文件只做总览和关键结论记录。

## 推荐阅读顺序

新增日期：2026-05-14；最近同步日期：2026-05-23

1. **本节** — 理解整体研究动机和目录结构。
2. [[section02_kcache_cosine_analysis]] — 理解 K-cache pairwise cosine 及压缩含义。
3. [[section03_kcache_norm_analysis]] — 查看 attention energy pruning 的当前证据。
4. [[section04_sparse_decode]] — 查看可运行的 block-selection sparse decode baseline。
5. [[section05_chunk_routing]] — 查看 oracle/router sparse chunk training。
6. [[section06_unet_synthetic_retrieval]] — 查看可控 synthetic retrieval 评估和 answer-only 训练。
7. [[section07_interval_subseq_retrieval]] — 查看 arithmetic subsequence next-token 训练和 attention score dump。
8. [[section08_pyramid_kv_compression]] — 在跑 compressed KV 继续预训练前先看风险记录。
9. [[section09_kcache_value_delta]] — 查看 K-cache 数值和 delta 分布 profiling。
10. [[section10_moe_selectivity]] — MoE 专家选择性合成数据实验。
11. [[section11_common_params]] — 常用默认参数与实践注意事项。
